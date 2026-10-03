"""
models/metadata_baseline.py
Quality/Metadata-Only Classifier for FORGE-Lite sanity checking.

Per newimp.md Phase B §2:
  "Test a quality/metadata-only classifier. If it predicts the label strongly,
   inspect dataset confounds; that result is a WARNING, not independent forgery evidence."

This classifier uses ONLY non-visual metadata features:
  - Blur score, brightness, contrast, face_area_ratio, compression quality
  - Head pose (yaw, pitch), video duration, codec type
  - NO image content, NO CNN features, NO temporal patterns

If this classifier achieves high accuracy on your dataset:
  → The real/fake split may be confounded by recording/processing conditions
  → "All real videos are clean; all fakes are compressed" would be a shortcut
  → Your detector may be learning codec/quality shortcuts, not forgery evidence

Usage:
    baseline = MetadataOnlyBaseline()
    baseline.fit(train_records)
    report = baseline.evaluate(test_records)
    if report["is_confounded"]:
        raise DatasetConfoundWarning(...)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction from manifest records
# ─────────────────────────────────────────────────────────────────────────────

def extract_metadata_features(record: dict) -> np.ndarray:
    """
    Extract numeric metadata features from a manifest record.
    These features MUST NOT include any image content.

    Returns: (10,) float array of quality/metadata features
    """
    tq = record.get("track_quality", {}) or {}
    frs = record.get("frame_records", []) or []

    # Aggregate from frame records if available
    if frs:
        blur = float(np.mean([f.get("blur_score", 0.5) for f in frs]))
        bright = float(np.mean([f.get("brightness", 0.5) for f in frs]))
        contrast = float(np.mean([f.get("contrast", 0.5) for f in frs]))
        area = float(np.mean([f.get("face_area_ratio", 0.1) for f in frs]))
        yaw = float(np.mean([abs(f.get("pose_yaw_deg", 0.0)) for f in frs])) / 90.0
        pitch = float(np.mean([abs(f.get("pose_pitch_deg", 0.0)) for f in frs])) / 90.0
        comp = float(np.mean([f.get("composite_quality", 0.5) for f in frs]))
        lmk = float(np.mean([f.get("landmark_confidence", 0.8) for f in frs]))
        n_frames = min(len(frs) / 300.0, 1.0)  # normalize by 300 max
        occlusion = float(np.mean([f.get("occlusion", 0.0) for f in frs]))
    else:
        blur = float(tq.get("mean_blur", 0.5))
        bright = float(tq.get("mean_brightness", 0.5))
        contrast = 0.5
        area = float(tq.get("mean_face_area_ratio", 0.1))
        yaw = min(abs(float(tq.get("max_pose_yaw", 0.0))) / 90.0, 1.0)
        pitch = 0.0
        comp = float(tq.get("mean_compression", 0.8))
        lmk = 1.0 - float(tq.get("landmark_jitter", 0.0))
        n_frames = 0.0
        occlusion = float(tq.get("mean_occlusion", 0.0))

    # Duration feature (capped at 300s)
    duration = min(float(record.get("duration_seconds", 0.0)) / 300.0, 1.0)

    return np.array([
        blur, bright, contrast, area, yaw, pitch,
        comp, lmk, occlusion, n_frames, duration,
    ], dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Metadata-Only Baseline Classifier
# ─────────────────────────────────────────────────────────────────────────────

class MetadataOnlyBaseline:
    """
    Logistic regression trained ONLY on quality/metadata features — no image content.

    This is a sanity check, NOT a detection system.

    Interpretation:
      AUROC > 0.75 on metadata alone  → dataset is likely confounded
      AUROC 0.60–0.75                 → partial confound, investigate
      AUROC < 0.60                    → metadata alone is not predictive (good)

    The confound threshold can be set in the config.
    """

    CONFOUND_AUROC_THRESHOLD = 0.72  # newimp.md: "inspect dataset confounds"
    STRONG_CONFOUND_THRESHOLD = 0.80  # clear confound — real/fake split is biased

    def __init__(
        self,
        confound_threshold: float = CONFOUND_AUROC_THRESHOLD,
        C: float = 1.0,
        max_iter: int = 1000,
        random_state: int = 42,
    ) -> None:
        self.confound_threshold = confound_threshold
        self.C = C
        self.max_iter = max_iter
        self.random_state = random_state
        self._model = None
        self._feature_names = [
            "blur", "brightness", "contrast", "face_area_ratio",
            "pose_yaw_norm", "pose_pitch_norm", "compression_quality",
            "landmark_confidence", "occlusion", "n_frames_norm", "duration_norm",
        ]
        self._is_fitted = False

    def fit(self, records: list[dict]) -> "MetadataOnlyBaseline":
        """
        Train metadata-only logistic regression on training records.

        Args:
            records: manifest records with track_quality / frame_records
        """
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import StandardScaler
            from sklearn.pipeline import Pipeline
        except ImportError:
            logger.error("scikit-learn required for MetadataOnlyBaseline")
            return self

        X, y = self._extract(records)
        if len(X) == 0:
            logger.warning("No records to fit MetadataOnlyBaseline")
            return self

        logger.info(
            "MetadataOnlyBaseline: fitting on %d records (%d real, %d fake) "
            "using ONLY quality/metadata features (no image content)",
            len(X), int((y == 0).sum()), int((y == 1).sum())
        )

        self._model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                C=self.C, max_iter=self.max_iter,
                random_state=self.random_state, class_weight="balanced"
            )),
        ])
        self._model.fit(X, y)
        self._is_fitted = True
        return self

    def predict_proba(self, records: list[dict]) -> np.ndarray:
        """Return fake probability scores for records."""
        if not self._is_fitted or self._model is None:
            raise RuntimeError("Call fit() before predict_proba()")
        X, _ = self._extract(records)
        return self._model.predict_proba(X)[:, 1]

    def evaluate(self, records: list[dict]) -> dict[str, Any]:
        """
        Evaluate on test records and return confound analysis report.

        Returns dict with:
          auroc, confound_level, is_confounded, feature_importances,
          interpretation, majority_baseline
        """
        from sklearn.metrics import roc_auc_score

        X, y = self._extract(records)
        if len(X) < 10 or len(set(y.tolist())) < 2:
            return {"error": "insufficient data for evaluation", "is_confounded": False}

        # Majority baseline
        majority_rate = max((y == 0).mean(), (y == 1).mean())

        # Metadata-only prediction
        scores = self.predict_proba(records)
        try:
            auroc = float(roc_auc_score(y, scores))
        except Exception:
            auroc = float("nan")

        # Confound level classification
        if auroc >= self.STRONG_CONFOUND_THRESHOLD:
            confound_level = "STRONG"
            is_confounded = True
            interpretation = (
                f"AUROC={auroc:.3f} on metadata alone is STRONG evidence of dataset confounds. "
                "Real/fake split is likely biased by recording conditions (e.g., all fakes are "
                "compressed, all reals are clean). Your detector may be learning codec shortcuts."
            )
        elif auroc >= self.confound_threshold:
            confound_level = "MODERATE"
            is_confounded = True
            interpretation = (
                f"AUROC={auroc:.3f} on metadata alone suggests PARTIAL confound. "
                "Investigate compression, resolution, or recording-device distribution "
                "differences between real and fake splits."
            )
        else:
            confound_level = "NONE"
            is_confounded = False
            interpretation = (
                f"AUROC={auroc:.3f} on metadata alone is below threshold ({self.confound_threshold}). "
                "Metadata features are not strongly predictive — no obvious confound detected."
            )

        # Feature importances (from logistic regression coefficients)
        feature_importances: dict[str, float] = {}
        if self._model is not None:
            clf = self._model.named_steps.get("clf")
            if hasattr(clf, "coef_"):
                coefs = clf.coef_[0]
                feature_importances = {
                    name: float(abs(coef))
                    for name, coef in zip(self._feature_names, coefs)
                }

        report = {
            "metadata_only_auroc": auroc,
            "majority_baseline": float(majority_rate),
            "confound_level": confound_level,
            "is_confounded": is_confounded,
            "interpretation": interpretation,
            "feature_importances": dict(
                sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)
            ),
            "n_records": len(records),
            "n_real": int((y == 0).sum()),
            "n_fake": int((y == 1).sum()),
            "confound_threshold": self.confound_threshold,
        }

        # Log prominently
        if is_confounded:
            logger.warning(
                "⚠️  DATASET CONFOUND WARNING [%s]: metadata-only AUROC=%.3f. %s",
                confound_level, auroc, interpretation
            )
        else:
            logger.info("Metadata confound check OK: AUROC=%.3f [%s]", auroc, confound_level)

        return report

    def save(self, output_dir: str) -> None:
        """Save model and confound report."""
        import pickle
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        if self._model is not None:
            with open(out / "metadata_baseline.pkl", "wb") as f:
                pickle.dump(self._model, f)
        logger.info("MetadataOnlyBaseline saved to %s", out)

    def _extract(self, records: list[dict]) -> tuple[np.ndarray, np.ndarray]:
        """Extract features and labels from records."""
        X, y = [], []
        for r in records:
            try:
                feat = extract_metadata_features(r)
                label = int(r.get("label_int", r.get("label", 0)))
                X.append(feat)
                y.append(label)
            except Exception as e:
                logger.debug("Failed to extract metadata for record %s: %s",
                             r.get("video_id", "?"), e)
        if not X:
            return np.zeros((0, len(self._feature_names))), np.zeros(0, dtype=int)
        return np.array(X), np.array(y, dtype=int)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(
        description="FORGE-Lite Metadata Confound Check. "
                    "If this classifier achieves high AUROC, your dataset is confounded."
    )
    parser.add_argument("--manifest", required=True, help="Path to manifest JSON/JSONL")
    parser.add_argument("--output-dir", default="reports/metadata_baseline")
    parser.add_argument("--confound-threshold", type=float, default=0.72)
    parser.add_argument("--simulate", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    baseline = MetadataOnlyBaseline(confound_threshold=args.confound_threshold)

    if args.simulate:
        # Simulate records with and without confound
        rng = np.random.default_rng(42)
        n = 500
        records: list[dict] = []
        for i in range(n):
            label = int(i % 2)
            # Simulate confound: fakes are more compressed
            comp = rng.uniform(0.2, 0.5) if label == 1 else rng.uniform(0.7, 1.0)
            records.append({
                "video_id": f"vid_{i}",
                "label_int": label,
                "track_quality": {
                    "mean_blur": float(rng.uniform(0.3, 0.9)),
                    "mean_brightness": float(rng.uniform(0.3, 0.8)),
                    "mean_face_area_ratio": float(rng.uniform(0.05, 0.4)),
                    "mean_compression": float(comp),
                    "max_pose_yaw": float(rng.uniform(-30, 30)),
                    "landmark_jitter": float(rng.uniform(0.0, 0.2)),
                    "mean_occlusion": float(rng.uniform(0.0, 0.3)),
                    "continuity": float(rng.uniform(0.7, 1.0)),
                },
                "duration_seconds": float(rng.choice([3.0, 8.0, 20.0])),
            })

        # Train/test split
        train = records[:400]
        test = records[400:]
        baseline.fit(train)
        report = baseline.evaluate(test)

    else:
        import json
        manifest = Path(args.manifest)
        if manifest.suffix == ".jsonl":
            with open(manifest, encoding="utf-8") as f:
                all_records = [json.loads(line) for line in f if line.strip()]
        else:
            with open(manifest, encoding="utf-8") as f:
                all_records = json.load(f)

        n = len(all_records)
        split_idx = int(n * 0.8)
        train_records = all_records[:split_idx]
        test_records = all_records[split_idx:]

        baseline.fit(train_records)
        report = baseline.evaluate(test_records)

    # Print and save
    print("\n" + "═" * 60)
    print("  FORGE-Lite Metadata Confound Check")
    print("═" * 60)
    print(f"  Metadata-only AUROC : {report.get('metadata_only_auroc', float('nan')):.4f}")
    print(f"  Majority baseline   : {report.get('majority_baseline', float('nan'))*100:.1f}%")
    print(f"  Confound level      : {report.get('confound_level', 'N/A')}")
    print(f"  Interpretation      : {report.get('interpretation', '')}")
    print("\n  Top metadata predictors:")
    for feat, imp in list(report.get("feature_importances", {}).items())[:5]:
        print(f"    {feat}: {imp:.4f}")
    print("═" * 60 + "\n")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "confound_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    baseline.save(args.output_dir)


if __name__ == "__main__":
    main()
