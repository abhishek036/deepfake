"""
evaluation/diagnostic_export.py
Full per-video diagnostic export as required by newimp.md §3.

Exports one row per evaluated video with ALL fields required for auditing:
  video_id, source_group_id, identity_group_if_known, dataset,
  generator_family, label, fake_score, predicted_label, threshold,
  face_count, usable_frames, duration, face_size, quality_flags,
  preprocessing_version, model_version

Also computes:
  - Majority-class baseline accuracy
  - Missing-face rate and unsupported-input rate (over ALL attempted inputs)
  - Source-grouped bootstrap CIs (resamples at source level, not frame level)

Decision gate from newimp.md:
  "Do not buy model complexity until this audit identifies what is actually failing."
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional

import numpy as np
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, confusion_matrix

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Per-video record (full schema from newimp.md §3)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class VideoRecord:
    """Full diagnostic record for one evaluated video."""

    # Identity
    video_id: str
    source_group_id: str                # real/fake pairs sharing same source/origin
    identity_group_if_known: str = "unknown"

    # Provenance
    dataset: str = "unknown"
    generator_family: str = "unknown"
    manipulation_family: str = "unknown"
    split: str = "unknown"

    # Ground truth
    label: int = 0                      # 0=real, 1=fake
    label_str: str = "real"

    # Prediction
    fake_score: float = 0.0             # [0, 1] calibrated
    predicted_label: int = 0
    threshold: float = 0.5
    abstained: bool = False
    abstention_reason: str = ""

    # Input quality
    face_count: int = 0                 # detected faces in clip
    usable_frames: int = 0              # frames that passed quality gate
    total_frames: int = 0
    duration_seconds: float = 0.0
    face_size: float = 0.0              # mean face_area_ratio
    quality_flags: str = ""             # comma-separated flags

    # Uncertainty / evidence
    uncertainty: float = 0.0
    evidence_coverage: float = 1.0
    branch_disagreement: float = 0.0

    # Metadata
    preprocessing_version: str = "prep-1.0"
    model_version: str = "forge-lite-1.0"
    aggregation_config: str = "mean"    # how frame scores → video score

    # Input status
    face_detected: bool = True          # False → missing-face case
    supported: bool = True              # False → unsupported input case

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostic Exporter
# ─────────────────────────────────────────────────────────────────────────────

class DiagnosticExporter:
    """
    Collects per-video records and produces:
      1. Full CSV export (one row per video)
      2. Audit metrics (majority baseline, confusion matrix, etc.)
      3. Source-group-level bootstrap CIs
      4. Score distribution analysis per label
      5. Missing-face and unsupported-input rates over ALL attempted inputs

    Usage:
        exporter = DiagnosticExporter(threshold=0.5)
        exporter.add(record)
        report = exporter.compute_audit_report()
        exporter.save(output_dir)
    """

    # Fields written to CSV (ordered, per newimp.md §3)
    CSV_FIELDS = [
        "video_id", "source_group_id", "identity_group_if_known",
        "dataset", "generator_family", "manipulation_family", "split",
        "label", "label_str", "fake_score", "predicted_label",
        "threshold", "abstained", "abstention_reason",
        "face_count", "usable_frames", "total_frames", "duration_seconds",
        "face_size", "quality_flags",
        "uncertainty", "evidence_coverage", "branch_disagreement",
        "preprocessing_version", "model_version", "aggregation_config",
        "face_detected", "supported",
    ]

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold
        self.records: list[VideoRecord] = []

        # Track ALL attempted inputs — not just successful ones
        self._total_attempted: int = 0
        self._missing_face_count: int = 0
        self._unsupported_count: int = 0

    def add(self, record: VideoRecord) -> None:
        """Add a single evaluated video record."""
        self.records.append(record)
        self._total_attempted += 1
        if not record.face_detected:
            self._missing_face_count += 1
        if not record.supported:
            self._unsupported_count += 1

    def add_attempted_but_failed(
        self,
        video_id: str,
        reason: str = "no_face",       # "no_face" | "unsupported" | "error"
        dataset: str = "unknown",
    ) -> None:
        """
        Register an input that was attempted but could not be evaluated.
        These count toward the denominator for missing-face / unsupported rates.
        Per newimp.md: report over ALL attempted inputs, not just successful ones.
        """
        self._total_attempted += 1
        rec = VideoRecord(
            video_id=video_id,
            source_group_id=video_id,
            dataset=dataset,
            face_detected=(reason != "no_face"),
            supported=(reason != "unsupported"),
            quality_flags=reason,
            abstained=True,
            abstention_reason=reason,
        )
        self.records.append(rec)
        if reason == "no_face":
            self._missing_face_count += 1
        elif reason == "unsupported":
            self._unsupported_count += 1

    # ── Core audit ────────────────────────────────────────────────────────────

    def compute_audit_report(self) -> dict[str, Any]:
        """
        Compute full audit report including all metrics from newimp.md §3:
          - Video AUROC, AP, balanced accuracy, confusion matrix
          - Fake recall/TPR and genuine FPR
          - Accuracy at FROZEN threshold (must be set outside test set)
          - Per-domain, per-generator breakdowns
          - Majority-class baseline
          - Confidence intervals grouped by independent source
          - Missing-face / unsupported rates over all attempted inputs
        """
        all_recs = [r for r in self.records if r.face_detected and r.supported]
        if not all_recs:
            logger.warning("No valid records to audit.")
            return {}

        labels = np.array([r.label for r in all_recs])
        scores = np.array([r.fake_score for r in all_recs])
        abstained = np.array([r.abstained for r in all_recs])
        source_groups = [r.source_group_id for r in all_recs]

        report: dict[str, Any] = {}

        # ── 1. Video-level core metrics ───────────────────────────────────────
        try:
            report["video_auroc"] = float(roc_auc_score(labels, scores))
        except Exception:
            report["video_auroc"] = float("nan")

        # Accuracy at the FROZEN threshold (not picked on this test set)
        preds = (scores >= self.threshold).astype(int)
        report["accuracy_at_frozen_threshold"] = float((preds == labels).mean())
        report["threshold_used"] = self.threshold

        # Balanced accuracy (not distorted by class imbalance)
        try:
            report["balanced_accuracy"] = float(balanced_accuracy_score(labels, preds))
        except Exception:
            report["balanced_accuracy"] = float("nan")

        # Confusion matrix
        cm = confusion_matrix(labels, preds, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        report["confusion_matrix"] = {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
        report["fake_recall_tpr"] = float(tp / max(tp + fn, 1))
        report["genuine_fpr"] = float(fp / max(fp + tn, 1))
        report["precision"] = float(tp / max(tp + fp, 1))

        # ── 2. Majority-class baseline ────────────────────────────────────────
        n_fake = int(labels.sum())
        n_real = len(labels) - n_fake
        majority_pct = max(n_fake, n_real) / max(len(labels), 1)
        report["majority_class_baseline"] = float(majority_pct)
        report["class_balance"] = {"n_real": n_real, "n_fake": n_fake,
                                   "pct_fake": float(n_fake / max(len(labels), 1))}
        report["gain_over_majority_baseline"] = float(
            report["accuracy_at_frozen_threshold"] - majority_pct
        )

        # ── 3. Score distributions per label ─────────────────────────────────
        real_scores = scores[labels == 0]
        fake_scores = scores[labels == 1]
        report["score_distribution"] = {
            "real_mean": float(real_scores.mean()) if len(real_scores) > 0 else float("nan"),
            "real_std": float(real_scores.std()) if len(real_scores) > 0 else float("nan"),
            "fake_mean": float(fake_scores.mean()) if len(fake_scores) > 0 else float("nan"),
            "fake_std": float(fake_scores.std()) if len(fake_scores) > 0 else float("nan"),
            "score_overlap_warning": (
                float(real_scores.mean()) > 0.4 or float(fake_scores.mean()) < 0.6
                if len(real_scores) > 0 and len(fake_scores) > 0 else True
            ),
        }

        # ── 4. Selective accuracy and abstention ─────────────────────────────
        non_abs = ~abstained
        report["abstention_rate"] = float(abstained.mean())
        if non_abs.sum() > 0:
            sel_preds = (scores[non_abs] >= self.threshold).astype(int)
            report["selective_accuracy"] = float((sel_preds == labels[non_abs]).mean())
            report["n_evaluated"] = int(non_abs.sum())
        else:
            report["selective_accuracy"] = float("nan")
            report["n_evaluated"] = 0

        # ── 5. Per-domain breakdown ───────────────────────────────────────────
        report["per_domain"] = self._slice_by(all_recs, "dataset", labels, scores)
        report["per_generator"] = self._slice_by(all_recs, "generator_family", labels, scores)
        report["per_manipulation"] = self._slice_by(all_recs, "manipulation_family", labels, scores)

        # ── 6. Source-group bootstrap CI ─────────────────────────────────────
        report["source_group_bootstrap_ci"] = self._source_group_bootstrap_ci(
            labels, scores, source_groups
        )

        # ── 7. Missing-face / unsupported rates (over ALL attempted inputs) ──
        total = max(self._total_attempted, 1)
        report["input_coverage"] = {
            "total_attempted": self._total_attempted,
            "missing_face_count": self._missing_face_count,
            "missing_face_rate": float(self._missing_face_count / total),
            "unsupported_count": self._unsupported_count,
            "unsupported_rate": float(self._unsupported_count / total),
            "evaluated_count": len(all_recs),
            "evaluated_rate": float(len(all_recs) / total),
        }

        # ── 8. Near-constant prediction check ────────────────────────────────
        score_var = float(np.var(scores))
        report["near_constant_prediction_warning"] = score_var < 0.001
        report["score_variance"] = score_var

        # ── 9. Failure slices — per compression, face-size, duration ─────────
        report["failure_analysis"] = self._failure_slices(all_recs, labels, scores)

        return report

    def _slice_by(
        self,
        recs: list[VideoRecord],
        attr: str,
        labels: np.ndarray,
        scores: np.ndarray,
    ) -> dict[str, Any]:
        """Compute AUROC per unique value of a categorical attribute."""
        groups: dict[str, list[int]] = {}
        for i, r in enumerate(recs):
            key = getattr(r, attr, "unknown")
            groups.setdefault(key, []).append(i)

        result = {}
        auc_values = []
        for key, indices in groups.items():
            g_labels = labels[indices]
            g_scores = scores[indices]
            try:
                auc = float(roc_auc_score(g_labels, g_scores)) \
                    if len(set(g_labels.tolist())) >= 2 else float("nan")
            except Exception:
                auc = float("nan")
            result[key] = {
                "n": len(indices),
                "n_real": int((g_labels == 0).sum()),
                "n_fake": int((g_labels == 1).sum()),
                "auroc": auc,
            }
            if not np.isnan(auc):
                auc_values.append(auc)

        if auc_values:
            result["__macro_mean__"] = float(np.mean(auc_values))
            result["__worst__"] = float(np.min(auc_values))
        return result

    def _source_group_bootstrap_ci(
        self,
        labels: np.ndarray,
        scores: np.ndarray,
        source_groups: list[str],
        n_resamples: int = 1000,
        confidence_level: float = 0.95,
        seed: int = 42,
    ) -> dict[str, Any]:
        """
        Bootstrap CI that resamples at SOURCE GROUP level, not individual videos.
        Per newimp.md: correlated frames/clips from the same source are NOT
        independent test samples.
        """
        unique_sources = list(set(source_groups))
        if len(unique_sources) < 5:
            logger.warning(
                "Source-group bootstrap: only %d unique sources — CI may be unreliable",
                len(unique_sources)
            )

        # Build source → indices map
        src_map: dict[str, list[int]] = {}
        for i, sg in enumerate(source_groups):
            src_map.setdefault(sg, []).append(i)

        rng = np.random.default_rng(seed)
        bootstrap_aucs = []

        for _ in range(n_resamples):
            # Resample sources (not individual videos)
            sampled_sources = rng.choice(unique_sources, size=len(unique_sources), replace=True)
            idx = [i for s in sampled_sources for i in src_map[s]]
            b_labels = labels[idx]
            b_scores = scores[idx]
            if len(set(b_labels.tolist())) < 2:
                continue
            try:
                bootstrap_aucs.append(float(roc_auc_score(b_labels, b_scores)))
            except Exception:
                continue

        if not bootstrap_aucs:
            return {"note": "insufficient data for source-group bootstrap"}

        arr = np.array(bootstrap_aucs)
        alpha = 1 - confidence_level
        return {
            "resampling_unit": "source_group",
            "n_unique_sources": len(unique_sources),
            "n_resamples": n_resamples,
            "confidence_level": confidence_level,
            "auroc_mean": float(arr.mean()),
            "auroc_lower_ci": float(np.percentile(arr, 100 * alpha / 2)),
            "auroc_upper_ci": float(np.percentile(arr, 100 * (1 - alpha / 2))),
            "auroc_std": float(arr.std()),
        }

    def _failure_slices(
        self,
        recs: list[VideoRecord],
        labels: np.ndarray,
        scores: np.ndarray,
    ) -> dict[str, Any]:
        """
        Per newimp.md §8: report per-domain, per-generator, duration, compression,
        face-size and lighting slices.
        """
        result: dict[str, Any] = {}

        # Duration slices: [0-5s, 5-15s, 15-60s, 60s+]
        dur_bins = [(0, 5), (5, 15), (15, 60), (60, 9999)]
        dur_slices = {}
        for lo, hi in dur_bins:
            idx = [i for i, r in enumerate(recs) if lo <= r.duration_seconds < hi]
            if len(idx) >= 2:
                g_labels = labels[idx]
                g_scores = scores[idx]
                try:
                    auc = float(roc_auc_score(g_labels, g_scores)) \
                        if len(set(g_labels.tolist())) >= 2 else float("nan")
                except Exception:
                    auc = float("nan")
                dur_slices[f"{lo}-{hi}s"] = {"n": len(idx), "auroc": auc}
        result["duration_slices"] = dur_slices

        # Face-size slices: [tiny <0.05, small <0.15, medium <0.30, large]
        size_bins = [(0, 0.05), (0.05, 0.15), (0.15, 0.30), (0.30, 1.0)]
        size_slices = {}
        for lo, hi in size_bins:
            idx = [i for i, r in enumerate(recs) if lo <= r.face_size < hi]
            if len(idx) >= 2:
                g_labels = labels[idx]
                g_scores = scores[idx]
                try:
                    auc = float(roc_auc_score(g_labels, g_scores)) \
                        if len(set(g_labels.tolist())) >= 2 else float("nan")
                except Exception:
                    auc = float("nan")
                size_slices[f"face_size_{lo:.2f}-{hi:.2f}"] = {"n": len(idx), "auroc": auc}
        result["face_size_slices"] = size_slices

        return result

    # ── Save outputs ──────────────────────────────────────────────────────────

    def save(self, output_dir: str) -> None:
        """Save CSV export and JSON audit report."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # CSV export
        csv_path = out / "diagnostic_export.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.CSV_FIELDS, extrasaction="ignore")
            writer.writeheader()
            for r in self.records:
                writer.writerow(r.to_dict())
        logger.info("Diagnostic CSV saved: %s (%d rows)", csv_path, len(self.records))

        # JSON audit report
        report = self.compute_audit_report()
        json_path = out / "audit_report.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)
        logger.info("Audit report saved: %s", json_path)

        # Print summary to console
        self._print_summary(report)

    def _print_summary(self, report: dict[str, Any]) -> None:
        """Print key audit findings to console."""
        sep = "=" * 70
        dash = "-" * 70
        print("\n" + sep)
        print("  FORGE-Lite Diagnostic Audit Summary")
        print(sep)
        cb = report.get("class_balance", {})
        print(f"  Class balance     : {cb.get('n_real', 0)} real / {cb.get('n_fake', 0)} fake "
              f"({cb.get('pct_fake', 0)*100:.1f}% fake)")
        maj = report.get("majority_class_baseline", float("nan"))
        acc = report.get("accuracy_at_frozen_threshold", float("nan"))
        gain = report.get("gain_over_majority_baseline", float("nan"))
        print(f"  Majority baseline : {maj*100:.1f}%")
        print(f"  Model accuracy    : {acc*100:.1f}%  (gain: {gain*100:+.1f}pp)")
        auc = report.get("video_auroc", float("nan"))
        print(f"  Video AUROC       : {auc:.4f}")
        bal = report.get("balanced_accuracy", float("nan"))
        print(f"  Balanced accuracy : {bal*100:.1f}%")
        cm = report.get("confusion_matrix", {})
        print(f"  Confusion matrix  : TP={cm.get('tp',0)} TN={cm.get('tn',0)} "
              f"FP={cm.get('fp',0)} FN={cm.get('fn',0)}")
        tpr = report.get("fake_recall_tpr", float("nan"))
        fpr = report.get("genuine_fpr", float("nan"))
        print(f"  Fake recall (TPR) : {tpr*100:.1f}%  | Genuine FPR: {fpr*100:.1f}%")
        abst = report.get("abstention_rate", float("nan"))
        sel_acc = report.get("selective_accuracy", float("nan"))
        print(f"  Abstention rate   : {abst*100:.1f}%  | Selective acc: {sel_acc*100:.1f}%")
        ic = report.get("input_coverage", {})
        print(f"  Missing-face rate : {ic.get('missing_face_rate', 0)*100:.1f}%  "
              f"({ic.get('missing_face_count',0)} / {ic.get('total_attempted',0)} attempted)")
        print(f"  Unsupported rate  : {ic.get('unsupported_rate', 0)*100:.1f}%")
        bci = report.get("source_group_bootstrap_ci", {})
        if "auroc_mean" in bci:
            print(f"  Source-group CI   : AUROC {bci['auroc_mean']:.4f} "
                  f"[{bci['auroc_lower_ci']:.4f}, {bci['auroc_upper_ci']:.4f}] "
                  f"(n_sources={bci['n_unique_sources']})")
        if report.get("near_constant_prediction_warning"):
            print("  [WARNING] near-constant predictions -- check label mapping, "
                  "logits/sigmoid handling, checkpoint or evaluation-mode bugs")
        if report.get("score_distribution", {}).get("score_overlap_warning"):
            print("  [WARNING] real/fake score distributions overlap significantly")
        print(sep + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI for simulation / testing
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="FORGE-Lite Diagnostic Export")
    parser.add_argument("--results-json", default=None,
                        help="Existing ForgeEvaluator JSON results to import")
    parser.add_argument("--output-dir", default="reports/diagnostic")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="FROZEN threshold — must be set outside test set")
    parser.add_argument("--simulate", action="store_true",
                        help="Generate synthetic records for testing")
    parser.add_argument("--n-simulate", type=int, default=500)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    exporter = DiagnosticExporter(threshold=args.threshold)

    if args.simulate:
        rng = np.random.default_rng(42)
        datasets = ["ff_c23", "ff_c40", "celeb_df_v2", "dfdc_preview", "wild_deepfake"]
        generators = ["face2face", "faceswap", "deepfakes", "neuraltextures", "unknown_gen"]

        # Simulate a few failed inputs
        for i in range(20):
            exporter.add_attempted_but_failed(
                video_id=f"failed_{i:04d}",
                reason="no_face" if i < 10 else "unsupported",
                dataset=rng.choice(datasets),
            )

        for i in range(args.n_simulate):
            label = int(rng.integers(0, 2))
            ds = rng.choice(datasets)
            gen = rng.choice(generators)
            # Simulate domain shift — cross-domain is harder
            base_mean = 0.75 if label == 1 else 0.25
            if ds in ("celeb_df_v2", "wild_deepfake"):
                base_mean = 0.62 if label == 1 else 0.38  # harder domain
            score = float(np.clip(rng.normal(base_mean, 0.18), 0.0, 1.0))
            source_id = f"src_{i // 4:04d}"  # 4 videos per source group

            rec = VideoRecord(
                video_id=f"vid_{i:05d}",
                source_group_id=source_id,
                dataset=ds,
                generator_family=gen,
                manipulation_family="face_swap" if label == 1 else "none",
                label=label,
                label_str="fake" if label == 1 else "real",
                fake_score=score,
                predicted_label=int(score >= args.threshold),
                threshold=args.threshold,
                abstained=bool(abs(score - 0.5) < 0.1),
                face_count=rng.integers(1, 20),
                usable_frames=rng.integers(8, 32),
                total_frames=32,
                duration_seconds=float(rng.choice([3.0, 8.0, 20.0, 90.0])),
                face_size=float(rng.uniform(0.02, 0.45)),
                uncertainty=float(rng.uniform(0.0, 0.4)),
                evidence_coverage=float(rng.uniform(0.5, 1.0)),
            )
            exporter.add(rec)

    elif args.results_json:
        with open(args.results_json) as f:
            raw = json.load(f)
        for i, r in enumerate(raw):
            rec = VideoRecord(
                video_id=r.get("video_id", f"vid_{i}"),
                source_group_id=r.get("video_id", f"vid_{i}"),
                dataset=r.get("dataset", "unknown"),
                generator_family=r.get("generator_family", "unknown"),
                manipulation_family=r.get("manipulation_family", "unknown"),
                label=int(r.get("label", 0)),
                label_str="fake" if r.get("label", 0) == 1 else "real",
                fake_score=float(r.get("score", 0.5)),
                predicted_label=int(float(r.get("score", 0.5)) >= args.threshold),
                threshold=args.threshold,
                abstained=bool(r.get("abstained", False)),
                uncertainty=float(r.get("uncertainty", 0.0)),
            )
            exporter.add(rec)

    exporter.save(args.output_dir)


if __name__ == "__main__":
    main()
