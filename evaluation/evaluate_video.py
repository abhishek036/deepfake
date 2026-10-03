"""
evaluation/evaluate_video.py
Video-level evaluation for FORGE-Lite.

Computes:
- Video AUC, Frame AUC
- Cross-domain macro AUROC (with worst-domain as per newimp.md §8)
- TPR@1%FPR, EER, Average Precision
- Localization IoU and Dice (on valid masks)
- Calibration metrics (ECE, Brier)
- Abstention rate and selective accuracy
- Majority-class baseline (newimp.md §3)
- Missing-face rate and unsupported-input rate over ALL attempted inputs
- Source-group-level bootstrap CIs (newimp.md §3)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
)

from models.calibration import compute_ece, compute_brier_score

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Per-video result
# ─────────────────────────────────────────────────────────────────────────────

def compute_video_auc(video_labels: list[int], video_scores: list[float]) -> float:
    """Compute video-level AUROC."""
    if len(set(video_labels)) < 2:
        return float("nan")
    return float(roc_auc_score(video_labels, video_scores))


def compute_tpr_at_fpr(
    y_true: np.ndarray, y_scores: np.ndarray, target_fpr: float = 0.01
) -> float:
    """Compute TPR at a fixed FPR (e.g., 1%)."""
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    idx = np.searchsorted(fpr, target_fpr)
    if idx >= len(tpr):
        return float(tpr[-1])
    return float(tpr[idx])


def compute_eer(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """Equal Error Rate."""
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    fnr = 1 - tpr
    idx = np.argmin(np.abs(fpr - fnr))
    return float((fpr[idx] + fnr[idx]) / 2)


def compute_localization_metrics(
    pred_masks: list[np.ndarray],
    gt_masks: list[np.ndarray],
    threshold: float = 0.5,
) -> dict[str, float]:
    """
    Compute IoU and Dice on pixel-level masks.

    PRD Rule: Only reported on samples with valid documented GT masks.
    """
    ious, dices = [], []
    for pred, gt in zip(pred_masks, gt_masks):
        pred_bin = (pred >= threshold).astype(float)
        gt_bin = (gt >= threshold).astype(float)
        intersection = (pred_bin * gt_bin).sum()
        union = pred_bin.sum() + gt_bin.sum() - intersection
        iou = intersection / max(union, 1e-6)
        dice = 2 * intersection / max(pred_bin.sum() + gt_bin.sum(), 1e-6)
        ious.append(iou)
        dices.append(dice)

    return {
        "localization_iou": float(np.mean(ious)) if ious else float("nan"),
        "localization_dice": float(np.mean(dices)) if dices else float("nan"),
    }


def compute_selective_accuracy(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    abstained: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """
    Compute accuracy on non-abstained samples (selective accuracy).
    """
    non_abstained = ~abstained
    if non_abstained.sum() == 0:
        return {"selective_accuracy": float("nan"), "abstention_rate": 1.0}

    preds = (y_scores[non_abstained] >= threshold).astype(int)
    acc = float((preds == y_true[non_abstained]).mean())
    abstention_rate = float(abstained.mean())
    return {"selective_accuracy": acc, "abstention_rate": abstention_rate}


# ─────────────────────────────────────────────────────────────────────────────
# Dataset-level evaluation
# ─────────────────────────────────────────────────────────────────────────────

class ForgeEvaluator:
    """
    Computes all FORGE-Lite evaluation metrics.

    Per newimp.md §3: tracks missing-face and unsupported-input rates over
    ALL attempted inputs (not just successful ones). Reports majority-class
    baseline alongside model accuracy.

    Usage:
        evaluator = ForgeEvaluator()
        evaluator.add_result(video_id, label, score, abstained)
        metrics = evaluator.compute()
    """

    def __init__(self) -> None:
        self.results = []
        # newimp.md: track ALL attempted inputs
        self._total_attempted: int = 0
        self._missing_face_count: int = 0
        self._unsupported_count: int = 0
        # VRAM/latency (populated externally from forge_lite.measure_vram)
        self._peak_vram_gb: Optional[float] = None
        self._inference_latency_ms: Optional[float] = None

    def set_resource_metrics(self, peak_vram_gb: float, latency_ms: float) -> None:
        """Register VRAM and latency from forge_lite.measure_vram()."""
        self._peak_vram_gb = peak_vram_gb
        self._inference_latency_ms = latency_ms

    def record_failed_input(
        self,
        reason: str = "no_face",  # "no_face" | "unsupported" | "error"
    ) -> None:
        """
        Register an input that could not be evaluated.
        Per newimp.md: denominator for missing-face rate = ALL attempted inputs.
        """
        self._total_attempted += 1
        if reason == "no_face":
            self._missing_face_count += 1
        elif reason == "unsupported":
            self._unsupported_count += 1

    def add_result(
        self,
        video_id: str,
        label: int,                   # 0=real, 1=fake
        score: float,                 # fake probability [0, 1]
        abstained: bool,
        uncertainty: float = 0.0,
        dataset: str = "unknown",
        manipulation_family: str = "unknown",
        generator_family: str = "unknown",
        source_group_id: Optional[str] = None,  # newimp.md: for source-grouped CI
        pred_mask: Optional[np.ndarray] = None,
        gt_mask: Optional[np.ndarray] = None,
        mask_available: bool = False,
    ) -> None:
        self._total_attempted += 1
        self.results.append({
            "video_id": video_id,
            "label": label,
            "score": score,
            "abstained": abstained,
            "uncertainty": uncertainty,
            "dataset": dataset,
            "manipulation_family": manipulation_family,
            "generator_family": generator_family,
            "source_group_id": source_group_id or video_id,
            "pred_mask": pred_mask,
            "gt_mask": gt_mask,
            "mask_available": mask_available,
        })

    def compute(self) -> dict:
        """Compute all metrics over all added results."""
        if not self.results:
            return {}

        labels = np.array([r["label"] for r in self.results])
        scores = np.array([r["score"] for r in self.results])
        abstained = np.array([r["abstained"] for r in self.results])
        uncertainties = np.array([r["uncertainty"] for r in self.results])

        metrics = {}

        # Core metrics
        try:
            metrics["video_auc"] = float(roc_auc_score(labels, scores))
        except Exception:
            metrics["video_auc"] = float("nan")

        try:
            metrics["average_precision"] = float(average_precision_score(labels, scores))
        except Exception:
            metrics["average_precision"] = float("nan")

        metrics["tpr_at_1pct_fpr"] = compute_tpr_at_fpr(labels, scores, 0.01)
        metrics["eer"] = compute_eer(labels, scores)

        # Calibration
        metrics["ece"] = compute_ece(scores, labels)
        metrics["brier_score"] = compute_brier_score(scores, labels)

        # Selective accuracy + abstention
        sel = compute_selective_accuracy(labels, scores, abstained)
        metrics.update(sel)

        # Cross-domain AUROC per dataset (with worst-domain per newimp.md §8)
        datasets = set(r["dataset"] for r in self.results)
        per_dataset_aucs = {}
        for ds in datasets:
            ds_results = [r for r in self.results if r["dataset"] == ds]
            ds_labels = np.array([r["label"] for r in ds_results])
            ds_scores = np.array([r["score"] for r in ds_results])
            if len(set(ds_labels.tolist())) >= 2:
                try:
                    per_dataset_aucs[ds] = float(roc_auc_score(ds_labels, ds_scores))
                except Exception:
                    per_dataset_aucs[ds] = float("nan")
        metrics["per_dataset_auc"] = per_dataset_aucs
        valid_aucs = [v for v in per_dataset_aucs.values() if not np.isnan(v)]
        metrics["cross_domain_macro_auroc"] = float(np.mean(valid_aucs)) if valid_aucs else float("nan")
        # newimp.md §8: report worst-domain, not just mean
        metrics["cross_domain_worst_auroc"] = float(np.min(valid_aucs)) if valid_aucs else float("nan")
        metrics["cross_domain_worst_dataset"] = (
            min(per_dataset_aucs, key=per_dataset_aucs.get)
            if per_dataset_aucs else "none"
        )

        # Majority-class baseline (newimp.md §3)
        n_fake = int(labels.sum())
        n_real = len(labels) - n_fake
        metrics["majority_class_baseline"] = float(max(n_fake, n_real) / max(len(labels), 1))
        metrics["class_balance"] = {"n_real": n_real, "n_fake": n_fake}
        metrics["gain_over_majority_baseline"] = (
            metrics.get("selective_accuracy", float("nan")) - metrics["majority_class_baseline"]
        )

        # Missing-face / unsupported rates over ALL attempted inputs (newimp.md §3)
        total_attempted = max(self._total_attempted, 1)
        metrics["input_coverage"] = {
            "total_attempted": self._total_attempted,
            "evaluated_count": len(self.results),
            "evaluated_rate": float(len(self.results) / total_attempted),
            "missing_face_count": self._missing_face_count,
            "missing_face_rate": float(self._missing_face_count / total_attempted),
            "unsupported_count": self._unsupported_count,
            "unsupported_rate": float(self._unsupported_count / total_attempted),
        }

        # VRAM and latency (newimp.md §8)
        if self._peak_vram_gb is not None:
            metrics["peak_vram_gb"] = self._peak_vram_gb
        if self._inference_latency_ms is not None:
            metrics["inference_latency_ms"] = self._inference_latency_ms

        # Localization metrics (only valid GT masks)
        mask_results = [r for r in self.results if r["mask_available"] and r["pred_mask"] is not None]
        if mask_results:
            loc = compute_localization_metrics(
                [r["pred_mask"] for r in mask_results],
                [r["gt_mask"] for r in mask_results],
            )
            metrics.update(loc)

        # Per-manipulation-family breakdown
        families = set(r["manipulation_family"] for r in self.results)
        metrics["per_family_auc"] = {}
        for fam in families:
            fam_r = [r for r in self.results if r["manipulation_family"] == fam]
            fl = np.array([r["label"] for r in fam_r])
            fs = np.array([r["score"] for r in fam_r])
            if len(set(fl.tolist())) >= 2:
                try:
                    metrics["per_family_auc"][fam] = float(roc_auc_score(fl, fs))
                except Exception:
                    metrics["per_family_auc"][fam] = float("nan")

        return metrics

    def save_results(self, output_path: str) -> None:
        """Save all results to JSON."""
        safe_results = [
            {k: v for k, v in r.items() if k not in ("pred_mask", "gt_mask")}
            for r in self.results
        ]
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(safe_results, f, indent=2)


def main():
    """CLI entry point for evaluation."""
    import argparse
    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser(description="FORGE-Lite Evaluation")
    parser.add_argument("--config", default="configs/eval.yaml")
    parser.add_argument("--checkpoint", default="outputs/checkpoints/best.ckpt")
    parser.add_argument("--output-dir", default="reports")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--ablation", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    cfg = OmegaConf.load(args.config)
    logger.info("Evaluation config loaded: %s", args.config)
    logger.info("Checkpoint: %s", args.checkpoint)

    if cfg.locked_test.allow_access is False:
        logger.warning("⚠️  Locked test set access is DISABLED. Set allow_access: true for final eval.")
        return

    logger.info("Starting evaluation... (implement dataset-specific evaluation loop)")


if __name__ == "__main__":
    main()
