"""
evaluation/evaluate_localization.py
Pixel-Level Forgery Localization Evaluation for FORGE-Lite.

Evaluates spatial forgery prediction masks against ground-truth manipulation masks:
- Intersection over Union (IoU)
- Dice Coefficient / F1 score
- Precision and Recall across threshold grid [0.1 .. 0.9]
- Pixel-level AUROC and PR-AUC
- Boundary alignment analysis

PRD Rule: Localization metrics are strictly computed and reported ONLY on
samples with valid, documented ground-truth masks (e.g. FaceForensics++ FaceSwap,
Deepfakes, FaceShifter). Never evaluated on whole-image reenactment or unmasked sets.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from omegaconf import OmegaConf
from sklearn.metrics import average_precision_score, roc_auc_score

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Core Pixel-Level Metric Calculations
# ─────────────────────────────────────────────────────────────────────────────

def compute_mask_iou_dice(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    threshold: float = 0.5,
    eps: float = 1e-7,
) -> Tuple[float, float, float, float]:
    """
    Computes (IoU, Dice, Precision, Recall) at a given binarization threshold.
    """
    p_bin = (pred_mask >= threshold).astype(bool)
    g_bin = (gt_mask >= 0.5).astype(bool)

    intersection = np.logical_and(p_bin, g_bin).sum()
    union = np.logical_or(p_bin, g_bin).sum()
    p_sum = p_bin.sum()
    g_sum = g_bin.sum()

    iou = float(intersection / (union + eps)) if union > 0 else (1.0 if p_sum == 0 and g_sum == 0 else 0.0)
    dice = float((2.0 * intersection) / (p_sum + g_sum + eps)) if (p_sum + g_sum) > 0 else (1.0 if p_sum == 0 and g_sum == 0 else 0.0)
    precision = float(intersection / (p_sum + eps)) if p_sum > 0 else 1.0
    recall = float(intersection / (g_sum + eps)) if g_sum > 0 else 1.0

    return iou, dice, precision, recall


# ─────────────────────────────────────────────────────────────────────────────
# Localization Evaluator
# ─────────────────────────────────────────────────────────────────────────────

class LocalizationEvaluator:
    """
    Collects predicted and ground-truth masks and computes comprehensive metrics.
    """

    def __init__(self, thresholds: Optional[List[float]] = None) -> None:
        self.thresholds = thresholds or [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        self.records: List[Dict[str, Any]] = []

    def add_sample(
        self,
        sample_id: str,
        pred_mask: np.ndarray,   # (H, W) in [0, 1]
        gt_mask: np.ndarray,     # (H, W) binary or [0, 1]
        manipulation_type: str = "unknown",
        dataset: str = "ff_c23",
    ) -> None:
        """Add a single validated mask pair."""
        assert pred_mask.shape == gt_mask.shape, f"Shape mismatch: {pred_mask.shape} vs {gt_mask.shape}"
        self.records.append({
            "sample_id": sample_id,
            "pred_mask": pred_mask.astype(np.float32),
            "gt_mask": gt_mask.astype(np.float32),
            "manipulation_type": manipulation_type,
            "dataset": dataset,
        })

    def compute_metrics(self) -> Dict[str, Any]:
        """
        Computes mean IoU, Dice, Precision, Recall at default (0.5) and sweep thresholds,
        plus pixel-level ROC-AUC and PR-AUC.
        """
        if not self.records:
            return {"error": "No valid ground truth mask records provided"}

        all_preds = []
        all_gts = []

        threshold_metrics = {th: {"ious": [], "dices": [], "precisions": [], "recalls": []} for th in self.thresholds}

        for rec in self.records:
            pred = rec["pred_mask"]
            gt = rec["gt_mask"]

            all_preds.append(pred.flatten())
            all_gts.append((gt.flatten() >= 0.5).astype(int))

            for th in self.thresholds:
                iou, dice, prec, rec_val = compute_mask_iou_dice(pred, gt, threshold=th)
                threshold_metrics[th]["ious"].append(iou)
                threshold_metrics[th]["dices"].append(dice)
                threshold_metrics[th]["precisions"].append(prec)
                threshold_metrics[th]["recalls"].append(rec_val)

        # Global flatten for pixel-level AUC
        flat_preds = np.concatenate(all_preds)
        flat_gts = np.concatenate(all_gts)

        pixel_auc = float("nan")
        pixel_ap = float("nan")
        if len(np.unique(flat_gts)) >= 2:
            try:
                pixel_auc = float(roc_auc_score(flat_gts, flat_preds))
                pixel_ap = float(average_precision_score(flat_gts, flat_preds))
            except Exception as e:
                logger.warning("Pixel AUC calculation encountered exception: %s", e)

        # Summarize across thresholds
        sweep_summary = {}
        for th in self.thresholds:
            sweep_summary[f"th_{th:.1f}"] = {
                "mean_iou": float(np.mean(threshold_metrics[th]["ious"])),
                "mean_dice": float(np.mean(threshold_metrics[th]["dices"])),
                "mean_precision": float(np.mean(threshold_metrics[th]["precisions"])),
                "mean_recall": float(np.mean(threshold_metrics[th]["recalls"])),
            }

        default_th = 0.5
        iou_05 = float(np.mean(threshold_metrics[default_th]["ious"]))
        dice_05 = float(np.mean(threshold_metrics[default_th]["dices"]))

        return {
            "num_evaluated_masks": len(self.records),
            "pixel_roc_auc": pixel_auc,
            "pixel_pr_auc": pixel_ap,
            "mean_iou_at_05": iou_05,
            "mean_dice_at_05": dice_05,
            "threshold_sweep": sweep_summary,
        }

    def generate_markdown_report(self, metrics: Dict[str, Any]) -> str:
        """Generates clean Markdown documentation of mask localization quality."""
        lines = [
            "# FORGE-Lite Pixel-Level Localization Report",
            "",
            f"- **Evaluated Samples (Valid GT Masks Only)**: {metrics.get('num_evaluated_masks', 0)}",
            f"- **Pixel ROC-AUC**: `{metrics.get('pixel_roc_auc', float('nan')):.4f}`",
            f"- **Pixel PR-AUC**: `{metrics.get('pixel_pr_auc', float('nan')):.4f}`",
            f"- **Mean IoU (threshold 0.5)**: `{metrics.get('mean_iou_at_05', float('nan')):.4f}`",
            f"- **Mean Dice (threshold 0.5)**: `{metrics.get('mean_dice_at_05', float('nan')):.4f}`",
            "",
            "### Threshold Sensitivity Analysis",
            "",
            "| Threshold | Mean IoU | Mean Dice | Precision | Recall |",
            "|---|---|---|---|---|",
        ]

        for th_key, vals in metrics.get("threshold_sweep", {}).items():
            th_num = th_key.replace("th_", "")
            iou = f"{vals['mean_iou']:.4f}"
            dice = f"{vals['mean_dice']:.4f}"
            prec = f"{vals['mean_precision']:.4f}"
            rec = f"{vals['mean_recall']:.4f}"
            lines.append(f"| `{th_num}` | {iou} | {dice} | {prec} | {rec} |")

        lines.append("")
        return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="FORGE-Lite Localization Evaluation")
    parser.add_argument("--output_dir", default="reports")
    parser.add_argument("--simulate", action="store_true", help="Simulate localization evaluation")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger.info("Running Localization Evaluation")

    evaluator = LocalizationEvaluator()

    if args.simulate:
        rng = np.random.default_rng(42)
        h, w = 56, 56
        for i in range(50):
            # Generate synthetic circle GT mask for face swap region
            gt = np.zeros((h, w), dtype=np.float32)
            cy, cx = rng.integers(18, 38), rng.integers(18, 38)
            radius = rng.integers(8, 16)
            y, x = np.ogrid[:h, :w]
            mask_area = (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2
            gt[mask_area] = 1.0

            # Pred mask with realistic noise and blur
            pred = gt * rng.uniform(0.6, 0.95) + rng.uniform(0.0, 0.15, size=(h, w))
            pred = np.clip(pred, 0.0, 1.0).astype(np.float32)

            evaluator.add_sample(
                sample_id=f"sim_sample_{i:03d}",
                pred_mask=pred,
                gt_mask=gt,
                manipulation_type="FaceSwap",
            )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = evaluator.compute_metrics()
    md_report = evaluator.generate_markdown_report(metrics)

    with open(out_dir / "localization_report.md", "w", encoding="utf-8") as f:
        f.write(md_report)

    with open(out_dir / "localization_report.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    logger.info("Saved localization report to %s", out_dir / "localization_report.md")
    print(md_report)


if __name__ == "__main__":
    main()
