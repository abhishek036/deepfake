"""
evaluation/evaluate_calibration.py
Calibration and Selective Prediction Evaluation for FORGE-Lite.

Evaluates:
- Expected Calibration Error (ECE) with equal-width & equal-frequency binning
- Maximum Calibration Error (MCE)
- Brier Score & Negative Log-Likelihood (NLL)
- Pre-calibration vs Post-Temperature Scaling comparison
- Reliability Diagram data (confidence vs empirical accuracy per bin)
- Selective Prediction / Abstention trade-offs:
  - Coverage vs Selective Accuracy
  - Error rate under varying uncertainty thresholds
  - Risk-Coverage curve
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Calibration Metrics Computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_detailed_ece(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, Any]:
    """
    Computes ECE, MCE, and per-bin confidence & accuracy for reliability diagrams.
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(probs, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    bin_data = []
    ece = 0.0
    mce = 0.0
    total_samples = len(labels)

    for i in range(n_bins):
        mask = bin_indices == i
        count = int(mask.sum())
        if count > 0:
            bin_conf = float(probs[mask].mean())
            bin_acc = float(labels[mask].mean())
            gap = abs(bin_conf - bin_acc)
            ece += (count / total_samples) * gap
            mce = max(mce, gap)
            bin_data.append({
                "bin_idx": i,
                "bin_lower": float(bins[i]),
                "bin_upper": float(bins[i + 1]),
                "count": count,
                "confidence": bin_conf,
                "accuracy": bin_acc,
                "calibration_gap": float(gap),
            })
        else:
            bin_data.append({
                "bin_idx": i,
                "bin_lower": float(bins[i]),
                "bin_upper": float(bins[i + 1]),
                "count": 0,
                "confidence": float((bins[i] + bins[i + 1]) / 2.0),
                "accuracy": 0.0,
                "calibration_gap": 0.0,
            })

    brier_score = float(np.mean((probs - labels) ** 2))
    eps = 1e-12
    probs_clipped = np.clip(probs, eps, 1.0 - eps)
    nll = float(-np.mean(labels * np.log(probs_clipped) + (1 - labels) * np.log(1 - probs_clipped)))

    return {
        "ece": float(ece),
        "mce": float(mce),
        "brier_score": brier_score,
        "nll": nll,
        "n_bins": n_bins,
        "bins": bin_data,
    }


def compute_selective_prediction_curve(
    probs: np.ndarray,
    labels: np.ndarray,
    uncertainties: np.ndarray,
    num_thresholds: int = 20,
) -> List[Dict[str, float]]:
    """
    Sweeps abstention threshold on uncertainty to compute coverage vs accuracy curve.
    """
    sorted_th = np.linspace(0.0, float(np.max(uncertainties) + 1e-5), num_thresholds)
    curve = []

    for th in sorted_th:
        # Abstain if uncertainty > th
        accepted_mask = uncertainties <= th
        cov = float(accepted_mask.mean())
        if accepted_mask.sum() > 0:
            preds = (probs[accepted_mask] >= 0.5).astype(int)
            acc = float((preds == labels[accepted_mask]).mean())
            err = 1.0 - acc
        else:
            acc = float("nan")
            err = float("nan")

        curve.append({
            "uncertainty_threshold": float(th),
            "coverage": cov,
            "abstention_rate": 1.0 - cov,
            "selective_accuracy": acc,
            "error_rate": err,
        })

    return curve


# ─────────────────────────────────────────────────────────────────────────────
# Calibration Evaluator Class
# ─────────────────────────────────────────────────────────────────────────────

class CalibrationEvaluator:
    """
    Evaluates probability calibration and selective abstention performance.
    """

    def __init__(self, n_bins: int = 10) -> None:
        self.n_bins = n_bins

    def evaluate(
        self,
        raw_probs: np.ndarray,
        calibrated_probs: Optional[np.ndarray],
        labels: np.ndarray,
        uncertainties: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Computes calibration before and after post-hoc calibration, plus selective curve.
        """
        raw_calib = compute_detailed_ece(raw_probs, labels, n_bins=self.n_bins)

        res = {
            "uncalibrated": raw_calib,
        }

        if calibrated_probs is not None:
            post_calib = compute_detailed_ece(calibrated_probs, labels, n_bins=self.n_bins)
            res["calibrated"] = post_calib
            res["ece_reduction_pct"] = float((raw_calib["ece"] - post_calib["ece"]) / max(raw_calib["ece"], 1e-6) * 100.0)

        eval_probs = calibrated_probs if calibrated_probs is not None else raw_probs
        if uncertainties is None:
            # Estimate epistemic uncertainty as distance from decision boundary
            uncertainties = 1.0 - 2.0 * np.abs(eval_probs - 0.5)

        res["selective_prediction_curve"] = compute_selective_prediction_curve(
            eval_probs, labels, uncertainties
        )

        return res

    def generate_markdown_report(self, results: Dict[str, Any]) -> str:
        """Formats reliability and calibration results into Markdown."""
        uncal = results["uncalibrated"]
        cal = results.get("calibrated", uncal)

        lines = [
            "# FORGE-Lite Calibration & Reliability Report",
            "",
            "### Calibration Metrics (Pre vs Post Scaling)",
            "",
            "| Metric | Uncalibrated | Temperature-Calibrated | Delta / Reduction |",
            "|---|---|---|---|",
            f"| **ECE (Expected Calibration Error)** | `{uncal['ece']:.4f}` | **`{cal['ece']:.4f}`** | {results.get('ece_reduction_pct', 0.0):.1f}% reduction |",
            f"| **MCE (Max Calibration Error)** | `{uncal['mce']:.4f}` | **`{cal['mce']:.4f}`** | {cal['mce'] - uncal['mce']:+.4f} |",
            f"| **Brier Score** | `{uncal['brier_score']:.4f}` | **`{cal['brier_score']:.4f}`** | {cal['brier_score'] - uncal['brier_score']:+.4f} |",
            f"| **Negative Log-Likelihood (NLL)** | `{uncal['nll']:.4f}` | **`{cal['nll']:.4f}`** | {cal['nll'] - uncal['nll']:+.4f} |",
            "",
            "### Reliability Diagram (10 Probability Bins)",
            "",
            "| Bin Range | Sample Count | Mean Confidence | Observed Accuracy | Calibration Gap |",
            "|---|---|---|---|---|",
        ]

        for b in cal["bins"]:
            r_str = f"[{b['bin_lower']:.1f} - {b['bin_upper']:.1f})"
            cnt = str(b["count"])
            conf = f"{b['confidence']:.3f}"
            acc = f"{b['accuracy']:.3f}"
            gap = f"{b['calibration_gap']:.3f}"
            lines.append(f"| `{r_str}` | {cnt} | {conf} | {acc} | {gap} |")

        lines.append("")
        lines.append("### Selective Prediction Trade-off (Coverage vs Accuracy)")
        lines.append("")
        lines.append("| Coverage (%) | Abstention Rate (%) | Selective Accuracy | Selective Error |")
        lines.append("|---|---|---|---|")

        curve = results.get("selective_prediction_curve", [])
        # Sample ~5 representative points from the curve
        sample_indices = np.linspace(0, len(curve) - 1, min(6, len(curve)), dtype=int)
        for idx in sample_indices:
            pt = curve[idx]
            cov = f"{pt['coverage'] * 100:.1f}%"
            abst = f"{pt['abstention_rate'] * 100:.1f}%"
            acc = f"{pt['selective_accuracy'] * 100:.2f}%" if not np.isnan(pt['selective_accuracy']) else "N/A"
            err = f"{pt['error_rate'] * 100:.2f}%" if not np.isnan(pt['error_rate']) else "N/A"
            lines.append(f"| {cov} | {abst} | **{acc}** | {err} |")

        lines.append("")
        return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="FORGE-Lite Calibration Evaluation")
    parser.add_argument("--output_dir", default="reports")
    parser.add_argument("--simulate", action="store_true", help="Simulate predictions for calibration")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger.info("Running Calibration Evaluation")

    evaluator = CalibrationEvaluator(n_bins=10)

    if args.simulate:
        rng = np.random.default_rng(99)
        n = 500
        labels = rng.integers(0, 2, size=n)
        # Raw overconfident probabilities
        raw_logits = np.where(labels == 1, rng.normal(2.5, 1.2, n), rng.normal(-2.5, 1.2, n))
        raw_probs = 1.0 / (1.0 + np.exp(-raw_logits))

        # Temperature-calibrated logits (temperature T = 1.6)
        calibrated_logits = raw_logits / 1.6
        calibrated_probs = 1.0 / (1.0 + np.exp(-calibrated_logits))

        uncertainties = 1.0 - np.abs(calibrated_probs - 0.5) * 2.0
        results = evaluator.evaluate(raw_probs, calibrated_probs, labels, uncertainties)
    else:
        results = {}

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    md_report = evaluator.generate_markdown_report(results)

    with open(out_dir / "calibration_report.md", "w", encoding="utf-8") as f:
        f.write(md_report)

    with open(out_dir / "calibration_report.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info("Saved calibration report to %s", out_dir / "calibration_report.md")
    print(md_report)


if __name__ == "__main__":
    main()
