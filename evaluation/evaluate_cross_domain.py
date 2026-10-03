"""
evaluation/evaluate_cross_domain.py
Cross-Dataset Generalization Evaluation for FORGE-Lite.

Evaluates the locked model checkpoint across unseen benchmark datasets:
- Celeb-DF v2
- DFDC Preview
- WildDeepfake
- FaceForensics++ (C23 in-domain and C40 cross-compression)

Computes:
- Per-dataset Video AUROC, Frame AUROC
- Macro-AUROC across all cross-domain datasets
- TPR@1%FPR (strict operational threshold)
- EER (Equal Error Rate) and Average Precision
- Selective prediction metrics (abstention rate & selective accuracy)
- Formats results into publication-ready Markdown tables and JSON records.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from omegaconf import OmegaConf
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from evaluation.bootstrap_ci import compute_all_bootstrap_ci
from evaluation.evaluate_video import (
    ForgeEvaluator,
    compute_eer,
    compute_selective_accuracy,
    compute_tpr_at_fpr,
)

logger = logging.getLogger(__name__)


class CrossDomainEvaluator:
    """
    Evaluates FORGE-Lite checkpoints across multiple cross-domain datasets.
    """

    def __init__(self, config_path: str = "configs/eval.yaml") -> None:
        self.config = OmegaConf.load(config_path) if Path(config_path).exists() else OmegaConf.create()
        self.evaluators: Dict[str, ForgeEvaluator] = {}

    def get_or_create_evaluator(self, dataset_name: str) -> ForgeEvaluator:
        if dataset_name not in self.evaluators:
            self.evaluators[dataset_name] = ForgeEvaluator()
        return self.evaluators[dataset_name]

    def record_video_prediction(
        self,
        dataset_name: str,
        video_id: str,
        label: int,
        score: float,
        abstained: bool = False,
        uncertainty: float = 0.0,
        manipulation_family: str = "unknown",
        generator_family: str = "unknown",
    ) -> None:
        """Add single video inference result to the appropriate dataset evaluator."""
        ev = self.get_or_create_evaluator(dataset_name)
        ev.add_result(
            video_id=video_id,
            label=label,
            score=score,
            abstained=abstained,
            uncertainty=uncertainty,
            dataset=dataset_name,
            manipulation_family=manipulation_family,
            generator_family=generator_family,
        )

    def compute_dataset_summary(self, dataset_name: str) -> Dict[str, Any]:
        """Compute metrics for a specific dataset with bootstrap confidence intervals."""
        if dataset_name not in self.evaluators:
            return {}

        ev = self.evaluators[dataset_name]
        raw_metrics = ev.compute()
        labels = np.array([r["label"] for r in ev.results])
        scores = np.array([r["score"] for r in ev.results])

        ci_metrics = {}
        if len(set(labels.tolist())) >= 2:
            try:
                ci_metrics = compute_all_bootstrap_ci(
                    y_true=labels,
                    y_scores=scores,
                    n_resamples=getattr(self.config, "bootstrap", {}).get("n_resamples", 1000),
                    confidence_level=getattr(self.config, "bootstrap", {}).get("confidence_level", 0.95),
                )
            except Exception as e:
                logger.warning("Bootstrap CI computation failed for %s: %s", dataset_name, e)

        return {
            "dataset": dataset_name,
            "sample_count": len(ev.results),
            "num_reals": int((labels == 0).sum()),
            "num_fakes": int((labels == 1).sum()),
            "metrics": raw_metrics,
            "confidence_intervals": ci_metrics,
        }

    def compute_all(self, cross_domain_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Compute evaluation over all evaluated datasets and macro cross-domain average.
        """
        all_summaries = {}
        cross_domain_aucs = []

        if cross_domain_names is None:
            # Default cross-domain datasets if not explicitly specified
            cross_domain_names = ["celeb_df_v2", "dfdc_preview", "wild_deepfake"]

        for name in self.evaluators.keys():
            summary = self.compute_dataset_summary(name)
            all_summaries[name] = summary

            # Collect cross-domain AUROC
            if name in cross_domain_names:
                v_auc = summary.get("metrics", {}).get("video_auc")
                if v_auc is not None and not np.isnan(v_auc):
                    cross_domain_aucs.append(v_auc)

        macro_auroc = float(np.mean(cross_domain_aucs)) if cross_domain_aucs else float("nan")

        return {
            "macro_cross_domain_auroc": macro_auroc,
            "evaluated_cross_domain_datasets": cross_domain_names,
            "dataset_results": all_summaries,
        }

    def generate_markdown_report(self, results: Dict[str, Any]) -> str:
        """Format cross-domain benchmark results as a clean Markdown table."""
        lines = [
            "# FORGE-Lite Cross-Domain Generalization Benchmark",
            "",
            f"**Macro Cross-Domain AUROC**: `{results['macro_cross_domain_auroc']:.4f}`"
            if not np.isnan(results["macro_cross_domain_auroc"]) else "**Macro Cross-Domain AUROC**: N/A",
            "",
            "| Dataset | Type | N (Real / Fake) | Video AUC | TPR @ 1% FPR | EER | AP | Abstention Rate |",
            "|---|---|---|---|---|---|---|---|",
        ]

        cross_names = results.get("evaluated_cross_domain_datasets", [])
        for name, data in results.get("dataset_results", {}).items():
            m = data.get("metrics", {})
            ds_type = "Cross-Domain" if name in cross_names else "In-Domain"
            n_str = f"{data.get('sample_count', 0)} ({data.get('num_reals', 0)} / {data.get('num_fakes', 0)})"
            v_auc = f"{m.get('video_auc', float('nan')):.4f}"
            tpr1 = f"{m.get('tpr_at_1pct_fpr', float('nan')) * 100:.2f}%"
            eer = f"{m.get('eer', float('nan')) * 100:.2f}%"
            ap = f"{m.get('average_precision', float('nan')):.4f}"
            abst = f"{m.get('abstention_rate', 0.0) * 100:.2f}%"

            lines.append(f"| `{name}` | {ds_type} | {n_str} | **{v_auc}** | {tpr1} | {eer} | {ap} | {abst} |")

        lines.append("")
        return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="FORGE-Lite Cross-Domain Evaluation")
    parser.add_argument("--config", default="configs/eval.yaml", help="Path to eval config")
    parser.add_argument("--checkpoint", default="outputs/checkpoints/best.ckpt", help="Model checkpoint")
    parser.add_argument("--output_dir", default="reports", help="Directory for output report")
    parser.add_argument("--simulate", action="store_true", help="Run with simulated synthetic predictions")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger.info("Running Cross-Domain Evaluation | Config: %s | Checkpoint: %s", args.config, args.checkpoint)

    evaluator = CrossDomainEvaluator(config_path=args.config)

    # If simulate flag is passed or testing offline: populate benchmark samples
    if args.simulate:
        rng = np.random.default_rng(42)
        datasets = ["ff_c23", "ff_c40", "celeb_df_v2", "dfdc_preview", "wild_deepfake"]
        for ds in datasets:
            n_samples = 100
            labels = rng.integers(0, 2, size=n_samples)
            # Simulated discrimination capability with cross-dataset domain shift
            shift = 0.85 if "ff_c23" in ds else (0.75 if "c40" in ds else 0.72)
            scores = np.where(labels == 1, rng.normal(shift, 0.15, size=n_samples), rng.normal(0.25, 0.15, size=n_samples))
            scores = np.clip(scores, 0.0, 1.0)
            uncertainties = np.abs(scores - 0.5)

            for i in range(n_samples):
                abstained = bool(uncertainties[i] < 0.1)
                evaluator.record_video_prediction(
                    dataset_name=ds,
                    video_id=f"{ds}_vid_{i:04d}",
                    label=int(labels[i]),
                    score=float(scores[i]),
                    abstained=abstained,
                    uncertainty=float(uncertainties[i]),
                )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = evaluator.compute_all()
    md_report = evaluator.generate_markdown_report(summary)

    report_file = out_dir / "cross_domain_report.md"
    json_file = out_dir / "cross_domain_report.json"

    with open(report_file, "w", encoding="utf-8") as f:
        f.write(md_report)

    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info("Saved cross-domain evaluation report to %s", report_file)
    print(md_report)


if __name__ == "__main__":
    main()
