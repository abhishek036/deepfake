"""
evaluation/ablation_runner.py
Runs all 12 ablation configurations prescribed by the PRD.

Ablation studies validate which components actually help generalisation.
PRD Rule: all ablations use the SAME train/val split, calibration set, and
random seed. Never tune on the held-out test set.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from data.dataset import ForgeDataset, collate_forge
from evaluation.evaluate_video import ForgeEvaluator
from models.forge_lite import FORGELite

logger = logging.getLogger(__name__)


@dataclass
class AblationConfig:
    name: str
    semantic: bool = True
    forensic: bool = True
    temporal: bool = True
    rppg: bool = False
    av: bool = False
    consistency_loss: bool = True
    quality_aware_fusion: bool = True
    unweighted_fusion: bool = False
    metric_learning: bool = True


# 12 PRD-prescribed configurations
ABLATION_CONFIGS = [
    AblationConfig("rgb_only", semantic=True, forensic=False, temporal=False),
    AblationConfig("forensic_only", semantic=False, forensic=True, temporal=False),
    AblationConfig("temporal_only", semantic=False, forensic=False, temporal=True),
    AblationConfig("semantic_forensic", semantic=True, forensic=True, temporal=False),
    AblationConfig("semantic_temporal", semantic=True, forensic=False, temporal=True),
    AblationConfig("semantic_forensic_temporal", semantic=True, forensic=True, temporal=True),
    AblationConfig("no_consistency_loss", consistency_loss=False),
    AblationConfig("no_source_pair_training", metric_learning=False),
    AblationConfig("no_quality_aware_fusion", quality_aware_fusion=False),
    AblationConfig("no_rppg", rppg=False),
    AblationConfig("unweighted_fusion", unweighted_fusion=True, quality_aware_fusion=False),
    AblationConfig("full_forge_lite", rppg=True, av=True),
]


class AblationRunner:
    """Runs all 12 ablation experiments and compiles a results table."""

    def __init__(
        self,
        base_checkpoint: str,
        manifest_path: str,
        output_dir: str = "reports/ablation",
        device: str = "cuda",
        max_samples: Optional[int] = None,
    ) -> None:
        self.base_checkpoint = base_checkpoint
        self.manifest_path = manifest_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.max_samples = max_samples

    def run_all(self) -> dict[str, dict]:
        """Run all 12 ablation configs. Returns results table."""
        results = {}
        for cfg in ABLATION_CONFIGS:
            logger.info("Running ablation: %s", cfg.name)
            try:
                metrics = self._run_single(cfg)
                results[cfg.name] = metrics
                logger.info("  → video_auc=%.4f | cross_domain=%.4f",
                            metrics.get("video_auc", 0), metrics.get("cross_domain_macro_auroc", 0))
            except Exception as e:
                logger.error("Ablation '%s' failed: %s", cfg.name, e)
                results[cfg.name] = {"error": str(e)}

        # Save table
        table_path = self.output_dir / "ablation_table.json"
        with open(table_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info("Ablation table saved: %s", table_path)

        # Print summary
        self._print_table(results)
        return results

    def _run_single(self, ablation_cfg: AblationConfig) -> dict:
        """Run a single ablation configuration."""
        # Build model with this config's branch flags
        model = FORGELite(
            enable_semantic=ablation_cfg.semantic,
            enable_forensic=ablation_cfg.forensic,
            enable_temporal=ablation_cfg.temporal,
            enable_rppg=ablation_cfg.rppg,
            enable_av=ablation_cfg.av,
            enable_localization=False,
            unweighted_fusion=ablation_cfg.unweighted_fusion,
        ).to(self.device).eval()

        # Load base checkpoint weights (strict=False — branches may be absent)
        if Path(self.base_checkpoint).exists():
            ckpt = torch.load(self.base_checkpoint, map_location="cpu")
            model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)

        # Evaluate on val set (NOT test set)
        val_ds = ForgeDataset(
            self.manifest_path, split="val",
            window_size=16, augment=False,
            max_samples=self.max_samples,
        )
        val_loader = DataLoader(val_ds, batch_size=4, shuffle=False,
                                 collate_fn=collate_forge, num_workers=0)

        evaluator = ForgeEvaluator()

        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}
                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    out = model(
                        frames=batch["frames"],
                        frame_diffs=batch["frame_diffs"],
                        face_quality=batch["face_quality"],
                        track_quality=batch["track_quality"],
                        track_continuity=batch["track_continuity"],
                    )

                scores = torch.sigmoid(out["fake_logit"]).cpu().numpy()
                labels = batch["label"].cpu().numpy().astype(int)
                uncertainties = out["uncertainty"].cpu().numpy()
                datasets = batch.get("dataset", ["unknown"] * len(scores))
                manip = batch.get("manipulation_family", ["unknown"] * len(scores))

                for i in range(len(scores)):
                    evaluator.add_result(
                        video_id=f"val_{i}",
                        label=int(labels[i]),
                        score=float(scores[i]),
                        abstained=False,
                        uncertainty=float(uncertainties[i]),
                        dataset=datasets[i] if isinstance(datasets, list) else "val",
                        manipulation_family=manip[i] if isinstance(manip, list) else "unknown",
                    )

        return evaluator.compute()

    def _print_table(self, results: dict) -> None:
        """Print a summary table to console."""
        print("\n" + "═" * 80)
        print(f"{'Config':<35} {'Video AUC':>10} {'Cross-Domain':>14} {'AP':>10}")
        print("─" * 80)
        for name, metrics in results.items():
            if "error" in metrics:
                print(f"{'  ' + name:<35} {'ERROR':>10}")
                continue
            auc = metrics.get("video_auc", float("nan"))
            cd = metrics.get("cross_domain_macro_auroc", float("nan"))
            ap = metrics.get("average_precision", float("nan"))
            print(f"{'  ' + name:<35} {auc:>10.4f} {cd:>14.4f} {ap:>10.4f}")
        print("═" * 80 + "\n")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="outputs/checkpoints/best.ckpt")
    parser.add_argument("--manifest", default="data/manifests/faceforensics.jsonl")
    parser.add_argument("--output-dir", default="reports/ablation")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--single", default=None, help="Run only this ablation config name")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    runner = AblationRunner(
        base_checkpoint=args.checkpoint,
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        device=device,
        max_samples=args.max_samples,
    )

    if args.single:
        config = next((c for c in ABLATION_CONFIGS if c.name == args.single), None)
        if config is None:
            print(f"Unknown ablation: {args.single}. Valid: {[c.name for c in ABLATION_CONFIGS]}")
            return
        metrics = runner._run_single(config)
        print(json.dumps(metrics, indent=2, default=str))
    else:
        runner.run_all()


if __name__ == "__main__":
    main()
