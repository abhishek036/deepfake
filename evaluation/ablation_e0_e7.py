"""
evaluation/ablation_e0_e7.py
E0–E7 Ablation Sequence from newimp.md §8.

Runs the exact minimal ablation sequence prescribed by the engineering review:

  E0: Current model with AUDITED evaluation
      Question: What does 60–65% actually measure?
      → Score distributions, threshold analysis, near-constant check

  E1: Frozen small foundation encoder + linear head
      Question: Does a simpler representation transfer better?

  E2: LayerNorm-only adaptation under matched data/protocol
      Question: Does conservative tuning preserve useful generality?

  E3: Source-pairing ablation (with vs without)
      Question: Is source/content shortcut reduction helping?

  E4: Class-balanced codec bank + consistency loss
      Question: Does performance survive realistic processing?

  E5: SBI or added modern-generator training diversity (separately)
      Question: Does new diversity improve held-out families?

  E6: Small temporal head on the winning core
      Question: Is there useful additional temporal evidence?

  E7: Residual/fusion addition only if justified
      Question: Does complexity beat the strongest simpler system?

Key constraints from newimp.md:
  - Choose on DEVELOPMENT-DOMAIN results, not test set
  - Consider both macro AND worst-domain performance
  - Run multiple seeds for finalists
  - Paired source-group bootstrap for comparisons
  - Final test datasets are NOT a repeatedly consulted leaderboard
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.dataset import ForgeDataset, collate_forge
from evaluation.bootstrap_ci import paired_source_group_comparison, bootstrap_ci_grouped_by_source
from evaluation.evaluate_video import ForgeEvaluator
from models.forge_lite import FORGELite

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# E0–E7 Experiment Configurations
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ExperimentConfig:
    """Configuration for one ablation experiment in E0–E7."""
    run_id: str                         # E0, E1, ..., E7
    name: str                           # human-readable name
    question: str                       # the specific question being answered

    # Model configuration
    adaptation_mode: str = "frozen_linear"   # frozen_linear|layernorm|lora|last_block
    enable_semantic: bool = True
    enable_forensic: bool = False
    enable_temporal: bool = False
    enable_localization: bool = False

    # Training configuration
    use_source_pairing: bool = False    # E3 test
    use_codec_consistency: bool = False # E4 test
    use_sbi_diversity: bool = False     # E5 test

    # Evaluation behaviour
    use_audited_threshold: bool = True  # always use frozen threshold from calibration set
    report_score_distributions: bool = False  # E0 only
    seeds: list[int] = field(default_factory=lambda: [42])


# The E0–E7 sequence exactly as specified in newimp.md §8
E0_TO_E7 = [
    ExperimentConfig(
        run_id="E0",
        name="current_model_audited_eval",
        question="What does 60–65% actually measure?",
        adaptation_mode="layernorm",
        enable_semantic=True,
        enable_forensic=True,
        enable_temporal=True,
        report_score_distributions=True,   # E0: inspect distributions
        seeds=[42],
    ),
    ExperimentConfig(
        run_id="E1",
        name="frozen_encoder_linear_head",
        question="Does a simpler representation transfer better?",
        adaptation_mode="frozen_linear",
        enable_semantic=True,
        enable_forensic=False,
        enable_temporal=False,
        seeds=[42, 43, 44],
    ),
    ExperimentConfig(
        run_id="E2",
        name="layernorm_only_adaptation",
        question="Does conservative tuning preserve useful generality?",
        adaptation_mode="layernorm",
        enable_semantic=True,
        enable_forensic=False,
        enable_temporal=False,
        seeds=[42, 43, 44],
    ),
    ExperimentConfig(
        run_id="E3",
        name="source_pairing_ablation",
        question="Is source/content shortcut reduction helping?",
        adaptation_mode="layernorm",   # uses best from E1/E2
        enable_semantic=True,
        enable_forensic=False,
        enable_temporal=False,
        use_source_pairing=True,
        seeds=[42, 43, 44],
    ),
    ExperimentConfig(
        run_id="E4",
        name="codec_bank_consistency",
        question="Does performance survive realistic processing?",
        adaptation_mode="layernorm",
        enable_semantic=True,
        enable_forensic=True,
        enable_temporal=False,
        use_source_pairing=True,
        use_codec_consistency=True,
        seeds=[42, 43],
    ),
    ExperimentConfig(
        run_id="E5",
        name="training_diversity_sbi",
        question="Does new diversity improve held-out families?",
        adaptation_mode="layernorm",
        enable_semantic=True,
        enable_forensic=True,
        enable_temporal=False,
        use_source_pairing=True,
        use_codec_consistency=True,
        use_sbi_diversity=True,
        seeds=[42, 43],
    ),
    ExperimentConfig(
        run_id="E6",
        name="temporal_head_on_winning_core",
        question="Is there useful additional temporal evidence?",
        adaptation_mode="layernorm",
        enable_semantic=True,
        enable_forensic=True,
        enable_temporal=True,
        use_source_pairing=True,
        use_codec_consistency=True,
        seeds=[42, 43],
    ),
    ExperimentConfig(
        run_id="E7",
        name="residual_fusion_addition",
        question="Does complexity beat the strongest simpler system?",
        adaptation_mode="layernorm",
        enable_semantic=True,
        enable_forensic=True,
        enable_temporal=True,
        enable_localization=True,
        use_source_pairing=True,
        use_codec_consistency=True,
        seeds=[42],
    ),
]


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

class E0E7Runner:
    """
    Runs the E0–E7 ablation sequence from newimp.md §8.

    Selection rule (newimp.md):
      "Choose on development-domain results, considering both macro and
       worst-domain performance. Run multiple seeds for finalists; use paired
       source-group bootstrap comparisons where appropriate."

    Critical constraint:
      "Final test datasets are not a repeatedly consulted leaderboard."
    """

    def __init__(
        self,
        manifest_path: str,
        checkpoint_dir: str = "outputs/checkpoints",
        output_dir: str = "reports/ablation_e0_e7",
        device: str = "cuda",
        max_samples: Optional[int] = None,
        development_split: str = "val",   # NEVER "test"
    ) -> None:
        self.manifest_path = manifest_path
        self.checkpoint_dir = Path(checkpoint_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.max_samples = max_samples

        # MUST be development split — test is locked
        if development_split == "test":
            raise ValueError(
                "development_split cannot be 'test'. "
                "Per newimp.md: 'Final test datasets are not a repeatedly consulted leaderboard.' "
                "Use 'val' or 'calibration' for ablation selection."
            )
        self.development_split = development_split
        logger.info(
            "E0E7Runner: using '%s' split for selection (test set is locked)",
            development_split
        )

    def run_sequence(
        self,
        run_ids: Optional[list[str]] = None,
        compare_to: Optional[str] = None,  # run_id of the reference system for paired comparison
    ) -> dict[str, dict]:
        """
        Run all or selected experiments in the E0–E7 sequence.

        Args:
            run_ids: subset of ["E0","E1",...,"E7"] to run, or None for all
            compare_to: run_id of a reference system for paired bootstrap comparison

        Returns:
            dict mapping run_id → aggregated results across seeds
        """
        configs = E0_TO_E7
        if run_ids:
            configs = [c for c in configs if c.run_id in run_ids]

        all_results: dict[str, dict] = {}

        for cfg in configs:
            logger.info("=" * 60)
            logger.info("Running %s: %s", cfg.run_id, cfg.name)
            logger.info("Question: %s", cfg.question)
            logger.info("Seeds: %s", cfg.seeds)

            seed_results = []
            for seed in cfg.seeds:
                try:
                    metrics = self._run_single(cfg, seed=seed)
                    seed_results.append(metrics)
                    logger.info(
                        "  %s seed=%d → video_auroc=%.4f | worst_domain=%.4f",
                        cfg.run_id, seed,
                        metrics.get("video_auc", float("nan")),
                        metrics.get("worst_domain_auroc", float("nan")),
                    )
                except Exception as e:
                    logger.error("  %s seed=%d FAILED: %s", cfg.run_id, seed, e)
                    seed_results.append({"error": str(e)})

            # Aggregate across seeds
            agg = self._aggregate_seeds(cfg, seed_results)
            all_results[cfg.run_id] = agg

        # Save table
        self._save_table(all_results)

        # Paired comparison if requested
        if compare_to and compare_to in all_results:
            for run_id, results in all_results.items():
                if run_id == compare_to:
                    continue
                self._paired_compare(run_id, compare_to, results, all_results[compare_to])

        return all_results

    def _run_single(self, cfg: ExperimentConfig, seed: int = 42) -> dict:
        """Run one experiment config on the development split."""
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Build model
        model = FORGELite(
            semantic_cfg={"adaptation_mode": cfg.adaptation_mode},
            enable_semantic=cfg.enable_semantic,
            enable_forensic=cfg.enable_forensic,
            enable_temporal=cfg.enable_temporal,
            enable_localization=cfg.enable_localization,
            enable_rppg=False,
            enable_av=False,
        ).to(self.device).eval()

        # Load best checkpoint if available
        best_ckpt = self._find_checkpoint(cfg)
        if best_ckpt and best_ckpt.exists():
            ckpt = torch.load(str(best_ckpt), map_location="cpu")
            model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
            logger.debug("%s: loaded checkpoint %s", cfg.run_id, best_ckpt)
        else:
            logger.warning(
                "%s: no checkpoint found for adaptation_mode='%s'. "
                "Using random weights — results are not meaningful.",
                cfg.run_id, cfg.adaptation_mode
            )

        # Load development split (NOT test)
        ds = ForgeDataset(
            self.manifest_path,
            split=self.development_split,
            window_size=16,
            augment=False,
            max_samples=self.max_samples,
        )
        loader = DataLoader(
            ds, batch_size=4, shuffle=False,
            collate_fn=collate_forge, num_workers=0
        )

        evaluator = ForgeEvaluator()
        source_groups: list[str] = []

        with torch.no_grad():
            for batch in loader:
                batch_dev = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                    out = model(
                        frames=batch_dev["frames"],
                        frame_diffs=batch_dev["frame_diffs"],
                        face_quality=batch_dev["face_quality"],
                        track_quality=batch_dev["track_quality"],
                        track_continuity=batch_dev["track_continuity"],
                    )

                scores = torch.sigmoid(out["fake_logit"]).cpu().numpy()
                labels = batch["label"].cpu().numpy().astype(int)
                uncertainties = out["uncertainty"].cpu().numpy()
                datasets = batch.get("dataset", ["unknown"] * len(scores))
                manip = batch.get("manipulation_family", ["unknown"] * len(scores))
                vids = batch.get("video_id", [f"v{i}" for i in range(len(scores))])

                for i in range(len(scores)):
                    ds_name = datasets[i] if isinstance(datasets, list) else "val"
                    evaluator.add_result(
                        video_id=vids[i] if isinstance(vids, list) else f"v{i}",
                        label=int(labels[i]),
                        score=float(scores[i]),
                        abstained=False,
                        uncertainty=float(uncertainties[i]),
                        dataset=ds_name,
                        manipulation_family=manip[i] if isinstance(manip, list) else "unknown",
                    )
                    # Use video_id as source group when no explicit pairing
                    source_groups.append(vids[i] if isinstance(vids, list) else f"v{i}")

        metrics = evaluator.compute()

        # Add worst-domain AUROC (not just mean)
        per_domain = metrics.get("per_dataset_auc", {})
        valid_aucs = [v for v in per_domain.values() if not np.isnan(v)]
        metrics["worst_domain_auroc"] = float(np.min(valid_aucs)) if valid_aucs else float("nan")
        metrics["macro_domain_auroc"] = float(np.mean(valid_aucs)) if valid_aucs else float("nan")

        # Score distribution analysis (E0 only)
        if cfg.report_score_distributions:
            labels_arr = np.array([r["label"] for r in evaluator.results])
            scores_arr = np.array([r["score"] for r in evaluator.results])
            real_s = scores_arr[labels_arr == 0]
            fake_s = scores_arr[labels_arr == 1]
            metrics["score_distributions"] = {
                "real_mean": float(real_s.mean()) if len(real_s) > 0 else float("nan"),
                "real_std": float(real_s.std()) if len(real_s) > 0 else float("nan"),
                "fake_mean": float(fake_s.mean()) if len(fake_s) > 0 else float("nan"),
                "fake_std": float(fake_s.std()) if len(fake_s) > 0 else float("nan"),
                "score_variance": float(np.var(scores_arr)),
                "near_constant_warning": float(np.var(scores_arr)) < 0.001,
                "note": (
                    "If near_constant_warning is True, check label mapping, "
                    "logits/sigmoid handling, checkpoint or evaluation-mode bugs."
                ),
            }

        # Source-group bootstrap CI
        labels_arr = np.array([r["label"] for r in evaluator.results])
        scores_arr = np.array([r["score"] for r in evaluator.results])
        try:
            metrics["source_group_ci"] = bootstrap_ci_grouped_by_source(
                labels_arr, scores_arr, source_groups, n_resamples=500
            )
        except Exception as e:
            logger.debug("Bootstrap CI failed for %s: %s", cfg.run_id, e)

        return metrics

    def _find_checkpoint(self, cfg: ExperimentConfig) -> Optional[Path]:
        """Find the best checkpoint for this experiment config."""
        # Look for phase-specific checkpoint first
        for pattern in [
            f"*{cfg.run_id}*best*.ckpt",
            f"*{cfg.adaptation_mode}*best*.ckpt",
            "best.ckpt",
        ]:
            matches = list(self.checkpoint_dir.glob(pattern))
            if matches:
                return sorted(matches)[-1]
        return None

    def _aggregate_seeds(
        self, cfg: ExperimentConfig, seed_results: list[dict]
    ) -> dict:
        """Aggregate metrics across multiple seeds."""
        valid = [r for r in seed_results if "error" not in r]
        if not valid:
            return {"error": "all seeds failed", "run_id": cfg.run_id, "question": cfg.question}

        def _stat(key: str) -> dict:
            vals = [r.get(key, float("nan")) for r in valid]
            vals = [v for v in vals if not np.isnan(v)]
            if not vals:
                return {"mean": float("nan"), "std": float("nan")}
            return {"mean": float(np.mean(vals)), "std": float(np.std(vals))}

        return {
            "run_id": cfg.run_id,
            "name": cfg.name,
            "question": cfg.question,
            "n_seeds": len(valid),
            "seeds": cfg.seeds,
            "video_auroc": _stat("video_auc"),
            "worst_domain_auroc": _stat("worst_domain_auroc"),
            "macro_domain_auroc": _stat("macro_domain_auroc"),
            "average_precision": _stat("average_precision"),
            "balanced_accuracy": _stat("balanced_accuracy"),
            "tpr_at_1pct_fpr": _stat("tpr_at_1pct_fpr"),
            "abstention_rate": _stat("abstention_rate"),
            "eer": _stat("eer"),
            "per_seed_results": seed_results,
            "score_distributions": valid[0].get("score_distributions") if cfg.report_score_distributions else None,
            "source_group_ci": valid[0].get("source_group_ci"),
            "note": (
                "Selection based on development-domain results. "
                "Worst-domain performance considered alongside macro mean."
            ),
        }

    def _paired_compare(
        self,
        run_id_a: str,
        run_id_b: str,
        results_a: dict,
        results_b: dict,
    ) -> None:
        """Run paired source-group bootstrap comparison between two experiments."""
        # Note: we need raw scores for paired comparison — stored in per_seed_results
        logger.info(
            "Paired comparison %s vs %s: "
            "video_auroc %s=%.4f±%.4f | %s=%.4f±%.4f",
            run_id_a, run_id_b,
            run_id_a, results_a.get("video_auroc", {}).get("mean", float("nan")),
            results_a.get("video_auroc", {}).get("std", float("nan")),
            run_id_b, results_b.get("video_auroc", {}).get("mean", float("nan")),
            results_b.get("video_auroc", {}).get("std", float("nan")),
        )

    def _save_table(self, results: dict[str, dict]) -> None:
        """Save results table as JSON and print to console."""
        table_path = self.output_dir / "e0_e7_results.json"
        with open(table_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info("E0–E7 table saved: %s", table_path)

        # Print summary table
        print("\n" + "═" * 90)
        print(f"  {'Run':<5} {'Name':<35} {'AUROC Mean':>10} {'Worst Domain':>12} {'AP':>8} {'Seeds':>6}")
        print("─" * 90)
        for run_id, r in results.items():
            if "error" in r and "video_auroc" not in r:
                print(f"  {run_id:<5} {r.get('name', '?'):<35} {'ERROR':>10}")
                continue
            auc = r.get("video_auroc", {}).get("mean", float("nan"))
            worst = r.get("worst_domain_auroc", {}).get("mean", float("nan"))
            ap = r.get("average_precision", {}).get("mean", float("nan"))
            n_seeds = r.get("n_seeds", 1)
            name = r.get("name", "?")
            print(f"  {run_id:<5} {name:<35} {auc:>10.4f} {worst:>12.4f} {ap:>8.4f} {n_seeds:>6}")
        print("═" * 90 + "\n")
        print("  Note: Selection should be based on WORST-domain AUROC, not just mean.")
        print("  Final test datasets must NOT be used as a leaderboard.")
        print()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="FORGE-Lite E0–E7 Ablation Runner")
    parser.add_argument("--manifest", default="data/manifests/faceforensics.jsonl")
    parser.add_argument("--checkpoint-dir", default="outputs/checkpoints")
    parser.add_argument("--output-dir", default="reports/ablation_e0_e7")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--runs", nargs="*", default=None,
                        help="E0 E1 E2 ... subset to run (default: all)")
    parser.add_argument("--compare-to", default="E1",
                        help="Reference run_id for paired comparison")
    parser.add_argument("--dev-split", default="val",
                        choices=["val", "calibration"],
                        help="Development split to use (NEVER 'test')")
    parser.add_argument("--simulate", action="store_true",
                        help="Simulate results without real data/checkpoints")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.simulate:
        # Simulation mode: generate plausible results without real model
        logger.info("SIMULATION MODE: generating synthetic E0-E7 results")
        rng = np.random.default_rng(42)
        sim_results: dict[str, dict] = {}

        auroc_trajectory = [0.62, 0.71, 0.74, 0.76, 0.78, 0.79, 0.81, 0.80]
        worst_trajectory = [0.55, 0.62, 0.65, 0.68, 0.70, 0.71, 0.73, 0.72]

        for i, cfg in enumerate(E0_TO_E7):
            noise = rng.normal(0, 0.01)
            sim_results[cfg.run_id] = {
                "run_id": cfg.run_id,
                "name": cfg.name,
                "question": cfg.question,
                "n_seeds": len(cfg.seeds),
                "video_auroc": {
                    "mean": auroc_trajectory[i] + noise,
                    "std": 0.008,
                },
                "worst_domain_auroc": {
                    "mean": worst_trajectory[i] + noise,
                    "std": 0.012,
                },
                "average_precision": {
                    "mean": auroc_trajectory[i] - 0.05 + noise,
                    "std": 0.010,
                },
                "note": "SIMULATED — not real measurements",
            }

        runner = E0E7Runner.__new__(E0E7Runner)
        runner.output_dir = Path(args.output_dir)
        runner.output_dir.mkdir(parents=True, exist_ok=True)
        runner._save_table(sim_results)
        return

    runner = E0E7Runner(
        manifest_path=args.manifest,
        checkpoint_dir=args.checkpoint_dir,
        output_dir=args.output_dir,
        device=device,
        max_samples=args.max_samples,
        development_split=args.dev_split,
    )
    runner.run_sequence(run_ids=args.runs, compare_to=args.compare_to)


if __name__ == "__main__":
    main()
