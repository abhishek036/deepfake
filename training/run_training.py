"""
training/run_training.py
========================
FORGE-Lite Complete Training Orchestrator

Runs all training phases in sequence:
  Phase 1: Semantic branch (DINO backbone)
  Phase 2: Forensic branch (SRM CNN)
  Phase 3: Joint spatial fusion (Semantic + Forensic)
  Phase 4: Temporal branch
  Phase 5: Full fusion with localization

Usage:
    python -m training.run_training [--config configs/train.yaml]
                                    [--manifest-dir data/manifests]
                                    [--smoke-test]
                                    [--phase 1|2|3|4|5|all]
                                    [--resume-from CHECKPOINT_PATH]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

# ── Windows fix: bypass shared-memory page-file limit (Error 1455) ────────────
# Must be set BEFORE any DataLoader is created.
import torch
import torch.multiprocessing as _tmp
_tmp.set_sharing_strategy("file_system")
del _tmp

# Enable cuDNN autotuner for fastest convolution algorithms
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(ROOT / "training" / "training.log", mode="a"),
    ],
)
logger = logging.getLogger("run_training")


def _verify_manifests(manifest_dir: Path) -> dict:
    """Verify all required manifests exist and count records."""
    info = {}
    for ds in ["faceforensics", "celeb_df", "wild_deepfake", "ffhq", "df40",
                "lav_df", "fakeavceleb", "talkingheadbench"]:
        mpath = manifest_dir / f"{ds}.jsonl"
        if mpath.exists():
            count = sum(1 for _ in open(mpath, encoding="utf-8") if _.strip())
            info[ds] = count
            logger.info("  Manifest %s: %d records", ds, count)
        else:
            logger.debug("  Manifest %s: not found (optional)", ds)
            info[ds] = 0
    return info


def _get_primary_manifest(manifest_dir: Path) -> str:
    """Return the primary training manifest path."""
    combined = manifest_dir / "combined.jsonl"
    if combined.exists():
        return str(combined)
    for ds in ["faceforensics", "celeb_df", "wild_deepfake", "ffhq", "df40"]:
        m = manifest_dir / f"{ds}.jsonl"
        if m.exists():
            return str(m)
    raise FileNotFoundError(f"No manifests found in {manifest_dir}")


_ALL_DATASETS = [
    "faceforensics", "celeb_df", "wild_deepfake", "ffhq", "df40", "df40_efs",
    "lav_df", "fakeavceleb", "talkingheadbench",
]


def _create_combined_manifest(manifest_dir: Path) -> str:
    """Merge all available dataset manifests into one combined.jsonl."""
    combined_path = manifest_dir / "combined.jsonl"
    total = 0
    with open(combined_path, "w", encoding="utf-8") as fout:
        for ds in _ALL_DATASETS:
            mpath = manifest_dir / f"{ds}.jsonl"
            if not mpath.exists():
                continue
            with open(mpath, encoding="utf-8") as fin:
                for line in fin:
                    line = line.strip()
                    if line:
                        fout.write(line + "\n")
                        total += 1
            logger.info("  Added %s to combined manifest", ds)
    logger.info("Combined manifest: %d total records -> %s", total, combined_path)
    return str(combined_path)


def phase_1_semantic(cfg, manifest_path, smoke_test):
    """Phase 1: Train DINO semantic branch."""
    logger.info("=" * 60)
    logger.info("PHASE 1: Semantic Branch (DINO)")
    logger.info("=" * 60)
    from training.train_spatial import run_semantic_branch
    results = run_semantic_branch(cfg, manifest_path, smoke_test=smoke_test)
    return results


def phase_2_forensic(cfg, manifest_path, smoke_test):
    """Phase 2: Train forensic SRM-CNN branch."""
    logger.info("=" * 60)
    logger.info("PHASE 2: Forensic Branch (SRM-CNN)")
    logger.info("=" * 60)
    from training.train_spatial import run_forensic_branch
    results = run_forensic_branch(cfg, manifest_path, smoke_test=smoke_test)
    return results


def phase_3_joint_spatial(cfg, manifest_path, smoke_test):
    """Phase 3: Joint semantic + forensic."""
    logger.info("=" * 60)
    logger.info("PHASE 3: Joint Spatial (Semantic + Forensic)")
    logger.info("=" * 60)
    from training.train_spatial import run_joint_spatial_branch
    results = run_joint_spatial_branch(cfg, manifest_path, smoke_test=smoke_test)
    return results


def phase_4_temporal(cfg, manifest_path, smoke_test):
    """Phase 4: Temporal branch training."""
    logger.info("=" * 60)
    logger.info("PHASE 4: Temporal Branch")
    logger.info("=" * 60)
    from training.train_temporal import run_temporal_branch
    results = run_temporal_branch(cfg, manifest_path, smoke_test=smoke_test)
    return results


def phase_5_fusion(cfg, manifest_path, smoke_test):
    """Phase 5: Full fusion with localization."""
    logger.info("=" * 60)
    logger.info("PHASE 5: Full Fusion + Localization")
    logger.info("=" * 60)
    from training.train_fusion import run_full_fusion
    results = run_full_fusion(cfg, manifest_path, smoke_test=smoke_test)
    return results


def phase_6_av(cfg, manifest_path, smoke_test, resume_from=None):
    """Phase 6: Audio-Visual Branch Training."""
    logger.info("=" * 60)
    logger.info("PHASE 6: Audio-Visual Branch")
    logger.info("=" * 60)
    manifest_dir = Path(manifest_path).parent
    for candidate in ["talkingheadbench.jsonl", "lav_df.jsonl", "lav-df.jsonl", "fakeavceleb.jsonl"]:
        av_p = manifest_dir / candidate
        if av_p.exists():
            manifest_path = str(av_p)
            logger.info("Using dedicated Audio-Visual dataset manifest: %s", manifest_path)
            break
    if resume_from is None:
        cand = Path("outputs/checkpoints") / f"{cfg.experiment_name}_av_branch_best.ckpt"
        if cand.exists():
            resume_from = str(cand)
            logger.info("Fine-tuning from pre-trained AV checkpoint: %s", resume_from)
    from training.train_av import run_av_training
    results = run_av_training(cfg, manifest_path, resume_checkpoint=resume_from, smoke_test=smoke_test)
    return results


PHASE_MAP = {
    1: phase_1_semantic,
    2: phase_2_forensic,
    3: phase_3_joint_spatial,
    4: phase_4_temporal,
    5: phase_5_fusion,
    6: phase_6_av,
}


def main():
    ap = argparse.ArgumentParser(description="FORGE-Lite Training Orchestrator")
    ap.add_argument("--config", default="configs/train.yaml", help="Training config YAML")
    ap.add_argument("--manifest-dir", default="data/manifests", help="Manifest directory")
    ap.add_argument("--smoke-test", action="store_true", default=False,
                    help="Quick smoke-test mode (2 batches, 1 epoch)")
    ap.add_argument("--phase", default="all",
                    help="Which phase(s) to run: 1|2|3|4|5|6|all (default: all)")
    ap.add_argument("--combined", action="store_true", default=False,
                    help="Merge all dataset manifests into one combined.jsonl")
    ap.add_argument("--resume-from", default=None, help="Resume from checkpoint path")
    args = ap.parse_args()

    # ── Load config ──────────────────────────────────────────────────────────
    from omegaconf import OmegaConf
    cfg_path = ROOT / args.config
    if not cfg_path.exists():
        logger.error("Config not found: %s", cfg_path)
        sys.exit(1)
    cfg = OmegaConf.load(cfg_path)
    logger.info("Config loaded: %s", cfg_path)

    # ── Check manifests ───────────────────────────────────────────────────────
    manifest_dir = ROOT / args.manifest_dir
    logger.info("Manifest directory: %s", manifest_dir)
    manifest_info = _verify_manifests(manifest_dir)
    total_records = sum(manifest_info.values())

    if total_records == 0:
        logger.error("No manifest records found! Run preprocessing first:")
        logger.error("  python -m preprocessing.preprocess_all --dataset all")
        sys.exit(1)

    # ── Choose manifest path ──────────────────────────────────────────────────
    if args.combined or total_records > 0:
        manifest_path = _create_combined_manifest(manifest_dir)
    else:
        manifest_path = _get_primary_manifest(manifest_dir)

    logger.info("Training manifest: %s", manifest_path)

    # ── Determine which phases to run ─────────────────────────────────────────
    if args.phase.lower() == "all":
        phases_to_run = [1, 2, 3, 4, 5, 6]
    else:
        try:
            phases_to_run = [int(p) for p in args.phase.split(",")]
        except ValueError:
            logger.error("Invalid --phase argument: %s", args.phase)
            sys.exit(1)

    # ── Run phases ────────────────────────────────────────────────────────────
    results = {}
    start_time = time.time()
    for phase_num in phases_to_run:
        if phase_num not in PHASE_MAP:
            logger.warning("Unknown phase %d, skipping", phase_num)
            continue
        phase_fn = PHASE_MAP[phase_num]
        try:
            import inspect
            sig = inspect.signature(phase_fn)
            kwargs = {"smoke_test": args.smoke_test}
            if "resume_from" in sig.parameters:
                kwargs["resume_from"] = args.resume_from
            phase_result = phase_fn(cfg, manifest_path, **kwargs)
            results[f"phase_{phase_num}"] = phase_result or {}
            logger.info("Phase %d complete: %s", phase_num, phase_result)
        except KeyboardInterrupt:
            logger.warning("Training interrupted at phase %d", phase_num)
            break
        except Exception as e:
            logger.error("Phase %d failed: %s", phase_num, e, exc_info=True)
            results[f"phase_{phase_num}_error"] = str(e)
            # Ask user whether to continue
            logger.warning("Continuing to next phase despite error...")

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed = time.time() - start_time
    hours, rem = divmod(elapsed, 3600)
    minutes, secs = divmod(rem, 60)
    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE in %dh %dm %ds", int(hours), int(minutes), int(secs))
    logger.info("Results: %s", json.dumps(results, default=str, indent=2))
    logger.info("=" * 60)

    # Save results
    results_path = ROOT / "outputs" / "training_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, default=str, indent=2)
    logger.info("Results saved to %s", results_path)


if __name__ == "__main__":
    main()
