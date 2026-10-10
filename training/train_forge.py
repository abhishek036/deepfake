"""
training/train_forge.py
=======================
Unified Master Training Entry Point for FORGE-Lite.

Supports:
  --phase spatial       Joint Spatial (Semantic + Forensic) or sub-branches
  --phase semantic      Phase 1: DINOv2 Semantic Branch
  --phase forensic      Phase 2: SRM-CNN Forensic Branch
  --phase joint_spatial Phase 3: Joint Semantic + Forensic Branch
  --phase temporal      Phase 4: Temporal GRU Branch
  --phase fusion        Phase 5: Full Multimodal Fusion + Localization Head
  --phase av            Phase 6: Audio-Visual SyncNet Branch
  --phase all           Full sequential pipeline (Phases 1 -> 6)

Usage:
    python -m training.train_forge --phase spatial --config configs/train.yaml
    python -m training.train_forge --phase spatial --branch forensic --config configs/train.yaml
    python -m training.train_forge --phase temporal --config configs/train.yaml
    python -m training.train_forge --phase fusion --config configs/train.yaml
    python -m training.train_forge --phase all --config configs/train.yaml
    python -m training.train_forge --smoke-test --phase spatial
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional, List, Dict, Any

# ── Windows Fix: Prevent shared-memory page-file exhaustion (Error 1455) ──────
import torch
import torch.multiprocessing as _tmp

try:
    _tmp.set_sharing_strategy("file_system")
except Exception:
    pass
del _tmp

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(ROOT / "training" / "training.log", mode="a", encoding="utf-8"),
    ],
)
logger = logging.getLogger("train_forge")

_CORE_DATASETS = [
    "faceforensics", "celeb_df", "wild_deepfake", "ffhq", "df40", "df40_efs",
    "lav_df", "fakeavceleb", "talkingheadbench",
]


def ensure_combined_manifest(manifest_dir: Path, force_recreate: bool = False) -> str:
    """Ensure a combined manifest exists containing all processed records."""
    combined_path = manifest_dir / "combined.jsonl"
    if combined_path.exists() and not force_recreate:
        size_mb = combined_path.stat().st_size / (1024 * 1024)
        logger.info("Found existing combined manifest: %s (%.1f MB)", combined_path.name, size_mb)
        return str(combined_path)

    logger.info("Building combined manifest from available datasets in %s ...", manifest_dir)
    total_records = 0
    with open(combined_path, "w", encoding="utf-8") as fout:
        for ds in _CORE_DATASETS:
            mpath = manifest_dir / f"{ds}.jsonl"
            if not mpath.exists():
                continue
            count = 0
            with open(mpath, encoding="utf-8") as fin:
                for line in fin:
                    line = line.strip()
                    if line:
                        fout.write(line + "\n")
                        count += 1
            logger.info("  Merged %s (%d records)", ds, count)
            total_records += count

    logger.info("Created combined manifest with %d records -> %s", total_records, combined_path)
    return str(combined_path)


def resolve_manifest_path(manifest_arg: Optional[str], manifest_dir: Path) -> str:
    """Resolve which manifest file to feed into training."""
    if manifest_arg:
        p = Path(manifest_arg)
        if not p.is_absolute():
            p = ROOT / p
        if p.exists():
            return str(p)
        logger.warning("Specified manifest %s not found, searching manifest dir...", manifest_arg)

    # Check combined manifest
    combined = manifest_dir / "combined.jsonl"
    if combined.exists() and combined.stat().st_size > 0:
        return str(combined)

    # Try any single available dataset manifest
    for ds in ["df40", "faceforensics", "celeb_df", "wild_deepfake", "ffhq"]:
        cand = manifest_dir / f"{ds}.jsonl"
        if cand.exists() and cand.stat().st_size > 0:
            logger.info("Using dataset manifest: %s", cand)
            return str(cand)

    # Recreate combined if possible
    return ensure_combined_manifest(manifest_dir)


# ── Phase Runners ─────────────────────────────────────────────────────────────

def run_phase_1(cfg, manifest_path: str, smoke_test: bool = False, **kwargs) -> Dict[str, Any]:
    logger.info("=" * 70)
    logger.info("STARTING PHASE 1: Semantic Branch (DINOv2)")
    logger.info("=" * 70)
    from training.train_spatial import run_semantic_branch
    return run_semantic_branch(cfg, manifest_path, smoke_test=smoke_test)


def run_phase_2(cfg, manifest_path: str, smoke_test: bool = False, **kwargs) -> Dict[str, Any]:
    logger.info("=" * 70)
    logger.info("STARTING PHASE 2: Forensic Branch (SRM-CNN)")
    logger.info("=" * 70)
    from training.train_spatial import run_forensic_branch
    return run_forensic_branch(cfg, manifest_path, smoke_test=smoke_test)


def run_phase_3(cfg, manifest_path: str, smoke_test: bool = False, **kwargs) -> Dict[str, Any]:
    logger.info("=" * 70)
    logger.info("STARTING PHASE 3: Joint Spatial Fusion (Semantic + Forensic)")
    logger.info("=" * 70)
    from training.train_spatial import run_joint_spatial_branch
    return run_joint_spatial_branch(cfg, manifest_path, smoke_test=smoke_test)


def run_phase_4(cfg, manifest_path: str, smoke_test: bool = False, resume_from: Optional[str] = None, **kwargs) -> Dict[str, Any]:
    logger.info("=" * 70)
    logger.info("STARTING PHASE 4: Temporal Branch (GRU)")
    logger.info("=" * 70)
    from training.train_temporal import run_temporal_branch
    return run_temporal_branch(cfg, manifest_path, spatial_checkpoint=resume_from, smoke_test=smoke_test)


def run_phase_5(cfg, manifest_path: str, smoke_test: bool = False, resume_from: Optional[str] = None, **kwargs) -> Dict[str, Any]:
    logger.info("=" * 70)
    logger.info("STARTING PHASE 5: Full Multimodal Fusion + Localization")
    logger.info("=" * 70)
    from training.train_fusion import run_fusion_training
    return run_fusion_training(cfg, manifest_path, temporal_checkpoint=resume_from, smoke_test=smoke_test)


def run_phase_6(cfg, manifest_path: str, smoke_test: bool = False, resume_from: Optional[str] = None, **kwargs) -> Dict[str, Any]:
    logger.info("=" * 70)
    logger.info("STARTING PHASE 6: Audio-Visual Branch (SyncNet)")
    logger.info("=" * 70)
    manifest_dir = Path(manifest_path).parent
    for candidate in ["talkingheadbench.jsonl", "lav_df.jsonl", "lav-df.jsonl", "fakeavceleb.jsonl"]:
        av_p = manifest_dir / candidate
        if av_p.exists():
            manifest_path = str(av_p)
            logger.info("Using dedicated Audio-Visual dataset: %s", manifest_path)
            break
    from training.train_av import run_av_training
    return run_av_training(cfg, manifest_path, resume_checkpoint=resume_from, smoke_test=smoke_test)


PHASE_REGISTRY = {
    1: run_phase_1,
    2: run_phase_2,
    3: run_phase_3,
    4: run_phase_4,
    5: run_phase_5,
    6: run_phase_6,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="FORGE-Lite Unified Training Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--phase",
        default="spatial",
        help="Training phase: spatial | semantic | forensic | joint_spatial | temporal | fusion | av | all (default: spatial)",
    )
    parser.add_argument(
        "--branch",
        default="joint",
        choices=["semantic", "forensic", "joint", "all"],
        help="Branch for spatial phase: semantic (1), forensic (2), joint (3), or all (1->2->3) (default: joint)",
    )
    parser.add_argument(
        "--config", "--config-train",
        dest="config",
        default="configs/train.yaml",
        help="Path to training config YAML (default: configs/train.yaml)",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="Path to jsonl manifest (default: auto-detected or combined.jsonl)",
    )
    parser.add_argument(
        "--manifest-dir",
        default="data/manifests",
        help="Directory where manifests live (default: data/manifests)",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        default=False,
        help="Run fast smoke-test (1 epoch, 2 batches) to verify end-to-end gradient flow",
    )
    parser.add_argument(
        "--resume-from",
        default=None,
        help="Path to checkpoint to initialize / resume weights from",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size from config",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override epoch count from config",
    )
    parser.add_argument(
        "--force-recombine",
        action="store_true",
        default=False,
        help="Rebuild combined.jsonl manifest before starting",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load configuration
    from omegaconf import OmegaConf

    cfg_file = ROOT / args.config
    if not cfg_file.exists():
        logger.error("Configuration file not found: %s", cfg_file)
        sys.exit(1)

    cfg = OmegaConf.load(cfg_file)
    logger.info("Loaded configuration from: %s", cfg_file)

    # CLI Overrides
    if args.batch_size is not None:
        cfg.batch_sizes.spatial = args.batch_size
        cfg.batch_sizes.temporal = args.batch_size
        cfg.batch_sizes.fusion = args.batch_size
        logger.info("Overriding batch_size -> %d", args.batch_size)

    if args.epochs is not None:
        cfg.epochs.semantic_branch = args.epochs
        cfg.epochs.forensic_branch = args.epochs
        cfg.epochs.temporal_branch = args.epochs
        cfg.epochs.fusion = args.epochs
        logger.info("Overriding epochs -> %d", args.epochs)

    manifest_dir = ROOT / args.manifest_dir
    manifest_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = resolve_manifest_path(args.manifest, manifest_dir)
    logger.info("Selected training manifest: %s", manifest_path)

    # Determine phase execution sequence
    phase_input = str(args.phase).lower().strip()
    phases_to_run: List[int] = []

    if phase_input in ("spatial", "joint_spatial", "joint"):
        if args.branch == "semantic":
            phases_to_run = [1]
        elif args.branch == "forensic":
            phases_to_run = [2]
        elif args.branch == "all":
            phases_to_run = [1, 2, 3]
        else:
            # Default for 'spatial' is Joint Spatial (Phase 3)
            phases_to_run = [3]
    elif phase_input in ("semantic", "1"):
        phases_to_run = [1]
    elif phase_input in ("forensic", "2"):
        phases_to_run = [2]
    elif phase_input in ("temporal", "4"):
        phases_to_run = [4]
    elif phase_input in ("fusion", "5"):
        phases_to_run = [5]
    elif phase_input in ("av", "6"):
        phases_to_run = [6]
    elif phase_input == "all":
        phases_to_run = [1, 2, 3, 4, 5, 6]
    else:
        # Check comma separated integers
        try:
            phases_to_run = [int(p.strip()) for p in phase_input.split(",")]
        except ValueError:
            logger.error("Unknown phase specification: '%s'", args.phase)
            logger.info("Valid phases: spatial, semantic, forensic, joint_spatial, temporal, fusion, av, all")
            sys.exit(1)

    logger.info("Executing training sequence: Phases %s", phases_to_run)
    logger.info("Balanced sampling: %s | Precision: %s | Device: %s",
                cfg.get("class_balance", {}).get("balanced_sampling", True),
                cfg.get("precision", "fp16"),
                "cuda" if torch.cuda.is_available() else "cpu")

    results: Dict[str, Any] = {}
    start_time = time.time()

    for p_num in phases_to_run:
        runner_fn = PHASE_REGISTRY.get(p_num)
        if not runner_fn:
            logger.warning("Phase %d not found in registry, skipping", p_num)
            continue

        try:
            phase_res = runner_fn(
                cfg=cfg,
                manifest_path=manifest_path,
                smoke_test=args.smoke_test,
                resume_from=args.resume_from,
            )
            results[f"phase_{p_num}"] = phase_res or {}
            logger.info("Finished Phase %d successfully!", p_num)
        except KeyboardInterrupt:
            logger.warning("Training manually aborted by user at Phase %d", p_num)
            break
        except Exception as e:
            logger.error("Phase %d encountered an error: %s", p_num, e, exc_info=True)
            results[f"phase_{p_num}_error"] = str(e)
            if p_num != phases_to_run[-1]:
                logger.warning("Attempting to proceed to next phase...")

    elapsed = time.time() - start_time
    hours, rem = divmod(elapsed, 3600)
    minutes, secs = divmod(rem, 60)

    logger.info("=" * 70)
    logger.info("FORGE-Lite Training Session Completed in %02dh %02dm %02ds", int(hours), int(minutes), int(secs))
    logger.info("Phase Results: %s", json.dumps(results, indent=2, default=str))
    logger.info("=" * 70)

    # Save summary
    out_dir = ROOT / cfg.get("output_dir", "outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "train_forge_summary.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)


if __name__ == "__main__":
    main()
