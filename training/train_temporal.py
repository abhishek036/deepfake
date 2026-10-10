"""
training/train_temporal.py
Step 9 (PRD): Train temporal GRU encoder with frozen spatial branches.

Loads best spatial checkpoint, freezes semantic + forensic branches,
then trains temporal encoder with full multi-statistic aggregation.

Key: per-frame suspiciousness scores are supervised with video-level labels
broadcast to all frames (temporal consistency loss).
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from data.dataset import ForgeDataset, collate_forge, make_dataloader
from models.forge_lite import FORGELite
from training.trainer import Trainer
from training.losses import ForgeLoss

logger = logging.getLogger(__name__)


def run_temporal_training(
    cfg,
    manifest_path: str,
    spatial_checkpoint: str = None,
    smoke_test: bool = False,
) -> dict:
    """
    Train temporal branch with frozen spatial branches.

    Args:
        spatial_checkpoint: Path to best spatial phase checkpoint to resume from
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 2 if smoke_test else cfg.batch_sizes.temporal

    model = FORGELite(
        enable_semantic=True,
        enable_forensic=True,
        enable_temporal=True,
        enable_localization=False,
    ).to(device)

    # Load spatial checkpoint if available
    if not spatial_checkpoint:
        ckpt_dir = Path(cfg.checkpointing.save_dir)
        candidates = [
            ckpt_dir / "forge-lite-v1_joint_spatial_joint_spatial_best.ckpt",
            ckpt_dir / "forge-lite-v1_forensic_forensic_only_best.ckpt",
        ]
        for c in candidates:
            if c.exists():
                spatial_checkpoint = str(c)
                break

    if spatial_checkpoint and Path(spatial_checkpoint).exists():
        ckpt = torch.load(spatial_checkpoint, map_location="cpu")
        state = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state, strict=False)
        logger.info("Loaded spatial checkpoint: %s", spatial_checkpoint)
    else:
        logger.warning("No spatial checkpoint provided — training from scratch")

    # Freeze semantic and forensic branches
    if hasattr(model, "semantic_branch"):
        for p in model.semantic_branch.parameters():
            p.requires_grad = False
        logger.info("Semantic branch frozen")
    if hasattr(model, "forensic_branch"):
        for p in model.forensic_branch.parameters():
            p.requires_grad = False
        logger.info("Forensic branch frozen")

    # Temporal windows for training the GRU (default 8 for 2x faster processing within 8GB VRAM)
    window_size = 4 if smoke_test else cfg.get("temporal_window_size", 8)

    train_ds = ForgeDataset(
        manifest_path, split="train",
        window_size=window_size, window_stride=window_size // 2,
        augment=True, codec_bank_p=0.4 if not smoke_test else 0.0,
        max_samples=cfg.smoke_test.max_batches * batch_size if smoke_test else None,
    )
    val_max = cfg.smoke_test.max_batches * batch_size if smoke_test else cfg.get("max_val_samples", 5000)
    val_ds = ForgeDataset(
        manifest_path, split="val",
        window_size=window_size, augment=False,
        max_samples=val_max,
    )

    nw = 0 if smoke_test else cfg.num_workers
    pw = bool(cfg.get("persistent_workers", False)) and nw > 0
    train_loader = make_dataloader(train_ds, batch_size=batch_size, shuffle=True,
                                   num_workers=nw, persistent_workers=pw,
                                   pin_memory=cfg.get("pin_memory", True) and not smoke_test,
                                   cfg=cfg, is_train=True)
    val_loader = make_dataloader(val_ds, batch_size=batch_size, shuffle=False,
                                 num_workers=nw, persistent_workers=pw,
                                 pin_memory=cfg.get("pin_memory", True),
                                 cfg=cfg, is_train=False)

    # Only temporal + fusion params
    temporal_params = list(model.temporal_encoder.parameters()) + list(model.fusion.parameters())
    optimizer = torch.optim.AdamW(
        temporal_params,
        lr=cfg.lr.temporal_branch,
        weight_decay=cfg.lr.weight_decay,
    )

    loss_fn = ForgeLoss(
        use_consistency=True,
        use_metric=False,
        use_localization=False,
        use_temporal=True,   # broadcast video label to frame-level supervision
        temporal_weight=cfg.loss_weights.temporal_consistency,
    )

    trainer = Trainer(
        model=model, optimizer=optimizer, loss_fn=loss_fn, device=device,
        precision=cfg.precision,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        checkpoint_dir=cfg.checkpointing.save_dir,
        experiment_name=f"{cfg.experiment_name}_temporal",
        use_wandb=cfg.logging.use_wandb and not smoke_test,
        early_stopping_patience=cfg.early_stopping.patience,
        warmup_epochs=1 if smoke_test else cfg.lr.warmup_epochs,
    )

    epochs = cfg.smoke_test.epochs if smoke_test else cfg.epochs.temporal_branch
    return trainer.train(train_loader, val_loader, epochs=epochs, phase="temporal_only")


# Alias for run_training orchestrator
run_temporal_branch = run_temporal_training


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-train", default="configs/train.yaml")
    parser.add_argument("--manifest", default="data/manifests/faceforensics.jsonl")
    parser.add_argument("--spatial-checkpoint", default=None)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    cfg = OmegaConf.load(args.config_train)
    result = run_temporal_training(cfg, args.manifest, args.spatial_checkpoint, args.smoke_test)
    logger.info("Temporal training complete: %s", result)


if __name__ == "__main__":
    main()
