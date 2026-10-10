"""
training/train_fusion.py
Step 10 (PRD): End-to-end joint fine-tuning of all branches + fusion head.

Loads temporal checkpoint, jointly fine-tunes all components with:
- Full composite loss (classification + consistency + metric + localization + temporal)
- Codec consistency training
- Source-pair metric learning
- Localization head training on valid masks

After this phase: run calibration training (train_calibration.py).
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


def run_fusion_training(
    cfg,
    manifest_path: str,
    temporal_checkpoint: str = None,
    smoke_test: bool = False,
) -> dict:
    """
    End-to-end fine-tuning: all branches + fusion head + localization head.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 2 if smoke_test else cfg.batch_sizes.fusion

    model = FORGELite(
        enable_semantic=True,
        enable_forensic=True,
        enable_temporal=True,
        enable_localization=True,
    ).to(device)

    # Load best checkpoint from previous phases
    if not temporal_checkpoint:
        ckpt_dir = Path(cfg.checkpointing.save_dir)
        candidates = [
            ckpt_dir / "forge-lite-v1_temporal_temporal_only_best.ckpt",
            ckpt_dir / "forge-lite-v1_joint_spatial_joint_spatial_best.ckpt",
            ckpt_dir / "forge-lite-v1_forensic_forensic_only_best.ckpt",
        ]
        for c in candidates:
            if c.exists():
                temporal_checkpoint = str(c)
                break

    if temporal_checkpoint and Path(temporal_checkpoint).exists():
        ckpt = torch.load(temporal_checkpoint, map_location="cpu")
        model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
        logger.info("Loaded checkpoint: %s", temporal_checkpoint)

    window_size = 4 if smoke_test else cfg.get("fusion_window_size", 8)

    train_ds = ForgeDataset(
        manifest_path, split="train",
        window_size=window_size, window_stride=window_size // 2,
        augment=True,
        codec_bank_p=cfg.consistency_training.codec_configs_to_use and 0.5,
        max_samples=cfg.smoke_test.max_batches * batch_size if smoke_test else None,
    )
    val_max = cfg.smoke_test.max_batches * batch_size if smoke_test else cfg.get("max_val_samples", 5000)
    val_ds = ForgeDataset(
        manifest_path, split="val", window_size=window_size, augment=False,
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

    # All parameters with differential LR
    lw = cfg.loss_weights
    param_groups = [
        {"params": model.get_trainable_parameters("semantic_only"), "lr": cfg.lr.last_block_adaptation},
        {"params": model.get_trainable_parameters("forensic_only"), "lr": cfg.lr.forensic_branch},
        {"params": model.get_trainable_parameters("temporal_only"), "lr": cfg.lr.temporal_branch},
        {"params": model.get_trainable_parameters("fusion"), "lr": cfg.lr.fusion},
    ]

    optimizer = torch.optim.AdamW(param_groups, weight_decay=cfg.lr.weight_decay)

    loss_fn = ForgeLoss(
        classification_weight=lw.classification,
        consistency_weight=lw.consistency,
        metric_weight=lw.metric,
        localization_weight=lw.localization,
        temporal_weight=lw.temporal_consistency,
        focal_alpha=lw.focal_alpha,
        focal_gamma=lw.focal_gamma,
        use_consistency=cfg.consistency_training.enabled,
        use_metric=True,
        use_localization=True,
        use_temporal=True,
    )

    trainer = Trainer(
        model=model, optimizer=optimizer, loss_fn=loss_fn, device=device,
        precision=cfg.precision,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        checkpoint_dir=cfg.checkpointing.save_dir,
        experiment_name=f"{cfg.experiment_name}_fusion",
        use_wandb=cfg.logging.use_wandb and not smoke_test,
        early_stopping_patience=cfg.early_stopping.patience,
        warmup_epochs=1 if smoke_test else cfg.lr.warmup_epochs,
        min_lr=cfg.lr.min_lr,
    )

    epochs = cfg.smoke_test.epochs if smoke_test else cfg.epochs.fusion
    return trainer.train(
        train_loader, val_loader,
        epochs=epochs, phase="fusion",
        val_metric_key="val/auc",
    )


def run_calibration_training(
    cfg,
    manifest_path: str,
    fusion_checkpoint: str,
    smoke_test: bool = False,
) -> dict:
    """
    Step 14 (PRD): Temperature scaling calibration.
    Fits temperature on a held-out calibration set (val split).
    Must NOT use the locked test set.
    """
    from models.calibration import CalibrationLayer
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = FORGELite(
        enable_semantic=True, enable_forensic=True,
        enable_temporal=True, enable_localization=False,
    ).to(device).eval()

    if Path(fusion_checkpoint).exists():
        ckpt = torch.load(fusion_checkpoint, map_location="cpu")
        model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
        logger.info("Loaded fusion checkpoint for calibration: %s", fusion_checkpoint)

    val_ds = ForgeDataset(manifest_path, split="val", window_size=8, augment=False)
    nw = cfg.get("num_workers", 2)
    pw = bool(cfg.get("persistent_workers", False)) and nw > 0
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False,
                             collate_fn=collate_forge, num_workers=nw,
                             persistent_workers=pw, pin_memory=True)

    # Collect logits on validation set
    all_logits, all_labels = [], []
    from tqdm import tqdm
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Calibrating"):
            batch = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            with torch.amp.autocast("cuda"):
                out = model(
                    frames=batch["frames"], frame_diffs=batch["frame_diffs"],
                    face_quality=batch["face_quality"], track_quality=batch["track_quality"],
                    track_continuity=batch["track_continuity"],
                )
            all_logits.append(out["fake_logit"].cpu())
            all_labels.append(batch["label"].cpu())

    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)

    # Fit temperature
    cal = CalibrationLayer()
    temperature = cal.calibrate(logits, labels)
    logger.info("Calibration complete: temperature=%.4f", temperature)

    # Save calibrated temperature to checkpoint
    ckpt_out = Path(cfg.checkpointing.save_dir) / "calibration_temperature.pt"
    torch.save({"temperature": temperature}, ckpt_out)
    logger.info("Saved calibration: %s", ckpt_out)
    return {"temperature": temperature}


# Alias for run_training orchestrator
run_full_fusion = run_fusion_training


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-train", default="configs/train.yaml")
    parser.add_argument("--manifest", default="data/manifests/faceforensics.jsonl")
    parser.add_argument("--temporal-checkpoint", default="outputs/checkpoints/best.ckpt")
    parser.add_argument("--mode", default="fusion", choices=["fusion", "calibration"])
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    cfg = OmegaConf.load(args.config_train)

    if args.mode == "fusion":
        result = run_fusion_training(cfg, args.manifest, args.temporal_checkpoint, args.smoke_test)
    else:
        result = run_calibration_training(cfg, args.manifest, args.temporal_checkpoint, args.smoke_test)
    logger.info("Done: %s", result)


if __name__ == "__main__":
    main()
