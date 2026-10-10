"""
training/train_spatial.py
Step 7–8 (PRD): Train semantic (DINO) and forensic branches independently,
then jointly on FaceForensics++ C23.

Phase A (semantic): DINOv2 backbone with best adaptation mode
Phase B (forensic): SRM CNN with codec consistency loss
Phase C (joint): semantic + forensic, no temporal yet
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from data.dataset import ForgeDataset, collate_forge, make_dataloader, get_pos_weight
from models.forge_lite import FORGELite
from training.trainer import Trainer
from training.losses import ForgeLoss

logger = logging.getLogger(__name__)


def run_semantic_branch(cfg, manifest_path: str, smoke_test: bool = False) -> dict:
    """Train DINO semantic branch with codec consistency."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 2 if smoke_test else cfg.batch_sizes.spatial

    # Adaptation mode ablation: try all 4 modes and select by cross-domain AUC
    # PRD: never select by in-domain AUC alone
    best_mode = cfg.get("semantic_adaptation_mode", "layernorm")

    model = FORGELite(
        enable_semantic=True,
        enable_forensic=False,
        enable_temporal=False,
        enable_localization=False,
        semantic_cfg={"adaptation_mode": best_mode},
    ).to(device)

    train_ds = ForgeDataset(manifest_path, split="train", window_size=1,
                             augment=True, codec_bank_p=0.5 if not smoke_test else 0.0,
                             max_samples=cfg.smoke_test.max_batches * batch_size if smoke_test else None)
    val_ds = ForgeDataset(manifest_path, split="val", window_size=1, augment=False,
                           max_samples=cfg.smoke_test.max_batches * batch_size if smoke_test else None)

    nw = 0 if smoke_test else cfg.num_workers
    pw = bool(cfg.get("persistent_workers", False)) and nw > 0
    pf = 2 if nw > 0 else None
    train_loader = make_dataloader(train_ds, batch_size=batch_size, shuffle=True,
                                   num_workers=nw, persistent_workers=pw,
                                   pin_memory=cfg.get("pin_memory", True),
                                   cfg=cfg, is_train=True)
    val_loader = make_dataloader(val_ds, batch_size=batch_size, shuffle=False,
                                 num_workers=nw, persistent_workers=pw,
                                 pin_memory=cfg.get("pin_memory", True),
                                 cfg=cfg, is_train=False)

    params = model.get_trainable_parameters("semantic_only") + (list(model.fusion.parameters()) if hasattr(model, "fusion") else [])
    optimizer = torch.optim.AdamW(params, lr=cfg.lr.layernorm_adaptation,
                                   weight_decay=cfg.lr.weight_decay)
    pos_weight = get_pos_weight(cfg).to(device)
    loss_fn = ForgeLoss(use_consistency=True, use_metric=True,
                         use_localization=False, use_temporal=False,
                         pos_weight=pos_weight)

    trainer = Trainer(model=model, optimizer=optimizer, loss_fn=loss_fn, device=device,
                       precision=cfg.precision,
                       gradient_accumulation_steps=1 if smoke_test else cfg.gradient_accumulation_steps,
                       checkpoint_dir=cfg.checkpointing.save_dir,
                       experiment_name=f"{cfg.experiment_name}_semantic",
                       use_wandb=cfg.logging.use_wandb and not smoke_test,
                       early_stopping_patience=cfg.early_stopping.patience,
                       warmup_epochs=1 if smoke_test else cfg.lr.warmup_epochs)

    epochs = cfg.smoke_test.epochs if smoke_test else cfg.epochs.semantic_branch
    return trainer.train(train_loader, val_loader, epochs=epochs, phase="semantic_only")


def run_forensic_branch(cfg, manifest_path: str, smoke_test: bool = False) -> dict:
    """Train forensic CNN branch with codec consistency loss."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 2 if smoke_test else cfg.batch_sizes.spatial

    model = FORGELite(
        enable_semantic=False,
        enable_forensic=True,
        enable_temporal=False,
        enable_localization=False,
    ).to(device)

    train_ds = ForgeDataset(manifest_path, split="train", window_size=1,
                             augment=True, codec_bank_p=0.5 if not smoke_test else 0.0,
                             max_samples=cfg.smoke_test.max_batches * batch_size if smoke_test else None)
    val_ds = ForgeDataset(manifest_path, split="val", window_size=1, augment=False,
                           max_samples=cfg.smoke_test.max_batches * batch_size if smoke_test else None)

    nw = 0 if smoke_test else cfg.num_workers
    pw = bool(cfg.get("persistent_workers", False)) and nw > 0
    train_loader = make_dataloader(train_ds, batch_size=batch_size, shuffle=True,
                                   num_workers=nw, persistent_workers=pw,
                                   pin_memory=cfg.get("pin_memory", True),
                                   cfg=cfg, is_train=True)
    val_loader = make_dataloader(val_ds, batch_size=batch_size, shuffle=False,
                                 num_workers=nw, persistent_workers=pw,
                                 pin_memory=cfg.get("pin_memory", True),
                                 cfg=cfg, is_train=False)

    params = model.get_trainable_parameters("forensic_only") + (list(model.fusion.parameters()) if hasattr(model, "fusion") else [])
    optimizer = torch.optim.AdamW(params, lr=cfg.lr.forensic_branch,
                                   weight_decay=cfg.lr.weight_decay)
    pos_weight = get_pos_weight(cfg).to(device)
    # Consistency training critical for forensic branch: teaches compression ≠ forgery
    loss_fn = ForgeLoss(use_consistency=True, use_metric=False,
                         use_localization=False, use_temporal=False,
                         pos_weight=pos_weight)

    trainer = Trainer(model=model, optimizer=optimizer, loss_fn=loss_fn, device=device,
                       precision=cfg.precision,
                       gradient_accumulation_steps=1 if smoke_test else cfg.gradient_accumulation_steps,
                       checkpoint_dir=cfg.checkpointing.save_dir,
                       experiment_name=f"{cfg.experiment_name}_forensic",
                       use_wandb=cfg.logging.use_wandb and not smoke_test,
                       early_stopping_patience=cfg.early_stopping.patience,
                       warmup_epochs=1 if smoke_test else cfg.lr.warmup_epochs)

    epochs = cfg.smoke_test.epochs if smoke_test else cfg.epochs.forensic_branch
    return trainer.train(train_loader, val_loader, epochs=epochs, phase="forensic_only")


def run_joint_spatial_branch(cfg, manifest_path: str, smoke_test: bool = False) -> dict:
    """Train semantic + forensic branches jointly."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 2 if smoke_test else cfg.batch_sizes.spatial
    best_mode = cfg.get("semantic_adaptation_mode", "layernorm")

    model = FORGELite(
        enable_semantic=True,
        enable_forensic=True,
        enable_temporal=False,
        enable_localization=False,
        semantic_cfg={"adaptation_mode": best_mode},
    ).to(device)

    train_ds = ForgeDataset(manifest_path, split="train", window_size=1,
                             augment=True, codec_bank_p=0.5 if not smoke_test else 0.0,
                             max_samples=cfg.smoke_test.max_batches * batch_size if smoke_test else None)
    val_ds = ForgeDataset(manifest_path, split="val", window_size=1, augment=False,
                           max_samples=cfg.smoke_test.max_batches * batch_size if smoke_test else None)

    # Load best checkpoints from phase 1 & 2 if available
    ckpt_dir = Path(cfg.checkpointing.save_dir)
    sem_ckpt = ckpt_dir / "forge-lite-v1_semantic_semantic_only_best.ckpt"
    for_ckpt = ckpt_dir / "forge-lite-v1_forensic_forensic_only_best.ckpt"
    if sem_ckpt.exists():
        try:
            model.load_state_dict(torch.load(sem_ckpt, map_location=device)["model_state_dict"], strict=False)
            logger.info("Loaded Phase 1 weights from %s", sem_ckpt)
        except Exception as e:
            logger.warning("Could not load Phase 1 weights: %s", e)
    if for_ckpt.exists():
        try:
            model.load_state_dict(torch.load(for_ckpt, map_location=device)["model_state_dict"], strict=False)
            logger.info("Loaded Phase 2 weights from %s", for_ckpt)
        except Exception as e:
            logger.warning("Could not load Phase 2 weights: %s", e)

    nw = 0 if smoke_test else cfg.num_workers
    pw = bool(cfg.get("persistent_workers", False)) and nw > 0
    train_loader = make_dataloader(train_ds, batch_size=batch_size, shuffle=True,
                                   num_workers=nw, persistent_workers=pw,
                                   pin_memory=cfg.get("pin_memory", True),
                                   cfg=cfg, is_train=True)
    val_loader = make_dataloader(val_ds, batch_size=batch_size, shuffle=False,
                                 num_workers=nw, persistent_workers=pw,
                                 pin_memory=cfg.get("pin_memory", True),
                                 cfg=cfg, is_train=False)

    params = model.get_trainable_parameters("joint_spatial")
    optimizer = torch.optim.AdamW(params, lr=cfg.lr.layernorm_adaptation,
                                   weight_decay=cfg.lr.weight_decay)
    pos_weight = get_pos_weight(cfg).to(device)
    loss_fn = ForgeLoss(use_consistency=True, use_metric=True,
                         use_localization=False, use_temporal=False,
                         pos_weight=pos_weight)

    trainer = Trainer(model=model, optimizer=optimizer, loss_fn=loss_fn, device=device,
                       precision=cfg.precision,
                       gradient_accumulation_steps=1 if smoke_test else cfg.gradient_accumulation_steps,
                       checkpoint_dir=cfg.checkpointing.save_dir,
                       experiment_name=f"{cfg.experiment_name}_joint_spatial",
                       use_wandb=cfg.logging.use_wandb and not smoke_test,
                       early_stopping_patience=cfg.early_stopping.patience,
                       warmup_epochs=1 if smoke_test else cfg.lr.warmup_epochs)

    epochs = cfg.smoke_test.epochs if smoke_test else cfg.epochs.semantic_branch
    return trainer.train(train_loader, val_loader, epochs=epochs, phase="joint_spatial")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-train", default="configs/train.yaml")
    parser.add_argument("--manifest", default="data/manifests/faceforensics.jsonl")
    parser.add_argument("--branch", default="semantic", choices=["semantic", "forensic"])
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    cfg = OmegaConf.load(args.config_train)

    if args.branch == "semantic":
        result = run_semantic_branch(cfg, args.manifest, smoke_test=args.smoke_test)
    else:
        result = run_forensic_branch(cfg, args.manifest, smoke_test=args.smoke_test)
    logger.info("Spatial branch training complete: %s", result)


if __name__ == "__main__":
    main()
