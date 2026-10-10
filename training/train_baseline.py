"""
training/train_baseline.py
Step 6 (PRD): Train baseline binary classifier from a single DINO CLS token.
No forensic/temporal branches. Establishes the performance floor.

Purpose: sets the ceiling for what a frozen backbone with linear head can do.
If this baseline is already near SOTA, the multi-stream ensemble adds less.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from data.dataset import ForgeDataset, collate_forge
from models.dino_branch import DINOSemanticBranch
from training.trainer import Trainer
from training.losses import ForgeLoss

logger = logging.getLogger(__name__)


class BaselineModel(nn.Module):
    """
    Baseline: frozen DINOv2 ViT-S/14 CLS token → linear binary classifier.
    No temporal, no forensic. Simplest possible setup.
    """

    def __init__(self, adaptation_mode: str = "layernorm") -> None:
        super().__init__()
        self.backbone = DINOSemanticBranch(adaptation_mode=adaptation_mode)
        self.classifier = nn.Linear(384, 1)

    def forward(
        self,
        frames: torch.Tensor,               # (B, T, 3, 224, 224)
        frame_diffs: torch.Tensor = None,
        face_quality: torch.Tensor = None,
        track_quality: torch.Tensor = None,
        track_continuity: torch.Tensor = None,
        **kwargs,
    ) -> dict:
        B, T = frames.shape[:2]
        # Mean-pool over all frames for a video-level CLS vector
        frames_flat = frames.view(B * T, 3, 224, 224)
        with torch.cuda.amp.autocast():
            dino_out = self.backbone(frames_flat, return_patch_tokens=False)
        cls = dino_out["cls"].view(B, T, -1).mean(dim=1)  # (B, 384)
        fake_logit = self.classifier(cls).squeeze(-1)      # (B,)
        return {
            "fake_logit": fake_logit,
            "fake_score": torch.sigmoid(fake_logit),
            "uncertainty": torch.zeros_like(fake_logit),
            "evidence_coverage": torch.ones_like(fake_logit),
            "branch_disagreement": torch.zeros_like(fake_logit),
            "branch_weights": torch.zeros(B, 5, device=fake_logit.device),
            "frame_scores": torch.zeros(B, T, device=fake_logit.device),
            "temporal_scores": torch.zeros(B, 5, device=fake_logit.device),
            "pred_mask": None,
            "semantic_embedding": cls,
            "forensic_embedding": torch.zeros(B, 256, device=fake_logit.device),
            "temporal_embedding": torch.zeros(B, 256, device=fake_logit.device),
            "decisions": None,
        }


def run_baseline(cfg, manifest_path: str, smoke_test: bool = False) -> dict:
    """Run baseline training and return best metrics."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Build dataset
    batch_size = 2 if smoke_test else cfg.batch_sizes.baseline
    max_batches = cfg.smoke_test.max_batches if smoke_test else None

    train_ds = ForgeDataset(
        manifest_path=manifest_path,
        split="train",
        window_size=1,  # single frame for baseline
        augment=True,
        max_samples=max_batches * batch_size if max_batches else None,
    )
    val_ds = ForgeDataset(
        manifest_path=manifest_path,
        split="val",
        window_size=1,
        augment=False,
        max_samples=max_batches * batch_size if max_batches else None,
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=collate_forge, num_workers=0 if smoke_test else cfg.num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_forge, num_workers=0 if smoke_test else cfg.num_workers
    )

    # Build model
    model = BaselineModel(
        adaptation_mode="layernorm" if not smoke_test else "frozen_linear"
    ).to(device)

    # Optimizer (only trainable params)
    trainable = model.backbone.get_trainable_parameters() + list(model.classifier.parameters())
    optimizer = torch.optim.AdamW(
        trainable,
        lr=cfg.lr.layernorm_adaptation,
        weight_decay=cfg.lr.weight_decay,
    )

    loss_fn = ForgeLoss(
        use_consistency=False,
        use_metric=False,
        use_localization=False,
        use_temporal=False,
    )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        precision=cfg.precision,
        gradient_accumulation_steps=1 if smoke_test else cfg.gradient_accumulation_steps,
        checkpoint_dir=cfg.checkpointing.save_dir,
        experiment_name=f"{cfg.experiment_name}_baseline",
        use_wandb=cfg.logging.use_wandb and not smoke_test,
        early_stopping_patience=cfg.early_stopping.patience,
        warmup_epochs=1 if smoke_test else cfg.lr.warmup_epochs,
    )

    epochs = cfg.smoke_test.epochs if smoke_test else cfg.epochs.baseline
    return trainer.train(train_loader, val_loader, epochs=epochs, phase="baseline")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-train", default="configs/train.yaml")
    parser.add_argument("--manifest", default="data/manifests/faceforensics.jsonl")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    cfg = OmegaConf.load(args.config_train)
    result = run_baseline(cfg, args.manifest, smoke_test=args.smoke_test)
    logger.info("Baseline complete: %s", result)


if __name__ == "__main__":
    main()
