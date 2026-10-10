"""
training/adversarial_train.py
Step 13 (PRD): Adversarial training with PGD for robustness.

Applies PGD-5 perturbations during training (on fraction of batches).
Evaluates with PGD-20 during validation.

Warning (PRD): adversarial training can slightly reduce clean accuracy.
Monitor both clean and adversarial AUC. If clean accuracy drops > 5%,
reduce adversarial fraction.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from data.dataset import ForgeDataset, collate_forge
from models.forge_lite import FORGELite
from training.trainer import Trainer
from training.losses import ForgeLoss

logger = logging.getLogger(__name__)


def pgd_attack(
    model: nn.Module,
    frames: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float = 0.03,
    step_size: float = 0.007,
    num_steps: int = 5,
    random_start: bool = True,
    **model_kwargs,
) -> torch.Tensor:
    """
    Project Gradient Descent (PGD) adversarial attack.

    Generates adversarial examples in L-inf ball of radius epsilon.
    Applied to frame inputs only (not quality vectors).

    Args:
        model: FORGELite model
        frames: (B, T, 3, 224, 224) clean frames
        labels: (B,) binary labels
        epsilon: L-inf perturbation bound
        step_size: PGD step size
        num_steps: PGD iterations
        random_start: initialize from random perturbation

    Returns:
        adversarial frames (B, T, 3, 224, 224), same dtype as input
    """
    B, T = frames.shape[:2]
    adv_frames = frames.clone().detach()

    if random_start:
        delta = torch.empty_like(adv_frames).uniform_(-epsilon, epsilon)
        adv_frames = (adv_frames + delta).clamp(0, 1)

    for _ in range(num_steps):
        adv_frames.requires_grad_(True)
        adv_diffs = adv_frames[:, 1:] - adv_frames[:, :-1]

        with torch.cuda.amp.autocast():
            out = model(frames=adv_frames, frame_diffs=adv_diffs, **model_kwargs)

        loss = F.binary_cross_entropy_with_logits(
            out["fake_logit"], labels.float()
        )
        loss.backward()

        grad = adv_frames.grad.detach()
        adv_frames = adv_frames.detach() + step_size * grad.sign()
        # Project back into epsilon ball
        adv_frames = (adv_frames - frames).clamp(-epsilon, epsilon) + frames
        adv_frames = adv_frames.clamp(0, 1).detach()

    return adv_frames


class AdversarialTrainer(Trainer):
    """
    Extends Trainer with PGD adversarial batch injection.
    """

    def __init__(
        self,
        *args,
        pgd_epsilon: float = 0.03,
        pgd_step_size: float = 0.007,
        pgd_steps_train: int = 5,
        apply_to_fraction: float = 0.5,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.pgd_epsilon = pgd_epsilon
        self.pgd_step_size = pgd_step_size
        self.pgd_steps_train = pgd_steps_train
        self.apply_to_fraction = apply_to_fraction

    def _train_epoch(self, loader: DataLoader, epoch: int, phase: str) -> dict:
        self.model.train()
        total_loss = 0.0
        adv_batches = 0
        n_batches = 0
        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(loader):
            batch = self._to_device(batch)
            frames = batch["frames"]
            labels = batch["label"]
            face_quality = batch["face_quality"]
            track_quality = batch["track_quality"]
            track_continuity = batch["track_continuity"]

            # Apply PGD to a fraction of batches
            import random
            if random.random() < self.apply_to_fraction:
                adv_frames = pgd_attack(
                    model=self.model,
                    frames=frames,
                    labels=labels,
                    epsilon=self.pgd_epsilon,
                    step_size=self.pgd_step_size,
                    num_steps=self.pgd_steps_train,
                    random_start=True,
                    face_quality=face_quality,
                    track_quality=track_quality,
                    track_continuity=track_continuity,
                )
                frames = adv_frames
                adv_batches += 1

            adv_diffs = frames[:, 1:] - frames[:, :-1]

            with torch.cuda.amp.autocast(enabled=(self.precision == "fp16")):
                outputs = self.model(
                    frames=frames, frame_diffs=adv_diffs,
                    face_quality=face_quality, track_quality=track_quality,
                    track_continuity=track_continuity,
                )
                losses = self.loss_fn(
                    fake_logit=outputs["fake_logit"],
                    labels=labels,
                    semantic_embedding=outputs.get("semantic_embedding"),
                )
                loss = losses["total"] / self.grad_accum_steps

            self.scaler.scale(loss).backward()
            if (batch_idx + 1) % self.grad_accum_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            total_loss += float(losses["total"].item())
            n_batches += 1

        logger.info("Adversarial batches: %d / %d (%.0f%%)",
                    adv_batches, n_batches, 100 * adv_batches / max(n_batches, 1))
        return {"total_loss": total_loss / max(n_batches, 1), "adv_fraction": adv_batches / max(n_batches, 1)}


def run_adversarial_training(
    cfg,
    manifest_path: str,
    fusion_checkpoint: str,
    smoke_test: bool = False,
) -> dict:
    """Train with PGD adversarial augmentation."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 2 if smoke_test else cfg.batch_sizes.adversarial

    model = FORGELite(
        enable_semantic=True, enable_forensic=True,
        enable_temporal=True, enable_localization=False,
    ).to(device)

    if Path(fusion_checkpoint).exists():
        ckpt = torch.load(fusion_checkpoint, map_location="cpu")
        model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
        logger.info("Loaded fusion checkpoint: %s", fusion_checkpoint)

    window_size = 4 if smoke_test else 16
    train_ds = ForgeDataset(manifest_path, split="train", window_size=window_size, augment=True,
                             max_samples=cfg.smoke_test.max_batches * batch_size if smoke_test else None)
    val_ds = ForgeDataset(manifest_path, split="val", window_size=window_size, augment=False,
                           max_samples=cfg.smoke_test.max_batches * batch_size if smoke_test else None)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               collate_fn=collate_forge, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                             collate_fn=collate_forge, num_workers=0)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=cfg.lr.adversarial, weight_decay=cfg.lr.weight_decay)
    loss_fn = ForgeLoss(use_consistency=False, use_metric=False,
                         use_localization=False, use_temporal=True)

    adv_cfg = cfg.adversarial
    trainer = AdversarialTrainer(
        model=model, optimizer=optimizer, loss_fn=loss_fn, device=device,
        precision=cfg.precision,
        gradient_accumulation_steps=1,
        checkpoint_dir=cfg.checkpointing.save_dir,
        experiment_name=f"{cfg.experiment_name}_adversarial",
        use_wandb=cfg.logging.use_wandb and not smoke_test,
        early_stopping_patience=cfg.early_stopping.patience,
        pgd_epsilon=adv_cfg.pgd_epsilon,
        pgd_step_size=adv_cfg.pgd_step_size,
        pgd_steps_train=2 if smoke_test else adv_cfg.pgd_steps_train,
        apply_to_fraction=0.2 if smoke_test else adv_cfg.apply_to_fraction,
    )

    epochs = cfg.smoke_test.epochs if smoke_test else cfg.epochs.adversarial
    return trainer.train(train_loader, val_loader, epochs=epochs, phase="adversarial")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-train", default="configs/train.yaml")
    parser.add_argument("--manifest", default="data/manifests/faceforensics.jsonl")
    parser.add_argument("--fusion-checkpoint", default="outputs/checkpoints/best.ckpt")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    cfg = OmegaConf.load(args.config_train)
    result = run_adversarial_training(cfg, args.manifest, args.fusion_checkpoint, args.smoke_test)
    logger.info("Adversarial training complete: %s", result)


if __name__ == "__main__":
    main()
