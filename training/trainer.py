"""
training/trainer.py
Shared trainer class for all FORGE-Lite training phases.

Handles:
- AMP fp16 training
- Gradient accumulation
- Wandb logging
- Checkpoint management
- LR scheduling (cosine warmup)
- Early stopping
- Phase-specific parameter groups
"""

from __future__ import annotations

import logging
import math
import os
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

logger = logging.getLogger(__name__)


class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 0.001, mode: str = "max") -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_value = float("-inf") if mode == "max" else float("inf")
        self.counter = 0
        self.should_stop = False

    def __call__(self, value: float) -> bool:
        improved = (
            value > self.best_value + self.min_delta
            if self.mode == "max"
            else value < self.best_value - self.min_delta
        )
        if improved:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


class CosineWarmupScheduler:
    """Cosine LR schedule with linear warmup."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        min_lr: float = 1e-6,
    ) -> None:
        import math
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]
        self._step = 0

    def step(self, epoch: int) -> None:
        import math
        if epoch < self.warmup_epochs:
            factor = (epoch + 1) / max(self.warmup_epochs, 1)
        else:
            progress = (epoch - self.warmup_epochs) / max(self.total_epochs - self.warmup_epochs, 1)
            factor = self.min_lr + 0.5 * (1 - self.min_lr) * (1 + math.cos(math.pi * progress))
        for base_lr, pg in zip(self.base_lrs, self.optimizer.param_groups):
            pg["lr"] = base_lr * factor


class Trainer:
    """
    Generic trainer for FORGE-Lite training phases.

    Usage:
        trainer = Trainer(model, optimizer, loss_fn, cfg)
        trainer.train(train_loader, val_loader, epochs=30, phase="semantic_only")
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
        device: str = "cuda",
        precision: str = "fp16",
        gradient_accumulation_steps: int = 4,
        clip_grad_norm: float = 1.0,
        checkpoint_dir: str = "outputs/checkpoints",
        experiment_name: str = "forge-lite",
        use_wandb: bool = True,
        early_stopping_patience: int = 10,
        warmup_epochs: int = 2,
        min_lr: float = 1e-6,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.precision = precision
        self.grad_accum_steps = gradient_accumulation_steps
        self.clip_grad_norm = clip_grad_norm
        self.checkpoint_dir = Path(checkpoint_dir)
        self.experiment_name = experiment_name
        self.warmup_epochs = warmup_epochs
        self.min_lr = min_lr

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # AMP scaler
        device_type = "cuda" if "cuda" in str(device) else "cpu"
        self.scaler = torch.amp.GradScaler(device_type, enabled=(precision == "fp16" and device_type == "cuda"))

        # Early stopping
        self.early_stopper = EarlyStopping(patience=early_stopping_patience)

        # Wandb
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        self._best_val_metric = float("-inf")
        self._best_ckpt_path = None

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        epochs: int,
        phase: str = "fusion",
        val_metric_key: str = "val/auc",
        warmup_epochs: Optional[int] = None,
    ) -> dict:
        """
        Run training for a given number of epochs.

        Returns dict of best metrics.
        """
        warmup = warmup_epochs or self.warmup_epochs
        scheduler = CosineWarmupScheduler(
            self.optimizer, warmup, epochs, self.min_lr
        )

        logger.info("Training phase='%s' | epochs=%d | device=%s", phase, epochs, self.device)
        history = {"train_loss": [], "val_metric": []}

        for epoch in range(epochs):
            t0 = time.time()
            train_metrics = self._train_epoch(train_loader, epoch, phase)
            epoch_time = time.time() - t0

            val_metrics = {}
            if val_loader is not None:
                val_metrics = self._val_epoch(val_loader, epoch, phase)

            scheduler.step(epoch)

            # Logging
            log_dict = {
                "epoch": epoch,
                "epoch_time_s": epoch_time,
                **{f"train/{k}": v for k, v in train_metrics.items()},
                **val_metrics,
            }
            if self.use_wandb:
                wandb.log(log_dict)
            logger.info(
                "Epoch %3d/%3d | train_loss=%.4f | %s | %.1fs",
                epoch + 1, epochs,
                train_metrics.get("total_loss", 0),
                " | ".join(f"{k}={v:.4f}" for k, v in val_metrics.items()),
                epoch_time,
            )

            # Checkpoint
            val_metric = val_metrics.get(val_metric_key, 0.0)
            history["train_loss"].append(train_metrics.get("total_loss", 0))
            history["val_metric"].append(val_metric)

            if val_metric > self._best_val_metric:
                self._best_val_metric = val_metric
                self._save_checkpoint(epoch, val_metric, phase, is_best=True)

            if epoch % 5 == 0:
                self._save_checkpoint(epoch, val_metric, phase, is_best=False)

            # Early stopping
            if self.early_stopper(val_metric):
                logger.info("Early stopping at epoch %d", epoch)
                break

        return {"best_val_metric": self._best_val_metric, "history": history}

    def _train_epoch(self, loader: DataLoader, epoch: int, phase: str) -> dict:
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        self.model.zero_grad(set_to_none=True)

        for batch_idx, batch in enumerate(tqdm(loader, desc=f"Train epoch {epoch}", leave=False)):
            batch = self._to_device(batch)

            device_type = "cuda" if "cuda" in str(self.device) else "cpu"
            with torch.amp.autocast(device_type=device_type, enabled=(self.precision == "fp16")):
                outputs = self._forward(batch, phase)
                losses = self.loss_fn(
                    fake_logit=outputs["fake_logit"],
                    labels=batch["label"],
                    pred_mask=outputs.get("pred_mask"),
                    gt_mask=batch.get("mask"),
                    mask_available=batch.get("mask_available"),
                    frame_scores=outputs.get("frame_scores"),
                    semantic_embedding=outputs.get("semantic_embedding"),
                    branch_logits=outputs.get("branch_logits"),
                    availability_mask=outputs.get("availability_mask"),
                )
                loss = losses["total"] / self.grad_accum_steps

            loss_val = float(losses["total"].item())
            if math.isnan(loss_val) or math.isinf(loss_val):
                logger.warning("NaN/Inf loss at batch %d (epoch %d), skipping step", batch_idx, epoch)
                self.model.zero_grad(set_to_none=True)
                continue

            self.scaler.scale(loss).backward()

            if (batch_idx + 1) % self.grad_accum_steps == 0:
                self.scaler.unscale_(self.optimizer)
                trainable_params = [
                    p for group in self.optimizer.param_groups for p in group["params"] if p.grad is not None
                ]
                if trainable_params:
                    torch.nn.utils.clip_grad_norm_(trainable_params, self.clip_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.model.zero_grad(set_to_none=True)

            total_loss += loss_val
            n_batches += 1

        return {"total_loss": total_loss / max(n_batches, 1)}

    @torch.no_grad()
    def _val_epoch(self, loader: DataLoader, epoch: int, phase: str) -> dict:
        from sklearn.metrics import roc_auc_score
        self.model.eval()
        all_scores, all_labels = [], []

        device_type = "cuda" if "cuda" in str(self.device) else "cpu"
        for batch in tqdm(loader, desc=f"Val epoch {epoch}", leave=False):
            batch = self._to_device(batch)
            with torch.amp.autocast(device_type=device_type, enabled=(self.precision == "fp16")):
                outputs = self._forward(batch, phase)
            scores = torch.sigmoid(outputs["fake_logit"]).cpu().numpy()
            labels = batch["label"].cpu().numpy()
            all_scores.extend(scores.tolist())
            all_labels.extend(labels.tolist())

        try:
            auc = float(roc_auc_score(all_labels, all_scores))
        except Exception:
            auc = 0.5

        return {"val/auc": auc}

    def _forward(self, batch: dict, phase: str) -> dict:
        """Forward pass — extracts the right inputs for the current phase."""
        frames = batch["frames"]
        frame_diffs = batch["frame_diffs"]
        face_quality = batch["face_quality"]
        track_quality = batch["track_quality"]
        track_continuity = batch["track_continuity"]

        return self.model(
            frames=frames,
            frame_diffs=frame_diffs,
            face_quality=face_quality,
            track_quality=track_quality,
            track_continuity=track_continuity,
            mask=batch.get("mask"),
            mask_available=batch.get("mask_available"),
            geometry_features=batch.get("geometry_features"),
            audio_mel=batch.get("audio_mel"),
            av_available=batch.get("audio_available"),
        )

    def _to_device(self, batch: dict) -> dict:
        return {
            k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

    def _save_checkpoint(
        self, epoch: int, metric: float, phase: str, is_best: bool
    ) -> None:
        import os
        ckpt_data = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metric": metric,
            "phase": phase,
        }
        suffix = "best" if is_best else f"epoch{epoch:04d}"
        path = self.checkpoint_dir / f"{self.experiment_name}_{phase}_{suffix}.ckpt"
        torch.save(ckpt_data, path)

        if is_best:
            best_path = self.checkpoint_dir / "best.ckpt"
            # Windows-safe atomic overwrite: write to a temp file then rename.
            # shutil.copy2() can be blocked by antivirus / file locks on Windows.
            tmp_path = best_path.with_suffix(".ckpt.tmp")
            try:
                torch.save(ckpt_data, tmp_path)
                os.replace(tmp_path, best_path)          # atomic on same filesystem
            except Exception as e:
                logger.warning("Could not update best.ckpt (%s) — best epoch ckpt still saved as %s", e, path)
                if tmp_path.exists():
                    try:
                        tmp_path.unlink()
                    except Exception:
                        pass

        logger.debug("Saved checkpoint: %s", path)

    def load_checkpoint(self, path: str) -> dict:
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        logger.info("Loaded checkpoint from %s (epoch=%d)", path, ckpt.get("epoch", -1))
        return ckpt


def main():
    """CLI entry point for training."""
    import argparse
    from omegaconf import OmegaConf
    parser = argparse.ArgumentParser(description="FORGE-Lite Trainer")
    parser.add_argument("--config-train", default="configs/train.yaml")
    parser.add_argument("--config-model", default="configs/model.yaml")
    parser.add_argument("--config-data", default="configs/data.yaml")
    parser.add_argument("--phase", default="fusion",
                        choices=["baseline", "semantic_only", "forensic_only",
                                 "temporal_only", "fusion", "adversarial", "calibration"])
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--max-batches", type=int, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    train_cfg = OmegaConf.load(args.config_train)
    logger.info("Training phase: %s | smoke_test: %s", args.phase, args.smoke_test)

    # Import and run phase-specific training script
    if args.phase == "baseline":
        from training.train_baseline import run_baseline
        run_baseline(train_cfg, smoke_test=args.smoke_test)
    else:
        logger.info("Phase '%s' requires calling the specific train_*.py script.", args.phase)


if __name__ == "__main__":
    main()
