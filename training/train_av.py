"""
training/train_av.py
====================
Audio-Visual Branch Training Runner for FORGE-Lite.

Trains the dual-stream SyncNet-inspired AudioVisualBranch:
- Audio Stream (2D ConvNet on 80-band log-Mel spectrograms)
- Visual Lip Stream (3D/2D Spatiotemporal ConvNet on mouth crops)
- Normalized joint projection space for cosine similarity and desynchronization detection.

Supervised with binary classification loss and AV-contrastive loss.
Supports both dedicated AV datasets (LAV-DF, FakeAVCeleb) and multimodal batches.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import ForgeDataset, collate_forge, make_dataloader
from models.forge_lite import FORGELite
from training.trainer import Trainer
from training.losses import ForgeLoss

logger = logging.getLogger("train_av")


class AVContrastiveLoss(nn.Module):
    """
    Audio-Visual sync contrastive loss.
    For genuine synced video (label 0): encourages high cosine similarity between audio & visual embeddings.
    For desynced or audio-replaced deepfake (label 1): penalizes high cosine similarity.
    """

    def __init__(self, margin: float = 0.5) -> None:
        super().__init__()
        self.margin = margin

    def forward(
        self,
        cosine_sim: torch.Tensor,       # (B, 1) in [-1, 1]
        labels: torch.Tensor,           # (B,) 0=real, 1=fake
        audio_available: torch.Tensor,  # (B,) bool
    ) -> torch.Tensor:
        mask = audio_available.bool().squeeze()
        if not mask.any():
            return torch.tensor(0.0, device=cosine_sim.device, requires_grad=True)

        sim = cosine_sim.squeeze(-1)[mask]
        lbl = labels[mask].float()

        # Real (0): maximize similarity -> minimize 1 - sim
        loss_real = (1.0 - sim) * (1.0 - lbl)
        # Fake (1): minimize similarity -> clamp(sim - margin, min=0)
        loss_fake = F.relu(sim - self.margin) * lbl

        return (loss_real + loss_fake).mean()


def run_av_training(
    cfg,
    manifest_path: str,
    resume_checkpoint: Optional[str] = None,
    smoke_test: bool = False,
) -> dict:
    """
    Train or fine-tune the AudioVisualBranch.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 2 if smoke_test else cfg.batch_sizes.get("av", 8)
    epochs = 1 if smoke_test else cfg.epochs.get("av_branch", 10)
    lr = float(cfg.lr.get("av_branch", 5.0e-4))

    logger.info("Initializing FORGE-Lite with active AV branch...")
    model = FORGELite(
        enable_semantic=False,
        enable_forensic=False,
        enable_temporal=False,
        enable_av=True,
        enable_localization=False,
    ).to(device)

    if resume_checkpoint and Path(resume_checkpoint).exists():
        ckpt = torch.load(resume_checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
        logger.info("Loaded resume checkpoint: %s", resume_checkpoint)

    # Prepare datasets
    window_size = 5  # AV branch operates on 5-frame lip windows
    train_ds = ForgeDataset(
        manifest_path,
        split="train",
        window_size=window_size,
        augment=True,
        max_samples=cfg.smoke_test.max_batches * batch_size if smoke_test else None,
    )
    val_max = cfg.smoke_test.max_batches * batch_size if smoke_test else cfg.get("max_val_samples", 5000)
    val_ds = ForgeDataset(
        manifest_path,
        split="val",
        window_size=window_size,
        augment=False,
        max_samples=val_max,
    )

    nw = 0 if smoke_test else cfg.get("num_workers", 2)
    pw = bool(cfg.get("persistent_workers", False)) and nw > 0

    train_loader = make_dataloader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=nw,
        persistent_workers=pw,
        pin_memory=True if device == "cuda" else False,
        cfg=cfg,
        is_train=True,
    )
    val_loader = make_dataloader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=nw,
        persistent_workers=pw,
        pin_memory=True if device == "cuda" else False,
        cfg=cfg,
        is_train=False,
    )

    # Optimizer & criterion
    av_params = model.get_trainable_parameters("av_only")
    if not av_params:
        logger.warning("No trainable parameters found in av_branch! Using all requires_grad.")
        av_params = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW(av_params, lr=lr, weight_decay=0.01)
    bce_loss = nn.BCELoss()
    contrastive_loss = AVContrastiveLoss(margin=0.3)

    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda" and cfg.get("precision", "fp16") == "fp16"))

    checkpoint_dir = Path(cfg.checkpointing.save_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_auc = 0.0

    logger.info("Starting AV Branch Training (%d epochs, %d batches/epoch)...", epochs, len(train_loader))

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in tqdm(train_loader, desc=f"AV Epoch {epoch}/{epochs}", leave=False):
            frames = batch["frames"].to(device)
            labels = batch["label"].to(device)
            audio_mel = batch.get("audio_mel")
            if audio_mel is not None:
                audio_mel = audio_mel.to(device)
            audio_available = batch.get("audio_available").to(device)

            optimizer.zero_grad()

            with torch.amp.autocast(device_type=device, enabled=(device == "cuda" and cfg.get("precision", "fp16") == "fp16")):
                av_out = model.av_branch(
                    audio_mel=audio_mel,
                    lip_crops=frames,
                    audio_available=audio_available,
                )
                pred_score = av_out["av_score"].squeeze(-1)  # (B,)
                cosine_sim = av_out["cosine_sim"]            # (B, 1)

            # Compute loss in fp32 outside autocast for numerical stability
            loss_bce = F.binary_cross_entropy(
                pred_score.float().clamp(1e-6, 1.0 - 1e-6),
                labels.float(),
            )
            loss_cont = contrastive_loss(cosine_sim.float(), labels, audio_available)
            loss = loss_bce + 0.3 * loss_cont

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += float(loss.item())
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)

        # Validation
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                frames = batch["frames"].to(device)
                labels = batch["label"].to(device)
                audio_mel = batch.get("audio_mel")
                if audio_mel is not None:
                    audio_mel = audio_mel.to(device)
                audio_available = batch.get("audio_available").to(device)

                with torch.amp.autocast(device_type=device, enabled=(device == "cuda" and cfg.get("precision", "fp16") == "fp16")):
                    av_out = model.av_branch(
                        audio_mel=audio_mel,
                        lip_crops=frames,
                        audio_available=audio_available,
                    )
                    scores = av_out["av_score"].squeeze(-1).cpu().numpy()

                val_preds.extend(scores.tolist())
                val_targets.extend(labels.cpu().numpy().tolist())

        if len(set(val_targets)) > 1:
            try:
                from sklearn.metrics import roc_auc_score
                val_auc = float(roc_auc_score(val_targets, val_preds))
            except Exception:
                val_auc = 0.5
        else:
            val_auc = 0.5

        logger.info(
            "Epoch %02d/%02d — Loss: %.4f | Val AUC: %.4f%s",
            epoch,
            epochs,
            avg_loss,
            val_auc,
            " (BEST)" if val_auc > best_auc else "",
        )

        if epoch == 1 or val_auc >= best_auc:
            best_auc = val_auc
            save_path = checkpoint_dir / f"{cfg.experiment_name}_av_branch_best.ckpt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_auc": val_auc,
                    "phase": "av_branch",
                },
                save_path,
            )
            logger.info("Saved best AV checkpoint: %s", save_path)

    return {"best_val_auc": best_auc, "last_loss": avg_loss}
