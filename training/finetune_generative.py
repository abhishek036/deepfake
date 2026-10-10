"""
training/finetune_generative.py
Fine-tunes FORGE-Lite spatial & forensic branches on a balanced multi-generator dataset:
- StyleGAN3 synthetic faces + Unsplash authentic photos (from FINAL_DATASET.csv)
- FaceForensics++ C23 real & deepfake crops (prevents catastrophic forgetting)

Optimizes:
- DINOv2 LayerNorm adaptation + projection
- SRM-CNN forensic residual layers (for high-frequency GAN spectral fingerprints)
- Reliability-Aware Fusion MLP
- Post-hoc Temperature Scaling calibration on validation split
"""

from __future__ import annotations

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import json
import logging
import math
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score

# Ensure root in path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.forge_lite import FORGELite
from data.augmentation import frames_to_tensor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("finetune_generative")


class GenerativeForensicDataset(Dataset):
    """
    Dataset combining StyleGAN3 generative faces, Unsplash reals,
    and FaceForensics++ video crops.
    """

    def __init__(self, records: List[Dict], augment: bool = True):
        self.records = records
        self.augment = augment

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rec = self.records[idx]
        crop_p = Path(rec.get("crop_path") or rec.get("source_path"))
        if not crop_p.is_absolute():
            crop_p = REPO_ROOT / crop_p

        img = cv2.imread(str(crop_p))
        if img is None:
            # Fallback black frame if image missing
            img = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            if img.shape[0] != 224 or img.shape[1] != 224:
                img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)

        if self.augment:
            if random.random() > 0.5:
                img = cv2.flip(img, 1)  # horizontal flip
            if random.random() > 0.3:
                # Slight brightness / contrast
                alpha = random.uniform(0.85, 1.15)
                beta = random.uniform(-15, 15)
                img = np.clip(alpha * img + beta, 0, 255).astype(np.uint8)
            if random.random() > 0.4:
                # JPEG compression artifact simulation
                quality = random.randint(65, 95)
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
                _, encimg = cv2.imencode(".jpg", img, encode_param)
                img = cv2.imdecode(encimg, 1)

        # Single-frame representation (T=1) for spatial & forensic fine-tuning (avoids 8x VRAM explosion)
        frames_t = frames_to_tensor([img])  # (1, 3, 224, 224)
        frame_diffs = torch.zeros(1, 3, 224, 224)

        label = float(rec["label"])
        return {
            "frames": frames_t,
            "frame_diffs": frame_diffs,
            "label": torch.tensor(label, dtype=torch.float32),
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    frames = torch.stack([b["frames"] for b in batch])  # (B, 1, 3, 224, 224)
    frame_diffs = torch.stack([b["frame_diffs"] for b in batch])
    labels = torch.stack([b["label"] for b in batch])
    return {
        "frames": frames,
        "frame_diffs": frame_diffs,
        "label": labels,
    }


def load_manifest_records(path: Path) -> List[Dict]:
    records = []
    if not path.exists():
        return records
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line.strip()))
    return records


def calibrate_temperature(model: FORGELite, val_loader: DataLoader, device: torch.device) -> float:
    """Optimize post-hoc temperature scaling parameter on validation set via NLL."""
    model.eval()
    logits_list = []
    labels_list = []

    with torch.no_grad():
        for batch in val_loader:
            frames = batch["frames"].to(device)
            frame_diffs = batch["frame_diffs"].to(device)
            labels = batch["label"].to(device)

            B = frames.shape[0]
            face_q = torch.zeros(B, 10, device=device)
            track_q = torch.zeros(B, 5, device=device)
            cont = torch.ones(B, device=device)

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                out = model(
                    frames=frames,
                    frame_diffs=frame_diffs,
                    face_quality=face_q,
                    track_quality=track_q,
                    track_continuity=cont,
                    return_decisions=False,
                )
            logits_list.append(out["fake_logit"])
            labels_list.append(labels)

    if not logits_list:
        return 1.0

    logits = torch.cat(logits_list, dim=0)
    labels = torch.cat(labels_list, dim=0)

    # Temperature parameter
    log_temp = nn.Parameter(torch.zeros(1, device=device))
    optimizer = torch.optim.LBFGS([log_temp], lr=0.01, max_iter=50)
    criterion = nn.BCEWithLogitsLoss()

    def eval_fn():
        optimizer.zero_grad()
        t = torch.exp(log_temp)
        loss = criterion(logits / t, labels)
        loss.backward()
        return loss

    optimizer.step(eval_fn)
    best_t = float(torch.exp(log_temp).item())
    best_t = max(0.85, min(best_t, 3.0))  # clamp to robust range
    logger.info("Optimized temperature T = %.4f", best_t)
    return best_t


def run_finetune(
    epochs: int = 8,
    batch_size: int = 16,
    grad_accum_steps: int = 2,
    lr: float = 2e-4,
    checkpoint_in: str = "outputs/checkpoints/best.ckpt",
    checkpoint_out: str = "outputs/checkpoints/best.ckpt",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Starting fine-tuning on %s (batch_size=%d, accum=%d)...", device, batch_size, grad_accum_steps)

    if device.type == "cuda":
        torch.cuda.empty_cache()

    # 1. Load manifests
    train_fd = load_manifest_records(REPO_ROOT / "data" / "manifests" / "final_dataset_train.jsonl")
    val_fd = load_manifest_records(REPO_ROOT / "data" / "manifests" / "final_dataset_val.jsonl")

    # Add 400 balanced FaceForensics++ crops to train set to preserve video manipulation capabilities
    ff_manifest = load_manifest_records(REPO_ROOT / "data" / "manifests" / "faceforensics.jsonl")
    if ff_manifest:
        reals_ff = [r for r in ff_manifest if int(r.get("label", 0)) == 0][:200]
        fakes_ff = [r for r in ff_manifest if int(r.get("label", 0)) == 1][:200]
        train_records = train_fd + reals_ff + fakes_ff
    else:
        train_records = train_fd

    random.seed(42)
    random.shuffle(train_records)
    random.shuffle(val_fd)

    logger.info("Training records: %d (FINAL_DATASET: %d, FF++: %d)",
                len(train_records), len(train_fd), len(train_records) - len(train_fd))
    logger.info("Validation records: %d", len(val_fd))

    if len(train_records) == 0:
        logger.error("No training records available! Ensure prepare_balanced_manifests.py has finished.")
        return

    train_ds = GenerativeForensicDataset(train_records, augment=True)
    val_ds = GenerativeForensicDataset(val_fd, augment=False)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=2, pin_memory=(device.type == "cuda")
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=2, pin_memory=(device.type == "cuda")
    )

    # 2. Load Model - Note: enable_temporal=False during spatial fine-tuning saves ~8x VRAM
    model = FORGELite(
        enable_semantic=True,
        enable_forensic=True,
        enable_temporal=False,
        enable_localization=False,
    ).to(device)

    in_ckpt_p = Path(checkpoint_in)
    if in_ckpt_p.exists():
        try:
            ckpt = torch.load(in_ckpt_p, map_location=device)
            state_dict = ckpt.get("model_state_dict", ckpt)
            model.load_state_dict(state_dict, strict=False)
            logger.info("Loaded baseline weights from %s", checkpoint_in)
        except Exception as e:
            logger.warning("Could not load checkpoint: %s. Initializing fresh heads.", e)

    # 3. Parameter Selection
    # Train: DINO LayerNorm & adapt, SRM forensic CNN, Fusion MLP
    trainable_params = []
    if hasattr(model, "semantic_branch"):
        trainable_params += model.semantic_branch.get_trainable_parameters()
    if hasattr(model, "forensic_branch"):
        trainable_params += list(model.forensic_branch.parameters())
    trainable_params += list(model.fusion.parameters())

    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion = nn.BCEWithLogitsLoss()

    best_val_auc = 0.0
    best_state_dict = None

    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    # 4. Training Loop
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0
        optimizer.zero_grad()

        for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", unit="batch")):
            frames = batch["frames"].to(device)
            frame_diffs = batch["frame_diffs"].to(device)
            labels = batch["label"].to(device)

            B = frames.shape[0]
            face_q = torch.zeros(B, 10, device=device)
            track_q = torch.zeros(B, 5, device=device)
            cont = torch.ones(B, device=device)

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                out = model(
                    frames=frames,
                    frame_diffs=frame_diffs,
                    face_quality=face_q,
                    track_quality=track_q,
                    track_continuity=cont,
                    return_decisions=False,
                )
                fake_logit = out["fake_logit"]
                loss = criterion(fake_logit, labels)
                loss = loss / grad_accum_steps

            scaler.scale(loss).backward()

            if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            total_loss += loss.item() * grad_accum_steps
            n_batches += 1

        scheduler.step()
        avg_train_loss = total_loss / max(1, n_batches)

        # Validation evaluation
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                frames = batch["frames"].to(device)
                frame_diffs = batch["frame_diffs"].to(device)
                labels = batch["label"].to(device)

                B = frames.shape[0]
                face_q = torch.zeros(B, 10, device=device)
                track_q = torch.zeros(B, 5, device=device)
                cont = torch.ones(B, device=device)

                with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                    out = model(
                        frames=frames,
                        frame_diffs=frame_diffs,
                        face_quality=face_q,
                        track_quality=track_q,
                        track_continuity=cont,
                        return_decisions=False,
                    )
                    prob = torch.sigmoid(out["fake_logit"]).cpu().numpy()

                val_preds.extend(prob.tolist())
                val_targets.extend(labels.cpu().numpy().tolist())

        try:
            val_auc = float(roc_auc_score(val_targets, val_preds))
            val_acc = float(accuracy_score(val_targets, [1 if p >= 0.5 else 0 for p in val_preds]))
        except Exception:
            val_auc = 0.5
            val_acc = 0.5

        logger.info(
            "Epoch %d/%d | Train Loss: %.4f | Val AUC: %.4f | Val Acc: %.1f%%",
            epoch, epochs, avg_train_loss, val_auc, val_acc * 100
        )

        if val_auc >= best_val_auc:
            best_val_auc = val_auc
            best_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}

    # 5. Calibration & Save
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    best_t = calibrate_temperature(model, val_loader, device)

    out_p = Path(checkpoint_out)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    # Backup existing checkpoint
    if out_p.exists():
        backup_p = out_p.parent / "best_backup.ckpt"
        import shutil
        shutil.copy2(out_p, backup_p)
        logger.info("Backed up prior checkpoint to %s", backup_p)

    torch.save({"model_state_dict": model.state_dict(), "best_val_auc": best_val_auc}, out_p)
    logger.info("Saved fine-tuned checkpoint to %s (Val AUC: %.4f)", out_p, best_val_auc)

    # Save calibrated temperature
    cal_file = out_p.parent / "calibration_temperature.pt"
    torch.save({"temperature": best_t}, cal_file)
    logger.info("Saved calibrated temperature T=%.4f to %s", best_t, cal_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    args = parser.parse_args()

    run_finetune(
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum,
        lr=args.lr,
    )
