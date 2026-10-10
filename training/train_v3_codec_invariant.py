"""
training/train_v3_codec_invariant.py
Trains V3 Codec-Invariant Representation Learning by passing augmented frames through DINO.
"""

from __future__ import annotations

import argparse
import time
import logging
import sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from torch.amp import autocast, GradScaler
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.dataset import ForgeDataset, collate_forge
from models.forge_lite import FORGELite
from training.losses import ForgeLoss

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
logger = logging.getLogger("train_v3")

def run_v3_training(
    manifest_path: str = "data/manifests/train_manifest.json",
    base_checkpoint: str = "outputs/checkpoints/best_v2_modality_router.ckpt",
    epochs: int = 5,
    batch_size: int = 8,
    lr: float = 1e-4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    logger.info("Starting V3 Codec-Invariant Training")
    
    # 1. Initialize Model
    model = FORGELite(ablation_mode="conf_qual").to(device)
    
    ckpt_path = Path(REPO_ROOT) / base_checkpoint
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("model_state_dict", ckpt)
        load_res = model.load_state_dict(state_dict, strict=False)
        logger.info("Loaded V2 checkpoint. Missing keys: %s", load_res.missing_keys)
    else:
        logger.error(f"Base checkpoint not found at {ckpt_path}")
        return
        
    # 2. Freeze Rules for V3
    for name, param in model.named_parameters():
        param.requires_grad = False
        
    # Unfreeze only the Semantic Branch (DINOv2 adaptation)
    if hasattr(model, "semantic_branch"):
        for name, param in model.semantic_branch.named_parameters():
            if "adapter" in name or "norm" in name.lower() or "lora" in name.lower():
                param.requires_grad = True
    
    # Optional: Unfreeze semantic projection in fusion if it exists
    if hasattr(model.fusion, "semantic_proj"):
        for param in model.fusion.semantic_proj.parameters():
            param.requires_grad = True
            
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    logger.info(f"Trainable parameters: {sum(p.numel() for p in trainable_params)} / {sum(p.numel() for p in model.parameters())}")
    
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # 3. Dataset with Codec Augmentation Enabled
    full_ds = ForgeDataset(manifest_path, split="train", augment=True, codec_aug=True)
    n_val = max(int(0.15 * len(full_ds)), 20)
    train_ds, val_ds = random_split(full_ds, [len(full_ds) - n_val, n_val], generator=torch.Generator().manual_seed(42))
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_forge, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_forge, num_workers=0)
    
    # Loss: classification + codec_consistency + metric (SourcePair)
    loss_fn = ForgeLoss(
        classification_weight=1.0,
        consistency_weight=0.5, # lambda_c
        metric_weight=0.5,      # lambda_f
        use_consistency=True,
        use_metric=True,
    )
    scaler = GradScaler(enabled=(device == "cuda"))
    
    best_auc = -1.0
    for epoch in range(1, epochs + 1):
        model.eval()
        model.semantic_branch.train()
        if hasattr(model.fusion, "semantic_proj"):
            model.fusion.semantic_proj.train()
        
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch} [Train]", leave=False):
            tensors = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            optimizer.zero_grad()
            
            with autocast("cuda", enabled=(device == "cuda")):
                outputs = model(
                    frames=tensors["frames"], 
                    frames_aug=tensors.get("frames_aug"), # Pass the codec-augmented frames
                    frame_diffs=tensors.get("frame_diffs"),
                    face_quality=tensors.get("face_quality"), 
                    track_quality=tensors.get("track_quality"),
                    track_continuity=tensors.get("track_continuity"), 
                    audio_mel=tensors.get("audio_mel")
                )
                
                losses = loss_fn(
                    fake_logit=outputs["fake_logit"], 
                    labels=tensors["label"], 
                    embeddings_clean=outputs.get("semantic_embedding"),
                    embeddings_aug=outputs.get("semantic_embedding_aug"),
                    semantic_embedding=outputs.get("semantic_embedding")
                )
            
            scaler.scale(losses["total"]).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += losses["total"].item()
            
        scheduler.step()
        
        # 4. Validation (Evaluate on clean only to save time during train loop)
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch} [Val]", leave=False):
                tensors = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                with autocast("cuda", enabled=(device == "cuda")):
                    outputs = model(
                        frames=tensors["frames"], 
                        frame_diffs=tensors.get("frame_diffs"),
                        face_quality=tensors.get("face_quality"), 
                        track_quality=tensors.get("track_quality")
                    )
                preds = torch.sigmoid(outputs["fake_logit"]).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(tensors["label"].cpu().numpy())
                
        val_auc = roc_auc_score(all_labels, all_preds)
        logger.info(f"Epoch {epoch} | Train Loss: {train_loss/len(train_loader):.4f} | Val AUC: {val_auc:.4f}")
        
        if val_auc > best_auc:
            best_auc = val_auc
            out_file = Path(REPO_ROOT) / "outputs/checkpoints" / "best_v3_codec_invariant.ckpt"
            torch.save({"model_state_dict": model.state_dict(), "v3_epoch": epoch}, out_file)
            logger.info(f"Saved new best V3 model to {out_file}")

if __name__ == "__main__":
    run_v3_training()
