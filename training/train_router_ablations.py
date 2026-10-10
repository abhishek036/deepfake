"""
training/train_router_ablations.py
Trains router ablations (e.g. conf_only) to isolate performance drivers.
"""

from __future__ import annotations

import argparse
import time
import logging
import sys
from pathlib import Path
from typing import Dict, Any
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.amp import autocast, GradScaler
from sklearn.metrics import roc_auc_score, accuracy_score
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.dataset import ForgeDataset, collate_forge
from models.forge_lite import FORGELite
from training.losses import ForgeLoss

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
logger = logging.getLogger("train_router_ablations")

def run_ablation_training(
    ablation_mode: str,
    manifest_path: str = "data/manifests/train_manifest.json",
    base_checkpoint: str = "outputs/checkpoints/best.ckpt", # Always start from frozen baseline
    epochs: int = 5,
    batch_size: int = 8,
    lr: float = 5e-4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    logger.info(f"Starting Ablation Training: {ablation_mode}")
    
    # Instantiate Model with ablation mode
    model = FORGELite(ablation_mode=ablation_mode).to(device)
    
    # Load base checkpoint
    ckpt_path = Path(REPO_ROOT) / base_checkpoint
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("model_state_dict", ckpt)
        load_res = model.load_state_dict(state_dict, strict=False)
        logger.info("Loaded base checkpoint. Missing keys: %s", load_res.missing_keys)
        
    # Freeze backbones
    for name, param in model.named_parameters():
        if "fusion" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
            
    router_params = list(model.fusion.modality_router.parameters())
    fusion_other = [p for p in model.fusion.parameters() if id(p) not in set(id(rp) for rp in router_params)]
    
    optimizer = torch.optim.AdamW([
        {"params": router_params, "lr": lr, "weight_decay": 1e-3},
        {"params": fusion_other, "lr": 1e-4, "weight_decay": 1e-2},
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Dataset
    full_ds = ForgeDataset(manifest_path, split="train", augment=True)
    n_val = max(int(0.15 * len(full_ds)), 20)
    train_ds, val_ds = random_split(full_ds, [len(full_ds) - n_val, n_val], generator=torch.Generator().manual_seed(42))
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_forge, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_forge, num_workers=0)
    
    loss_fn = ForgeLoss(classification_weight=1.0, branch_weight=1.0, use_branch_logits=True)
    scaler = GradScaler(enabled=(device == "cuda"))
    
    best_auc = -1.0
    for epoch in range(1, epochs + 1):
        model.train()
        # Keep backbones in eval
        model.semantic_branch.eval()
        model.forensic_branch.eval()
        model.temporal_encoder.eval()
        
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch} [Train]", leave=False):
            tensors = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            optimizer.zero_grad()
            with autocast("cuda", enabled=(device == "cuda")):
                outputs = model(
                    frames=tensors["frames"], frame_diffs=tensors["frame_diffs"],
                    face_quality=tensors["face_quality"], track_quality=tensors["track_quality"],
                    track_continuity=tensors["track_continuity"], audio_mel=tensors.get("audio_mel")
                )
                losses = loss_fn(
                    fake_logit=outputs["fake_logit"], 
                    labels=tensors["label"], 
                    branch_logits=outputs.get("branch_logits"), 
                    availability_mask=outputs.get("availability_mask")
                )
            
            scaler.scale(losses["total"]).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += losses["total"].item()
            
        scheduler.step()
        
        # Val
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch} [Val]", leave=False):
                tensors = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                with autocast("cuda", enabled=(device == "cuda")):
                    outputs = model(
                        frames=tensors["frames"], frame_diffs=tensors["frame_diffs"],
                        face_quality=tensors["face_quality"], track_quality=tensors["track_quality"],
                        track_continuity=tensors["track_continuity"], audio_mel=tensors.get("audio_mel")
                    )
                preds = torch.sigmoid(outputs["fake_logit"]).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(tensors["label"].cpu().numpy())
                
        val_auc = roc_auc_score(all_labels, all_preds)
        logger.info(f"Epoch {epoch} | Train Loss: {train_loss/len(train_loader):.4f} | Val AUC: {val_auc:.4f}")
        best_auc = max(best_auc, val_auc)
        
    # Save
    out_file = Path(REPO_ROOT) / "outputs/checkpoints" / f"best_v2_{ablation_mode}.ckpt"
    torch.save({"model_state_dict": model.state_dict(), "ablation_mode": ablation_mode}, out_file)
    logger.info(f"Saved {ablation_mode} to {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True, choices=["conf_only", "conf_qual_calib", "full_ood"])
    args = parser.parse_args()
    run_ablation_training(args.mode)
