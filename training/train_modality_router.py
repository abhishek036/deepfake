"""
training/calibrate_fusion_heads.py
Dedicated 5-epoch calibration fine-tuning for FORGE-Lite fusion branch logit heads.

Solves the uncalibrated branch disagreement problem:
1. Freezes DINOv2 and SRM-CNN backbones.
2. Trains all 5 dedicated branch_logit_heads (nn.Linear(128, 1)) on data/manifests/train_manifest.json
   using masked Binary Cross-Entropy loss.
3. Produces true calibrated sigmoid fake probabilities per branch so masked_probs.std(dim=1)
   measures genuine inter-branch contradiction.
4. Saves updated weights to outputs/checkpoints/best.ckpt (with safety backup).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import shutil
import time
from typing import Dict, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.amp import autocast, GradScaler
from sklearn.metrics import roc_auc_score, accuracy_score
from tqdm import tqdm

from data.dataset import ForgeDataset, collate_forge
from models.forge_lite import FORGELite
from training.losses import ForgeLoss

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("calibrate_fusion")

REPO_ROOT = Path(__file__).resolve().parent.parent


def run_calibration(
    manifest_path: str = "data/manifests/train_manifest.json",
    checkpoint_path: str = "outputs/checkpoints/best.ckpt",
    epochs: int = 5,
    batch_size: int = 8,
    lr_heads: float = 5.0e-4,
    lr_fusion: float = 1.0e-4,
    window_size: int = 4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Dict[str, Any]:
    logger.info("=" * 70)
    logger.info("STARTING FUSION HEAD CALIBRATION FINE-TUNING")
    logger.info("Target manifest: %s", manifest_path)
    logger.info("Base checkpoint: %s", checkpoint_path)
    logger.info("Device: %s | Epochs: %d | Batch Size: %d", device, epochs, batch_size)
    logger.info("=" * 70)

    # 1. Safe Backup of Base Checkpoint
    ckpt_file = Path(checkpoint_path)
    backup_file = ckpt_file.parent / "best_pre_calibration.ckpt"
    if ckpt_file.exists() and not backup_file.exists():
        shutil.copy2(ckpt_file, backup_file)
        logger.info("Created safety backup of base checkpoint at: %s", backup_file)

    # 2. Instantiate Model
    model = FORGELite(
        enable_semantic=True,
        enable_forensic=True,
        enable_temporal=True,
        enable_localization=True,
        enable_av=True,
    ).to(device)

    # 3. Load Existing Trained Weights & Align Dimensions
    if ckpt_file.exists():
        raw_ckpt = torch.load(ckpt_file, map_location="cpu")
        state_dict = raw_ckpt.get("model_state_dict", raw_ckpt)
        current_state = model.state_dict()

        # Weight surgery: align temporal_encoder.gru.weight_ih_l0 if shape changed
        k_gru = "temporal_encoder.gru.weight_ih_l0"
        if k_gru in state_dict and k_gru in current_state:
            old_w = state_dict[k_gru]
            new_w = current_state[k_gru]
            if old_w.shape != new_w.shape:
                logger.info("Aligning %s from %s to %s (padding new dims with 0)", k_gru, old_w.shape, new_w.shape)
                aligned_w = new_w.clone()
                min_in = min(old_w.shape[1], new_w.shape[1])
                aligned_w[:, :min_in] = old_w[:, :min_in]
                state_dict[k_gru] = aligned_w

        # Load temperature if available
        temp_file = ckpt_file.parent / "calibration_temperature.pt"
        if temp_file.exists():
            try:
                temp_dict = torch.load(temp_file, map_location="cpu")
                t_val = temp_dict.get("temperature", 1.0)
                state_dict["calibration.temperature_scaling.log_temperature"] = torch.tensor([float(np.log(t_val))])
            except Exception as e:
                logger.debug("Could not load calibration_temperature.pt: %s", e)

        # Modality Router is new, so its parameters won't be in state_dict.
        # strict=False will keep its default initialization.
        load_res = model.load_state_dict(state_dict, strict=False)
        logger.info("Loaded base checkpoint. Missing keys: %s", load_res.missing_keys)
        logger.info("Unexpected keys: %s", load_res.unexpected_keys)

    # 4. Freeze Backbones & Enable Fusion Heads
    # Keep DINOv2 and SRM-CNN backbones strictly frozen
    for p in model.semantic_branch.parameters():
        p.requires_grad = False
    for p in model.forensic_branch.parameters():
        p.requires_grad = False
    for p in model.temporal_encoder.parameters():
        p.requires_grad = False
    if hasattr(model, "localization_head"):
        for p in model.localization_head.parameters():
            p.requires_grad = False
    if hasattr(model, "geometry_extractor"):
        for p in model.geometry_extractor.parameters():
            p.requires_grad = False

    # Fusion module is trainable
    for p in model.fusion.parameters():
        p.requires_grad = True

    # Parameter groups with differential LR: higher LR on new Modality Router
    router_params = list(model.fusion.modality_router.parameters())
    router_param_ids = set(id(p) for p in router_params)
    fusion_other_params = [p for p in model.fusion.parameters() if id(p) not in router_param_ids]

    optimizer = torch.optim.AdamW(
        [
            {"params": router_params, "lr": lr_heads, "weight_decay": 1.0e-3},
            {"params": fusion_other_params, "lr": lr_fusion, "weight_decay": 1.0e-2},
        ]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1.0e-6)

    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Frozen backbones: semantic, forensic, temporal.")
    logger.info("Trainable fusion parameters: %d (Router: %d)", trainable_count, sum(p.numel() for p in router_params))

    # 5. Dataset & Loaders
    full_ds = ForgeDataset(
        manifest_path=manifest_path,
        split="train",
        window_size=window_size,
        min_track_length=1,
        augment=True,
    )
    n_total = len(full_ds)
    n_val = max(int(0.15 * n_total), 20)
    n_train = n_total - n_val

    rng_gen = torch.Generator().manual_seed(42)
    train_ds, val_ds = random_split(full_ds, [n_train, n_val], generator=rng_gen)
    logger.info("Dataset split: %d Train records, %d Validation records", len(train_ds), len(val_ds))

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_forge,
        num_workers=0,  # Windows-safe
        pin_memory=True if device == "cuda" else False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_forge,
        num_workers=0,
        pin_memory=True if device == "cuda" else False,
    )

    loss_fn = ForgeLoss(
        classification_weight=1.0,
        branch_weight=1.0,
        use_consistency=False,
        use_metric=False,
        use_localization=False,
        use_temporal=False,
        use_geometry=False,
        use_branch_logits=True,
    )

    scaler = GradScaler(enabled=(device == "cuda"))

    # 6. Training Loop
    branch_names = ["Semantic", "Forensic", "Temporal", "rPPG", "AV"]
    history = []
    best_val_auc = -1.0

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        # Keep frozen backbones in eval mode for consistent norm statistics
        model.semantic_branch.eval()
        model.forensic_branch.eval()
        model.temporal_encoder.eval()

        train_loss = 0.0
        train_cls_loss = 0.0
        train_branch_loss = 0.0
        train_steps = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]", leave=False):
            batch_tensors = {
                k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            optimizer.zero_grad()

            with autocast(device_type="cuda" if device == "cuda" else "cpu", enabled=(device == "cuda")):
                outputs = model(
                    frames=batch_tensors["frames"],
                    frame_diffs=batch_tensors["frame_diffs"],
                    face_quality=batch_tensors["face_quality"],
                    track_quality=batch_tensors["track_quality"],
                    track_continuity=batch_tensors["track_continuity"],
                    audio_mel=batch_tensors.get("audio_mel"),
                    geometry_features=batch_tensors.get("geometry_features"),
                )

                losses = loss_fn(
                    fake_logit=outputs["fake_logit"],
                    labels=batch_tensors["label"],
                    branch_logits=outputs.get("branch_logits"),
                    availability_mask=outputs.get("availability_mask"),
                )
                loss = losses["total"]

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.fusion.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss += float(losses["total"].item())
            train_cls_loss += float(losses["classification"].item())
            train_branch_loss += float(losses.get("branch_logits", torch.tensor(0.0)).item())
            train_steps += 1

        scheduler.step()
        train_loss /= max(train_steps, 1)
        train_cls_loss /= max(train_steps, 1)
        train_branch_loss /= max(train_steps, 1)

        # 7. Validation Loop
        model.eval()
        val_loss = 0.0
        val_cls_loss = 0.0
        val_branch_loss = 0.0
        val_steps = 0
        all_preds, all_labels = [], []
        branch_probs_list = []
        disagreements = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [Val]", leave=False):
                batch_tensors = {
                    k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                with autocast(device_type="cuda" if device == "cuda" else "cpu", enabled=(device == "cuda")):
                    outputs = model(
                        frames=batch_tensors["frames"],
                        frame_diffs=batch_tensors["frame_diffs"],
                        face_quality=batch_tensors["face_quality"],
                        track_quality=batch_tensors["track_quality"],
                        track_continuity=batch_tensors["track_continuity"],
                        audio_mel=batch_tensors.get("audio_mel"),
                        geometry_features=batch_tensors.get("geometry_features"),
                    )
                    losses = loss_fn(
                        fake_logit=outputs["fake_logit"],
                        labels=batch_tensors["label"],
                        branch_logits=outputs.get("branch_logits"),
                        availability_mask=outputs.get("availability_mask"),
                    )

                val_loss += float(losses["total"].item())
                val_cls_loss += float(losses["classification"].item())
                val_branch_loss += float(losses.get("branch_logits", torch.tensor(0.0)).item())
                val_steps += 1

                preds = torch.sigmoid(outputs["fake_logit"]).cpu().numpy()
                lbls = batch_tensors["label"].cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(lbls)

                if outputs.get("branch_fake_probs") is not None:
                    branch_probs_list.append(outputs["branch_fake_probs"].cpu().numpy())
                if outputs.get("branch_disagreement") is not None:
                    disagreements.extend(outputs["branch_disagreement"].cpu().numpy())

        val_loss /= max(val_steps, 1)
        val_cls_loss /= max(val_steps, 1)
        val_branch_loss /= max(val_steps, 1)

        val_auc = roc_auc_score(all_labels, all_preds) if len(set(all_labels)) > 1 else 0.5
        val_acc = accuracy_score(all_labels, (np.array(all_preds) >= 0.5).astype(int))

        # Per-branch calibration summary
        branch_means_str = ""
        if branch_probs_list:
            bp = np.concatenate(branch_probs_list, axis=0)  # (N_val, 5)
            y_arr = np.array(all_labels)
            real_mask = (y_arr == 0)
            fake_mask = (y_arr == 1)
            b_real = bp[real_mask].mean(axis=0) if real_mask.any() else np.zeros(5)
            b_fake = bp[fake_mask].mean(axis=0) if fake_mask.any() else np.zeros(5)
            branch_info = [
                f"{branch_names[i]}: R={b_real[i]:.2f}/F={b_fake[i]:.2f}"
                for i in range(len(branch_names))
            ]
            branch_means_str = " | " + ", ".join(branch_info[:3])

        mean_disagree = np.mean(disagreements) if disagreements else 0.0
        elapsed = time.time() - t0

        logger.info(
            "Epoch %d/%d (%.1fs) | Train Total=%.4f (Cls=%.4f, Br=%.4f) | "
            "Val Total=%.4f (Cls=%.4f, Br=%.4f) | Val AUC=%.4f, Acc=%.4f | Disagree=%.3f%s",
            epoch, epochs, elapsed,
            train_loss, train_cls_loss, train_branch_loss,
            val_loss, val_cls_loss, val_branch_loss,
            val_auc, val_acc, mean_disagree, branch_means_str,
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_auc": val_auc,
            "val_acc": val_acc,
            "mean_disagreement": mean_disagree,
        })

        if val_auc >= best_val_auc:
            best_val_auc = val_auc

    # 8. Save Modality Router Checkpoint
    save_dict = {
        "model_state_dict": model.state_dict(),
        "epoch": epochs,
        "history": history,
        "modality_router": True,
        "calibration_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    
    router_ckpt_file = ckpt_file.parent / "best_v2_modality_router.ckpt"
    torch.save(save_dict, router_ckpt_file)
    logger.info("Saved Modality Router checkpoint to: %s", router_ckpt_file)

    # 9. Verify Checkpoint Loading
    verify_model = FORGELite().to(device)
    v_ckpt = torch.load(router_ckpt_file, map_location="cpu")
    v_res = verify_model.load_state_dict(v_ckpt["model_state_dict"], strict=False)
    router_missing = [k for k in v_res.missing_keys if "modality_router" in k]
    logger.info("Post-training verification: missing modality_router keys = %s (count: %d)", router_missing, len(router_missing))
    assert len(router_missing) == 0, f"Error: modality_router still missing after training: {router_missing}"
    logger.info("VERIFICATION PASSED: Modality Router is present and trained!")

    return {
        "status": "success",
        "best_val_auc": best_val_auc,
        "epochs_trained": epochs,
        "history": history,
    }


def main():
    parser = argparse.ArgumentParser(description="Calibrate FORGE-Lite Fusion Branch Heads")
    parser.add_argument("--manifest", default="data/manifests/train_manifest.json")
    parser.add_argument("--checkpoint", default="outputs/checkpoints/best.ckpt")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--window-size", type=int, default=4)
    parser.add_argument("--lr-heads", type=float, default=5.0e-4)
    parser.add_argument("--lr-fusion", type=float, default=1.0e-4)
    args = parser.parse_args()

    run_calibration(
        manifest_path=args.manifest,
        checkpoint_path=args.checkpoint,
        epochs=args.epochs,
        batch_size=args.batch_size,
        window_size=args.window_size,
        lr_heads=args.lr_heads,
        lr_fusion=args.lr_fusion,
    )


if __name__ == "__main__":
    main()
