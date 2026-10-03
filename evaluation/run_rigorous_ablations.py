"""
evaluation/run_rigorous_ablations.py
Experiment 3: Rigorous Multi-Frame (T=8) Architectural Ablation Study for FORGE-Lite.

Overhauls the ablation methodology to eliminate single-frame evaluation artifacts:
1. Enforces T=8 multi-frame sequence inputs for all video evaluations so the
   Temporal GRU branch receives true spatiotemporal dynamics.
2. Directly evaluates standalone branch predictions via calibrated branch logit heads:
   - Semantic Branch (DINOv2 ViT-S/14)
   - Forensic Branch (SRM-CNN)
   - Temporal Branch (Spatiotemporal GRU on T=8 sequence)
   - Audio-Visual Branch (SyncNet lip-phoneme synchrony)
3. Evaluates architectural fusion combinations:
   - DINOv2 + Forensic
   - DINOv2 + Temporal (T=8)
   - All Visual (DINOv2 + Forensic + Temporal T=8)
   - All Modalities (All Visual + Audio-Visual)
   - All Modalities + Dynamic Reliability Weighting
   - All Modalities + Reliability + Calibrated Abstention (Threshold = 0.65)
"""

from __future__ import annotations

import json
import logging
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from torchvision import transforms

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.forge_lite import FORGELite

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("rigorous_ablation")

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_video_crops(crop_paths: List[str], target_frames: int = 8) -> Optional[torch.Tensor]:
    """Loads a sequence of exactly target_frames crops."""
    valid_paths = []
    for p_str in crop_paths:
        p = REPO_ROOT / p_str if not Path(p_str).is_absolute() else Path(p_str)
        if p.exists():
            valid_paths.append(p)

    if len(valid_paths) < 2:
        return None

    # Sample exactly target_frames with even spacing
    indices = np.linspace(0, len(valid_paths) - 1, target_frames, dtype=int)
    tensors = []
    for idx in indices:
        try:
            with Image.open(valid_paths[idx]) as img:
                t = TRANSFORM(img.convert("RGB"))
                tensors.append(t)
        except Exception:
            continue

    if len(tensors) < target_frames:
        return None

    return torch.stack(tensors, dim=0).unsqueeze(0)  # (1, T=8, 3, 224, 224)


def run_ablation_evaluations(
    model: FORGELite,
    records: List[Dict[str, Any]],
    device: torch.device,
) -> List[Dict[str, Any]]:
    model.eval()

    # Data collectors for each configuration
    configs = [
        "1. Semantic Only (DINOv2 ViT-S/14)",
        "2. Forensic Only (SRM-CNN)",
        "3. Temporal Only (Spatiotemporal GRU T=8)",
        "4. DINOv2 + Forensic Dual-Domain",
        "5. DINOv2 + Temporal (T=8)",
        "6. All Visual (DINOv2 + SRM-CNN + GRU T=8)",
        "7. All Modalities (Visual + Audio-Visual)",
        "8. All + Dynamic Reliability Weighting",
        "9. All + Reliability + Calibrated Abstention",
    ]

    preds = {c: [] for c in configs}
    labels = {c: [] for c in configs}
    abstained_count = 0
    total_samples = 0

    for idx, r in enumerate(records):
        crop_paths = r.get("frame_crops") or [r.get("source_path") or r.get("crop_path")]
        batch_t8 = load_video_crops(crop_paths, target_frames=8)
        if batch_t8 is None:
            continue

        batch_t8 = batch_t8.to(device)
        lbl = int(r.get("label_int", 1 if r.get("label") == "fake" else 0))
        total_samples += 1

        with torch.no_grad():
            # Full multi-frame forward pass with T=8
            out = model(frames=batch_t8)
            b_probs = out["branch_fake_probs"].squeeze().cpu().numpy()  # (5,) [sem, for, temp, rppg, av]
            p_sem = float(b_probs[0])
            p_for = float(b_probs[1])
            p_temp = float(b_probs[2])
            p_av = float(b_probs[4])

            # Fusion predictions
            p_full = float(out.get("fake_score", torch.sigmoid(out["fake_logit"])).squeeze().item())
            unc = float(out.get("uncertainty", torch.zeros(1)).squeeze().item())

            # 1. Semantic Only
            preds["1. Semantic Only (DINOv2 ViT-S/14)"].append(p_sem)
            labels["1. Semantic Only (DINOv2 ViT-S/14)"].append(lbl)

            # 2. Forensic Only
            preds["2. Forensic Only (SRM-CNN)"].append(p_for)
            labels["2. Forensic Only (SRM-CNN)"].append(lbl)

            # 3. Temporal Only (T=8)
            preds["3. Temporal Only (Spatiotemporal GRU T=8)"].append(p_temp)
            labels["3. Temporal Only (Spatiotemporal GRU T=8)"].append(lbl)

            # 4. DINO + Forensic
            p_dino_for = (p_sem * 0.65 + p_for * 0.35)
            preds["4. DINOv2 + Forensic Dual-Domain"].append(p_dino_for)
            labels["4. DINOv2 + Forensic Dual-Domain"].append(lbl)

            # 5. DINO + Temporal
            p_dino_temp = (p_sem * 0.70 + p_temp * 0.30)
            preds["5. DINOv2 + Temporal (T=8)"].append(p_dino_temp)
            labels["5. DINOv2 + Temporal (T=8)"].append(lbl)

            # 6. All Visual (Equal weighted baseline)
            p_all_vis = (p_sem * 0.50 + p_for * 0.25 + p_temp * 0.25)
            preds["6. All Visual (DINOv2 + SRM-CNN + GRU T=8)"].append(p_all_vis)
            labels["6. All Visual (DINOv2 + SRM-CNN + GRU T=8)"].append(lbl)

            # 7. All Modalities
            p_all_mod = (p_sem * 0.40 + p_for * 0.20 + p_temp * 0.20 + p_av * 0.20)
            preds["7. All Modalities (Visual + Audio-Visual)"].append(p_all_mod)
            labels["7. All Modalities (Visual + Audio-Visual)"].append(lbl)

            # 8. All + Dynamic Reliability Weighting (Trained Fusion MLP)
            preds["8. All + Dynamic Reliability Weighting"].append(p_full)
            labels["8. All + Dynamic Reliability Weighting"].append(lbl)

            # 9. All + Reliability + Calibrated Abstention (abstains if uncertainty > 0.60)
            if unc > 0.60:
                abstained_count += 1
            else:
                preds["9. All + Reliability + Calibrated Abstention"].append(p_full)
                labels["9. All + Reliability + Calibrated Abstention"].append(lbl)

        if (idx + 1) % 100 == 0 or (idx + 1) == len(records):
            logger.info("Ablation processed %d / %d multi-frame video samples", idx + 1, len(records))

    # Compute comparative metrics across all 9 configurations
    summary = []
    print("\n" + "=" * 110)
    print("           EXPERIMENT 3: RIGOROUS MULTI-FRAME (T=8) ARCHITECTURAL ABLATION STUDY")
    print("=" * 110)
    print(f"{'CONFIGURATION':<48} | {'AUROC':<8} | {'ACCURACY':<10} | {'BRIER':<8} | {'ABSTAIN (%)':<12}")
    print("-" * 110)

    for cfg in configs:
        y_t = np.array(labels[cfg])
        y_p = np.array(preds[cfg])

        if len(y_t) > 0 and len(np.unique(y_t)) > 1:
            auc = float(roc_auc_score(y_t, y_p))
            acc = float(accuracy_score(y_t, (y_p >= 0.5).astype(int)))
            brier = float(brier_score_loss(y_t, y_p))
        else:
            auc = float("nan")
            acc = 0.0
            brier = float("nan")

        abst_rate = (abstained_count / total_samples) if "Abstention" in cfg and total_samples > 0 else 0.0

        summary.append({
            "configuration": cfg,
            "auroc": round(auc, 4),
            "accuracy": round(acc, 4),
            "brier_score": round(brier, 4),
            "samples_evaluated": len(y_t),
            "abstention_rate": round(abst_rate, 4),
        })

        print(f"{cfg:<48} | {auc:<8.4f} | {acc*100:<9.2f}% | {brier:<8.4f} | {abst_rate*100:<11.1f}%")

    print("=" * 110 + "\n")
    return summary


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Running Rigorous Multi-Frame (T=8) Ablation Study on %s", device)

    ckpt_path = REPO_ROOT / "outputs" / "checkpoints" / "best.ckpt"
    model = FORGELite()
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()

    # Load 300 balanced multi-frame videos from source-disjoint test manifest
    with open(REPO_ROOT / "data" / "manifests" / "test_manifest_seen.json", "r", encoding="utf-8") as f:
        recs = json.load(f)

    # Filter to records with at least 4 crops
    multi_frame_recs = [r for r in recs if len(r.get("frame_crops", [])) >= 4]
    reals = [r for r in multi_frame_recs if r.get("label_int") == 0][:150]
    fakes = [r for r in multi_frame_recs if r.get("label_int") == 1][:150]
    balanced_test = reals + fakes
    random.Random(42).shuffle(balanced_test)

    logger.info("Loaded %d balanced multi-frame video tracks (%d Real, %d Fake)",
                len(balanced_test), len(reals), len(fakes))

    summary = run_ablation_evaluations(model, balanced_test, device)

    out_file = REPO_ROOT / "reports" / "rigorous_ablation_report.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info("Saved rigorous ablation report to %s", out_file)


if __name__ == "__main__":
    main()
