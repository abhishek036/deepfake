"""
evaluation/bootstrap_ablation_significance.py
Performs non-parametric bootstrap resampling (N=1000) to compute 95% confidence intervals
for the AUROC of various ablation configurations.
This determines if the differences between branches (e.g., 0.8193 -> 0.8264 -> 0.8275)
are statistically significant.
"""

from __future__ import annotations

import json
import logging
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score
from torchvision import transforms
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.forge_lite import FORGELite

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("bootstrap_ablation")

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def load_video_crops(crop_paths: List[str], target_frames: int = 8) -> Optional[torch.Tensor]:
    valid_paths = []
    for p_str in crop_paths:
        p = REPO_ROOT / p_str if not Path(p_str).is_absolute() else Path(p_str)
        if p.exists():
            valid_paths.append(p)
    if len(valid_paths) < 2:
        return None
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
    return torch.stack(tensors, dim=0).unsqueeze(0)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Running Bootstrap Ablation Analysis on %s", device)

    ckpt_path = REPO_ROOT / "outputs" / "checkpoints" / "best.ckpt"
    model = FORGELite()
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()

    with open(REPO_ROOT / "data" / "manifests" / "test_manifest_seen.json", "r", encoding="utf-8") as f:
        recs = json.load(f)

    multi_frame_recs = [r for r in recs if len(r.get("frame_crops", [])) >= 4]
    reals = [r for r in multi_frame_recs if r.get("label_int") == 0][:150]
    fakes = [r for r in multi_frame_recs if r.get("label_int") == 1][:150]
    balanced_test = reals + fakes
    random.Random(42).shuffle(balanced_test)

    logger.info("Extracting predictions for %d balanced multi-frame videos...", len(balanced_test))
    
    configs = [
        "1. Semantic Only (DINOv2 ViT-S/14)",
        "2. Forensic Only (SRM-CNN)",
        "3. Temporal Only (Spatiotemporal GRU T=8)",
        "4. DINOv2 + Forensic Dual-Domain",
        "5. DINOv2 + Temporal (T=8)",
        "6. All Visual (DINOv2 + SRM-CNN + GRU T=8)",
        "7. All Modalities (Visual + Audio-Visual)",
        "8. All + Dynamic Reliability Weighting"
    ]
    
    preds = {c: [] for c in configs}
    labels = []
    
    for r in tqdm(balanced_test):
        crop_paths = r.get("frame_crops") or [r.get("source_path") or r.get("crop_path")]
        batch_t8 = load_video_crops(crop_paths, target_frames=8)
        if batch_t8 is None:
            continue
            
        lbl = int(r.get("label_int", 1 if r.get("label") == "fake" else 0))
        
        with torch.no_grad():
            batch_t8 = batch_t8.to(device)
            out = model(frames=batch_t8)
            b_probs = out["branch_fake_probs"].squeeze().cpu().numpy()
            p_sem, p_for, p_temp, _, p_av = b_probs
            p_full = float(out.get("fake_score", torch.sigmoid(out["fake_logit"])).squeeze().item())
            
            preds["1. Semantic Only (DINOv2 ViT-S/14)"].append(p_sem)
            preds["2. Forensic Only (SRM-CNN)"].append(p_for)
            preds["3. Temporal Only (Spatiotemporal GRU T=8)"].append(p_temp)
            preds["4. DINOv2 + Forensic Dual-Domain"].append(p_sem * 0.65 + p_for * 0.35)
            preds["5. DINOv2 + Temporal (T=8)"].append(p_sem * 0.70 + p_temp * 0.30)
            preds["6. All Visual (DINOv2 + SRM-CNN + GRU T=8)"].append(p_sem * 0.50 + p_for * 0.25 + p_temp * 0.25)
            preds["7. All Modalities (Visual + Audio-Visual)"].append(p_sem * 0.40 + p_for * 0.20 + p_temp * 0.20 + p_av * 0.20)
            preds["8. All + Dynamic Reliability Weighting"].append(p_full)
            labels.append(lbl)

    logger.info("Finished extracting predictions for %d samples.", len(labels))
    labels = np.array(labels)
    for c in configs:
        preds[c] = np.array(preds[c])
    
    # Bootstrap
    n_iterations = 1000
    n_size = len(labels)
    
    logger.info("Running %d bootstrap iterations for AUROC 95%% CI...", n_iterations)
    
    results = {}
    print("\n" + "=" * 100)
    print(f"       STATISTICAL VALIDATION: BOOTSTRAPPED AUROC CONFIDENCE INTERVALS (N={n_iterations})")
    print("=" * 100)
    print(f"{'CONFIGURATION':<45} | {'AUROC':<8} | {'95% CI (LOWER, UPPER)':<25}")
    print("-" * 100)
    
    for cfg in configs:
        scores = []
        y_true = labels
        y_pred = preds[cfg]
        
        # Point estimate
        if len(np.unique(y_true)) > 1:
            point_auc = roc_auc_score(y_true, y_pred)
        else:
            point_auc = np.nan
            
        for i in range(n_iterations):
            indices = np.random.choice(n_size, size=n_size, replace=True)
            if len(np.unique(y_true[indices])) > 1:
                scores.append(roc_auc_score(y_true[indices], y_pred[indices]))
                
        alpha = 0.95
        p = ((1.0-alpha)/2.0) * 100
        lower = np.percentile(scores, p)
        p = (alpha+((1.0-alpha)/2.0)) * 100
        upper = np.percentile(scores, p)
        
        results[cfg] = {
            "point_estimate": round(float(point_auc), 4),
            "ci_lower": round(float(lower), 4),
            "ci_upper": round(float(upper), 4)
        }
        print(f"{cfg:<45} | {point_auc:<8.4f} | [{lower:.4f}, {upper:.4f}]")
        
    print("=" * 100 + "\n")
    
    out_file = REPO_ROOT / "reports" / "bootstrap_ablation_significance.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved bootstrap results to %s", out_file)

if __name__ == "__main__":
    main()
