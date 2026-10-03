"""
evaluation/evaluate_v2_router_vs_v1_baseline.py
Experiment for Tier 2: Evidence-Aware Modality Router vs Current Fusion
"""

from __future__ import annotations

import json
import logging
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import io

import numpy as np
import torch
from PIL import Image, ImageFilter
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from torchvision import transforms
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.forge_lite import FORGELite

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
logger = logging.getLogger("eval_v2_vs_v1")

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def apply_perturbation(img: Image.Image, condition: str) -> Image.Image:
    if condition == "clean":
        return img
    elif condition == "jpeg_q50":
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=50)
        buf.seek(0)
        return Image.open(buf).convert("RGB")
    return img

def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(probs)
    if n == 0:
        return 0.0
    for i in range(n_bins):
        bin_lower, bin_upper = bin_boundaries[i], bin_boundaries[i + 1]
        in_bin = (probs >= bin_lower) & (probs < bin_upper if i < n_bins - 1 else probs <= bin_upper)
        prop = np.mean(in_bin)
        if prop > 0:
            avg_acc = np.mean(labels[in_bin])
            avg_conf = np.mean(probs[in_bin])
            ece += abs(avg_conf - avg_acc) * prop
    return float(ece)

def evaluate_model(model: FORGELite, records: List[Dict[str, Any]], condition: str, device: torch.device):
    model.eval()
    y_true, y_prob = [], []
    for r in tqdm(records, desc=f"Evaluating ({condition})", leave=False):
        crop_paths = r.get("frame_crops") or [r.get("source_path") or r.get("crop_path")]
        p_str = crop_paths[len(crop_paths) // 2]
        p = REPO_ROOT / p_str if not Path(p_str).is_absolute() else Path(p_str)
        if not p.exists(): continue
        try:
            with Image.open(p) as img:
                perturbed_img = apply_perturbation(img.convert("RGB"), condition)
                tensor = TRANSFORM(perturbed_img).unsqueeze(0).unsqueeze(0).to(device)
            lbl = int(r.get("label_int", 1 if r.get("label") == "fake" else 0))
        except:
            continue
            
        with torch.no_grad():
            out = model(frames=tensor)
            prob = float(out.get("fake_score", torch.sigmoid(out["fake_logit"])).squeeze().item())
        y_true.append(lbl)
        y_prob.append(prob)
        
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    
    if len(y_true) == 0:
        return {"auroc": np.nan, "auprc": np.nan, "balanced_accuracy": np.nan, "mcc": np.nan, "brier": np.nan, "ece": np.nan, "auroc_ci": [np.nan, np.nan]}
        
    auc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else np.nan
    auprc = float(average_precision_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else np.nan
    acc = float(balanced_accuracy_score(y_true, (y_prob >= 0.5).astype(int)))
    mcc = float(matthews_corrcoef(y_true, (y_prob >= 0.5).astype(int)))
    brier = float(brier_score_loss(y_true, y_prob))
    ece = compute_ece(y_prob, y_true)
    
    # Bootstrap AUROC CI
    n_iterations = 1000
    n_size = len(y_true)
    scores = []
    if len(np.unique(y_true)) > 1:
        for i in range(n_iterations):
            idx = np.random.choice(n_size, size=n_size, replace=True)
            if len(np.unique(y_true[idx])) > 1:
                scores.append(roc_auc_score(y_true[idx], y_prob[idx]))
        lower = np.percentile(scores, 2.5) if scores else np.nan
        upper = np.percentile(scores, 97.5) if scores else np.nan
    else:
        lower, upper = np.nan, np.nan
        
    return {
        "n_samples": len(y_true),
        "auroc": round(auc, 4),
        "auprc": round(auprc, 4),
        "balanced_accuracy": round(acc, 4),
        "mcc": round(mcc, 4),
        "brier": round(brier, 4),
        "ece": round(ece, 4),
        "auroc_ci": [round(float(lower), 4), round(float(upper), 4)]
    }

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Running Tier-2 Comparative Evaluation on %s", device)
    
    # Load test set (exactly the same as baseline)
    with open(REPO_ROOT / "data" / "manifests" / "test_manifest_seen.json", "r", encoding="utf-8") as f:
        seen = json.load(f)
    with open(REPO_ROOT / "data" / "manifests" / "test_manifest_unseen.json", "r", encoding="utf-8") as f:
        unseen = json.load(f)
        
    all_test = seen + unseen
    test_reals = [r for r in all_test if r.get("label_int") == 0]
    test_fakes = [r for r in all_test if r.get("label_int") == 1]
    
    n_bal = min(250, len(test_reals), len(test_fakes))
    rng = random.Random(42)
    test_subset = rng.sample(test_reals, n_bal) + rng.sample(test_fakes, n_bal)
    rng.shuffle(test_subset)
    
    av_subset = [r for r in test_subset if r.get("manipulation_family") == "TalkingHeadSynthesis" or r.get("dataset") == "talkingheadbench"]
    fs_subset = [r for r in test_subset if r.get("manipulation_family") == "FaceShifter" or (r.get("label_int") == 0 and r.get("dataset") == "faceforensics")]
    fswap_subset = [r for r in test_subset if r.get("manipulation_family") == "FaceSwap" or (r.get("label_int") == 0 and r.get("dataset") == "faceforensics")]
    
    # Balance subsets if possible
    def balance(subset):
        r = [x for x in subset if x.get("label_int")==0]
        f = [x for x in subset if x.get("label_int")==1]
        m = min(len(r), len(f))
        if m == 0: return subset
        return rng.sample(r, m) + rng.sample(f, m)
        
    subsets = {
        "Overall (Clean)": (test_subset, "clean"),
        "Overall (JPEG)": (test_subset, "jpeg_q50"),
        "AV TalkingHeadBench": (balance(av_subset), "clean"),
        "FaceShifter": (balance(fs_subset), "clean"),
        "FaceSwap": (balance(fswap_subset), "clean"),
    }
    
    # Evaluate v2
    model_v2 = FORGELite().to(device)
    ckpt_v2_path = REPO_ROOT / "outputs" / "checkpoints" / "best_v2_modality_router.ckpt"
    if ckpt_v2_path.exists():
        ckpt_v2 = torch.load(ckpt_v2_path, map_location=device, weights_only=False)
        model_v2.load_state_dict(ckpt_v2.get("model_state_dict", ckpt_v2), strict=False)
    
    results_v2 = {}
    for name, (recs, cond) in subsets.items():
        if len(recs) == 0: continue
        results_v2[name] = evaluate_model(model_v2, recs, cond, device)
        
    print("\n" + "=" * 120)
    print("                      TIER-2 EVALUATION: V2 MODALITY ROUTER")
    print("=" * 120)
    print(f"{'SUBSET':<25} | {'AUROC':<7} | {'95% CI':<20} | {'BAL ACC':<8} | {'MCC':<7} | {'BRIER':<7} | {'ECE':<7}")
    print("-" * 120)
    for name, res in results_v2.items():
        ci_str = f"[{res['auroc_ci'][0]:.4f}, {res['auroc_ci'][1]:.4f}]" if not np.isnan(res['auroc_ci'][0]) else "[N/A, N/A]"
        print(f"{name:<25} | {res['auroc']:<7.4f} | {ci_str:<20} | {res['balanced_accuracy']*100:<7.2f}% | {res['mcc']:<7.4f} | {res['brier']:<7.4f} | {res['ece']:<7.4f}")
    
    print("=" * 120 + "\n")
    
if __name__ == "__main__":
    main()
