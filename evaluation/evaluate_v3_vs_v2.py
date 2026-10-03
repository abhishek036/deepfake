"""
evaluation/evaluate_v3_vs_v2.py
Performs evaluation of V3 (Codec-Invariant) against V2 (Baseline Router) 
on identical test sets across explicit perturbation conditions, including WebP.
"""

from __future__ import annotations

import json
import logging
import random
import sys
import io
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import torch
from PIL import Image, ImageFilter
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    matthews_corrcoef,
    roc_auc_score,
)
from torchvision import transforms
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.forge_lite import FORGELite

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger("eval_v3")

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
    elif condition == "jpeg_q30":
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=30)
        buf.seek(0)
        return Image.open(buf).convert("RGB")
    elif condition == "lowres":
        w, h = img.size
        scale = 0.3
        small = img.resize((int(w * scale), int(h * scale)), Image.BILINEAR)
        return small.resize((w, h), Image.BICUBIC)
    elif condition == "blur":
        return img.filter(ImageFilter.GaussianBlur(radius=2.0))
    elif condition == "webp_q20": # Held-out codec
        buf = io.BytesIO()
        img.save(buf, format="WEBP", quality=20)
        buf.seek(0)
        return Image.open(buf).convert("RGB")
    return img

def get_predictions(model: FORGELite, records: List[Dict], condition: str, device: torch.device):
    model.eval()
    y_true, y_prob = [], []
    for r in tqdm(records, desc=f"Inference ({condition})", leave=False):
        crop_paths = r.get("frame_crops") or [r.get("source_path") or r.get("crop_path")]
        p_str = crop_paths[len(crop_paths) // 2]
        p = REPO_ROOT / p_str if not Path(p_str).is_absolute() else Path(p_str)
        if not p.exists(): continue
        try:
            with Image.open(p) as img:
                perturbed = apply_perturbation(img.convert("RGB"), condition)
                tensor = TRANSFORM(perturbed).unsqueeze(0).unsqueeze(0).to(device)
            lbl = int(r.get("label_int", 1 if r.get("label") == "fake" else 0))
        except:
            continue
            
        with torch.no_grad():
            out = model(frames=tensor)
            prob = float(out.get("fake_score", torch.sigmoid(out["fake_logit"])).squeeze().item())
        y_true.append(lbl)
        y_prob.append(prob)
    return np.array(y_true), np.array(y_prob)

def compute_metrics(y_true, y_prob):
    if len(np.unique(y_true)) <= 1:
        return {"auroc": np.nan, "acc": np.nan, "mcc": np.nan, "brier": np.nan}
    return {
        "auroc": float(roc_auc_score(y_true, y_prob)),
        "acc": float(balanced_accuracy_score(y_true, (y_prob >= 0.5).astype(int))),
        "mcc": float(matthews_corrcoef(y_true, (y_prob >= 0.5).astype(int))),
        "brier": float(brier_score_loss(y_true, y_prob))
    }

def run_evaluation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Models to evaluate
    models_config = {
        "V2 (Router Baseline)": {"ckpt": "outputs/checkpoints/best_v2_modality_router.ckpt", "mode": "conf_qual"},
        "V3 (Codec-Invariant)": {"ckpt": "outputs/checkpoints/best_v3_codec_invariant.ckpt", "mode": "conf_qual"}
    }
    
    # 2. Prepare subsets
    with open(REPO_ROOT / "data/manifests/test_manifest_seen.json", "r", encoding="utf-8") as f:
        seen = json.load(f)
    with open(REPO_ROOT / "data/manifests/test_manifest_unseen.json", "r", encoding="utf-8") as f:
        unseen = json.load(f)
        
    all_test = seen + unseen
    test_reals = [r for r in all_test if r.get("label_int") == 0]
    test_fakes = [r for r in all_test if r.get("label_int") == 1]
    
    rng = random.Random(42)
    n_bal = min(250, len(test_reals), len(test_fakes))
    test_subset = rng.sample(test_reals, n_bal) + rng.sample(test_fakes, n_bal)
    rng.shuffle(test_subset)
    
    def balance(subset):
        r = [x for x in subset if x.get("label_int")==0]
        f = [x for x in subset if x.get("label_int")==1]
        m = min(len(r), len(f))
        if m == 0: return subset
        return rng.sample(r, m) + rng.sample(f, m)
        
    subsets = {
        "Clean": (test_subset, "clean"),
        "JPEG Q50": (test_subset, "jpeg_q50"),
        "JPEG Q30": (test_subset, "jpeg_q30"),
        "Low-Res (0.3x)": (test_subset, "lowres"),
        "Blur (r=2.0)": (test_subset, "blur"),
        "HELD-OUT (WebP Q20)": (test_subset, "webp_q20"),
        "FaceShifter (Clean)": (balance([r for r in test_subset if r.get("manipulation_family") == "FaceShifter" or (r.get("label_int") == 0 and r.get("dataset") == "faceforensics")]), "clean"),
        "FaceSwap (Clean)": (balance([r for r in test_subset if r.get("manipulation_family") == "FaceSwap" or (r.get("label_int") == 0 and r.get("dataset") == "faceforensics")]), "clean"),
    }
    
    # 3. Collect Predictions
    predictions = {}
    for name, config in models_config.items():
        ckpt_path = REPO_ROOT / config["ckpt"]
        if not ckpt_path.exists():
            logger.warning(f"Skipping {name}, checkpoint not found: {ckpt_path}")
            continue
            
        model = FORGELite(ablation_mode=config["mode"]).to(device)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
        
        predictions[name] = {}
        for sub_name, (recs, cond) in subsets.items():
            if len(recs) == 0: continue
            y_true, y_prob = get_predictions(model, recs, cond, device)
            predictions[name][sub_name] = {"y_true": y_true, "y_prob": y_prob}
            
    # 4. Reporting
    print("\n" + "=" * 120)
    print("                               V3 VS V2 CODEC-INVARIANCE EVALUATION")
    print("=" * 120)
    
    print(f"{'Condition / Subset':<25} | {'V2 AUROC':<10} | {'V3 AUROC':<10} | {'Delta_Codec (V2)':<16} | {'Delta_Codec (V3)':<16}")
    print("-" * 120)
    
    v2 = "V2 (Router Baseline)"
    v3 = "V3 (Codec-Invariant)"
    
    if v2 not in predictions or v3 not in predictions:
        print("Required models missing.")
        return
        
    v2_clean = compute_metrics(predictions[v2]["Clean"]["y_true"], predictions[v2]["Clean"]["y_prob"])["auroc"]
    v3_clean = compute_metrics(predictions[v3]["Clean"]["y_true"], predictions[v3]["Clean"]["y_prob"])["auroc"]
    
    for sub_name in subsets.keys():
        v2_m = compute_metrics(predictions[v2][sub_name]["y_true"], predictions[v2][sub_name]["y_prob"])["auroc"]
        v3_m = compute_metrics(predictions[v3][sub_name]["y_true"], predictions[v3][sub_name]["y_prob"])["auroc"]
        
        if "Clean" in sub_name:
            d_v2, d_v3 = "---", "---"
        else:
            d_v2 = f"{(v2_m - v2_clean):+.4f}"
            d_v3 = f"{(v3_m - v3_clean):+.4f}"
            
        print(f"{sub_name:<25} | {v2_m:<10.4f} | {v3_m:<10.4f} | {d_v2:<16} | {d_v3:<16}")

if __name__ == "__main__":
    run_evaluation()
