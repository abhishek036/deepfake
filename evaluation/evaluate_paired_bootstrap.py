"""
evaluation/evaluate_paired_bootstrap.py
Performs a rigorous paired statistical bootstrap test across router ablations
on exactly the same subsets to determine if gains are statistically significant.
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
from PIL import Image
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger("paired_bootstrap")

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
        return {"auroc": np.nan, "auprc": np.nan, "acc": np.nan, "mcc": np.nan, "brier": np.nan, "ece": np.nan}
    return {
        "auroc": float(roc_auc_score(y_true, y_prob)),
        "auprc": float(average_precision_score(y_true, y_prob)),
        "acc": float(balanced_accuracy_score(y_true, (y_prob >= 0.5).astype(int))),
        "mcc": float(matthews_corrcoef(y_true, (y_prob >= 0.5).astype(int))),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "ece": compute_ece(y_prob, y_true)
    }

def run_paired_bootstrap():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Models to evaluate
    models_config = {
        "A (v1_baseline)": {"ckpt": "outputs/checkpoints/best.ckpt", "mode": "v1"},
        "B (v2_conf_only)": {"ckpt": "outputs/checkpoints/best_v2_conf_only.ckpt", "mode": "conf_only"},
        "C (v2_conf_qual)": {"ckpt": "outputs/checkpoints/best_v2_modality_router.ckpt", "mode": "conf_qual"}
    }
    
    # 2. Prepare identical subsets
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
        "Overall (Clean)": (test_subset, "clean"),
        "Overall (JPEG Q50)": (test_subset, "jpeg_q50"),
        "FaceShifter": (balance([r for r in test_subset if r.get("manipulation_family") == "FaceShifter" or (r.get("label_int") == 0 and r.get("dataset") == "faceforensics")]), "clean"),
        "FaceSwap": (balance([r for r in test_subset if r.get("manipulation_family") == "FaceSwap" or (r.get("label_int") == 0 and r.get("dataset") == "faceforensics")]), "clean"),
        "TalkingHeadBench": (balance([r for r in test_subset if (r.get("dataset") == "talkingheadbench" or r.get("manipulation_family") == "TalkingHeadSynthesis") or (r.get("label_int") == 0 and r.get("dataset") == "talkingheadbench")]), "clean")
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
            
    # 4. Bootstrap and Reporting
    print("\n" + "=" * 140)
    print("                               PAIRED STATISTICAL BOOTSTRAP (N=1000)")
    print("=" * 140)
    
    base_model = "A (v1_baseline)"
    if base_model not in predictions:
        print("Baseline not available.")
        return
        
    n_iterations = 1000
    for sub_name in subsets.keys():
        print(f"\nSubset: {sub_name}")
        print("-" * 140)
        
        base_y_true = predictions[base_model][sub_name]["y_true"]
        if len(np.unique(base_y_true)) <= 1:
            print(f"Skipping {sub_name}: Need both classes.")
            continue
            
        print(f"{'Model':<20} | {'AUROC':<7} | {'95% CI':<20} | {'BAL ACC':<8} | {'MCC':<7} | {'BRIER':<7} | {'Delta_AUROC vs Baseline (95% CI)':<30}")
        print("-" * 140)
        
        for name in models_config.keys():
            if name not in predictions: continue
            
            y_true = predictions[name][sub_name]["y_true"]
            y_prob = predictions[name][sub_name]["y_prob"]
            base_prob = predictions[base_model][sub_name]["y_prob"]
            
            mets = compute_metrics(y_true, y_prob)
            
            auc_scores = []
            delta_auc_scores = []
            n_size = len(y_true)
            
            np.random.seed(42)
            for _ in range(n_iterations):
                idx = np.random.choice(n_size, size=n_size, replace=True)
                if len(np.unique(y_true[idx])) > 1:
                    iter_auc = roc_auc_score(y_true[idx], y_prob[idx])
                    auc_scores.append(iter_auc)
                    if name != base_model:
                        iter_base_auc = roc_auc_score(base_y_true[idx], base_prob[idx])
                        delta_auc_scores.append(iter_auc - iter_base_auc)
                        
            ci_lower = np.percentile(auc_scores, 2.5) if auc_scores else np.nan
            ci_upper = np.percentile(auc_scores, 97.5) if auc_scores else np.nan
            ci_str = f"[{ci_lower:.4f}, {ci_upper:.4f}]"
            
            if name == base_model:
                delta_str = "---"
            else:
                d_low = np.percentile(delta_auc_scores, 2.5) if delta_auc_scores else np.nan
                d_up = np.percentile(delta_auc_scores, 97.5) if delta_auc_scores else np.nan
                emp_delta = mets['auroc'] - compute_metrics(base_y_true, base_prob)['auroc']
                sig = "*" if (d_low > 0 or d_up < 0) else "ns"
                delta_str = f"{emp_delta:+.4f} [{d_low:+.4f}, {d_up:+.4f}] ({sig})"
                
            print(f"{name:<20} | {mets['auroc']:<7.4f} | {ci_str:<20} | {mets['acc']*100:<7.2f}% | {mets['mcc']:<7.4f} | {mets['brier']:<7.4f} | {delta_str:<30}")

if __name__ == "__main__":
    run_paired_bootstrap()
