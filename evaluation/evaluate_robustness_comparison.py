"""
evaluation/evaluate_robustness_comparison.py
Experiment 2: Robust-Hardened Checkpoint vs. Original Baseline Benchmark.

Directly tests the hypothesis:
"Did multi-codec robustness training shrink the severe JPEG degradation gap?"

Compares:
- Model A: outputs/checkpoints/best.ckpt (Pre-Hardening Baseline)
- Model B: outputs/checkpoints/best_robust_hardened.ckpt (Post-Hardening Model)

Across 4 Perturbation Conditions:
1. Clean (Reference)
2. JPEG Compression (Quality = 50)
3. Low-Resolution Downsampling (0.5x bilinear followed by bicubic upsampling)
4. Gaussian Blur (sigma = 1.0)

Reports:
- AUROC, Accuracy, Brier Score per condition
- Retention Deltas: Delta_JPEG, Delta_LowRes, Delta_Blur
"""

from __future__ import annotations

import io
import json
import logging
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageFilter
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
logger = logging.getLogger("robustness_compare")

BASE_TRANSFORM = transforms.Compose([
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
    elif condition == "low_res_0.5x":
        w, h = img.size
        small = img.resize((max(16, w // 2), max(16, h // 2)), resample=Image.BILINEAR)
        return small.resize((w, h), resample=Image.BICUBIC)
    elif condition == "blur_sigma1":
        return img.filter(ImageFilter.GaussianBlur(radius=1.0))
    return img


def evaluate_model_under_condition(
    model: FORGELite,
    records: List[Dict[str, Any]],
    condition: str,
    device: torch.device,
) -> Dict[str, Any]:
    model.eval()
    y_true = []
    y_prob = []

    for r in records:
        crop_paths = r.get("frame_crops") or [r.get("source_path") or r.get("crop_path")]
        p_str = crop_paths[len(crop_paths) // 2]
        p = REPO_ROOT / p_str if not Path(p_str).is_absolute() else Path(p_str)
        if not p.exists():
            continue

        try:
            with Image.open(p) as raw_img:
                perturbed_img = apply_perturbation(raw_img.convert("RGB"), condition)
                tensor = BASE_TRANSFORM(perturbed_img).unsqueeze(0).unsqueeze(0).to(device)
            lbl = int(r.get("label_int", 1 if r.get("label") == "fake" else 0))
        except Exception:
            continue

        with torch.no_grad():
            out = model(frames=tensor)
            prob = float(out.get("fake_score", torch.sigmoid(out["fake_logit"])).squeeze().item())

        y_true.append(lbl)
        y_prob.append(prob)

    y_true = np.array(y_true)
    y_prob = np.array(y_prob)

    auc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float("nan")
    acc = float(accuracy_score(y_true, (y_prob >= 0.5).astype(int)))
    brier = float(brier_score_loss(y_true, y_prob))

    return {
        "condition": condition,
        "n_samples": len(y_true),
        "auroc": round(auc, 4),
        "accuracy": round(acc, 4),
        "brier_score": round(brier, 4),
    }


def load_checkpoint_model(ckpt_path: Path, device: torch.device) -> FORGELite:
    model = FORGELite()
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    return model


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Running Robustness Hardening Comparison on %s", device)

    ckpt_original = REPO_ROOT / "outputs" / "checkpoints" / "best.ckpt"
    ckpt_hardened = REPO_ROOT / "outputs" / "checkpoints" / "best_robust_hardened.ckpt"

    if not ckpt_hardened.exists():
        logger.error("Hardened checkpoint not found: %s", ckpt_hardened)
        return

    logger.info("Loading original baseline model...")
    model_orig = load_checkpoint_model(ckpt_original, device)
    logger.info("Loading robust-hardened model...")
    model_hard = load_checkpoint_model(ckpt_hardened, device)

    # Load 200 balanced test samples (100 Real, 100 Fake) from disjoint test partition
    with open(REPO_ROOT / "data" / "manifests" / "test_manifest_seen.json", "r", encoding="utf-8") as f:
        recs = json.load(f)

    reals = [r for r in recs if r.get("label_int") == 0][:100]
    fakes = [r for r in recs if r.get("label_int") == 1][:100]
    test_subset = reals + fakes

    conditions = ["clean", "jpeg_q50", "low_res_0.5x", "blur_sigma1"]

    results_orig = {}
    results_hard = {}

    for cond in conditions:
        logger.info("Evaluating condition '%s' on Original model...", cond)
        results_orig[cond] = evaluate_model_under_condition(model_orig, test_subset, cond, device)
        logger.info("Evaluating condition '%s' on Hardened model...", cond)
        results_hard[cond] = evaluate_model_under_condition(model_hard, test_subset, cond, device)

    # Compute retention deltas
    clean_auc_orig = results_orig["clean"]["auroc"]
    clean_auc_hard = results_hard["clean"]["auroc"]

    delta_jpeg_orig = results_orig["jpeg_q50"]["auroc"] - clean_auc_orig
    delta_jpeg_hard = results_hard["jpeg_q50"]["auroc"] - clean_auc_hard

    delta_lowres_orig = results_orig["low_res_0.5x"]["auroc"] - clean_auc_orig
    delta_lowres_hard = results_hard["low_res_0.5x"]["auroc"] - clean_auc_hard

    delta_blur_orig = results_orig["blur_sigma1"]["auroc"] - clean_auc_orig
    delta_blur_hard = results_hard["blur_sigma1"]["auroc"] - clean_auc_hard

    print("\n" + "=" * 105)
    print("           EXPERIMENT 2: ROBUST-HARDENED CHECKPOINT VS. ORIGINAL BENCHMARK")
    print("=" * 105)
    print(f"{'MODEL VARIANT':<24} | {'CLEAN AUC':<11} | {'JPEG Q50':<11} | {'LOW-RES':<11} | {'BLUR':<11} | {'DELTA (JPEG)':<14} | {'STATUS':<12}")
    print("-" * 105)

    print(f"{'Original Baseline':<24} | {clean_auc_orig:<11.4f} | {results_orig['jpeg_q50']['auroc']:<11.4f} | {results_orig['low_res_0.5x']['auroc']:<11.4f} | {results_orig['blur_sigma1']['auroc']:<11.4f} | {delta_jpeg_orig:<+14.4f} | {'Baseline':<12}")
    print(f"{'Robust-Hardened':<24} | {clean_auc_hard:<11.4f} | {results_hard['jpeg_q50']['auroc']:<11.4f} | {results_hard['low_res_0.5x']['auroc']:<11.4f} | {results_hard['blur_sigma1']['auroc']:<11.4f} | {delta_jpeg_hard:<+14.4f} | {'Hardened':<12}")
    print("=" * 105)

    comparison_report = {
        "original_baseline": results_orig,
        "robust_hardened": results_hard,
        "deltas": {
            "original_delta_jpeg": round(delta_jpeg_orig, 4),
            "hardened_delta_jpeg": round(delta_jpeg_hard, 4),
            "original_delta_lowres": round(delta_lowres_orig, 4),
            "hardened_delta_lowres": round(delta_lowres_hard, 4),
            "original_delta_blur": round(delta_blur_orig, 4),
            "hardened_delta_blur": round(delta_blur_hard, 4),
            "jpeg_gap_reduction": round(abs(delta_jpeg_orig) - abs(delta_jpeg_hard), 4),
        },
    }

    out_file = REPO_ROOT / "reports" / "robustness_comparison_report.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(comparison_report, f, indent=2)
    logger.info("Saved robustness comparison report to %s", out_file)


if __name__ == "__main__":
    main()
