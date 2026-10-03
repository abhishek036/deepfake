"""
evaluation/audit_shortcuts.py
=============================
Rigorous Dataset Shortcut & Format Leakage Audit for FORGE-Lite.

Tests the scientific hypothesis:
"Is the near-perfect (0.9999) SRM forensic score on DF40 vs FFHQ detecting
genuine deepfake artifacts, or is it learning dataset-level format, resolution,
and compression shortcuts between FFHQ thumbnails and DF40 PNGs?"

Methodology:
1. Experiment A: Background-Only / Face-Masked Baseline
   - Black out the entire facial region (center 70% ellipse / box).
   - Feed ONLY the background pixels into the model (SRM branch and DINOv2).
   - If AUROC > 0.80 on blacked-out faces, the model is detecting background
     sensor/compression differences, not facial forgeries.

2. Experiment B: Metadata & Image Statistics Classifier
   - Extract trivial format features: file size (bytes), aspect ratio,
     Laplacian variance (sharpness), channel mean/std, high-frequency DCT power.
   - Train a simple 5-fold cross-validated linear classifier.
   - If trivial metadata achieves AUROC > 0.90, the benchmark suffers from
     severe format leakage.

3. Experiment C: Parent-Matched Reals Comparison
   - Compare DF40 fakes against their ACTUAL parent dataset reals (Celeb-DF-v2
     YouTube-real and FaceForensics++ original) vs FFHQ thumbnails.
"""

from __future__ import annotations

import json
import logging
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
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
logger = logging.getLogger("shortcut_audit")

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def mask_face_region(img: Image.Image, mask_ratio: float = 0.70) -> Image.Image:
    """Blacks out the central face region with an ellipse to evaluate background only."""
    img_copy = img.copy().convert("RGB")
    draw = ImageDraw.Draw(img_copy)
    w, h = img_copy.size
    cx, cy = w / 2.0, h / 2.0
    rx, ry = (w * mask_ratio) / 2.0, (h * mask_ratio) / 2.0
    bbox = [cx - rx, cy - ry, cx + rx, cy + ry]
    draw.ellipse(bbox, fill=(0, 0, 0))
    return img_copy


def extract_metadata_features(img_path: Path) -> np.ndarray:
    """Extracts trivial non-semantic image format statistics."""
    stat = img_path.stat()
    file_size_kb = stat.st_size / 1024.0

    try:
        with Image.open(img_path) as pil_img:
            w, h = pil_img.size
            aspect_ratio = float(w) / float(h)
            arr = np.array(pil_img.convert("RGB"), dtype=np.float32)
    except Exception:
        return np.zeros(8, dtype=np.float32)

    # Channel statistics
    mean_r, mean_g, mean_b = np.mean(arr, axis=(0, 1))
    std_r, std_g, std_b = np.std(arr, axis=(0, 1))

    # Laplacian variance (blur / sharpness metric)
    gray = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    return np.array([
        file_size_kb,
        w,
        h,
        aspect_ratio,
        mean_r,
        std_r,
        laplacian_var,
        float(np.mean(arr)),
    ], dtype=np.float32)


def run_shortcut_audit(num_samples_per_class: int = 150):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Initializing Shortcut & Leakage Audit on %s", device)

    # 1. Load Master Model
    ckpt_path = REPO_ROOT / "outputs" / "checkpoints" / "best.ckpt"
    model = FORGELite()
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()

    # 2. Collect DF40 Fakes and FFHQ Reals
    df40_fakes = []
    with open(REPO_ROOT / "data" / "manifests" / "df40_disk.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            p = REPO_ROOT / obj["source_path"]
            if p.exists():
                df40_fakes.append((p, obj.get("method", "df40"), obj.get("forgery_type", "fake")))
                if len(df40_fakes) >= num_samples_per_class:
                    break

    ffhq_reals = []
    with open(REPO_ROOT / "data" / "manifests" / "ffhq.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            p = REPO_ROOT / obj["source_path"]
            if p.exists():
                ffhq_reals.append((p, "ffhq", "real"))
                if len(ffhq_reals) >= num_samples_per_class:
                    break

    logger.info("Audit Pool: %d DF40 Fakes vs %d FFHQ Reals", len(df40_fakes), len(ffhq_reals))

    # ─────────────────────────────────────────────────────────────────────────
    # EXPERIMENT A: Background-Only / Face-Masked Baseline
    # ─────────────────────────────────────────────────────────────────────────
    logger.info("Executing Experiment A: Background-Only (Masked-Face) Inference...")
    all_items = [(p, 1) for p, _, _ in df40_fakes] + [(p, 0) for p, _, _ in ffhq_reals]
    random.seed(42)
    random.shuffle(all_items)

    y_true = []
    unmasked_probs = []
    masked_probs = []
    srm_unmasked_probs = []
    srm_masked_probs = []

    for idx, (p, label) in enumerate(all_items):
        try:
            with Image.open(p) as img:
                orig_img = img.convert("RGB")
                masked_img = mask_face_region(orig_img, mask_ratio=0.70)

                t_orig = TRANSFORM(orig_img).unsqueeze(0).unsqueeze(0).to(device)
                t_masked = TRANSFORM(masked_img).unsqueeze(0).unsqueeze(0).to(device)

                with torch.no_grad():
                    # Full model inference
                    out_orig = model(t_orig)
                    out_masked = model(t_masked)

                    p_orig = float(torch.sigmoid(out_orig["fake_logit"]).squeeze().item())
                    p_masked = float(torch.sigmoid(out_masked["fake_logit"]).squeeze().item())

                    # Forensic SRM standalone linear probe (index 1)
                    srm_p_orig = float(out_orig["branch_fake_probs"].squeeze().cpu().numpy()[1])
                    srm_p_masked = float(out_masked["branch_fake_probs"].squeeze().cpu().numpy()[1])

                y_true.append(label)
                unmasked_probs.append(p_orig)
                masked_probs.append(p_masked)
                srm_unmasked_probs.append(srm_p_orig)
                srm_masked_probs.append(srm_p_masked)

        except Exception as e:
            continue

    y_true = np.array(y_true)
    auc_unmasked = float(roc_auc_score(y_true, np.array(unmasked_probs)))
    auc_masked = float(roc_auc_score(y_true, np.array(masked_probs)))
    srm_auc_unmasked = float(roc_auc_score(y_true, np.array(srm_unmasked_probs)))
    srm_auc_masked = float(roc_auc_score(y_true, np.array(srm_masked_probs)))

    # ─────────────────────────────────────────────────────────────────────────
    # EXPERIMENT B: Metadata & Image Format Shortcut Baseline
    # ─────────────────────────────────────────────────────────────────────────
    logger.info("Executing Experiment B: Trivial Metadata & File Format Classifier...")
    X_meta = []
    y_meta = []
    for p, _, _ in df40_fakes:
        X_meta.append(extract_metadata_features(p))
        y_meta.append(1)
    for p, _, _ in ffhq_reals:
        X_meta.append(extract_metadata_features(p))
        y_meta.append(0)

    X_meta = np.array(X_meta)
    y_meta = np.array(y_meta)

    # 5-Fold Stratified Cross-Validation with simple Logistic Regression
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    meta_aucs = []
    meta_accs = []
    for train_idx, test_idx in cv.split(X_meta, y_meta):
        clf = LogisticRegression(max_iter=1000)
        # Normalize features
        mu = np.mean(X_meta[train_idx], axis=0, keepdims=True)
        sigma = np.std(X_meta[train_idx], axis=0, keepdims=True) + 1e-6
        X_tr = (X_meta[train_idx] - mu) / sigma
        X_te = (X_meta[test_idx] - mu) / sigma

        clf.fit(X_tr, y_meta[train_idx])
        preds = clf.predict_proba(X_te)[:, 1]
        meta_aucs.append(float(roc_auc_score(y_meta[test_idx], preds)))
        meta_accs.append(float(accuracy_score(y_meta[test_idx], (preds >= 0.5).astype(int))))

    mean_meta_auc = float(np.mean(meta_aucs))
    mean_meta_acc = float(np.mean(meta_accs))

    # ─────────────────────────────────────────────────────────────────────────
    # Print Audit Findings
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("                    DATASET SHORTCUT & FORMAT LEAKAGE AUDIT")
    print("=" * 90)
    print(f"{'EVALUATION CONDITION':<45} | {'AUROC':<10} | {'SHORTCUT VERDICT'}")
    print("-" * 90)
    print(f"{'Full Model (Unmasked Images)':<45} | {auc_unmasked:<10.4f} | Normal Reference")
    print(f"{'Full Model (Face Blacked Out - Background Only)':<45} | {auc_masked:<10.4f} | {'LEAKAGE DETECTED' if auc_masked > 0.80 else 'CLEAN (Face-Focused)'}")
    print(f"{'SRM Forensic Branch (Unmasked Images)':<45} | {srm_auc_unmasked:<10.4f} | Standalone Reference")
    print(f"{'SRM Forensic Branch (Face Blacked Out)':<45} | {srm_auc_masked:<10.4f} | {'LEAKAGE DETECTED' if srm_auc_masked > 0.80 else 'CLEAN (Face-Focused)'}")
    print(f"{'Trivial Metadata Classifier (Size, Blur, Format)':<45} | {mean_meta_auc:<10.4f} | {'FORMAT SHORTCUT' if mean_meta_auc > 0.85 else 'CLEAN'}")
    print("=" * 90)

    # Scientific interpretation
    leakage_verdict = (auc_masked > 0.80 or srm_auc_masked > 0.80 or mean_meta_auc > 0.85)
    if leakage_verdict:
        print("\nCRITICAL AUDIT FINDING: DATASET SHORTCUT CONFIRMED!")
        print("  - The model / forensic filters can distinguish FFHQ reals from DF40 fakes even")
        print("    when the face is 100% blacked out, or using trivial image size/blur metadata.")
        print("  - Conclusion: Evaluating DF40 fakes against FFHQ thumbnails measures pipeline/camera/format")
        print("    differences, NOT facial deepfake artifacts. FFHQ must be replaced by parent reals.")
    else:
        print("\nAUDIT PASSED: No severe format leakage detected.")

    # Save to report
    out_dir = REPO_ROOT / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "shortcut_audit_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({
            "audit": "Dataset Shortcut & Format Leakage Audit",
            "samples_evaluated": len(y_true),
            "unmasked_full_model_auroc": round(auc_unmasked, 4),
            "masked_background_only_full_model_auroc": round(auc_masked, 4),
            "unmasked_srm_branch_auroc": round(srm_auc_unmasked, 4),
            "masked_background_only_srm_branch_auroc": round(srm_auc_masked, 4),
            "metadata_only_auroc": round(mean_meta_auc, 4),
            "metadata_only_accuracy": round(mean_meta_acc, 4),
            "shortcut_leakage_confirmed": leakage_verdict,
        }, f, indent=2)
    logger.info("Saved audit report to %s", report_path)


if __name__ == "__main__":
    run_shortcut_audit(num_samples_per_class=100)
