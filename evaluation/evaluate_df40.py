"""
evaluation/evaluate_df40.py
===========================
Comprehensive DF40 Forensic Benchmark & SRM Contribution Audit for FORGE-Lite.

Evaluates master checkpoint across all 4 DF40 generative families:
1. Fully Synthetic (Diffusion & GAN: pixart, sd2.1, RDDM, DiT, SiT, ddim, StyleGAN2/3)
2. Composited Face Swaps (simswap, inswap, blendface, faceswap, mobileswap, facedancer, fsgan)
3. Face Reenactment (fomm, facevid2vid, pirender, lia, mcnet, hyperreenact, MRAA)
4. Talking Head Audio/Visual Sync (sadtalker, wav2lip)

Balanced with authentic real faces (FFHQ & unseen controls).
Also runs the Forensic SRM Ablation Audit to measure SRM branch contribution on
fully_synthetic (diffusion) vs composited (face-swap) boundaries.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

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

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.forge_lite import FORGELite

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("df40_eval")

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_model_and_calibration(device: torch.device):
    ckpt_path = REPO_ROOT / "outputs" / "checkpoints" / "best.ckpt"
    logger.info("Loading master model from %s", ckpt_path.name)
    model = FORGELite()
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    logger.info("Loaded master state | Missing: %d, Unexpected: %d", len(missing), len(unexpected))
    model.to(device)
    model.eval()

    # Load calibration parameters
    calib_path = REPO_ROOT / "outputs" / "checkpoints" / "calibration_temperature.pt"
    temp, bias, thresh = 9.6651, 0.0, 0.672
    if calib_path.exists():
        cdata = torch.load(calib_path, map_location="cpu", weights_only=False)
        temp = float(cdata.get("temperature", 9.6651))
        bias = float(cdata.get("platt_b", 0.0))
        thresh = float(cdata.get("image_decision_threshold", 0.672))
    logger.info("Calibrator: Temp=%.4f | Platt_bias=%.4f | Threshold=%.4f", temp, bias, thresh)
    return model, temp, bias, thresh


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.672) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    n_total = len(y_true)
    n_real = int(np.sum(y_true == 0))
    n_fake = int(np.sum(y_true == 1))

    if n_real == 0 or n_fake == 0 or len(np.unique(y_true)) < 2:
        acc = float(accuracy_score(y_true, y_pred)) if n_total > 0 else 0.0
        return {
            "n_total": n_total, "n_real": n_real, "n_fake": n_fake,
            "auroc": float("nan"), "auprc": float("nan"),
            "accuracy": acc, "balanced_acc": acc,
            "tpr": float("nan"), "fpr": float("nan"), "specificity": float("nan"),
            "f1": float("nan"), "brier": float("nan"),
        }

    auc = float(roc_auc_score(y_true, y_prob))
    auprc = float(average_precision_score(y_true, y_prob))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    acc = float(accuracy_score(y_true, y_pred))

    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))

    tpr = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    brier = float(brier_score_loss(y_true, y_prob))

    return {
        "n_total": n_total, "n_real": n_real, "n_fake": n_fake,
        "auroc": round(auc, 4), "auprc": round(auprc, 4),
        "accuracy": round(acc, 4), "balanced_acc": round(bal_acc, 4),
        "tpr": round(tpr, 4), "fpr": round(fpr, 4), "specificity": round(spec, 4),
        "f1": round(f1, 4), "brier": round(brier, 4),
    }


def run_eval(samples_per_method: int = 50, audit_srm: bool = True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Initializing DF40 evaluation on %s", device)
    model, temp, bias, threshold = load_model_and_calibration(device)

    # 1. Load real baseline items (FFHQ)
    ffhq_manifest = REPO_ROOT / "data" / "manifests" / "ffhq.jsonl"
    real_paths = []
    if ffhq_manifest.exists():
        with open(ffhq_manifest, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                p = REPO_ROOT / obj["source_path"]
                if p.exists():
                    real_paths.append(p)
                    if len(real_paths) >= 2000:
                        break
    logger.info("Loaded %d real control faces from FFHQ", len(real_paths))
    random.seed(42)
    random.shuffle(real_paths)

    # 2. Group DF40 samples by method and family
    disk_manifest = REPO_ROOT / "data" / "manifests" / "df40_disk.jsonl"
    by_method = defaultdict(list)
    by_family = defaultdict(list)

    with open(disk_manifest, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            m = obj.get("method")
            ft = obj.get("forgery_type")
            p = REPO_ROOT / obj["source_path"]
            if len(by_method[m]) < samples_per_method:
                if p.exists():
                    by_method[m].append((p, ft))
                    by_family[ft].append((p, m))

    logger.info("Collected test samples across %d methods and %d families", len(by_method), len(by_family))
    for ft, items in by_family.items():
        logger.info("  Family '%s': %d fake samples", ft, len(items))

    # Helper for batch inference
    def infer_batch(paths: List[Path], disable_srm: bool = False) -> np.ndarray:
        probs = []
        batch_size = 32
        was_enabled = model.enable_forensic
        if disable_srm:
            model.enable_forensic = False
        try:
            for i in range(0, len(paths), batch_size):
                batch_paths = paths[i:i + batch_size]
                tensors = []
                for bp in batch_paths:
                    try:
                        with Image.open(bp) as img:
                            tensors.append(TRANSFORM(img.convert("RGB")))
                    except Exception:
                        continue
                if not tensors:
                    continue
                # Shape: (B, 1, 3, 224, 224)
                batch_t = torch.stack(tensors, dim=0).unsqueeze(1).to(device)
                with torch.no_grad():
                    out = model(batch_t)
                    raw_logits = out["fake_logit"].squeeze(-1).cpu().numpy()
                    cal_logits = (raw_logits / temp) + bias
                    p = 1.0 / (1.0 + np.exp(-cal_logits))
                    probs.extend(p.tolist() if isinstance(p, np.ndarray) else [p])
        finally:
            model.enable_forensic = was_enabled
        return np.array(probs)

    # 3. Evaluate Per-Family with Matched Real Controls
    family_results = {}
    real_idx = 0

    print("\n" + "=" * 105)
    print(f"{'DF40 FORGERY FAMILY':<22} | {'SAMPLES (R/F)':<15} | {'AUROC':<8} | {'ACC (%)':<8} | {'TPR (REC)':<10} | {'FPR':<8} | {'SPEC':<8}")
    print("-" * 105)

    all_y_true = []
    all_y_prob = []

    for ft, fake_tuples in sorted(by_family.items()):
        f_paths = [t[0] for t in fake_tuples]
        n_fake = len(f_paths)
        r_paths = real_paths[real_idx:real_idx + n_fake]
        real_idx += n_fake
        n_real = len(r_paths)

        r_probs = infer_batch(r_paths)
        f_probs = infer_batch(f_paths)

        y_true = np.concatenate([np.zeros(len(r_probs)), np.ones(len(f_probs))])
        y_prob = np.concatenate([r_probs, f_probs])

        all_y_true.extend(y_true.tolist())
        all_y_prob.extend(y_prob.tolist())

        metrics = compute_metrics(y_true, y_prob, threshold=threshold)
        family_results[ft] = metrics

        samp_str = f"{metrics['n_real']}R / {metrics['n_fake']}F"
        auc_str = f"{metrics['auroc']:.4f}"
        acc_str = f"{metrics['accuracy'] * 100:.2f}%"
        tpr_str = f"{metrics['tpr'] * 100:.2f}%"
        fpr_str = f"{metrics['fpr'] * 100:.2f}%"
        spec_str = f"{metrics['specificity'] * 100:.2f}%"
        print(f"{ft:<22} | {samp_str:<15} | {auc_str:<8} | {acc_str:<8} | {tpr_str:<10} | {fpr_str:<8} | {spec_str:<8}")

    overall_metrics = compute_metrics(np.array(all_y_true), np.array(all_y_prob), threshold=threshold)
    family_results["Overall Pooled DF40"] = overall_metrics
    print("-" * 105)
    samp_str = f"{overall_metrics['n_real']}R / {overall_metrics['n_fake']}F"
    auc_str = f"{overall_metrics['auroc']:.4f}"
    acc_str = f"{overall_metrics['accuracy'] * 100:.2f}%"
    tpr_str = f"{overall_metrics['tpr'] * 100:.2f}%"
    fpr_str = f"{overall_metrics['fpr'] * 100:.2f}%"
    spec_str = f"{overall_metrics['specificity'] * 100:.2f}%"
    print(f"{'Overall Pooled DF40':<22} | {samp_str:<15} | {auc_str:<8} | {acc_str:<8} | {tpr_str:<10} | {fpr_str:<8} | {spec_str:<8}")
    print("=" * 105)

    # 4. SRM Branch Marginal Contribution Audit (MASTER_PLAN §2 Risk 1)
    srm_audit_results = {}
    if audit_srm:
        logger.info("\nExecuting SRM Branch Forensic Audit on Fully Synthetic vs Composited...")
        for audit_fam in ["fully_synthetic", "composited"]:
            fake_paths = [t[0] for t in by_family[audit_fam]]
            ctrl_reals = real_paths[:len(fake_paths)]

            # Full model
            r_full = infer_batch(ctrl_reals, disable_srm=False)
            f_full = infer_batch(fake_paths, disable_srm=False)
            y_t = np.concatenate([np.zeros(len(r_full)), np.ones(len(f_full))])
            auc_full = float(roc_auc_score(y_t, np.concatenate([r_full, f_full])))

            # Ablated SRM model
            r_nosrm = infer_batch(ctrl_reals, disable_srm=True)
            f_nosrm = infer_batch(fake_paths, disable_srm=True)
            auc_nosrm = float(roc_auc_score(y_t, np.concatenate([r_nosrm, f_nosrm])))
            delta_auc = auc_full - auc_nosrm

            srm_audit_results[audit_fam] = {
                "auroc_with_srm": round(auc_full, 4),
                "auroc_without_srm": round(auc_nosrm, 4),
                "marginal_srm_delta": round(delta_auc, 4),
            }

        print("\n" + "=" * 70)
        print("SRM FORENSIC BRANCH MARGINAL CONTRIBUTION AUDIT (MASTER_PLAN §2)")
        print("-" * 70)
        print(f"{'FAMILY':<20} | {'WITH SRM':<12} | {'WITHOUT SRM':<14} | {'DELTA AUROC':<12}")
        print("-" * 70)
        for fam, sres in srm_audit_results.items():
            print(f"{fam:<20} | {sres['auroc_with_srm']:<12.4f} | {sres['auroc_without_srm']:<14.4f} | {sres['marginal_srm_delta']:+.4f}")
        print("=" * 70)

    # 5. Coverage vs. Error Curve (Selective Prediction on DF40)
    arr_t = np.array(all_y_true)
    arr_p = np.array(all_y_prob)
    uncertainty = 1.0 - 2.0 * np.abs(arr_p - 0.5)
    sorted_indices = np.argsort(uncertainty)

    coverage_levels = [1.00, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.60, 0.50]
    coverage_curve = []

    print("\n" + "=" * 80)
    print("CVPR 2027 SELECTIVE PREDICTION: COVERAGE VS. ERROR RATE ON DF40")
    print("-" * 80)
    print(f"{'COVERAGE (%)':<16} | {'ACCEPTED SAMPLES':<18} | {'ERROR RATE (%)':<16} | {'SELECTIVE ACC (%)':<18}")
    print("-" * 80)

    total_pts = len(sorted_indices)
    for cov in coverage_levels:
        k = max(1, int(round(total_pts * cov)))
        sel_idx = sorted_indices[:k]
        sel_t = arr_t[sel_idx]
        sel_p = arr_p[sel_idx]
        errors = int(np.sum((sel_p >= threshold) != sel_t))
        err_rate = float(errors / k)
        sel_acc = float(1.0 - err_rate)

        coverage_curve.append({
            "coverage": cov,
            "accepted_samples": k,
            "error_rate": round(err_rate, 4),
            "selective_accuracy": round(sel_acc, 4),
        })
        print(f"{cov * 100:<16.0f} | {k:<18} | {err_rate * 100:<16.2f}% | {sel_acc * 100:<18.2f}%")
    print("=" * 80 + "\n")

    # 6. Save Report
    out_dir = REPO_ROOT / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_data = {
        "benchmark": "DF40 Full Dataset Evaluation",
        "checkpoint": "best.ckpt",
        "calibration": {"temperature": temp, "platt_b": bias, "threshold": threshold},
        "family_metrics": family_results,
        "srm_audit": srm_audit_results,
        "coverage_vs_error": coverage_curve,
    }
    with open(out_dir / "df40_evaluation_report.json", "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
    logger.info("Saved complete report to reports/df40_evaluation_report.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples-per-method", type=int, default=30, help="Number of samples per generator method")
    parser.add_argument("--skip-srm-audit", action="store_true", help="Skip SRM ablation audit")
    args = parser.parse_args()
    run_eval(samples_per_method=args.samples_per_method, audit_srm=not args.skip_srm_audit)
