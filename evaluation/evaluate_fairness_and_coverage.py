"""
evaluation/evaluate_fairness_and_coverage.py
============================================
CVPR 2027 Demographic Fairness & Coverage-Error Evaluation for FORGE-Lite.

Implements two required CVPR 2027 experimental analyses:
1. Per-Demographic / Skin-Tone Fairness Evaluation (MASTER_PLAN §2 Risk 6 & §9):
   - Measures performance across all 5 demographic groups in FakeAVCeleb:
     * African
     * Asian (East)
     * Asian (South)
     * Caucasian (American)
     * Caucasian (European)
   - Reports per-group: AUROC, EER, TPR@1%FPR, TPR@0.1%FPR, and EER Gap (Disparity).

2. Selective Prediction & Coverage vs. Error Curve (MASTER_PLAN §9):
   - Evaluates Risk-Coverage dynamics using calibrated prediction margin & uncertainty.
   - Measures Error Rate at coverage intervals [100%, 95%, 90%, 85%, 80%, 75%, 70%].
   - Validates quality-gated abstention policy for high-stakes deployment.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import torch
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, roc_curve
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
logger = logging.getLogger("fairness_coverage")

TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def compute_eer(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Computes Equal Error Rate (EER) using ROC curve interpolation."""
    if len(np.unique(y_true)) < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=1)
    eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
    return float(eer)


def compute_tpr_at_fpr(y_true: np.ndarray, y_score: np.ndarray, target_fpr: float = 0.01) -> float:
    """Computes TPR at a fixed operational False Positive Rate."""
    if len(np.unique(y_true)) < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=1)
    # Find max TPR where FPR <= target_fpr
    valid_idx = np.where(fpr <= target_fpr)[0]
    if len(valid_idx) == 0:
        return 0.0
    return float(np.max(tpr[valid_idx]))


def load_video_frames(video_path: Path, max_frames: int = 8) -> Optional[torch.Tensor]:
    """Extracts evenly spaced face crops from an MP4 video."""
    if not video_path.exists():
        return None
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return None

    if total_frames > max_frames:
        frame_indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)
    else:
        frame_indices = np.arange(total_frames)

    frames = []
    current_idx = 0
    selected_set = set(frame_indices)

    while cap.isOpened() and len(frames) < len(frame_indices):
        ret, frame = cap.read()
        if not ret:
            break
        if current_idx in selected_set:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(TRANSFORM(frame_rgb))
        current_idx += 1

    cap.release()
    if not frames:
        return None
    # Shape: (1, T, 3, 224, 224)
    return torch.stack(frames, dim=0).unsqueeze(0)


def run_evaluation(samples_per_race_per_class: int = 30):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Initializing Fairness & Coverage Evaluation on %s", device)

    # 1. Load Master Checkpoint and Calibration
    ckpt_path = REPO_ROOT / "outputs" / "checkpoints" / "best.ckpt"
    model = FORGELite()
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()

    calib_path = REPO_ROOT / "outputs" / "checkpoints" / "calibration_temperature.pt"
    temp, bias, threshold = 9.6651, 0.0, 0.672
    if calib_path.exists():
        cdata = torch.load(calib_path, map_location="cpu", weights_only=False)
        temp = float(cdata.get("temperature", 9.6651))
        bias = float(cdata.get("platt_b", 0.0))
        thresh = float(cdata.get("image_decision_threshold", 0.672))
    logger.info("Loaded master model | Calibrator: Temp=%.4f, Bias=%.4f, Thresh=%.4f", temp, bias, threshold)

    # 2. Collect Stratified Balanced Dataset by Race
    manifest_path = REPO_ROOT / "data" / "manifests" / "fakeavceleb.jsonl"
    by_race = defaultdict(lambda: {0: [], 1: []})

    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            race = obj.get("race", "Unknown")
            lbl = int(obj.get("label_int", 1 if obj.get("label") == "fake" else 0))
            p = REPO_ROOT / obj["source_path"]
            if len(by_race[race][lbl]) < samples_per_race_per_class:
                if p.exists():
                    by_race[race][lbl].append((p, obj.get("video_id", "")))

    logger.info("Collected FakeAVCeleb demographic samples:")
    for race, classes in sorted(by_race.items()):
        logger.info("  %-25s: %d Real, %d Fake", race, len(classes[0]), len(classes[1]))

    # 3. Perform Video Inference Across All Demographics
    all_results = []

    for race, classes in sorted(by_race.items()):
        items = [(p, vid, 0, race) for p, vid in classes[0]] + [(p, vid, 1, race) for p, vid in classes[1]]
        random.seed(42)
        random.shuffle(items)

        for p, vid, lbl, r_name in items:
            t_tensor = load_video_frames(p, max_frames=8)
            if t_tensor is None:
                continue

            t_tensor = t_tensor.to(device)
            with torch.no_grad():
                out = model(t_tensor)
                raw_logit = float(out["fake_logit"].squeeze().item())
                cal_logit = (raw_logit / temp) + bias
                cal_prob = float(1.0 / (1.0 + np.exp(-cal_logit)))

                # Uncertainty metrics
                margin_unc = 1.0 - 2.0 * abs(cal_prob - 0.5)
                branch_disagree = float(out["branch_disagreement"].squeeze().item()) if "branch_disagreement" in out else 0.0
                total_unc = 0.7 * margin_unc + 0.3 * branch_disagree

            all_results.append({
                "video_id": vid,
                "race": r_name,
                "label": lbl,
                "prob": cal_prob,
                "uncertainty": total_unc,
                "error": int((cal_prob >= threshold) != lbl),
            })

    logger.info("Completed inference on %d total multi-frame videos", len(all_results))

    # 4. PART 1: Per-Demographic Fairness Table
    race_metrics = {}
    print("\n" + "=" * 115)
    print("CVPR 2027 DEMOGRAPHIC / SKIN-TONE FAIRNESS EVALUATION ON FAKEAVCELEB")
    print("-" * 115)
    print(f"{'DEMOGRAPHIC GROUP':<25} | {'SAMPLES (R/F)':<15} | {'AUROC':<8} | {'EER (%)':<8} | {'TPR@1%FPR':<10} | {'TPR@0.1%FPR':<12} | {'BAL ACC':<8}")
    print("-" * 115)

    eers = []
    aucs = []

    for race, _ in sorted(by_race.items()):
        sub = [r for r in all_results if r["race"] == race]
        if not sub:
            continue
        y_t = np.array([r["label"] for r in sub])
        y_p = np.array([r["prob"] for r in sub])

        n_real = int((y_t == 0).sum())
        n_fake = int((y_t == 1).sum())

        auc = float(roc_auc_score(y_t, y_p))
        eer = compute_eer(y_t, y_p)
        tpr1 = compute_tpr_at_fpr(y_t, y_p, target_fpr=0.01)
        tpr01 = compute_tpr_at_fpr(y_t, y_p, target_fpr=0.001)
        bal_acc = float(balanced_accuracy_score(y_t, (y_p >= threshold).astype(int)))

        race_metrics[race] = {
            "n_real": n_real, "n_fake": n_fake,
            "auroc": round(auc, 4), "eer": round(eer, 4),
            "tpr_at_1pct_fpr": round(tpr1, 4), "tpr_at_0_1pct_fpr": round(tpr01, 4),
            "balanced_accuracy": round(bal_acc, 4),
        }
        eers.append(eer)
        aucs.append(auc)

        samp_str = f"{n_real}R / {n_fake}F"
        print(f"{race:<25} | {samp_str:<15} | {auc:<8.4f} | {eer * 100:<8.2f}% | {tpr1 * 100:<10.2f}% | {tpr01 * 100:<12.2f}% | {bal_acc * 100:<8.2f}%")

    # Pooled Overall
    all_t = np.array([r["label"] for r in all_results])
    all_p = np.array([r["prob"] for r in all_results])
    pooled_auc = float(roc_auc_score(all_t, all_p))
    pooled_eer = compute_eer(all_t, all_p)
    pooled_tpr1 = compute_tpr_at_fpr(all_t, all_p, target_fpr=0.01)
    pooled_tpr01 = compute_tpr_at_fpr(all_t, all_p, target_fpr=0.001)
    pooled_bal_acc = float(balanced_accuracy_score(all_t, (all_p >= threshold).astype(int)))

    race_metrics["Pooled Demographics"] = {
        "n_real": int((all_t == 0).sum()), "n_fake": int((all_t == 1).sum()),
        "auroc": round(pooled_auc, 4), "eer": round(pooled_eer, 4),
        "tpr_at_1pct_fpr": round(pooled_tpr1, 4), "tpr_at_0_1pct_fpr": round(pooled_tpr01, 4),
        "balanced_accuracy": round(pooled_bal_acc, 4),
    }

    max_eer_gap = max(eers) - min(eers)
    print("-" * 115)
    samp_str = f"{int((all_t == 0).sum())}R / {int((all_t == 1).sum())}F"
    print(f"{'Pooled Demographics':<25} | {samp_str:<15} | {pooled_auc:<8.4f} | {pooled_eer * 100:<8.2f}% | {pooled_tpr1 * 100:<10.2f}% | {pooled_tpr01 * 100:<12.2f}% | {pooled_bal_acc * 100:<8.2f}%")
    print("=" * 115)
    print(f"Fairness Disparity: Max Inter-Group EER Gap = {max_eer_gap * 100:.2f}%\n")

    # 5. PART 2: Coverage vs. Error Curve (Selective Prediction)
    sorted_by_uncertainty = sorted(all_results, key=lambda x: x["uncertainty"])
    total_count = len(sorted_by_uncertainty)

    coverage_levels = [1.00, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.60, 0.50]
    coverage_table = []

    print("=" * 75)
    print("CVPR 2027 SELECTIVE PREDICTION: COVERAGE VS. ERROR RATE CURVE")
    print("-" * 75)
    print(f"{'COVERAGE (%)':<15} | {'ACCEPTED SAMPLES':<18} | {'ERROR RATE (%)':<16} | {'SELECTIVE ACC (%)':<18}")
    print("-" * 75)

    for cov in coverage_levels:
        k = max(1, int(round(total_count * cov)))
        accepted = sorted_by_uncertainty[:k]
        errors = sum(r["error"] for r in accepted)
        err_rate = float(errors / k)
        sel_acc = float(1.0 - err_rate)

        coverage_table.append({
            "coverage": cov,
            "accepted_count": k,
            "error_rate": round(err_rate, 4),
            "selective_accuracy": round(sel_acc, 4),
        })
        print(f"{cov * 100:<15.0f} | {k:<18} | {err_rate * 100:<16.2f}% | {sel_acc * 100:<18.2f}%")

    print("=" * 75)

    # 6. Save Complete Report
    out_dir = REPO_ROOT / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_file = out_dir / "fairness_and_coverage_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump({
            "demographic_fairness": {
                "max_eer_gap": round(max_eer_gap, 4),
                "metrics_by_group": race_metrics,
            },
            "coverage_vs_error": coverage_table,
        }, f, indent=2)
    logger.info("Saved complete report to %s", report_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples-per-group", type=int, default=30, help="Samples per demographic group per class")
    args = parser.parse_args()
    run_evaluation(samples_per_race_per_class=args.samples_per_group)
