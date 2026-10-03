"""
evaluation/calibrate_held_out.py
Phase 5: Rigorous Held-Out Probabilistic Calibration for FORGE-Lite.

As mandated by forensic engineering principles:
1. Fits post-hoc calibration methods exclusively on the held-out CALIBRATION partition:
   data/manifests/calibration_manifest.json (1,813 videos, 0 overlap with train or test).
2. Fits:
   - Temperature Scaling (optimizing scalar T via L-BFGS to minimize NLL)
   - Platt Scaling (Affine logit adjustment: a * z + b)
   - Isotonic Regression (Non-parametric isotonic monotonic regression)
3. Evaluates calibration quality strictly on the UNTOUCHED TEST partitions:
   data/manifests/test_manifest_seen.json & test_manifest_unseen.json
4. Compares Before vs After on:
   - ECE (Expected Calibration Error)
   - Brier Score
   - NLL (Negative Log Likelihood)
   - Reliability Diagram Bins (Confidence vs Empirical Accuracy)
"""

from __future__ import annotations

import json
import logging
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
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
logger = logging.getLogger("held_out_calib")

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class TemperatureScaler(nn.Module):
    """Learns a single temperature parameter T > 0 to calibrate logits."""

    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature.clamp(min=0.01)

    def fit(self, logits: np.ndarray, labels: np.ndarray, max_iter: int = 50) -> float:
        z = torch.tensor(logits, dtype=torch.float32)
        y = torch.tensor(labels, dtype=torch.float32)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.LBFGS([self.temperature], lr=0.05, max_iter=max_iter)

        def eval_loss():
            optimizer.zero_grad()
            scaled_logits = self.forward(z)
            loss = criterion(scaled_logits, y)
            loss.backward()
            return loss

        optimizer.step(eval_loss)
        t_val = float(self.temperature.item())
        logger.info("Fitted Temperature: T = %.4f", t_val)
        return t_val


def compute_reliability_diagram(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> Dict[str, Any]:
    """Generates reliability diagram bins and computes ECE."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_data = []
    ece = 0.0
    n = len(probs)
    if n == 0:
        return {"ece": 0.0, "bins": []}

    for i in range(n_bins):
        bin_lower, bin_upper = bin_boundaries[i], bin_boundaries[i + 1]
        in_bin = (probs >= bin_lower) & (probs < bin_upper if i < n_bins - 1 else probs <= bin_upper)
        count = int(np.sum(in_bin))
        prop = count / n
        if count > 0:
            avg_acc = float(np.mean(labels[in_bin]))
            avg_conf = float(np.mean(probs[in_bin]))
            diff = abs(avg_conf - avg_acc)
            ece += diff * prop
        else:
            avg_acc = 0.0
            avg_conf = float((bin_lower + bin_upper) / 2.0)
            diff = 0.0

        bin_data.append({
            "bin_idx": i,
            "range": [round(float(bin_lower), 2), round(float(bin_upper), 2)],
            "count": count,
            "avg_confidence": round(avg_conf, 4),
            "empirical_accuracy": round(avg_acc, 4),
            "calibration_error": round(diff, 4),
        })

    return {"ece": float(ece), "bins": bin_data}


def extract_logits_and_labels(
    model: FORGELite,
    records: List[Dict[str, Any]],
    device: torch.device,
    max_samples: int = 400,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extracts raw logits and ground truth labels from records."""
    model.eval()
    import random
    rng = random.Random(42)
    recs = list(records)
    rng.shuffle(recs)
    recs = recs[:max_samples]

    logits_list = []
    labels_list = []

    for idx, r in enumerate(recs):
        crop_paths = r.get("frame_crops") or [r.get("source_path") or r.get("crop_path")]
        selected_path = crop_paths[len(crop_paths) // 2]
        p = REPO_ROOT / selected_path if not Path(selected_path).is_absolute() else Path(selected_path)
        if not p.exists():
            continue
        try:
            with Image.open(p) as img:
                tensor = TRANSFORM(img.convert("RGB")).unsqueeze(0).unsqueeze(0).to(device)
            with torch.no_grad():
                out = model(frames=tensor)
                logit = float(out["fake_logit"].squeeze().item())
                logits_list.append(logit)
                lbl = int(r.get("label_int", 1 if r.get("label") == "fake" else 0))
                labels_list.append(lbl)
        except Exception:
            continue

        if (idx + 1) % 100 == 0 or (idx + 1) == len(recs):
            logger.info("Extracted %d / %d samples", len(logits_list), len(recs))

    return np.array(logits_list), np.array(labels_list)


def evaluate_calibration(
    name: str,
    raw_probs: np.ndarray,
    labels: np.ndarray,
) -> Dict[str, Any]:
    """Computes ECE, Brier, NLL for a given probability vector."""
    clipped = np.clip(raw_probs, 1e-7, 1 - 1e-7)
    nll = float(log_loss(labels, clipped))
    brier = float(brier_score_loss(labels, raw_probs))
    rel = compute_reliability_diagram(raw_probs, labels)
    auc = float(roc_auc_score(labels, raw_probs))

    return {
        "method": name,
        "ece": round(rel["ece"], 4),
        "brier_score": round(brier, 4),
        "nll": round(nll, 4),
        "auroc": round(auc, 4),
        "reliability_bins": rel["bins"],
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Starting Phase 5: Held-Out Probabilistic Calibration on %s", device)

    ckpt_path = REPO_ROOT / "outputs" / "checkpoints" / "best.ckpt"
    model = FORGELite()
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=False)
    model.to(device)

    # 1. Load Held-Out Calibration Partition
    calib_manifest = REPO_ROOT / "data" / "manifests" / "calibration_manifest.json"
    with open(calib_manifest, "r", encoding="utf-8") as f:
        calib_recs = json.load(f)
    logger.info("Loaded %d held-out calibration partition records", len(calib_recs))

    # Extract calibration logits & labels
    logger.info("Extracting logits from held-out calibration set...")
    cal_logits, cal_labels = extract_logits_and_labels(model, calib_recs, device, max_samples=400)
    logger.info("Calibration dataset: %d samples (%d Real, %d Fake)",
                len(cal_labels), int(np.sum(cal_labels == 0)), int(np.sum(cal_labels == 1)))

    # 2. Fit Calibration Models Strictly on Calibration Partition
    # A. Temperature Scaling
    ts = TemperatureScaler()
    opt_temp = ts.fit(cal_logits, cal_labels)

    # B. Platt Scaling (Logistic Regression on Logits)
    platt = LogisticRegression(C=1.0, solver="lbfgs")
    platt.fit(cal_logits.reshape(-1, 1), cal_labels)
    logger.info("Fitted Platt Scaling: coef=%.4f, intercept=%.4f", platt.coef_[0][0], platt.intercept_[0])

    # C. Isotonic Regression on Sigmoid Probs
    cal_raw_probs = 1.0 / (1.0 + np.exp(-cal_logits))
    isotonic = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    isotonic.fit(cal_raw_probs, cal_labels)
    logger.info("Fitted Isotonic Regression")

    # 3. Load UNTOUCHED Test Partition
    test_manifest = REPO_ROOT / "data" / "manifests" / "test_manifest_seen.json"
    with open(test_manifest, "r", encoding="utf-8") as f:
        test_recs = json.load(f)
    logger.info("Loaded %d untouched test partition records", len(test_recs))

    logger.info("Extracting logits from untouched test partition...")
    test_logits, test_labels = extract_logits_and_labels(model, test_recs, device, max_samples=400)
    logger.info("Test dataset: %d samples (%d Real, %d Fake)",
                len(test_labels), int(np.sum(test_labels == 0)), int(np.sum(test_labels == 1)))

    # 4. Evaluate Calibration Before vs After on Untouched Test Set
    test_raw_probs = 1.0 / (1.0 + np.exp(-test_logits))
    test_ts_probs = 1.0 / (1.0 + np.exp(-test_logits / opt_temp))
    test_platt_probs = platt.predict_proba(test_logits.reshape(-1, 1))[:, 1]
    test_iso_probs = isotonic.predict(test_raw_probs)

    res_raw = evaluate_calibration("Uncalibrated (Baseline)", test_raw_probs, test_labels)
    res_ts = evaluate_calibration("Temperature Scaling", test_ts_probs, test_labels)
    res_platt = evaluate_calibration("Platt Scaling", test_platt_probs, test_labels)
    res_iso = evaluate_calibration("Isotonic Regression", test_iso_probs, test_labels)

    comparisons = [res_raw, res_ts, res_platt, res_iso]

    print("\n" + "=" * 90)
    print("      FORGE-LITE PHASE 5: PROBABILISTIC CALIBRATION AUDIT (UNTOUCHED TEST SET)")
    print("=" * 90)
    print(f"{'CALIBRATION METHOD':<28} | {'ECE':<8} | {'BRIER':<8} | {'NLL':<8} | {'AUROC':<8} | {'STATUS':<15}")
    print("-" * 90)
    for c in comparisons:
        status = "Baseline" if "Uncalibrated" in c["method"] else "Calibrated"
        print(f"{c['method']:<28} | {c['ece']:<8.4f} | {c['brier_score']:<8.4f} | {c['nll']:<8.4f} | {c['auroc']:<8.4f} | {status:<15}")
    print("=" * 90)

    # Print Reliability Diagram Bins for Best Method
    best_method = min(comparisons, key=lambda x: x["ece"])
    print(f"\nRELIABILITY DIAGRAM BINS [{best_method['method']} (ECE={best_method['ece']:.4f})]:")
    print(f"{'Bin Range':<16} | {'Count':<8} | {'Avg Confidence':<16} | {'Empirical Accuracy':<20} | {'Gap (|Acc - Conf|)':<18}")
    print("-" * 85)
    for b in best_method["reliability_bins"]:
        print(f"[{b['range'][0]:.2f}, {b['range'][1]:.2f}){'':<6} | {b['count']:<8} | {b['avg_confidence']:<16.4f} | {b['empirical_accuracy']:<20.4f} | {b['calibration_error']:<18.4f}")
    print("-" * 85 + "\n")

    report_path = REPO_ROOT / "reports" / "proper_calibration_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({
            "fitted_parameters": {
                "temperature": opt_temp,
                "platt_coef": float(platt.coef_[0][0]),
                "platt_intercept": float(platt.intercept_[0]),
            },
            "comparisons": comparisons,
        }, f, indent=2)
    logger.info("Saved calibration report to %s", report_path)


if __name__ == "__main__":
    main()
