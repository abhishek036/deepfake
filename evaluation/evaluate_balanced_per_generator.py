"""
evaluation/evaluate_balanced_per_generator.py
Experiment 1: Balanced Per-Generator Forensic Evaluation for FORGE-Lite.

Eliminates the measurement problem of single-class fake-detection rates.
For every manipulation family, pairs fake samples with matched real controls
from the exact same source dataset:
- FF++ Deepfakes: Matched with FF++ Original Real
- FF++ Face2Face: Matched with FF++ Original Real
- FF++ FaceSwap: Matched with FF++ Original Real
- FF++ NeuralTextures: Matched with FF++ Original Real
- FF++ FaceShifter: Matched with FF++ Original Real
- Celeb-DF v2: Matched with Celeb-DF Original Real
- Generative / StyleGAN3: Matched with Authentic Real Faces
- TalkingHeadBench: Matched with Authentic Talking Head Speech Videos

Reports for every balanced generator family (50% Real / 50% Fake):
- AUROC
- AUPRC
- TPR (Recall @ 0.5)
- FPR (False Positive Rate @ 0.5)
- Balanced Accuracy
- F1-Score
- Matthews Correlation Coefficient (MCC)
- Brier Score
- Expected Calibration Error (ECE)
"""

from __future__ import annotations

import json
import logging
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
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
logger = logging.getLogger("balanced_per_gen")

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


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


def load_crops_as_tensor(crop_paths: List[str], max_frames: int = 8) -> Optional[torch.Tensor]:
    tensors = []
    if len(crop_paths) > max_frames:
        step = len(crop_paths) / max_frames
        selected_paths = [crop_paths[int(i * step)] for i in range(max_frames)]
    else:
        selected_paths = crop_paths

    for p_str in selected_paths:
        p = REPO_ROOT / p_str if not Path(p_str).is_absolute() else Path(p_str)
        if not p.exists():
            continue
        try:
            with Image.open(p) as img:
                t = TRANSFORM(img.convert("RGB"))
                tensors.append(t)
        except Exception:
            continue

    if not tensors:
        return None
    return torch.stack(tensors, dim=0).unsqueeze(0)


def run_inference_on_records(
    model: FORGELite,
    records: List[Dict[str, Any]],
    device: torch.device,
) -> List[Dict[str, Any]]:
    results = []
    for idx, r in enumerate(records):
        crops = r.get("frame_crops") or [r.get("source_path") or r.get("crop_path")]
        batch = load_crops_as_tensor(crops)
        if batch is None:
            continue
        batch = batch.to(device)
        with torch.no_grad():
            out = model(batch)
            fake_prob = float(out.get("fake_score", torch.sigmoid(out["fake_logit"])).squeeze().item())
            b_disagree = float(out["branch_disagreement"].squeeze().item()) if "branch_disagreement" in out else 0.0

        lbl = int(r.get("label_int", 1 if r.get("label") == "fake" else 0))
        results.append({
            "video_id": r.get("video_id", f"sample_{idx}"),
            "dataset": r.get("dataset", "unknown"),
            "manipulation_family": r.get("manipulation_family", "unknown"),
            "ground_truth": lbl,
            "fake_prob": fake_prob,
            "branch_disagreement": b_disagree,
        })
        if (idx + 1) % 100 == 0 or (idx + 1) == len(records):
            logger.info("Processed %d / %d samples", idx + 1, len(records))
    return results


def compute_metrics_suite(name: str, items: List[Dict[str, Any]]) -> Dict[str, Any]:
    y_true = np.array([it["ground_truth"] for it in items])
    y_prob = np.array([it["fake_prob"] for it in items])
    y_pred = (y_prob >= 0.5).astype(int)

    n_total = len(items)
    n_real = int(np.sum(y_true == 0))
    n_fake = int(np.sum(y_true == 1))

    if n_real == 0 or n_fake == 0:
        return {
            "family": name,
            "n_total": n_total,
            "n_real": n_real,
            "n_fake": n_fake,
            "auroc": float("nan"),
            "auprc": float("nan"),
            "tpr": float("nan"),
            "fpr": float("nan"),
            "balanced_accuracy": float("nan"),
            "f1": float("nan"),
            "mcc": float("nan"),
            "brier": float("nan"),
            "ece": float("nan"),
        }

    auc = float(roc_auc_score(y_true, y_prob))
    auprc = float(average_precision_score(y_true, y_prob))

    # Confusion matrix metrics
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))

    tpr = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    mcc = float(matthews_corrcoef(y_true, y_pred))
    brier = float(brier_score_loss(y_true, y_prob))
    ece = float(compute_ece(y_prob, y_true))

    return {
        "family": name,
        "n_total": n_total,
        "n_real": n_real,
        "n_fake": n_fake,
        "auroc": round(auc, 4),
        "auprc": round(auprc, 4),
        "tpr": round(tpr, 4),
        "fpr": round(fpr, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "f1": round(f1, 4),
        "mcc": round(mcc, 4),
        "brier": round(brier, 4),
        "ece": round(ece, 4),
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Running Balanced Per-Generator Evaluation on %s", device)

    ckpt_path = REPO_ROOT / "outputs" / "checkpoints" / "best.ckpt"
    model = FORGELite()
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()

    # Load Source-Disjoint Test Manifests
    with open(REPO_ROOT / "data" / "manifests" / "test_manifest_seen.json", "r", encoding="utf-8") as f:
        seen = json.load(f)
    with open(REPO_ROOT / "data" / "manifests" / "test_manifest_unseen.json", "r", encoding="utf-8") as f:
        unseen = json.load(f)

    all_test = seen + unseen
    logger.info("Loaded %d total test samples from source-disjoint partitions", len(all_test))

    # Organize by dataset & manipulation
    ff_reals = [r for r in all_test if r.get("dataset") == "faceforensics" and r.get("label_int") == 0]
    celeb_reals = [r for r in all_test if r.get("dataset") == "celeb_df" and r.get("label_int") == 0]
    gen_reals = [r for r in all_test if r.get("dataset") == "final_dataset_test" and r.get("label_int") == 0]
    thb_reals = [r for r in all_test if r.get("dataset") == "talkingheadbench" and r.get("label_int") == 0]

    rng = random.Random(42)

    def sample_matched(reals: List[Dict], fakes: List[Dict], n: int = 50) -> List[Dict]:
        k = min(len(reals), len(fakes), n)
        r_samp = rng.sample(reals, k)
        f_samp = rng.sample(fakes, k)
        combined = r_samp + f_samp
        rng.shuffle(combined)
        return combined

    # Build Matched Evaluation Pairs
    families = {
        "FF++ Deepfakes": sample_matched(
            ff_reals,
            [r for r in all_test if r.get("manipulation_family") == "Deepfakes"],
            n=50
        ),
        "FF++ Face2Face": sample_matched(
            ff_reals,
            [r for r in all_test if r.get("manipulation_family") == "Face2Face"],
            n=50
        ),
        "FF++ FaceSwap": sample_matched(
            ff_reals,
            [r for r in all_test if r.get("manipulation_family") == "FaceSwap"],
            n=50
        ),
        "FF++ NeuralTextures": sample_matched(
            ff_reals,
            [r for r in all_test if r.get("manipulation_family") == "NeuralTextures"],
            n=50
        ),
        "FF++ FaceShifter": sample_matched(
            ff_reals,
            [r for r in all_test if r.get("manipulation_family") == "FaceShifter"],
            n=50
        ),
        "Celeb-DF v2": sample_matched(
            celeb_reals,
            [r for r in all_test if r.get("manipulation_family") == "Celeb-DF-synthesis"],
            n=50
        ),
        "Generative / StyleGAN3": sample_matched(
            gen_reals,
            [r for r in all_test if r.get("manipulation_family") == "StyleGAN3"],
            n=50
        ),
        "TalkingHeadBench": sample_matched(
            thb_reals,
            [r for r in all_test if r.get("manipulation_family") == "TalkingHeadSynthesis" and r.get("label_int") == 1],
            n=50
        ),
    }

    # Execute Inference and Compute Metrics for each family
    family_metrics = {}
    pooled_records = []

    print("\n" + "=" * 120)
    print("                     EXPERIMENT 1: BALANCED PER-GENERATOR FORENSIC EVALUATION")
    print("                     (Strictly Pairwise Balanced: 50% Real Controls / 50% Fake)")
    print("=" * 120)
    print(f"{'MANIPULATION FAMILY':<24} | {'N (R/F)':<10} | {'AUROC':<7} | {'AUPRC':<7} | {'TPR':<7} | {'FPR':<7} | {'BAL ACC':<8} | {'F1':<7} | {'MCC':<7} | {'BRIER':<7} | {'ECE':<7}")
    print("-" * 120)

    for fam_name, records in families.items():
        results = run_inference_on_records(model, records, device)
        m = compute_metrics_suite(fam_name, results)
        family_metrics[fam_name] = m
        pooled_records.extend(results)

        s_cnt = f"{m['n_total']} ({m['n_real']}R/{m['n_fake']}F)"
        print(f"{fam_name:<24} | {s_cnt:<10} | {m['auroc']:<7.4f} | {m['auprc']:<7.4f} | {m['tpr']:<7.4f} | {m['fpr']:<7.4f} | {m['balanced_accuracy']*100:<7.2f}% | {m['f1']:<7.4f} | {m['mcc']:<7.4f} | {m['brier']:<7.4f} | {m['ece']:<7.4f}")

    # Overall Balanced Pooled Test (800 balanced samples: 400 Real / 400 Fake)
    overall_m = compute_metrics_suite("Overall Balanced Pooled", pooled_records)
    family_metrics["Overall Balanced Pooled"] = overall_m

    print("-" * 120)
    s_cnt = f"{overall_m['n_total']} ({overall_m['n_real']}R/{overall_m['n_fake']}F)"
    print(f"{'Overall Balanced Pooled':<24} | {s_cnt:<10} | {overall_m['auroc']:<7.4f} | {overall_m['auprc']:<7.4f} | {overall_m['tpr']:<7.4f} | {overall_m['fpr']:<7.4f} | {overall_m['balanced_accuracy']*100:<7.2f}% | {overall_m['f1']:<7.4f} | {overall_m['mcc']:<7.4f} | {overall_m['brier']:<7.4f} | {overall_m['ece']:<7.4f}")
    print("=" * 120 + "\n")

    out_file = REPO_ROOT / "reports" / "balanced_per_generator_report.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(family_metrics, f, indent=2)
    logger.info("Saved balanced per-generator report to %s", out_file)


if __name__ == "__main__":
    main()
