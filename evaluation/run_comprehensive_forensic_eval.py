"""
evaluation/run_comprehensive_forensic_eval.py
Unified Forensic Evaluation & Probabilistic Calibration Audit for FORGE-Lite.

Implements rigorous engineering validation:
1. Probabilistic Calibration: ECE (10 bins), MCE, Brier score, NLL, and Reliability Diagrams.
2. Checkpoint Comparison: Pre-fine-tuning vs Post-fine-tuning.
3. 5-Branch Forensic Audit: Semantic, Forensic, Temporal, rPPG, Audio-Visual individual calibration.
4. Disagreement vs Prediction Error Stratification: Low vs Med vs High disagreement tiers.
5. Domain Slices: Seen manipulations (FF++, Celeb-DF) vs Unseen (StyleGAN3, Diffusion, TalkingHeadBench).
6. Robustness Slices: Clean vs JPEG (Q=50) vs Blur (sigma=1.0) vs Downsampling (0.5x).
7. Selective Abstention Curves: Accuracy @ 0%, 5%, 10% abstention.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from torch.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import ForgeDataset, collate_forge
from models.forge_lite import FORGELite

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("comprehensive_eval")


# ─────────────────────────────────────────────────────────────────────────────
# Calibration Metrics Computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_calibration_metrics(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, Any]:
    """
    Computes ECE, MCE, Brier score, NLL, and per-bin reliability data.
    """
    probs = np.clip(probs, 1e-7, 1.0 - 1e-7)
    labels = labels.astype(int)
    total_samples = len(labels)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(probs, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    ece = 0.0
    mce = 0.0
    reliability_bins = []

    for i in range(n_bins):
        mask = bin_indices == i
        count = int(mask.sum())
        bin_lower = float(bins[i])
        bin_upper = float(bins[i + 1])
        if count > 0:
            bin_conf = float(probs[mask].mean())
            bin_acc = float(labels[mask].mean())
            gap = abs(bin_conf - bin_acc)
            ece += (count / total_samples) * gap
            mce = max(mce, gap)
        else:
            bin_conf = (bin_lower + bin_upper) / 2.0
            bin_acc = 0.0
            gap = 0.0

        reliability_bins.append({
            "bin": i + 1,
            "range": f"[{bin_lower:.1f}, {bin_upper:.1f})",
            "count": count,
            "confidence": bin_conf,
            "accuracy": bin_acc,
            "gap": gap,
        })

    brier_score = float(np.mean((probs - labels) ** 2))
    nll = float(-np.mean(labels * np.log(probs) + (1 - labels) * np.log(1.0 - probs)))

    return {
        "ece": float(ece),
        "mce": float(mce),
        "brier_score": brier_score,
        "nll": nll,
        "bins": reliability_bins,
    }


def compute_eer(labels: np.ndarray, scores: np.ndarray) -> float:
    """Compute Equal Error Rate where FPR == 1 - TPR."""
    if len(set(labels)) < 2:
        return 0.5
    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1.0 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))
    return float((fpr[idx] + fnr[idx]) / 2.0)


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation Data Curation
# ─────────────────────────────────────────────────────────────────────────────

def gather_evaluation_records(
    combined_path: Path = REPO_ROOT / "data" / "manifests" / "combined.jsonl",
    final_test_path: Path = REPO_ROOT / "data" / "manifests" / "final_dataset_test.jsonl",
    talkinghead_path: Path = REPO_ROOT / "data" / "manifests" / "talkingheadbench.jsonl",
    max_samples: int = 600,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Returns (seen_records, unseen_records).
    Seen: FF++ and Celeb-DF test split.
    Unseen: Final Dataset test (StyleGAN3, Diffusion) + TalkingHeadBench.
    """
    import random
    rng = random.Random(seed)

    seen_records = []
    unseen_records = []

    # 1. Seen manipulations (test split of combined.jsonl)
    if combined_path.exists():
        from collections import defaultdict
        src_groups = defaultdict(list)
        with open(combined_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                if r.get("split") == "test":
                    src_groups[r["source_path"]].append(r)

        for src, f_list in src_groups.items():
            valid = [f for f in f_list if (REPO_ROOT / f.get("crop_path", "")).exists()]
            if len(valid) >= 2:
                valid.sort(key=lambda x: x.get("frame_idx", 0))
                first = valid[0]
                lbl = int(first.get("label", 0))
                seen_records.append({
                    "video_id": src,
                    "dataset": first.get("dataset", "combined"),
                    "domain": "seen",
                    "manipulation_family": first.get("manipulation", "real" if lbl == 0 else "fake"),
                    "label_int": lbl,
                    "frame_crops": [f.get("crop_path") for f in valid],
                    "track_length": len(valid),
                    "split": "test",
                    "audio_path": None,
                    "mel_path": None,
                    "audio_available": False,
                    "track_continuity": 1.0,
                    "sufficient_for_analysis": True,
                })

    # 2. Unseen generative manipulations (final_dataset_test.jsonl)
    if final_test_path.exists():
        with open(final_test_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                cp = item.get("crop_path") or item.get("source_path")
                if cp and Path(cp).exists():
                    lbl = int(item.get("label", 0))
                    unseen_records.append({
                        "video_id": item.get("record_id", str(cp)),
                        "dataset": "final_dataset_test",
                        "domain": "unseen",
                        "manipulation_family": item.get("manipulation", "Generative"),
                        "difficulty": item.get("difficulty", "Medium"),
                        "label_int": lbl,
                        "frame_crops": [str(cp)],
                        "track_length": 1,
                        "split": "test",
                        "audio_path": None,
                        "mel_path": None,
                        "audio_available": False,
                        "track_continuity": 1.0,
                        "sufficient_for_analysis": True,
                    })

    # 3. Unseen Talking Heads
    if talkinghead_path.exists():
        with open(talkinghead_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                cp = item.get("crop_path") or item.get("source_path")
                if cp and Path(cp).exists():
                    lbl = int(item.get("label", 1))
                    unseen_records.append({
                        "video_id": item.get("record_id", str(cp)),
                        "dataset": "talkingheadbench",
                        "domain": "unseen",
                        "manipulation_family": "TalkingHeadSynthesis",
                        "label_int": lbl,
                        "frame_crops": [str(cp)],
                        "track_length": 1,
                        "split": "test",
                        "audio_path": item.get("audio_path"),
                        "mel_path": item.get("mel_path"),
                        "audio_available": bool(item.get("audio_path") or item.get("mel_path")),
                        "track_continuity": 1.0,
                        "sufficient_for_analysis": True,
                    })

    rng.shuffle(seen_records)
    rng.shuffle(unseen_records)

    half = max_samples // 2
    seen_selected = seen_records[:half]
    unseen_selected = unseen_records[:half]

    logger.info("Gathered %d Seen records and %d Unseen records", len(seen_selected), len(unseen_selected))
    return seen_selected, unseen_selected


# ─────────────────────────────────────────────────────────────────────────────
# Model Evaluation Pipeline
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_checkpoint(
    model: FORGELite,
    records: List[Dict],
    batch_size: int = 8,
    window_size: int = 4,
    device: str = "cuda",
    corruption: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Evaluates a FORGELite model on records, optionally applying corruptions.
    Returns dictionary with all metrics, branch predictions, and disagreement stats.
    """
    model.eval()

    # Temporary dataset wrapper
    class RecordDataset(torch.utils.data.Dataset):
        def __init__(self, recs):
            self.records = recs
            self.window_size = window_size
            self.split = "test"
            self.augment = False
            self.temporal_aug = None

        def __len__(self):
            return len(self.records)

        def __getitem__(self, idx):
            rec = self.records[idx]
            paths = rec["frame_crops"][:self.window_size]
            frames_bgr = []
            for p in paths:
                im = cv2.imread(str(p))
                if im is not None:
                    if im.shape[:2] != (224, 224):
                        im = cv2.resize(im, (224, 224))
                    frames_bgr.append(im)
            if not frames_bgr:
                frames_bgr = [np.zeros((224, 224, 3), dtype=np.uint8)]

            # Pad to window_size
            while len(frames_bgr) < self.window_size:
                frames_bgr.append(frames_bgr[-1].copy())

            # Apply corruption if requested
            if corruption == "jpeg":
                frames_bgr = [
                    cv2.imdecode(cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 50])[1], cv2.IMREAD_COLOR)
                    for f in frames_bgr
                ]
            elif corruption == "blur":
                frames_bgr = [cv2.GaussianBlur(f, (5, 5), sigmaX=1.0) for f in frames_bgr]
            elif corruption == "downsample":
                frames_bgr = [cv2.resize(cv2.resize(f, (112, 112)), (224, 224)) for f in frames_bgr]

            from data.augmentation import frames_to_tensor
            frames_tensor = frames_to_tensor(frames_bgr)
            diffs = frames_tensor[1:] - frames_tensor[:-1]
            if len(diffs) == 0:
                diffs = torch.zeros(1, 3, 224, 224)

            return {
                "frames": frames_tensor,
                "frame_diffs": diffs,
                "label": torch.tensor(rec["label_int"], dtype=torch.float32),
                "face_quality": torch.zeros(10),
                "track_quality": torch.zeros(5),
                "track_continuity": torch.tensor(1.0),
                "video_id": rec["video_id"],
                "domain": rec.get("domain", "unknown"),
                "manipulation": rec.get("manipulation_family", "unknown"),
            }

    ds = RecordDataset(records)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_forge, num_workers=0)

    all_scores = []
    all_uncertainties = []
    all_labels = []
    all_domains = []
    all_manipulations = []
    all_branch_probs = []
    all_disagreements = []
    all_availability = []

    for batch in tqdm(loader, desc=f"Evaluating (corr={corruption or 'clean'})", leave=False):
        b_frames = batch["frames"].to(device)
        b_diffs = batch["frame_diffs"].to(device)
        b_fq = batch["face_quality"].to(device)
        b_tq = batch["track_quality"].to(device)
        b_cont = batch["track_continuity"].to(device)

        with autocast(device_type="cuda" if device == "cuda" else "cpu", enabled=(device == "cuda")):
            out = model(
                frames=b_frames,
                frame_diffs=b_diffs,
                face_quality=b_fq,
                track_quality=b_tq,
                track_continuity=b_cont,
            )

        fake_probs = torch.sigmoid(out["fake_logit"]).cpu().numpy()
        uncertainties = out["uncertainty"].cpu().numpy()
        lbls = batch["label"].cpu().numpy()

        all_scores.extend(fake_probs)
        all_uncertainties.extend(uncertainties)
        all_labels.extend(lbls)
        all_domains.extend(batch["domain"])
        all_manipulations.extend(batch["manipulation"])

        if out.get("branch_fake_probs") is not None:
            all_branch_probs.append(out["branch_fake_probs"].cpu().numpy())
        if out.get("branch_disagreement") is not None:
            all_disagreements.extend(out["branch_disagreement"].cpu().numpy())
        if out.get("availability_mask") is not None:
            all_availability.append(out["availability_mask"].cpu().numpy())

    scores_arr = np.array(all_scores)
    labels_arr = np.array(all_labels)
    uncertainties_arr = np.array(all_uncertainties)
    disagreements_arr = np.array(all_disagreements) if all_disagreements else np.zeros_like(scores_arr)
    branch_probs_arr = np.concatenate(all_branch_probs, axis=0) if all_branch_probs else None
    avail_arr = np.concatenate(all_availability, axis=0) if all_availability else None

    # Overall Detection Metrics
    auc = float(roc_auc_score(labels_arr, scores_arr)) if len(set(labels_arr)) > 1 else 0.5
    ap = float(average_precision_score(labels_arr, scores_arr)) if len(set(labels_arr)) > 1 else 0.5
    preds_binary = (scores_arr >= 0.5).astype(int)
    acc = float(accuracy_score(labels_arr, preds_binary))
    f1 = float(f1_score(labels_arr, preds_binary, zero_division=0))
    eer = float(compute_eer(labels_arr, scores_arr))

    # Overall Calibration Metrics
    cal_metrics = compute_calibration_metrics(scores_arr, labels_arr, n_bins=10)

    # Branch-Specific Calibration Audit (all 5 branches)
    branch_names = ["Semantic (DINOv2)", "Forensic (SRM-CNN)", "Temporal (GRU)", "rPPG (Physio)", "Audio-Visual (Sync)"]
    branch_audit = []
    if branch_probs_arr is not None:
        real_mask = (labels_arr == 0)
        fake_mask = (labels_arr == 1)

        for i in range(5):
            b_p = branch_probs_arr[:, i]
            # check availability mask
            is_avail = avail_arr[:, i].astype(bool) if avail_arr is not None else np.ones(len(b_p), dtype=bool)
            avail_ratio = float(is_avail.mean())

            if is_avail.sum() > 0:
                p_avail = b_p[is_avail]
                y_avail = labels_arr[is_avail]
                r_mask_sub = (y_avail == 0)
                f_mask_sub = (y_avail == 1)

                mean_real = float(p_avail[r_mask_sub].mean()) if r_mask_sub.any() else 0.5
                mean_fake = float(p_avail[f_mask_sub].mean()) if f_mask_sub.any() else 0.5
                b_brier = float(np.mean((p_avail - y_avail) ** 2))
                b_cal = compute_calibration_metrics(p_avail, y_avail, n_bins=10)
                b_auc = float(roc_auc_score(y_avail, p_avail)) if len(set(y_avail)) > 1 else 0.5
            else:
                mean_real, mean_fake, b_brier, b_auc = 0.5, 0.5, 0.25, 0.5
                b_cal = {"ece": 0.0, "nll": 0.0}

            branch_audit.append({
                "branch_idx": i,
                "name": branch_names[i],
                "availability_rate": avail_ratio,
                "mean_prob_real": mean_real,
                "mean_prob_fake": mean_fake,
                "separation": mean_fake - mean_real,
                "brier_score": b_brier,
                "ece": b_cal["ece"],
                "auc": b_auc,
            })

    # Disagreement vs Error Stratification
    disagree_stratification = []
    disagree_thresholds = [
        ("Low Disagreement (< 0.05)", disagreements_arr < 0.05),
        ("Medium Disagreement (0.05 - 0.10)", (disagreements_arr >= 0.05) & (disagreements_arr < 0.10)),
        ("High Disagreement (>= 0.10)", disagreements_arr >= 0.10),
    ]
    for name, mask in disagree_thresholds:
        count = int(mask.sum())
        if count > 0:
            tier_acc = float(accuracy_score(labels_arr[mask], preds_binary[mask]))
            tier_err = 1.0 - tier_acc
            tier_brier = float(np.mean((scores_arr[mask] - labels_arr[mask]) ** 2))
        else:
            tier_acc, tier_err, tier_brier = 0.0, 0.0, 0.0

        disagree_stratification.append({
            "tier": name,
            "count": count,
            "percentage": count / len(scores_arr),
            "accuracy": tier_acc,
            "error_rate": tier_err,
            "brier_score": tier_brier,
        })

    # Domain Breakdown (Seen vs Unseen)
    domain_arr = np.array(all_domains)
    seen_mask = (domain_arr == "seen")
    unseen_mask = (domain_arr == "unseen")

    seen_auc = float(roc_auc_score(labels_arr[seen_mask], scores_arr[seen_mask])) if seen_mask.any() and len(set(labels_arr[seen_mask])) > 1 else 0.5
    unseen_auc = float(roc_auc_score(labels_arr[unseen_mask], scores_arr[unseen_mask])) if unseen_mask.any() and len(set(labels_arr[unseen_mask])) > 1 else 0.5
    seen_acc = float(accuracy_score(labels_arr[seen_mask], preds_binary[seen_mask])) if seen_mask.any() else 0.0
    unseen_acc = float(accuracy_score(labels_arr[unseen_mask], preds_binary[unseen_mask])) if unseen_mask.any() else 0.0

    # Selective Abstention Trade-offs (Acc @ 5% and 10% abstention)
    # Abstention sorting score: combination of uncertainty + branch disagreement
    abstain_score = uncertainties_arr + disagreements_arr
    sorted_order = np.argsort(abstain_score)  # lowest uncertainty/disagreement first

    def get_selective_acc(retain_fraction: float) -> float:
        cutoff = max(int(len(sorted_order) * retain_fraction), 1)
        retained_idx = sorted_order[:cutoff]
        return float(accuracy_score(labels_arr[retained_idx], preds_binary[retained_idx]))

    acc_at_0_abst = acc
    acc_at_5_abst = get_selective_acc(0.95)
    acc_at_10_abst = get_selective_acc(0.90)

    return {
        "samples": len(labels_arr),
        "real_count": int((labels_arr == 0).sum()),
        "fake_count": int((labels_arr == 1).sum()),
        "accuracy": acc,
        "roc_auc": auc,
        "pr_auc": ap,
        "f1": f1,
        "eer": eer,
        "ece": cal_metrics["ece"],
        "mce": cal_metrics["mce"],
        "brier_score": cal_metrics["brier_score"],
        "nll": cal_metrics["nll"],
        "reliability_bins": cal_metrics["bins"],
        "branch_audit": branch_audit,
        "disagreement_stratification": disagree_stratification,
        "seen_auc": seen_auc,
        "seen_acc": seen_acc,
        "unseen_auc": unseen_auc,
        "unseen_acc": unseen_acc,
        "acc_at_0_abst": acc_at_0_abst,
        "acc_at_5_abst": acc_at_5_abst,
        "acc_at_10_abst": acc_at_10_abst,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main Orchestrator & Report Printing
# ─────────────────────────────────────────────────────────────────────────────

def run_full_forensic_audit(
    pre_checkpoint: str = "outputs/checkpoints/best_pre_calibration.ckpt",
    post_checkpoint: str = "outputs/checkpoints/best.ckpt",
    num_samples: int = 500,
    batch_size: int = 8,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Dict[str, Any]:
    logger.info("=" * 80)
    logger.info("FORGE-Lite Rigorous Forensic Evaluation & Probabilistic Calibration Audit")
    logger.info("Device: %s | Target Samples: %d | Batch: %d", device, num_samples, batch_size)
    logger.info("=" * 80)

    # 1. Gather Seen and Unseen Evaluation Records
    seen_recs, unseen_recs = gather_evaluation_records(max_samples=num_samples)
    all_eval_recs = seen_recs + unseen_recs
    logger.info("Total evaluation set size: %d samples (%d Seen, %d Unseen)", len(all_eval_recs), len(seen_recs), len(unseen_recs))

    # 2. Evaluate Calibrated Model (Post-fine-tuning)
    logger.info("Loading Post-fine-tuning Calibrated Checkpoint: %s", post_checkpoint)
    post_model = FORGELite().to(device)
    post_ckpt = torch.load(post_checkpoint, map_location="cpu")
    post_model.load_state_dict(post_ckpt["model_state_dict"], strict=False)

    post_results = evaluate_checkpoint(post_model, all_eval_recs, batch_size=batch_size, device=device)

    # 3. Evaluate Pre-fine-tuning Checkpoint (for Before vs After comparison)
    pre_results = None
    if Path(pre_checkpoint).exists():
        logger.info("Loading Pre-fine-tuning Checkpoint for Baseline Comparison: %s", pre_checkpoint)
        pre_model = FORGELite().to(device)
        pre_ckpt = torch.load(pre_checkpoint, map_location="cpu")
        pre_model.load_state_dict(pre_ckpt.get("model_state_dict", pre_ckpt), strict=False)
        pre_results = evaluate_checkpoint(pre_model, all_eval_recs, batch_size=batch_size, device=device)

    # 4. Evaluate Corruptions (Robustness Slices) on Calibrated Model
    logger.info("Running Perturbation & Corruption Slices (JPEG Q=50, Blur sigma=1.0, Downsample 0.5x)...")
    res_jpeg = evaluate_checkpoint(post_model, all_eval_recs, batch_size=batch_size, device=device, corruption="jpeg")
    res_blur = evaluate_checkpoint(post_model, all_eval_recs, batch_size=batch_size, device=device, corruption="blur")
    res_down = evaluate_checkpoint(post_model, all_eval_recs, batch_size=batch_size, device=device, corruption="downsample")

    # 5. Format and Print Structured Forensic Audit Report
    print("\n" + "=" * 80)
    print("                      FORGE-LITE FORENSIC EVALUATION & CALIBRATION AUDIT")
    print("=" * 80)
    print(f"Total Samples:         {post_results['samples']} ({post_results['real_count']} Real, {post_results['fake_count']} Fake)")
    print(f"Foundation Backbone:   Meta DINOv2 ViT-S/14 (~22M params) + SRM-CNN (Dual-Domain)")
    print("-" * 80)
    print("CORE DETECTION PERFORMANCE:")
    print(f"  Accuracy:            {post_results['accuracy'] * 100:.2f}%")
    print(f"  ROC-AUC:             {post_results['roc_auc']:.4f}")
    print(f"  PR-AUC:              {post_results['pr_auc']:.4f}")
    print(f"  F1-Score:            {post_results['f1'] * 100:.2f}%")
    print(f"  EER:                 {post_results['eer'] * 100:.2f}%")
    print("-" * 80)
    print("PROBABILISTIC CALIBRATION:")
    print(f"  ECE (10-bin):        {post_results['ece']:.4f}" + (f" (Before Fine-tuning: {pre_results['ece']:.4f})" if pre_results else ""))
    print(f"  MCE:                 {post_results['mce']:.4f}" + (f" (Before Fine-tuning: {pre_results['mce']:.4f})" if pre_results else ""))
    print(f"  Brier Score:         {post_results['brier_score']:.4f}" + (f" (Before Fine-tuning: {pre_results['brier_score']:.4f})" if pre_results else ""))
    print(f"  Negative Log-Lik:    {post_results['nll']:.4f}" + (f" (Before Fine-tuning: {pre_results['nll']:.4f})" if pre_results else ""))
    print("-" * 80)
    print("SELECTIVE PREDICTION & ABSTENTION:")
    print(f"  Accuracy @ 0% Abstention:   {post_results['acc_at_0_abst'] * 100:.2f}%")
    print(f"  Accuracy @ 5% Abstention:   {post_results['acc_at_5_abst'] * 100:.2f}%")
    print(f"  Accuracy @ 10% Abstention:  {post_results['acc_at_10_abst'] * 100:.2f}%")
    print("-" * 80)
    print("DOMAIN & PERTURBATION SLICES:")
    print(f"  Seen Manipulation AUC:      {post_results['seen_auc']:.4f} (Acc: {post_results['seen_acc'] * 100:.2f}%)")
    print(f"  Unseen Manipulation AUC:    {post_results['unseen_auc']:.4f} (Acc: {post_results['unseen_acc'] * 100:.2f}%)")
    print(f"  JPEG (Q=50) Compressed AUC: {res_jpeg['roc_auc']:.4f} (Delta: {res_jpeg['roc_auc'] - post_results['roc_auc']:+.4f})")
    print(f"  Gaussian Blur (sigma=1) AUC:{res_blur['roc_auc']:.4f} (Delta: {res_blur['roc_auc'] - post_results['roc_auc']:+.4f})")
    print(f"  Downsampled (0.5x) AUC:     {res_down['roc_auc']:.4f} (Delta: {res_down['roc_auc'] - post_results['roc_auc']:+.4f})")
    print("-" * 80)
    print("ALL 5 FORENSIC BRANCH HEADS AUDIT:")
    print(f"  {'Branch Name':<22} | {'Avail%':<7} | {'P(Real)':<8} | {'P(Fake)':<8} | {'Sep':<6} | {'Brier':<7} | {'ECE':<6} | {'AUC':<6}")
    print("  " + "-" * 78)
    for b in post_results["branch_audit"]:
        print(f"  {b['name']:<22} | {b['availability_rate']*100:>5.1f}% | {b['mean_prob_real']:>8.4f} | {b['mean_prob_fake']:>8.4f} | {b['separation']:>+6.3f} | {b['brier_score']:>7.4f} | {b['ece']:>6.4f} | {b['auc']:>6.4f}")
    print("-" * 80)
    print("DISAGREEMENT VS PREDICTION ERROR CORRELATION:")
    print(f"  {'Disagreement Tier':<35} | {'Count':<7} | {'Accuracy':<10} | {'Error Rate':<10} | {'Brier':<7}")
    print("  " + "-" * 78)
    for d in post_results["disagreement_stratification"]:
        print(f"  {d['tier']:<35} | {d['count']:<7} | {d['accuracy']*100:>8.2f}% | {d['error_rate']*100:>8.2f}% | {d['brier_score']:>7.4f}")
    print("-" * 80)
    print("RELIABILITY DIAGRAM (10 Equal-Width Bins):")
    print(f"  {'Bin':<4} | {'Confidence Range':<18} | {'Count':<7} | {'Avg Confidence':<14} | {'Empirical Acc':<13} | {'Gap':<7}")
    print("  " + "-" * 78)
    for rb in post_results["reliability_bins"]:
        print(f"  {rb['bin']:<4} | {rb['range']:<18} | {rb['count']:<7} | {rb['confidence']:>14.4f} | {rb['accuracy']:>13.4f} | {rb['gap']:>7.4f}")
    print("=" * 80 + "\n")

    # Save full results JSON
    report_dict = {
        "post_calibration": post_results,
        "pre_calibration": pre_results,
        "robustness": {
            "jpeg_q50": res_jpeg["roc_auc"],
            "blur_sigma1": res_blur["roc_auc"],
            "downsample_0.5x": res_down["roc_auc"],
        },
        "audit_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    out_json = REPO_ROOT / "reports" / "comprehensive_forensic_audit.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2)
    logger.info("Saved full audit metrics to: %s", out_json)

    return report_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Comprehensive Forensic Evaluation")
    parser.add_argument("--num-samples", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    run_full_forensic_audit(
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        device=args.device,
    )
