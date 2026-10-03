"""
evaluation/evaluate_per_generator.py
Phase 2: Per-Dataset & Per-Generator Evaluation for FORGE-Lite.

Disaggregates model performance across distinct generative manipulation families:
- FaceForensics++: Deepfakes, Face2Face, FaceSwap, NeuralTextures
- Celeb-DF v2
- Diffusion / Generative Faces (final_dataset_test)
- TalkingHeadSynthesis (talkingheadbench)
- FaceShifter / StyleGAN3 if present in dataset records

Reports per-family:
- Sample count (Real vs Fake)
- AUROC
- Accuracy & Balanced Accuracy
- ECE & Brier Score
- Log Loss (NLL)
- Mean Fake Probability (Real vs Fake)
- Mean Branch Disagreement
"""

from __future__ import annotations

import json
import logging
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import accuracy_score, balanced_accuracy_score, brier_score_loss, log_loss, roc_auc_score
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
logger = logging.getLogger("per_generator_eval")

EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Computes Expected Calibration Error."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(probs)
    if n == 0:
        return 0.0
    for i in range(n_bins):
        bin_lower, bin_upper = bin_boundaries[i], bin_boundaries[i + 1]
        in_bin = (probs >= bin_lower) & (probs < bin_upper if i < n_bins - 1 else probs <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(labels[in_bin])
            avg_confidence_in_bin = np.mean(probs[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return float(ece)


def load_model(checkpoint_path: Path, device: torch.device) -> FORGELite:
    logger.info("Instantiating FORGELite backbone...")
    model = FORGELite()
    logger.info("Loading weights from %s", checkpoint_path)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    logger.info("Loaded state dict | Missing: %d, Unexpected: %d", len(missing), len(unexpected))
    model.to(device)
    model.eval()
    return model


def load_crops_as_tensor(crop_paths: List[str], max_frames: int = 8) -> Optional[torch.Tensor]:
    """Loads a sequence of crop images as a (1, T, 3, 224, 224) tensor."""
    tensors = []
    # Subsample evenly if more than max_frames
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
                t = EVAL_TRANSFORM(img.convert("RGB"))
                tensors.append(t)
        except Exception:
            continue

    if not tensors:
        return None
    # shape: (1, T, 3, H, W)
    return torch.stack(tensors, dim=0).unsqueeze(0)


def get_clean_family(r: Dict[str, Any]) -> str:
    fam = r.get("manipulation_family") or r.get("generator_family") or "Unknown"
    dset = r.get("dataset", "Unknown")
    fam_clean = fam.replace(".mp4", "").replace("_", " ").title()
    if "Deepfake" in fam_clean or "Dfd" in fam_clean:
        return "Deepfakes"
    elif "Face2Face" in fam_clean:
        return "Face2Face"
    elif "FaceSwap" in fam_clean or "Faceswap" in fam_clean:
        return "FaceSwap"
    elif "Neuraltextures" in fam_clean:
        return "NeuralTextures"
    elif "Celeb" in fam_clean or "Cdf" in fam_clean or "Celeb" in dset:
        return "Celeb-DF v2"
    elif "Talkinghead" in fam_clean.lower() or "talkinghead" in dset.lower():
        return "TalkingHeadSynthesis"
    elif "Generative" in fam_clean or "final_dataset" in dset:
        return "Generative / Diffusion"
    return fam_clean


def stratify_records(records: List[Dict[str, Any]], max_per_class_per_family: int = 100, seed: int = 42) -> List[Dict[str, Any]]:
    """Selects a balanced, representative stratified sample across families and labels."""
    import random
    rng = random.Random(seed)
    buckets = defaultdict(lambda: defaultdict(list))
    for r in records:
        fam = get_clean_family(r)
        lbl = int(r.get("label_int", 1 if r.get("label") == "fake" else 0))
        buckets[fam][lbl].append(r)

    stratified = []
    for fam, class_dict in sorted(buckets.items()):
        for lbl, item_list in class_dict.items():
            rng.shuffle(item_list)
            selected = item_list[:max_per_class_per_family]
            stratified.extend(selected)
            logger.info("Family '%s' (Label %d): Selected %d / %d samples", fam, lbl, len(selected), len(item_list))

    rng.shuffle(stratified)
    return stratified


def evaluate_records(
    model: FORGELite,
    records: List[Dict[str, Any]],
    device: torch.device,
    max_samples: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Runs inference across records and collects predictions."""
    results = []
    if max_samples and len(records) > max_samples:
        records = records[:max_samples]

    for idx, r in enumerate(records):
        crop_paths = r.get("frame_crops") or [r.get("source_path") or r.get("crop_path")]
        batch_tensor = load_crops_as_tensor(crop_paths)
        if batch_tensor is None:
            continue

        batch_tensor = batch_tensor.to(device)
        with torch.no_grad():
            outputs = model(batch_tensor)
            fake_prob = float(outputs.get("fake_score", torch.sigmoid(outputs["fake_logit"])).squeeze().item())
            logits = float(outputs["fake_logit"].squeeze().item())
            b_disagree = float(outputs["branch_disagreement"].squeeze().item()) if "branch_disagreement" in outputs else 0.0
            
            b_probs = []
            if "branch_fake_probs" in outputs:
                b_probs = outputs["branch_fake_probs"].squeeze().cpu().tolist()
                if not isinstance(b_probs, list):
                    b_probs = [b_probs]

        gt_label = int(r.get("label_int", 1 if r.get("label") == "fake" else 0))
        fam_clean = get_clean_family(r)
        dset = r.get("dataset", "Unknown")

        results.append({
            "video_id": r.get("video_id", f"sample_{idx}"),
            "dataset": dset,
            "family": fam_clean,
            "split": r.get("split", "test"),
            "ground_truth": gt_label,
            "fake_prob": fake_prob,
            "logit": logits,
            "branch_disagreement": b_disagree,
            "branch_fake_probs": b_probs,
        })

        if (idx + 1) % 100 == 0 or (idx + 1) == len(records):
            logger.info("Processed %d / %d samples", idx + 1, len(records))

    return results


def summarize_metrics(family_name: str, items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculates all forensic and calibration metrics for a subset."""
    labels = np.array([it["ground_truth"] for it in items])
    probs = np.array([it["fake_prob"] for it in items])
    disagreements = np.array([it["branch_disagreement"] for it in items])

    n_total = len(items)
    n_real = int(np.sum(labels == 0))
    n_fake = int(np.sum(labels == 1))

    preds = (probs >= 0.5).astype(int)
    acc = float(accuracy_score(labels, preds)) if n_total > 0 else 0.0

    # Handle AUC for single-class subsets
    if len(np.unique(labels)) >= 2:
        auc = float(roc_auc_score(labels, probs))
        bal_acc = float(balanced_accuracy_score(labels, preds))
        logloss = float(log_loss(labels, np.clip(probs, 1e-7, 1 - 1e-7)))
    else:
        auc = float("nan")
        bal_acc = acc
        logloss = float("nan")

    brier = float(brier_score_loss(labels, probs)) if n_total > 0 else float("nan")
    ece = float(compute_ece(probs, labels)) if n_total > 0 else float("nan")

    mean_prob_real = float(np.mean(probs[labels == 0])) if n_real > 0 else float("nan")
    mean_prob_fake = float(np.mean(probs[labels == 1])) if n_fake > 0 else float("nan")
    mean_disagree = float(np.mean(disagreements)) if n_total > 0 else 0.0

    return {
        "family": family_name,
        "n_total": n_total,
        "n_real": n_real,
        "n_fake": n_fake,
        "auroc": auc,
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "brier_score": brier,
        "ece": ece,
        "log_loss": logloss,
        "mean_fake_prob_real": mean_prob_real,
        "mean_fake_prob_fake": mean_prob_fake,
        "mean_disagreement": mean_disagree,
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    model = FORGELite(ablation_mode="conf_qual").to(device)
    ckpt_path = REPO_ROOT / "outputs" / "checkpoints" / "best_v3_codec_invariant.ckpt"
    logger.info("Loading V3 model from %s", ckpt_path.name)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=False)
    model.to(device)

    # Load Disjoint Test Partitions
    seen_path = REPO_ROOT / "data" / "manifests" / "test_manifest_seen.json"
    unseen_path = REPO_ROOT / "data" / "manifests" / "test_manifest_unseen.json"

    records = []
    if seen_path.exists():
        with open(seen_path, "r", encoding="utf-8") as f:
            seen_recs = json.load(f)
            for r in seen_recs:
                r["split_type"] = "seen"
            records.extend(seen_recs)
            logger.info("Loaded %d seen test records", len(seen_recs))

    if unseen_path.exists():
        with open(unseen_path, "r", encoding="utf-8") as f:
            unseen_recs = json.load(f)
            for r in unseen_recs:
                r["split_type"] = "unseen"
            records.extend(unseen_recs)
            logger.info("Loaded %d unseen test records", len(unseen_recs))

    if not records:
        logger.error("No test records found! Check split generation.")
        return

    # Stratify to ensure balanced representation across all families without redundant compute
    eval_records = stratify_records(records, max_per_class_per_family=100)
    logger.info("Total stratified evaluation test records: %d", len(eval_records))
    results = evaluate_records(model, eval_records, device)

    # Group by Generator Family
    by_family = defaultdict(list)
    for r in results:
        by_family[r["family"]].append(r)

    family_metrics = {}
    print("\n" + "=" * 100)
    print(f"{'GENERATOR / MANIPULATION FAMILY':<28} | {'SAMPLES':<10} | {'AUROC':<8} | {'ACC (%)':<8} | {'BRIER':<8} | {'ECE':<8} | {'DISAGREE':<8}")
    print("-" * 100)

    for fam, items in sorted(by_family.items(), key=lambda x: -len(x[1])):
        m = summarize_metrics(fam, items)
        family_metrics[fam] = m
        auc_str = f"{m['auroc']:.4f}" if not math.isnan(m["auroc"]) else "N/A*"
        acc_str = f"{m['accuracy'] * 100:.2f}%"
        brier_str = f"{m['brier_score']:.4f}"
        ece_str = f"{m['ece']:.4f}"
        dis_str = f"{m['mean_disagreement']:.4f}"
        samp_str = f"{m['n_total']} ({m['n_real']}R/{m['n_fake']}F)"
        print(f"{fam:<28} | {samp_str:<10} | {auc_str:<8} | {acc_str:<8} | {brier_str:<8} | {ece_str:<8} | {dis_str:<8}")

    # Overall Metrics
    overall = summarize_metrics("Overall Pooled Test", results)
    family_metrics["Overall Pooled Test"] = overall
    print("-" * 100)
    auc_str = f"{overall['auroc']:.4f}" if not math.isnan(overall["auroc"]) else "N/A"
    acc_str = f"{overall['accuracy'] * 100:.2f}%"
    brier_str = f"{overall['brier_score']:.4f}"
    ece_str = f"{overall['ece']:.4f}"
    dis_str = f"{overall['mean_disagreement']:.4f}"
    samp_str = f"{overall['n_total']} ({overall['n_real']}R/{overall['n_fake']}F)"
    print(f"{'Overall Pooled Test':<28} | {samp_str:<10} | {auc_str:<8} | {acc_str:<8} | {brier_str:<8} | {ece_str:<8} | {dis_str:<8}")
    print("=" * 100)
    print("*Note: Datasets with single-class distributions (e.g. 100% fake generative benchmarks) report Accuracy & Brier Score; AUROC is defined on pairwise classes.\n")

    out_path = REPO_ROOT / "reports" / "per_generator_evaluation_report_v3.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "metrics_per_generator": family_metrics,
            "individual_predictions": results,
        }, f, indent=2)
    logger.info("Saved per-generator evaluation report to: %s", out_path)


if __name__ == "__main__":
    main()
