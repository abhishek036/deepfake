"""
evaluation/evaluate_talkingheadbench.py
======================================
Benchmark Evaluation Script for TalkingHeadBench (2024 SOTA Talking Head Deepfakes).

Evaluates the trained FORGE-Lite Audio-Visual Branch on:
- LivePortrait (KwaiVGI implicit keypoints)
- Hallo2 (Fudan/Baidu hierarchical audio-driven portraits)
- AniPortraitAudio (Tencent diffusion-based talking head)
- EmoPortrait (emotion-conditioned expressive portraits)
- Hallo (audio-driven talking face)
- HDTF (genuine high-definition talking face controls)

Metrics:
- Overall ROC-AUC & Equal Error Rate (EER)
- Accuracy @ threshold 0.5
- Per-generator Detection Rate (Recall @ 0.5) & Mean Anomaly Score
- Audio-Visual Cosine Similarity (genuine sync vs generative desync)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from omegaconf import OmegaConf
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.dataset import ForgeDataset, collate_forge
from models.forge_lite import FORGELite

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("eval_thb")


def compute_eer(labels: np.ndarray, scores: np.ndarray) -> float:
    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.absolute(fnr - fpr))
    return float((fpr[eer_idx] + fnr[eer_idx]) / 2.0)


def evaluate_thb(
    manifest_path: str = "data/manifests/talkingheadbench.jsonl",
    checkpoint_path: str = "outputs/checkpoints/forge-lite-v1_av_branch_best.ckpt",
    split: str = "test",
    batch_size: int = 4,
) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Evaluating on device: %s | Split: %s", device, split)

    # 1. Load model
    logger.info("Initializing FORGE-Lite Audio-Visual branch...")
    model = FORGELite(
        enable_semantic=False,
        enable_forensic=False,
        enable_temporal=False,
        enable_av=True,
        enable_localization=False,
    ).to(device)

    ckpt_p = Path(checkpoint_path)
    if not ckpt_p.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_p}")

    ckpt = torch.load(ckpt_p, map_location="cpu", weights_only=True)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=False)
    model.eval()
    logger.info("Loaded weights from %s", ckpt_p)

    # 2. Dataset
    manifest_file = ROOT / manifest_path
    if not manifest_file.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_file}")

    # Read records to map indices to generator families
    records = []
    with open(manifest_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                if split == "all" or r.get("split") == split:
                    records.append(r)

    logger.info("Loaded %d records for evaluation (split=%s)", len(records), split)

    dataset = ForgeDataset(
        manifest_path=str(manifest_file),
        split=split,
        window_size=5,
        augment=False,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_forge,
        num_workers=0,
    )

    all_scores = []
    all_cosine_sims = []
    all_labels = []
    all_gens = []

    with torch.no_grad():
        idx_offset = 0
        for batch in tqdm(loader, desc=f"Evaluating THB [{split}]"):
            frames = batch["frames"].to(device)
            labels = batch["label"].to(device)
            audio_mel = batch.get("audio_mel")
            if audio_mel is not None:
                audio_mel = audio_mel.to(device)
            audio_available = batch.get("audio_available").to(device)

            with torch.amp.autocast(device_type=device, enabled=(device == "cuda")):
                av_out = model.av_branch(
                    audio_mel=audio_mel,
                    lip_crops=frames,
                    audio_available=audio_available,
                )
                pred_scores = av_out["av_score"].squeeze(-1).cpu().numpy()
                cos_sims = av_out["cosine_sim"].squeeze(-1).cpu().numpy()

            b_len = len(labels)
            all_scores.extend(pred_scores.tolist())
            all_cosine_sims.extend(cos_sims.tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

            for i in range(b_len):
                if idx_offset + i < len(records):
                    all_gens.append(records[idx_offset + i].get("generator_family", "unknown"))
                else:
                    all_gens.append("unknown")

            idx_offset += b_len

    y_true = np.array(all_labels)
    y_pred = np.array(all_scores)
    y_cos = np.array(all_cosine_sims)

    # Compute overall metrics
    has_both = len(np.unique(y_true)) > 1
    auc = float(roc_auc_score(y_true, y_pred)) if has_both else 0.5
    eer = compute_eer(y_true, y_pred) if has_both else 0.5
    acc = float(np.mean((y_pred >= 0.5) == (y_true == 1)))

    # Real vs Fake Cosine Similarity
    real_cos = float(np.mean(y_cos[y_true == 0])) if (y_true == 0).any() else 0.0
    fake_cos = float(np.mean(y_cos[y_true == 1])) if (y_true == 1).any() else 0.0

    # Per-generator breakdown
    unique_gens = sorted(set(all_gens))
    gen_stats = {}

    for g in unique_gens:
        mask = np.array([gen == g for gen in all_gens])
        g_scores = y_pred[mask]
        g_labels = y_true[mask]
        g_cos = y_cos[mask]

        is_fake_gen = (g != "HDTF_real" and not g.lower().startswith("real"))
        target_label = 1 if is_fake_gen else 0

        correct = np.sum((g_scores >= 0.5) == target_label)
        acc_gen = float(correct / len(g_scores)) if len(g_scores) > 0 else 0.0

        gen_stats[g] = {
            "count": int(len(g_scores)),
            "target": "fake" if target_label == 1 else "real",
            "mean_score": float(np.mean(g_scores)),
            "accuracy": acc_gen,
            "mean_cosine_sim": float(np.mean(g_cos)),
        }

    results = {
        "split": split,
        "total_samples": len(y_true),
        "roc_auc": auc,
        "eer": eer,
        "overall_accuracy": acc,
        "real_mean_cosine_sim": real_cos,
        "fake_mean_cosine_sim": fake_cos,
        "generators": gen_stats,
    }

    # Print clean terminal report
    print("\n" + "=" * 70)
    print(f"  TALKINGHEADBENCH EVALUATION REPORT (Split: {split.upper()})")
    print("=" * 70)
    print(f"  Total Samples Evaluated : {len(y_true)}")
    print(f"  ROC-AUC                : {auc * 100:.2f}%")
    print(f"  Equal Error Rate (EER) : {eer * 100:.2f}%")
    print(f"  Accuracy (@ 0.5)       : {acc * 100:.2f}%")
    print(f"  Real Cosine Similarity : {real_cos:+.4f} (High = authentic lip-speech sync)")
    print(f"  Fake Cosine Similarity : {fake_cos:+.4f} (Low = AI synthesis desynchrony)")
    print("-" * 70)
    print(f"  {'Generator / Source':<20} | {'Type':<6} | {'Count':<5} | {'Acc@0.5':<8} | {'Mean Score':<10} | {'Mean Sim'}")
    print("-" * 70)
    for g, s in gen_stats.items():
        print(f"  {g:<20} | {s['target']:<6} | {s['count']:<5} | {s['accuracy']*100:>6.1f}% | {s['mean_score']:>10.4f} | {s['mean_cosine_sim']:>+8.4f}")
    print("=" * 70 + "\n")

    # Save to outputs
    out_json = ROOT / "outputs" / f"talkingheadbench_eval_{split}.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info("Evaluation results saved to: %s", out_json)

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate FORGE-Lite AV branch on TalkingHeadBench")
    parser.add_argument("--manifest", default="data/manifests/talkingheadbench.jsonl")
    parser.add_argument("--checkpoint", default="outputs/checkpoints/forge-lite-v1_av_branch_best.ckpt")
    parser.add_argument("--split", default="all", choices=["train", "val", "test", "all"])
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    evaluate_thb(
        manifest_path=args.manifest,
        checkpoint_path=args.checkpoint,
        split=args.split,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
