"""
evaluation/evaluate_branch_pruning_df40.py
=========================================
Full Branch Pruning Ablation & Per-Branch Marginal AUROC Study for CVPR 2027.
Evaluates the master FORGE-Lite model against the DF40 benchmark across all 4 families:
1. Full Master Model (All Active Modalities + Dynamic Reliability Weighting)
2. Leave-One-Out Branch Ablations:
   - w/o Semantic Branch (DINOv2 context ablated)
   - w/o Forensic Branch (SRM residual & FFT ablated)
   - w/o Dynamic Reliability Fusion (Unweighted linear average)
3. Standalone Branch Direct Heads:
   - Standalone Semantic Branch (DINOv2 linear probe)
   - Standalone Forensic Branch (SRM-CNN linear probe)

Computes marginal AUROC drop (Delta AUROC) to rigorously measure each branch's
forensic necessity across Diffusion, Face-Swap, Reenactment, and Talking-Head.
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

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
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
logger = logging.getLogger("branch_pruning_df40")

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_master_model(device: torch.device):
    ckpt_path = REPO_ROOT / "outputs" / "checkpoints" / "best.ckpt"
    logger.info("Loading master checkpoint from %s", ckpt_path.name)
    model = FORGELite()
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()

    calib_path = REPO_ROOT / "outputs" / "checkpoints" / "calibration_temperature.pt"
    temp, bias, thresh = 9.6651, 0.0, 0.672
    if calib_path.exists():
        cdata = torch.load(calib_path, map_location="cpu", weights_only=False)
        temp = float(cdata.get("temperature", 9.6651))
        bias = float(cdata.get("platt_b", 0.0))
        thresh = float(cdata.get("image_decision_threshold", 0.672))
    return model, temp, bias, thresh


def run_pruning_ablation(samples_per_method: int = 20):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Initializing DF40 Branch Pruning Ablation on %s", device)
    model, temp, bias, threshold = load_master_model(device)

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
    random.seed(42)
    random.shuffle(real_paths)

    # 2. Group DF40 samples by family
    disk_manifest = REPO_ROOT / "data" / "manifests" / "df40_disk.jsonl"
    by_family = defaultdict(list)
    by_method = defaultdict(int)

    with open(disk_manifest, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            m = obj.get("method")
            ft = obj.get("forgery_type")
            if by_method[m] < samples_per_method:
                p = REPO_ROOT / obj["source_path"]
                if p.exists():
                    by_method[m] += 1
                    by_family[ft].append((p, m))

    all_fakes = []
    for ft, items in by_family.items():
        all_fakes.extend(items)
        logger.info("Family '%s': %d fake samples", ft, len(items))

    matched_reals = real_paths[:len(all_fakes)]
    logger.info("Total evaluation pool: %d Real + %d Fake = %d images", len(matched_reals), len(all_fakes), len(matched_reals) + len(all_fakes))

    eval_items = [(p, 0, "real", "real") for p in matched_reals] + [
        (p, 1, ft, m) for (p, m), ft in [(it, ft) for ft, items in by_family.items() for it in items]
    ]
    random.shuffle(eval_items)

    # Helper for inference under a specific ablation config
    def evaluate_configuration(config_name: str) -> Dict[str, Any]:
        logger.info("Evaluating configuration: %s", config_name)
        probs = []
        labels = []
        families = []

        batch_size = 32
        for i in range(0, len(eval_items), batch_size):
            batch = eval_items[i:i + batch_size]
            tensors = []
            valid_b = []
            for item in batch:
                try:
                    with Image.open(item[0]) as img:
                        tensors.append(TRANSFORM(img.convert("RGB")))
                        valid_b.append(item)
                except Exception:
                    continue

            if not tensors:
                continue

            batch_t = torch.stack(tensors, dim=0).unsqueeze(1).to(device)

            with torch.no_grad():
                # Apply ablation switch
                if config_name == "Full Master Model":
                    model.enable_semantic = True
                    model.enable_forensic = True
                    model.unweighted_fusion = False
                    out = model(batch_t)
                    raw_logits = out["fake_logit"].squeeze(-1).cpu().numpy()
                    cal_logits = (raw_logits / temp) + bias
                    p = 1.0 / (1.0 + np.exp(-cal_logits))

                elif config_name == "w/o Semantic Branch (DINOv2)":
                    model.enable_semantic = False
                    model.enable_forensic = True
                    model.unweighted_fusion = False
                    out = model(batch_t)
                    raw_logits = out["fake_logit"].squeeze(-1).cpu().numpy()
                    cal_logits = (raw_logits / temp) + bias
                    p = 1.0 / (1.0 + np.exp(-cal_logits))

                elif config_name == "w/o Forensic Branch (SRM+FFT)":
                    model.enable_semantic = True
                    model.enable_forensic = False
                    model.unweighted_fusion = False
                    out = model(batch_t)
                    raw_logits = out["fake_logit"].squeeze(-1).cpu().numpy()
                    cal_logits = (raw_logits / temp) + bias
                    p = 1.0 / (1.0 + np.exp(-cal_logits))

                elif config_name == "w/o Dynamic Reliability (Unweighted Average)":
                    model.enable_semantic = True
                    model.enable_forensic = True
                    model.unweighted_fusion = True
                    out = model(batch_t)
                    raw_logits = out["fake_logit"].squeeze(-1).cpu().numpy()
                    cal_logits = (raw_logits / temp) + bias
                    p = 1.0 / (1.0 + np.exp(-cal_logits))

                elif config_name == "Standalone Semantic Branch":
                    model.enable_semantic = True
                    model.enable_forensic = True
                    out = model(batch_t)
                    b_probs = out["branch_fake_probs"].squeeze().cpu().numpy()
                    # index 0 is semantic
                    p = b_probs[:, 0] if b_probs.ndim > 1 else np.array([b_probs[0]])

                elif config_name == "Standalone Forensic Branch":
                    model.enable_semantic = True
                    model.enable_forensic = True
                    out = model(batch_t)
                    b_probs = out["branch_fake_probs"].squeeze().cpu().numpy()
                    # index 1 is forensic
                    p = b_probs[:, 1] if b_probs.ndim > 1 else np.array([b_probs[1]])

                else:
                    raise ValueError(f"Unknown config: {config_name}")

                if isinstance(p, np.ndarray):
                    probs.extend(p.tolist())
                else:
                    probs.append(p)

                for item in valid_b:
                    labels.append(item[1])
                    families.append(item[2])

        y_true = np.array(labels)
        y_prob = np.array(probs)
        overall_auc = float(roc_auc_score(y_true, y_prob))
        y_pred = (y_prob >= threshold).astype(int)
        overall_acc = float(accuracy_score(y_true, y_pred))

        # Per family breakdown
        per_fam_auc = {}
        for fam in ["composited", "fully_synthetic", "reenactment", "talking_head"]:
            idx = [j for j, f in enumerate(families) if f in (fam, "real")]
            if len(idx) > 0 and len(np.unique(y_true[idx])) >= 2:
                per_fam_auc[fam] = float(roc_auc_score(y_true[idx], y_prob[idx]))
            else:
                per_fam_auc[fam] = float("nan")

        return {
            "overall_auroc": round(overall_auc, 4),
            "overall_acc": round(overall_acc, 4),
            "per_family_auroc": per_fam_auc,
        }

    # Run all ablation configurations
    configurations = [
        "Full Master Model",
        "w/o Forensic Branch (SRM+FFT)",
        "w/o Semantic Branch (DINOv2)",
        "w/o Dynamic Reliability (Unweighted Average)",
        "Standalone Semantic Branch",
        "Standalone Forensic Branch",
    ]

    results = {}
    for cfg in configurations:
        results[cfg] = evaluate_configuration(cfg)

    # Restore default model flags
    model.enable_semantic = True
    model.enable_forensic = True
    model.unweighted_fusion = False

    # Format into publication-grade table
    baseline_auc = results["Full Master Model"]["overall_auroc"]

    print("\n" + "=" * 115)
    print("CVPR 2027 PER-BRANCH MARGINAL AUROC & PRUNING ABLATION ON DF40")
    print("-" * 115)
    print(f"{'CONFIGURATION / BRANCH':<42} | {'POOLED AUROC':<12} | {'MARGINAL DELTA':<14} | {'DIFFUSION':<10} | {'FACE-SWAP':<10} | {'REENACT':<10}")
    print("-" * 115)

    for cfg, m in results.items():
        auc = m["overall_auroc"]
        delta = auc - baseline_auc
        delta_str = f"{delta:+.4f}" if cfg != "Full Master Model" else "BASELINE"
        pf = m["per_family_auroc"]
        diff_auc = f"{pf.get('fully_synthetic', float('nan')):.4f}"
        swap_auc = f"{pf.get('composited', float('nan')):.4f}"
        reen_auc = f"{pf.get('reenactment', float('nan')):.4f}"
        print(f"{cfg:<42} | {auc:<12.4f} | {delta_str:<14} | {diff_auc:<10} | {swap_auc:<10} | {reen_auc:<10}")

    print("=" * 115)
    print("*Diffusion: fully_synthetic (pixart, sd2.1, DiT, etc.) | Face-Swap: composited (simswap, inswap, etc.)\n")

    # Save to reports
    out_dir = REPO_ROOT / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "branch_pruning_df40_report.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "benchmark": "DF40 Branch Pruning Ablation",
            "samples_per_method": samples_per_method,
            "baseline_auroc": baseline_auc,
            "results": results,
        }, f, indent=2)
    logger.info("Saved complete report to %s", out_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples-per-method", type=int, default=20, help="Samples per generator method")
    args = parser.parse_args()
    run_pruning_ablation(samples_per_method=args.samples_per_method)
