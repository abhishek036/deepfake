"""
evaluation/test_final_dataset.py
Benchmark and test FORGE-Lite on FINAL_DATASET.csv.

Supports:
- Auto-download and caching of image links
- Balanced evaluation across REAL and FAKE splits
- Sliced metrics by difficulty (Easy/Medium/Hard), quality, and fake method
- Automated publication-quality Markdown report generation (reports/final_dataset_test_report.md)
- JSON export of granular forensic results (reports/final_dataset_test_results.json)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import sys

# Ensure UTF-8 stdout on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tqdm import tqdm

from data.setup_final_dataset import download_single_image
from service.inference import FORGEInference

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("test_final_dataset")


def prepare_evaluation_samples(
    csv_path: Path,
    split: str = "test",
    limit: int = 0,
    balanced: bool = True,
) -> pd.DataFrame:
    """Loads CSV and samples items for evaluation."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset CSV not found at: {csv_path}")

    df = pd.read_csv(csv_path)
    logger.info("Loaded '%s' with %d total records.", csv_path, len(df))

    if split != "all" and "dataset_split" in df.columns:
        df = df[df["dataset_split"] == split].reset_index(drop=True)
        logger.info("Filtered for split '%s': %d records.", split, len(df))

    if limit > 0 and len(df) > limit:
        if balanced and "label" in df.columns:
            # Sample equally from REAL and FAKE
            real_df = df[df["label"].str.upper() == "REAL"]
            fake_df = df[df["label"].str.upper() == "FAKE"]
            half = limit // 2
            n_real = min(half, len(real_df))
            n_fake = min(limit - n_real, len(fake_df))

            sampled_real = real_df.sample(n=n_real, random_state=42) if len(real_df) > n_real else real_df
            sampled_fake = fake_df.sample(n=n_fake, random_state=42) if len(fake_df) > n_fake else fake_df
            df = pd.concat([sampled_real, sampled_fake]).sample(frac=1.0, random_state=42).reset_index(drop=True)
            logger.info("Balanced sampling applied: %d REAL, %d FAKE (Total: %d).", n_real, n_fake, len(df))
        else:
            df = df.head(limit).reset_index(drop=True)
            logger.info("Sampled first %d records.", len(df))

    return df


def ensure_images_downloaded(
    df: pd.DataFrame,
    images_dir: Path,
    max_workers: int = 16,
) -> List[Path]:
    """Ensures all required images exist on disk, downloading missing ones concurrently."""
    import concurrent.futures

    images_dir.mkdir(parents=True, exist_ok=True)
    missing_tasks = []
    local_paths = []

    for _, row in df.iterrows():
        img_id = row["image_id"]
        url = row["image_url"]
        dest = images_dir / f"{img_id}.jpg"
        local_paths.append(dest)

        if not (dest.exists() and dest.stat().st_size > 1024):
            missing_tasks.append((img_id, url, dest))

    if missing_tasks:
        logger.info("Downloading %d missing images using %d workers...", len(missing_tasks), max_workers)
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_id = {
                executor.submit(download_single_image, item[0], item[1], item[2]): item[0]
                for item in missing_tasks
            }
            with tqdm(total=len(missing_tasks), desc="Downloading missing images", unit="img") as pbar:
                for future in concurrent.futures.as_completed(future_to_id):
                    pbar.update(1)

    return local_paths


def evaluate_final_dataset(
    csv_path: str = "FINAL_DATASET.csv",
    split: str = "test",
    limit: int = 0,
    balanced: bool = True,
    checkpoint_path: str = "outputs/checkpoints/best.ckpt",
    images_dir: str = "data/final_dataset/images",
    output_report: str = "reports/final_dataset_test_report.md",
    output_json: str = "reports/final_dataset_test_results.json",
    workers: int = 16,
    device: str = "cuda",
) -> Dict[str, Any]:
    """Executes benchmark on FINAL_DATASET.csv."""
    t_start = time.time()
    csv_p = Path(csv_path)
    img_dir_p = Path(images_dir)

    df = prepare_evaluation_samples(csv_p, split=split, limit=limit, balanced=balanced)
    local_paths = ensure_images_downloaded(df, images_dir=img_dir_p, max_workers=workers)
    df["local_path"] = local_paths
    df = df[df["local_path"].apply(lambda p: p.exists() and p.stat().st_size > 1024)].reset_index(drop=True)
    if len(df) == 0:
        logger.error("No valid images available on disk to evaluate!")
        return {}

    logger.info("Initializing FORGE-Lite inference engine on %s...", device)
    engine = FORGEInference(checkpoint_path=checkpoint_path, device=device)

    results: List[Dict[str, Any]] = []
    logger.info("Evaluating %d images through FORGE-Lite pipeline...", len(df))

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Inference", unit="img"):
        img_id = row["image_id"]
        local_p = row["local_path"]
        gt_label_str = str(row["label"]).strip().upper()
        gt_label = 0 if gt_label_str == "REAL" else 1

        if not local_p.exists():
            logger.warning("Image %d not found on disk: %s", img_id, local_p)
            continue

        t0 = time.time()
        try:
            res = engine.analyze_image(str(local_p))
            elapsed = time.time() - t0

            decision = res.get("decision", "uncertain")
            risk = float(res.get("calibrated_risk", 0.5))
            unc = float(res.get("uncertainty", 1.0))
            cov = float(res.get("evidence_coverage", 0.0))
            blur = float(res.get("quality", {}).get("blur", 0.0))
            pose = float(res.get("quality", {}).get("pose", 0.0))

            results.append({
                "image_id": img_id,
                "file_path": str(local_p),
                "gt_label_str": gt_label_str,
                "gt_label": gt_label,
                "decision": decision,
                "fake_score": float(res.get("fake_score", 0.5)),
                "calibrated_risk": risk,
                "uncertainty": unc,
                "evidence_coverage": cov,
                "blur": blur,
                "pose": pose,
                "difficulty": str(row.get("detection_difficulty", "Unknown")),
                "quality": str(row.get("image_quality", "Unknown")),
                "fake_method": str(row.get("fake_method", "None")),
                "category": str(row.get("category", "Unknown")),
                "latency_sec": round(elapsed, 4),
            })
        except Exception as e:
            logger.error("Error evaluating image_id %s (%s): %s", img_id, local_p, e)

    if not results:
        logger.error("No results recorded!")
        return {}

    # ─────────────────────────────────────────────────────────────────────────────
    # Compute Metrics
    # ─────────────────────────────────────────────────────────────────────────────
    n_total = len(results)
    y_true = [r["gt_label"] for r in results]
    y_scores = [r["calibrated_risk"] for r in results]

    # Partition decisions
    abstained_items = [r for r in results if r["decision"] in ("insufficient_evidence", "abstain")]
    decided_items = [r for r in results if r["decision"] not in ("insufficient_evidence", "abstain")]

    abstention_rate = (len(abstained_items) / n_total) * 100.0 if n_total > 0 else 0.0

    # Predictions on decided items
    # decision == 'likely_fake' -> 1, 'likely_real' -> 0
    d_y_true = [r["gt_label"] for r in decided_items]
    d_y_pred = [1 if r["decision"] in ("likely_fake", "fake") else 0 for r in decided_items]
    d_y_scores = [r["calibrated_risk"] for r in decided_items]

    acc = (accuracy_score(d_y_true, d_y_pred) * 100.0) if decided_items else 0.0
    prec = (precision_score(d_y_true, d_y_pred, zero_division=0) * 100.0) if decided_items else 0.0
    rec = (recall_score(d_y_true, d_y_pred, zero_division=0) * 100.0) if decided_items else 0.0
    f1 = (f1_score(d_y_true, d_y_pred, zero_division=0) * 100.0) if decided_items else 0.0

    # Confusion matrix on decided items: [[TN, FP], [FN, TP]]
    if len(decided_items) > 0 and len(set(d_y_true)) > 1:
        tn, fp, fn, tp = confusion_matrix(d_y_true, d_y_pred, labels=[0, 1]).ravel()
        specificity = (tn / (tn + fp) * 100.0) if (tn + fp) > 0 else 0.0
        fpr = (fp / (tn + fp) * 100.0) if (tn + fp) > 0 else 0.0
    else:
        tn = fp = fn = tp = 0
        specificity = fpr = 0.0

    # Global ROC-AUC and Average Precision
    try:
        if len(set(y_true)) > 1:
            auc = float(roc_auc_score(y_true, y_scores))
            ap = float(average_precision_score(y_true, y_scores))
        else:
            auc = float("nan")
            ap = float("nan")
    except Exception:
        auc = float("nan")
        ap = float("nan")

    mean_latency = float(np.mean([r["latency_sec"] for r in results]))

    # Subgroup / Slice Metrics
    def compute_slice_metrics(subset_items: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not subset_items:
            return {"count": 0, "acc": "-", "auc": "-", "abstain": "-"}
        sub_d = [r for r in subset_items if r["decision"] not in ("insufficient_evidence", "abstain")]
        sub_acc = (
            accuracy_score(
                [r["gt_label"] for r in sub_d],
                [1 if r["decision"] in ("likely_fake", "fake") else 0 for r in sub_d],
            )
            * 100.0
            if sub_d
            else float("nan")
        )
        try:
            sub_y = [r["gt_label"] for r in subset_items]
            sub_s = [r["calibrated_risk"] for r in subset_items]
            sub_auc = (
                float(roc_auc_score(sub_y, sub_s))
                if len(set(sub_y)) > 1
                else float("nan")
            )
        except Exception:
            sub_auc = float("nan")
        sub_abstain = (len(subset_items) - len(sub_d)) / len(subset_items) * 100.0
        return {
            "count": len(subset_items),
            "acc": f"{sub_acc:.1f}%" if not np.isnan(sub_acc) else "N/A",
            "auc": f"{sub_auc:.4f}" if not np.isnan(sub_auc) else "N/A",
            "abstain": f"{sub_abstain:.1f}%",
        }

    difficulty_slices = {}
    for diff in ["Easy", "Medium", "Hard"]:
        items = [r for r in results if r["difficulty"].lower() == diff.lower()]
        difficulty_slices[diff] = compute_slice_metrics(items)

    quality_slices = {}
    for q_val in ["High", "Medium", "Low"]:
        items = [r for r in results if r["quality"].lower() == q_val.lower()]
        if items:
            quality_slices[q_val] = compute_slice_metrics(items)

    method_slices = {}
    methods = sorted(list(set(r["fake_method"] for r in results)))
    for meth in methods:
        items = [r for r in results if r["fake_method"] == meth]
        method_slices[meth] = compute_slice_metrics(items)

    # ─────────────────────────────────────────────────────────────────────────────
    # Build Markdown Report
    # ─────────────────────────────────────────────────────────────────────────────
    report_lines = [
        "# FORGE-Lite Benchmark: FINAL_DATASET.csv Evaluation Report",
        "",
        f"**Dataset CSV**: `{csv_path}`",
        f"**Split Evaluated**: `{split.upper()}` | **Balanced Sampling**: `{balanced}`",
        f"**Items Tested**: **{n_total}** ({len(decided_items)} decided, {len(abstained_items)} abstained)",
        f"**Model Checkpoint**: `{checkpoint_path}`",
        f"**Evaluation Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "---",
        "",
        "## 1. Executive Performance Summary",
        "",
        "| Forensic Evaluation Metric | Score | Target / Benchmark Standard |",
        "|---|:---:|:---:|",
        f"| **Overall Accuracy (on decided media)** | **{acc:.1f}%** | $\\ge 90.0\\%$ |",
        f"| **ROC-AUC (Global Discriminative Power)** | **{f'{auc:.4f}' if not np.isnan(auc) else 'N/A'}** | $\\ge 0.9200$ |",
        f"| **Average Precision (PR-AUC)** | **{f'{ap:.4f}' if not np.isnan(ap) else 'N/A'}** | $\\ge 0.9000$ |",
        f"| **Fake Detection Sensitivity (Recall)** | **{rec:.1f}%** | $\\ge 90.0\\%$ |",
        f"| **Authentic Verification (Specificity)** | **{specificity:.1f}%** | $\\ge 90.0\\%$ |",
        f"| **False Positive Rate (FPR)** | **{fpr:.1f}%** | $\\le 10.0\\%$ |",
        f"| **F1-Score** | **{f1:.1f}%** | $\\ge 90.0\\%$ |",
        f"| **Selective Prediction Abstention Rate** | **{abstention_rate:.1f}%** | 10% – 25% |",
        f"| **Average Inference Latency** | **{mean_latency:.3f}s / image** | $< 0.500s$ |",
        "",
        "---",
        "",
        "## 2. Confusion Matrix (Decided Samples)",
        "",
        "| | Predicted Real (Authentic) | Predicted Fake (Manipulated) |",
        "|---|:---:|:---:|",
        f"| **Actual Real** | **{tn}** (True Negative) | **{fp}** (False Positive) |",
        f"| **Actual Fake** | **{fn}** (False Negative) | **{tp}** (True Positive) |",
        "",
        "---",
        "",
        "## 3. Slice Analysis: Detection Difficulty",
        "",
        "| Difficulty Level | Total Samples | Accuracy | ROC-AUC | Abstention Rate |",
        "|---|:---:|:---:|:---:|:---:|",
    ]

    for diff, metrics in difficulty_slices.items():
        report_lines.append(
            f"| **{diff}** | {metrics['count']} | {metrics['acc']} | {metrics['auc']} | {metrics['abstain']} |"
        )

    report_lines.extend([
        "",
        "---",
        "",
        "## 4. Slice Analysis: Generator & Manipulation Method",
        "",
        "| Generator / Method | Samples | Accuracy | ROC-AUC | Abstention Rate |",
        "|---|:---:|:---:|:---:|:---:|",
    ])

    for meth, metrics in method_slices.items():
        name = "Authentic (Real Media)" if meth in ("None", "nan", "") else meth
        report_lines.append(
            f"| **{name}** | {metrics['count']} | {metrics['acc']} | {metrics['auc']} | {metrics['abstain']} |"
        )

    report_lines.extend([
        "",
        "---",
        "",
        "## 5. Granular Per-Sample Forensic Results",
        "",
        "| Image ID | Ground Truth | Decision | Match? | Risk | Uncertainty | Difficulty | Generator | Latency |",
        "|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ])

    for r in results[:100]:  # Cap first 100 in table for clean markdown readability
        gt = r["gt_label_str"]
        dec = r["decision"]
        is_abstained = dec in ("insufficient_evidence", "abstain")
        pred_label = "FAKE" if dec in ("likely_fake", "fake") else ("REAL" if dec in ("likely_real", "real") else "ABSTAIN")

        if is_abstained:
            match_badge = "⚠️ Abstain"
        elif pred_label == gt:
            match_badge = "✅ Correct"
        else:
            match_badge = "❌ Error"

        dec_badge = (
            "🚨 Likely Fake" if pred_label == "FAKE"
            else ("✅ Likely Real" if pred_label == "REAL" else "⚠️ Abstained")
        )

        report_lines.append(
            f"| `{r['image_id']}` | **{gt}** | {dec_badge} | {match_badge} | `{r['calibrated_risk']*100:.1f}%` | `{r['uncertainty']:.3f}` | {r['difficulty']} | {r['fake_method']} | `{r['latency_sec']}s` |"
        )

    if len(results) > 100:
        report_lines.append(f"\n*(Truncated: {len(results) - 100} additional records saved in JSON report)*\n")

    # Save Markdown Report
    rep_p = Path(output_report)
    rep_p.parent.mkdir(parents=True, exist_ok=True)
    with open(rep_p, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    # Save JSON results
    json_p = Path(output_json)
    json_p.parent.mkdir(parents=True, exist_ok=True)
    clean_results = [
        {k: v for k, v in r.items() if not isinstance(v, (np.ndarray, np.generic))}
        for r in results
    ]
    summary_data = {
        "summary": {
            "total_items": n_total,
            "decided_items": len(decided_items),
            "abstained_items": len(abstained_items),
            "accuracy": round(acc, 2),
            "auc": round(auc, 4) if not np.isnan(auc) else None,
            "average_precision": round(ap, 4) if not np.isnan(ap) else None,
            "fake_sensitivity": round(rec, 2),
            "authentic_specificity": round(specificity, 2),
            "false_positive_rate": round(fpr, 2),
            "abstention_rate": round(abstention_rate, 2),
            "mean_latency_sec": round(mean_latency, 4),
            "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        },
        "slice_difficulty": difficulty_slices,
        "slice_methods": method_slices,
        "items": clean_results,
    }
    with open(json_p, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    total_time = time.time() - t_start
    print("\n" + "=" * 76)
    print(">>> FORGE-Lite Benchmark Complete on FINAL_DATASET.csv")
    print(f"    Evaluated: {n_total} items in {total_time:.1f}s ({mean_latency*1000:.1f} ms / image)")
    print("-" * 76)
    print(f"    Accuracy (Decided):        {acc:.1f}%")
    print(f"    ROC-AUC:                  {f'{auc:.4f}' if not np.isnan(auc) else 'N/A'}")
    print(f"    Sensitivity (Fake Recall): {rec:.1f}%")
    print(f"    Specificity (Real Rec.):   {specificity:.1f}%")
    print(f"    False Positive Rate (FPR): {fpr:.1f}%")
    print(f"    Abstention Rate:          {abstention_rate:.1f}%")
    print(f"    Confusion Matrix:         TP={tp}, TN={tn}, FP={fp}, FN={fn}")
    print("-" * 76)
    print(f"    [Markdown Report] {output_report}")
    print(f"    [JSON Data]       {output_json}")
    print("=" * 76 + "\n")

    return summary_data


def main():
    parser = argparse.ArgumentParser(description="Run FORGE-Lite benchmark on FINAL_DATASET.csv")
    parser.add_argument("--csv", type=str, default="FINAL_DATASET.csv", help="Path to FINAL_DATASET.csv")
    parser.add_argument("--split", type=str, default="test", choices=["test", "val", "train", "all"], help="Split to test")
    parser.add_argument("--limit", type=int, default=20, help="Number of items to test (default: 20, 0 = all)")
    parser.add_argument("--no-balanced", action="store_true", help="Disable 50/50 balanced sampling between Real and Fake")
    parser.add_argument("--checkpoint", type=str, default="outputs/checkpoints/best.ckpt", help="Model checkpoint")
    parser.add_argument("--images-dir", type=str, default="data/final_dataset/images", help="Cache directory for images")
    parser.add_argument("--output-report", type=str, default="reports/final_dataset_test_report.md", help="Markdown report path")
    parser.add_argument("--output-json", type=str, default="reports/final_dataset_test_results.json", help="JSON output path")
    parser.add_argument("--workers", type=int, default=16, help="Download threads (default: 16)")
    parser.add_argument("--device", type=str, default="cuda", help="Inference device (cuda or cpu)")
    args = parser.parse_args()

    evaluate_final_dataset(
        csv_path=args.csv,
        split=args.split,
        limit=args.limit,
        balanced=not args.no_balanced,
        checkpoint_path=args.checkpoint,
        images_dir=args.images_dir,
        output_report=args.output_report,
        output_json=args.output_json,
        workers=args.workers,
        device=args.device,
    )


if __name__ == "__main__":
    main()
