"""
evaluation/test_trainsets.py
Benchmark FORGE-Lite on the dataset in C:\deepfake\trainsets.

CSV: trainsets/selfie_and_video.csv
Files directory: trainsets/files/

Supports:
- Manual execution by user from terminal:
  python evaluation/test_trainsets.py
- Filtering by type:
  python evaluation/test_trainsets.py --type video
  python evaluation/test_trainsets.py --type photo
- Quick test with limit:
  python evaluation/test_trainsets.py --limit 10
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

# Add repo root to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
from service.inference import FORGEInference

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def evaluate_trainsets(
    csv_path: str = "trainsets/selfie_and_video.csv",
    files_dir: str = "trainsets/files",
    checkpoint_path: str = "outputs/checkpoints/best.ckpt",
    output_report: str = "reports/trainsets_evaluation_report.md",
    media_type: str = "all",
    limit: int = 0,
):
    csv_file = Path(csv_path)
    if not csv_file.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_file)
    logger.info("Loaded CSV with %d records from %s", len(df), csv_path)

    # Filter by media type if requested
    if media_type in ("photo", "video"):
        df = df[df["type"].str.lower() == media_type].reset_index(drop=True)
        logger.info("Filtered to %d records of type '%s'", len(df), media_type)

    if limit > 0:
        df = df.head(limit).reset_index(drop=True)
        logger.info("Limited evaluation to first %d records", limit)

    logger.info("Initializing FORGE-Lite inference engine (%s)...", checkpoint_path)
    engine = FORGEInference(checkpoint_path=checkpoint_path)

    results = []
    logger.info("--- Starting Benchmark on %d items ---", len(df))

    for idx, row in df.iterrows():
        rel_path = str(row["files"]).lstrip("/\\").replace("/", os.sep)
        m_type = str(row["type"]).strip().lower()
        full_path = os.path.join(files_dir, rel_path)

        if not os.path.exists(full_path):
            logger.warning("[%2d/%2d] File missing on disk: %s", idx + 1, len(df), full_path)
            continue

        t0 = time.time()
        try:
            if m_type == "video" or full_path.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                res = engine.analyze_video(full_path)
            else:
                res = engine.analyze_image(full_path)
            el = time.time() - t0

            decision = res.get("decision", "uncertain")
            risk = float(res.get("calibrated_risk", 0.5))
            unc = float(res.get("uncertainty", 1.0))
            cov = float(res.get("evidence_coverage", 0.0))
            blur = float(res.get("quality", {}).get("blur", 0.0))
            pose = float(res.get("quality", {}).get("pose", 0.0))

            results.append({
                "index": idx + 1,
                "file": rel_path,
                "type": m_type,
                "decision": decision,
                "fake_score": float(res.get("fake_score", 0.5)),
                "calibrated_risk": risk,
                "uncertainty": unc,
                "coverage": cov,
                "blur": blur,
                "pose": pose,
                "abstention_reason": res.get("abstention_reason"),
                "time_sec": round(el, 2),
            })

            logger.info(
                "[%2d/%2d] [%-5s] %-15s -> %-21s (Risk: %5.1f%%, Unc: %.3f, Time: %.2fs)",
                idx + 1, len(df), m_type, rel_path, decision.upper(), risk * 100, unc, el
            )

        except Exception as e:
            logger.error("[%2d/%2d] Error evaluating %s: %s", idx + 1, len(df), rel_path, e)

    # Compute Statistics
    n_total = len(results)
    if n_total == 0:
        logger.error("No results recorded.")
        return

    n_real = sum(1 for r in results if r["decision"] in ("likely_real", "real"))
    n_fake = sum(1 for r in results if r["decision"] in ("likely_fake", "fake"))
    n_abstain = sum(1 for r in results if r["decision"] in ("insufficient_evidence", "abstain"))
    decided = n_real + n_fake

    specificity = (n_real / decided * 100.0) if decided > 0 else 0.0
    fpr = (n_fake / decided * 100.0) if decided > 0 else 0.0
    abstention_rate = (n_abstain / n_total * 100.0) if n_total > 0 else 0.0
    mean_risk = float(np.mean([r["calibrated_risk"] for r in results]))
    mean_latency = float(np.mean([r["time_sec"] for r in results]))

    # Video breakdown
    v_items = [r for r in results if r["type"] == "video"]
    v_decided = sum(1 for r in v_items if r["decision"] not in ("insufficient_evidence", "abstain"))
    v_real = sum(1 for r in v_items if r["decision"] in ("likely_real", "real"))
    v_fake = sum(1 for r in v_items if r["decision"] in ("likely_fake", "fake"))
    v_abstain = sum(1 for r in v_items if r["decision"] in ("insufficient_evidence", "abstain"))
    v_spec = (v_real / v_decided * 100.0) if v_decided > 0 else 0.0
    v_fpr = (v_fake / v_decided * 100.0) if v_decided > 0 else 0.0

    # Photo breakdown
    p_items = [r for r in results if r["type"] == "photo"]
    p_decided = sum(1 for r in p_items if r["decision"] not in ("insufficient_evidence", "abstain"))
    p_real = sum(1 for r in p_items if r["decision"] in ("likely_real", "real"))
    p_fake = sum(1 for r in p_items if r["decision"] in ("likely_fake", "fake"))
    p_abstain = sum(1 for r in p_items if r["decision"] in ("insufficient_evidence", "abstain"))
    p_spec = (p_real / p_decided * 100.0) if p_decided > 0 else 0.0
    p_fpr = (p_fake / p_decided * 100.0) if p_decided > 0 else 0.0

    # Build Markdown Report
    report_lines = [
        "# FORGE-Lite Benchmark: Trainsets Evaluation Report",
        "",
        f"**CSV Manifest**: `{csv_path}`",
        f"**Media Root**: `{files_dir}`",
        f"**Test Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Model Checkpoint**: `{checkpoint_path}` (Robust Consensus Aggregation, Clamped $T=0.85$)",
        "",
        "---",
        "",
        "## 1. Summary Performance Metrics",
        "",
        "| Evaluation Metric | Consolidated (All Media) | Videos Only | Photos Only | Target |",
        "|---|---|---|---|---|",
        f"| **Total Processed** | **{n_total}** | **{len(v_items)}** | **{len(p_items)}** | — |",
        f"| **Authentic Verification (Specificity)** | **{specificity:.1f}%** | **{v_spec:.1f}%** | **{p_spec:.1f}%** | $\\ge 90.0\\%$ |",
        f"| **False Positive Rate (FPR)** | **{fpr:.1f}%** | **{v_fpr:.1f}%** | **{p_fpr:.1f}%** | $\\le 10.0\\%$ |",
        f"| **Selective Abstention Rate** | **{abstention_rate:.1f}%** | **{(v_abstain / max(1, len(v_items)) * 100):.1f}%** | **{(p_abstain / max(1, len(p_items)) * 100):.1f}%** | 10–25% on extreme poses |",
        f"| **Mean Calibrated Synthetic Risk** | **{mean_risk * 100:.1f}%** | **{(float(np.mean([r['calibrated_risk'] for r in v_items])) if v_items else 0.0)*100:.1f}%** | **{(float(np.mean([r['calibrated_risk'] for r in p_items])) if p_items else 0.0)*100:.1f}%** | Lower is better |",
        f"| **Average Inference Latency** | **{mean_latency:.2f}s** | **{(float(np.mean([r['time_sec'] for r in v_items])) if v_items else 0.0):.2f}s** | **{(float(np.mean([r['time_sec'] for r in p_items])) if p_items else 0.0):.2f}s** | Sub-second |",
        "",
        "---",
        "",
        "## 2. Detailed Per-Item Breakdown",
        "",
        "| # | File Path | Type | Decision | Calibrated Risk | Uncertainty | Evidence Coverage | Latency |",
        "|---|---|---|---|---|---|---|---|",
    ]

    for r in results:
        badge = (
            "✅ Likely Real" if r["decision"] in ("likely_real", "real")
            else ("⚠️ Abstained" if r["decision"] in ("insufficient_evidence", "abstain")
            else "🚨 Likely Fake")
        )
        report_lines.append(
            f"| {r['index']} | `{r['file']}` | `{r['type']}` | {badge} | `{r['calibrated_risk'] * 100:.1f}%` | `{r['uncertainty']:.3f}` | `{r['coverage'] * 100:.0f}%` | `{r['time_sec']}s` |"
        )

    out_path = Path(output_report)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    logger.info("Report saved to %s", output_report)
    print("\n" + "="*70)
    print(f"BENCHMARK COMPLETE ({n_total} items evaluated)")
    print(f"  Specificity (Real Verified): {specificity:.1f}%")
    print(f"  False Positive Rate:         {fpr:.1f}%")
    print(f"  Abstention Rate:             {abstention_rate:.1f}%")
    print(f"  Mean Calibrated Risk:        {mean_risk * 100:.1f}%")
    print(f"  Average Latency:             {mean_latency:.2f}s / item")
    print(f"  Full report saved to:        {output_report}")
    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="FORGE-Lite Benchmark on trainsets dataset")
    parser.add_argument("--csv", type=str, default="trainsets/selfie_and_video.csv", help="Path to CSV manifest")
    parser.add_argument("--files_dir", type=str, default="trainsets/files", help="Base directory of media files")
    parser.add_argument("--checkpoint", type=str, default="outputs/checkpoints/best.ckpt", help="Model checkpoint")
    parser.add_argument("--output_report", type=str, default="reports/trainsets_evaluation_report.md", help="Output report path")
    parser.add_argument("--type", type=str, default="all", choices=["all", "photo", "video"], help="Filter by media type")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of items to test (0 = all)")
    args = parser.parse_args()

    evaluate_trainsets(
        csv_path=args.csv,
        files_dir=args.files_dir,
        checkpoint_path=args.checkpoint,
        output_report=args.output_report,
        media_type=args.type,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
