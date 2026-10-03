"""
evaluation/slice_analysis.py
Subgroup / Slice Performance Analysis for FORGE-Lite.

Breaks down detection metrics across critical operational dimensions:
1. Face Size Ratio: [0, 0.05, 0.15, 0.30, 1.0] (small, medium, large, closeup)
2. Blur Score: [0, 0.2, 0.5, 0.8, 1.0] (sharp to heavily blurred)
3. Compression Quality: [0, 0.3, 0.6, 0.9, 1.0] (c23 high-quality to c40 low-quality)
4. Head Pose Yaw: [0, 15, 30, 45, 90] degrees (frontal to extreme profile)
5. Video Duration: [0, 5, 15, 60, 9999] seconds (short clip to long interview)

newimp.md §5 additions:
6. Lighting condition: [dark, dim, normal, bright, overexposed]
7. Skin tone (ITA proxy): [very_light, light, medium, dark, very_dark]

Identifies disparate failure modes, blind spots, and operational boundaries
as required by the PRD and newimp.md §5.

newimp.md requirement: report WORST-DOMAIN performance, not just mean.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from omegaconf import OmegaConf
from sklearn.metrics import roc_auc_score

from evaluation.evaluate_video import compute_eer, compute_tpr_at_fpr

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Slice Analyzer
# ─────────────────────────────────────────────────────────────────────────────

class SliceAnalyzer:
    """
    Performs multidimensional slice analysis across metadata attributes.
    """

    def __init__(self, config_path: str = "configs/eval.yaml") -> None:
        self.config = OmegaConf.load(config_path) if Path(config_path).exists() else OmegaConf.create()
        slice_cfg = getattr(self.config, "slices", {})

        self.bins_def = {
            "face_size": list(slice_cfg.get("face_size_bins", [0.0, 0.05, 0.15, 0.30, 1.0])),
            "blur": list(slice_cfg.get("blur_bins", [0.0, 0.2, 0.5, 0.8, 1.0])),
            "compression": list(slice_cfg.get("compression_bins", [0.0, 0.3, 0.6, 0.9, 1.0])),
            "pose_yaw": list(slice_cfg.get("pose_bins", [0.0, 15.0, 30.0, 45.0, 90.0])),
            "video_duration": list(slice_cfg.get("video_length_bins", [0.0, 5.0, 15.0, 60.0, 9999.0])),
            # newimp.md §5: lighting and skin-tone slices
            "brightness": list(slice_cfg.get("brightness_bins", [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])),
            "skin_tone_ita": list(slice_cfg.get("ita_bins", [-90.0, -35.0, 10.0, 28.0, 55.0, 90.0])),
        }

        self.records: List[Dict[str, Any]] = []

    def add_video_record(
        self,
        video_id: str,
        label: int,
        score: float,
        abstained: bool = False,
        uncertainty: float = 0.0,
        face_size: float = 0.15,
        blur: float = 0.1,
        compression: float = 0.2,
        pose_yaw: float = 10.0,
        video_duration: float = 12.0,
        # newimp.md §5 additions
        brightness: float = 0.5,        # mean frame brightness [0, 1]
        skin_tone_ita: float = 28.0,    # Individual Typology Angle proxy
    ) -> None:
        """Add evaluated video metadata and prediction."""
        self.records.append({
            "video_id": video_id,
            "label": label,
            "score": score,
            "abstained": abstained,
            "uncertainty": uncertainty,
            "face_size": face_size,
            "blur": blur,
            "compression": compression,
            "pose_yaw": pose_yaw,
            "video_duration": video_duration,
            "brightness": brightness,
            "skin_tone_ita": skin_tone_ita,
        })

    def _bin_values(self, values: np.ndarray, bin_edges: List[float]) -> np.ndarray:
        return np.digitize(values, bin_edges[1:-1])

    def evaluate_dimension(self, dimension_name: str) -> List[Dict[str, Any]]:
        """Compute metrics for every bin of a specified slice dimension."""
        if dimension_name not in self.bins_def:
            raise ValueError(f"Unknown slice dimension: {dimension_name}")

        bin_edges = self.bins_def[dimension_name]
        num_bins = len(bin_edges) - 1

        values = np.array([r[dimension_name] for r in self.records])
        labels = np.array([r["label"] for r in self.records])
        scores = np.array([r["score"] for r in self.records])
        abstained = np.array([r["abstained"] for r in self.records])

        bin_indices = self._bin_values(values, bin_edges)

        results = []
        for i in range(num_bins):
            low, high = bin_edges[i], bin_edges[i + 1]
            mask = bin_indices == i
            count = int(mask.sum())
            reals = int((labels[mask] == 0).sum())
            fakes = int((labels[mask] == 1).sum())

            if count > 0 and len(np.unique(labels[mask])) >= 2:
                try:
                    auc = float(roc_auc_score(labels[mask], scores[mask]))
                except Exception:
                    auc = float("nan")
                tpr1 = compute_tpr_at_fpr(labels[mask], scores[mask], target_fpr=0.01)
                eer = compute_eer(labels[mask], scores[mask])
            else:
                auc = float("nan")
                tpr1 = float("nan")
                eer = float("nan")

            abst_rate = float(abstained[mask].mean()) if count > 0 else 0.0

            results.append({
                "bin_index": i,
                "range": f"[{low} - {high})",
                "sample_count": count,
                "reals": reals,
                "fakes": fakes,
                "video_auc": auc,
                "tpr_at_1pct_fpr": tpr1,
                "eer": eer,
                "abstention_rate": abst_rate,
            })

        return results

    def compute_all_slices(self) -> Dict[str, List[Dict[str, Any]]]:
        """Runs slice evaluations across all operational dimensions."""
        return {dim: self.evaluate_dimension(dim) for dim in self.bins_def.keys()}

    def compute_summary_with_worst_domain(
        self, all_slices: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """
        Compute summary including WORST-domain performance.

        Per newimp.md §8: "Choose on development-domain results, considering
        BOTH macro AND worst-domain performance."
        """
        summary = {
            "dimensions": {},
        }
        worst_overall_auc = float("inf")
        worst_overall_condition = ""

        for dim, results in all_slices.items():
            valid = [r for r in results if not np.isnan(r.get("video_auc", float("nan")))]
            if not valid:
                continue
            aucs = [r["video_auc"] for r in valid]
            worst_bin = min(valid, key=lambda x: x["video_auc"])
            summary["dimensions"][dim] = {
                "macro_mean_auc": float(np.mean(aucs)),
                "worst_bin_auc": float(worst_bin["video_auc"]),
                "worst_bin_range": worst_bin["range"],
                "worst_bin_n": worst_bin["sample_count"],
            }
            if worst_bin["video_auc"] < worst_overall_auc:
                worst_overall_auc = worst_bin["video_auc"]
                worst_overall_condition = f"{dim}={worst_bin['range']}"

        summary["worst_condition_across_all_slices"] = worst_overall_condition
        summary["worst_condition_auc"] = float(worst_overall_auc) \
            if worst_overall_auc < float("inf") else float("nan")
        summary["note"] = (
            "Worst-domain performance must be considered alongside macro mean. "
            "A model that averages well but fails on one condition is not deployment-ready."
        )
        return summary

    def generate_markdown_report(self, all_slices: Dict[str, List[Dict[str, Any]]]) -> str:
        """Formats slice analysis into clean Markdown tables with worst-domain highlighting."""
        # Compute worst-domain summary first
        worst_summary = self.compute_summary_with_worst_domain(all_slices)

        lines = [
            "# FORGE-Lite Operational Slice Analysis",
            "",
            "Identifies performance disparities across face scale, quality, pose, duration, "
            "lighting, and skin tone.",
            "",
            "> **newimp.md requirement**: Report WORST-domain performance alongside macro mean.",
            "",
            "## Worst-Domain Summary",
            "",
            f"**Worst condition overall**: `{worst_summary.get('worst_condition_across_all_slices', 'N/A')}`  ",
            f"**Worst condition AUROC**: `{worst_summary.get('worst_condition_auc', float('nan')):.4f}`",
            "",
            "| Dimension | Macro Mean AUC | Worst Bin AUC | Worst Bin Range | Worst Bin N |",
            "|---|---|---|---|---|",
        ]
        for dim, s in worst_summary.get("dimensions", {}).items():
            lines.append(
                f"| `{dim}` | {s['macro_mean_auc']:.4f} | **{s['worst_bin_auc']:.4f}** | "
                f"`{s['worst_bin_range']}` | {s['worst_bin_n']} |"
            )
        lines.append("")
        lines.append(
            "> " + worst_summary.get("note", "")
        )
        lines.append("")
        lines.append("---")
        lines.append("")
        lines.append("## Per-Dimension Breakdowns")
        lines.append("")

        title_map = {
            "face_size": "1. Face Area Ratio (Scale)",
            "blur": "2. Image Blur Score",
            "compression": "3. Compression Artifact Severity",
            "pose_yaw": "4. Head Pose (Yaw Degrees)",
            "video_duration": "5. Video Duration (Seconds)",
            "brightness": "6. Lighting / Brightness (newimp.md §5)",
            "skin_tone_ita": "7. Skin Tone / ITA Proxy (newimp.md §5)",
        }

        for dim, results in all_slices.items():
            # Find worst bin to highlight
            valid_aucs = [r["video_auc"] for r in results if not np.isnan(r.get("video_auc", float("nan")))]
            worst_auc = min(valid_aucs) if valid_aucs else float("nan")

            lines.append(f"### {title_map.get(dim, dim)}")
            lines.append("")
            lines.append("| Bin Range | Samples (Real / Fake) | Video AUC | TPR @ 1% FPR | EER | Abstention Rate |")
            lines.append("|---|---|---|---|---|---|")

            for r in results:
                n_str = f"{r['sample_count']} ({r['reals']} / {r['fakes']})"
                auc_val = r.get("video_auc", float("nan"))
                is_worst = (not np.isnan(auc_val)) and (abs(auc_val - worst_auc) < 1e-6)
                worst_mark = " [WORST]" if is_worst else ""

                auc_str = f"**{auc_val:.4f}**{worst_mark}" if not np.isnan(auc_val) else "N/A"
                tpr_str = f"{r['tpr_at_1pct_fpr'] * 100:.2f}%" if not np.isnan(r['tpr_at_1pct_fpr']) else "N/A"
                eer_str = f"{r['eer'] * 100:.2f}%" if not np.isnan(r['eer']) else "N/A"
                abst_str = f"{r['abstention_rate'] * 100:.1f}%"
                lines.append(f"| `{r['range']}` | {n_str} | {auc_str} | {tpr_str} | {eer_str} | {abst_str} |")

            lines.append("")

        return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="FORGE-Lite Slice Analysis")
    parser.add_argument("--config", default="configs/eval.yaml")
    parser.add_argument("--output_dir", default="reports")
    parser.add_argument("--simulate", action="store_true", help="Simulate slice records")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger.info("Initializing Operational Slice Analysis")

    analyzer = SliceAnalyzer(config_path=args.config)

    if args.simulate:
        rng = np.random.default_rng(888)
        n = 300
        labels = rng.integers(0, 2, size=n)

        for i in range(n):
            face_sz = float(rng.choice([0.03, 0.10, 0.22, 0.45]))
            blur_val = float(rng.uniform(0.05, 0.95))
            comp_val = float(rng.uniform(0.1, 0.9))
            yaw_val = float(rng.choice([8.0, 22.0, 38.0, 65.0]))
            dur_val = float(rng.choice([3.5, 10.0, 30.0, 120.0]))

            # Base score + realistic penalty for tiny faces or high blur
            base_mean = 0.85 if labels[i] == 1 else 0.15
            if face_sz < 0.05 or blur_val > 0.8 or yaw_val > 45.0:
                base_mean = 0.55 if labels[i] == 1 else 0.45

            score = float(np.clip(rng.normal(base_mean, 0.15), 0.0, 1.0))
            abstained = bool(abs(score - 0.5) < 0.08)

            analyzer.add_video_record(
                video_id=f"slice_vid_{i:04d}",
                label=int(labels[i]),
                score=score,
                abstained=abstained,
                face_size=face_sz,
                blur=blur_val,
                compression=comp_val,
                pose_yaw=yaw_val,
                video_duration=dur_val,
            )

    all_slices = analyzer.compute_all_slices()
    md_report = analyzer.generate_markdown_report(all_slices)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "slice_analysis_report.md", "w", encoding="utf-8") as f:
        f.write(md_report)

    with open(out_dir / "slice_analysis_report.json", "w", encoding="utf-8") as f:
        json.dump(all_slices, f, indent=2)

    logger.info("Saved slice analysis report to %s", out_dir / "slice_analysis_report.md")
    print(md_report)


if __name__ == "__main__":
    main()
