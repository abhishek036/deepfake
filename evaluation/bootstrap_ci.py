"""
evaluation/bootstrap_ci.py
Bootstrap confidence intervals for all FORGE-Lite metrics.

PRD requirement: report 95% CI via bootstrap (n=1000 resamples) for
video AUC, cross-domain macro-AUROC, TPR@1%FPR, EER, AP.

newimp.md §3 / §8 additions:
  - Source-group-level resampling: resample at source group, not individual
    videos. Correlated frames/clips from the same source are NOT independent
    test samples. Thousands of correlated frames ≠ thousands of independent cases.
  - Paired source-group comparison for ablation comparisons.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

logger = logging.getLogger(__name__)


def bootstrap_ci(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_resamples: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> dict[str, float]:
    """
    Compute bootstrap confidence interval for a scalar metric.

    Args:
        y_true: binary labels
        y_scores: predicted scores
        metric_fn: function(y_true, y_scores) → float
        n_resamples: bootstrap samples
        confidence_level: CI width (e.g., 0.95)
        seed: reproducibility

    Returns:
        {metric, lower_ci, upper_ci, std}
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    bootstrap_scores = []

    for _ in range(n_resamples):
        indices = rng.integers(0, n, size=n)
        y_t = y_true[indices]
        y_s = y_scores[indices]
        if len(np.unique(y_t)) < 2:
            continue
        try:
            bootstrap_scores.append(metric_fn(y_t, y_s))
        except Exception:
            continue

    if not bootstrap_scores:
        return {"metric": float("nan"), "lower_ci": float("nan"),
                "upper_ci": float("nan"), "std": float("nan")}

    scores = np.array(bootstrap_scores)
    alpha = 1 - confidence_level
    lower = float(np.percentile(scores, 100 * alpha / 2))
    upper = float(np.percentile(scores, 100 * (1 - alpha / 2)))
    return {
        "metric": float(np.mean(scores)),
        "lower_ci": lower,
        "upper_ci": upper,
        "std": float(np.std(scores)),
        "n_resamples": n_resamples,
        "confidence_level": confidence_level,
    }


def compute_all_bootstrap_cis(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    n_resamples: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> dict[str, dict]:
    """
    Compute bootstrap CIs for all standard metrics.
    """
    from evaluation.evaluate_video import compute_tpr_at_fpr, compute_eer

    metrics = {}

    metrics["video_auc"] = bootstrap_ci(
        y_true, y_scores,
        lambda t, s: roc_auc_score(t, s),
        n_resamples, confidence_level, seed
    )

    metrics["average_precision"] = bootstrap_ci(
        y_true, y_scores,
        lambda t, s: average_precision_score(t, s),
        n_resamples, confidence_level, seed
    )

    metrics["tpr_at_1pct_fpr"] = bootstrap_ci(
        y_true, y_scores,
        lambda t, s: compute_tpr_at_fpr(t, s, 0.01),
        n_resamples, confidence_level, seed
    )

    metrics["eer"] = bootstrap_ci(
        y_true, y_scores,
        lambda t, s: compute_eer(t, s),
        n_resamples, confidence_level, seed
    )

    return metrics


# Alias for singular form
compute_all_bootstrap_ci = compute_all_bootstrap_cis


# ─────────────────────────────────────────────────────────────────────────────
# Source-group-level bootstrap (newimp.md §3, §8)
# ─────────────────────────────────────────────────────────────────────────────

def bootstrap_ci_grouped_by_source(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    source_groups: list[str],
    metric_fn=None,
    n_resamples: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> dict[str, float]:
    """
    Bootstrap CI that resamples at SOURCE GROUP level.

    Per newimp.md §3 and §8:
      "Correlated frames/clips from the same source are NOT independent test samples.
       Thousands of correlated frames do not equal thousands of independent cases."

    Args:
        y_true: binary labels (video level)
        y_scores: predicted scores
        source_groups: source group ID for each video
        metric_fn: callable(y_true, y_scores) → float. Defaults to roc_auc_score.
        n_resamples: number of bootstrap resamples
        confidence_level: CI width
        seed: reproducibility

    Returns:
        dict with metric, lower_ci, upper_ci, std, n_unique_sources
    """
    if metric_fn is None:
        metric_fn = roc_auc_score

    unique_sources = list(set(source_groups))
    n_sources = len(unique_sources)

    if n_sources < 5:
        logger.warning(
            "Source-group bootstrap: only %d unique sources — CI may be unreliable. "
            "At least 30 independent sources recommended.",
            n_sources,
        )

    # Build source → indices map
    src_map: dict[str, list[int]] = {}
    for i, sg in enumerate(source_groups):
        src_map.setdefault(sg, []).append(i)

    rng = np.random.default_rng(seed)
    bootstrap_values: list[float] = []

    for _ in range(n_resamples):
        # Resample SOURCES (not individual videos)
        sampled = rng.choice(unique_sources, size=n_sources, replace=True)
        indices = [i for s in sampled for i in src_map[s]]
        b_true = y_true[indices]
        b_scores = y_scores[indices]
        if len(set(b_true.tolist())) < 2:
            continue
        try:
            bootstrap_values.append(float(metric_fn(b_true, b_scores)))
        except Exception:
            continue

    if not bootstrap_values:
        return {
            "metric": float("nan"), "lower_ci": float("nan"),
            "upper_ci": float("nan"), "std": float("nan"),
            "n_unique_sources": n_sources,
            "resampling_unit": "source_group",
        }

    arr = np.array(bootstrap_values)
    alpha = 1 - confidence_level
    return {
        "metric": float(arr.mean()),
        "lower_ci": float(np.percentile(arr, 100 * alpha / 2)),
        "upper_ci": float(np.percentile(arr, 100 * (1 - alpha / 2))),
        "std": float(arr.std()),
        "n_resamples": n_resamples,
        "confidence_level": confidence_level,
        "n_unique_sources": n_sources,
        "resampling_unit": "source_group",
    }


def paired_source_group_comparison(
    y_true: np.ndarray,
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    source_groups: list[str],
    n_resamples: int = 1000,
    seed: int = 42,
) -> dict[str, float]:
    """
    Paired bootstrap test comparing two systems on the same source-grouped data.

    Per newimp.md §8: "Run multiple seeds for finalists; use paired source-group
    bootstrap comparisons where appropriate."

    Returns p-value (one-sided: P(A_auroc > B_auroc)) and CI on AUROC difference.
    """
    unique_sources = list(set(source_groups))
    n_sources = len(unique_sources)
    src_map: dict[str, list[int]] = {}
    for i, sg in enumerate(source_groups):
        src_map.setdefault(sg, []).append(i)

    rng = np.random.default_rng(seed)
    deltas: list[float] = []

    for _ in range(n_resamples):
        sampled = rng.choice(unique_sources, size=n_sources, replace=True)
        indices = [i for s in sampled for i in src_map[s]]
        b_true = y_true[indices]
        b_a = scores_a[indices]
        b_b = scores_b[indices]
        if len(set(b_true.tolist())) < 2:
            continue
        try:
            auc_a = float(roc_auc_score(b_true, b_a))
            auc_b = float(roc_auc_score(b_true, b_b))
            deltas.append(auc_a - auc_b)
        except Exception:
            continue

    if not deltas:
        return {"delta_auroc": float("nan"), "p_value": float("nan"),
                "lower_ci": float("nan"), "upper_ci": float("nan")}

    arr = np.array(deltas)
    # Observed delta
    try:
        obs_auc_a = float(roc_auc_score(y_true, scores_a))
        obs_auc_b = float(roc_auc_score(y_true, scores_b))
        obs_delta = obs_auc_a - obs_auc_b
    except Exception:
        obs_delta = float("nan")

    # p-value: fraction of bootstrap samples where A does not outperform B
    p_value = float((arr <= 0).mean())

    return {
        "delta_auroc": obs_delta,
        "bootstrap_mean_delta": float(arr.mean()),
        "lower_ci": float(np.percentile(arr, 2.5)),
        "upper_ci": float(np.percentile(arr, 97.5)),
        "p_value_a_beats_b": p_value,
        "n_unique_sources": n_sources,
        "n_resamples": n_resamples,
    }


def print_ci_table(metrics: dict[str, dict]) -> None:
    """Print a formatted CI table."""
    print("\n" + "─" * 70)
    print(f"{'Metric':<25} {'Value':>10} {'95% CI':>22} {'Std':>8}")
    print("─" * 70)
    for name, ci in metrics.items():
        val = ci.get("metric", float("nan"))
        lo = ci.get("lower_ci", float("nan"))
        hi = ci.get("upper_ci", float("nan"))
        std = ci.get("std", float("nan"))
        print(f"{name:<25} {val:>10.4f} ({lo:.4f}, {hi:.4f})      {std:>8.4f}")
    print("─" * 70 + "\n")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-json", required=True, help="ForgeEvaluator saved results JSON")
    parser.add_argument("--n-resamples", type=int, default=1000)
    parser.add_argument("--output", default="reports/bootstrap_ci.json")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    with open(args.results_json) as f:
        results = json.load(f)

    y_true = np.array([r["label"] for r in results])
    y_scores = np.array([r["score"] for r in results])

    ci_metrics = compute_all_bootstrap_cis(
        y_true, y_scores, n_resamples=args.n_resamples
    )

    print_ci_table(ci_metrics)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(ci_metrics, f, indent=2)
    logger.info("Bootstrap CI saved: %s", args.output)


if __name__ == "__main__":
    main()
