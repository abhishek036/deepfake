"""
evaluation/dataset_id_baseline.py
=================================
Quantifies the dataset-identity shortcut in multi-dataset deepfake training sets.

Computes the AUROC achieved by predicting forgery status P(fake | dataset_id)
using ONLY the categorical dataset identifier, without observing any image pixels.

Usage:
  python evaluation/dataset_id_baseline.py --manifest data/manifests/combined.jsonl --out reports/dataset_id_baseline.json
Self-test:
  python evaluation/dataset_id_baseline.py --self-test
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, brier_score_loss, roc_auc_score


def sha256_file(p: Path | str) -> str | None:
    p = Path(p)
    if not p.exists():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(8 << 20):
            h.update(chunk)
    return h.hexdigest()


def bootstrap_ci(y_true: np.ndarray, y_score: np.ndarray, n_boot: int = 1000, seed: int = 42) -> tuple[float, float, float]:
    rng = np.random.RandomState(seed)
    n = len(y_true)
    point = float(roc_auc_score(y_true, y_score))
    scores = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        yt = y_true[idx]
        if len(np.unique(yt)) < 2:
            continue
        scores.append(float(roc_auc_score(yt, y_score[idx])))
    if not scores:
        return point, point, point
    low = float(np.percentile(scores, 2.5))
    high = float(np.percentile(scores, 97.5))
    return point, low, high


def evaluate_dataset_id_baseline(records: list[dict], n_folds: int = 5, seed: int = 42) -> dict:
    rng = random.Random(seed)
    shuffled = records.copy()
    rng.shuffle(shuffled)

    # 5-fold cross-validation out-of-fold posterior estimation
    n = len(shuffled)
    fold_size = n // n_folds
    oof_preds = np.zeros(n, dtype=np.float64)
    y_true = np.zeros(n, dtype=np.int32)

    for i, r in enumerate(shuffled):
        lbl = r.get("label")
        y_true[i] = 1 if lbl in (1, "fake", "1", True) else 0

    for fold in range(n_folds):
        val_start = fold * fold_size
        val_end = n if fold == n_folds - 1 else (fold + 1) * fold_size
        val_idx = list(range(val_start, val_end))
        train_idx = [j for j in range(n) if j < val_start or j >= val_end]

        counts = defaultdict(lambda: [0, 0])
        for j in train_idx:
            ds = str(shuffled[j].get("dataset", "unknown")).lower()
            counts[ds][y_true[j]] += 1

        # Laplace smoothing: (fake + 1) / (total + 2)
        p_fake = {ds: (c[1] + 1) / (sum(c) + 2) for ds, c in counts.items()}
        global_prior = (sum(c[1] for c in counts.values()) + 1) / (sum(sum(c) for c in counts.values()) + 2)

        for j in val_idx:
            ds = str(shuffled[j].get("dataset", "unknown")).lower()
            oof_preds[j] = p_fake.get(ds, global_prior)

    # Overall dataset statistics
    total_counts = defaultdict(lambda: [0, 0])
    for j in range(n):
        ds = str(shuffled[j].get("dataset", "unknown")).lower()
        total_counts[ds][y_true[j]] += 1

    per_dataset_table = {}
    for ds, (n_real, n_fake) in sorted(total_counts.items(), key=lambda x: -(x[1][0] + x[1][1])):
        tot = n_real + n_fake
        per_dataset_table[ds] = {
            "n_total": tot,
            "n_real": n_real,
            "n_fake": n_fake,
            "p_fake": round(n_fake / tot, 4),
        }

    auc_point, auc_low, auc_high = bootstrap_ci(y_true, oof_preds, n_boot=1000, seed=seed)
    y_bin = (oof_preds >= 0.5).astype(np.int32)
    bal_acc = float(balanced_accuracy_score(y_true, y_bin))
    brier = float(brier_score_loss(y_true, oof_preds))

    return {
        "n_samples": n,
        "n_real": int(np.sum(y_true == 0)),
        "n_fake": int(np.sum(y_true == 1)),
        "auroc": round(auc_point, 4),
        "auroc_ci95": [round(auc_low, 4), round(auc_high, 4)],
        "balanced_accuracy": round(bal_acc, 4),
        "brier_score": round(brier, 4),
        "per_dataset": per_dataset_table,
    }


def run_self_test():
    print("[self-test] Generating synthetic confounded multi-dataset records...")
    rng = random.Random(42)
    synthetic = []
    # Dataset A: 95% fakes
    for _ in range(500):
        synthetic.append({"dataset": "A", "label": 1 if rng.random() < 0.95 else 0})
    # Dataset B: 5% fakes (95% reals)
    for _ in range(500):
        synthetic.append({"dataset": "B", "label": 1 if rng.random() < 0.05 else 0})
    # Dataset C: 50% fakes
    for _ in range(200):
        synthetic.append({"dataset": "C", "label": 1 if rng.random() < 0.50 else 0})

    res = evaluate_dataset_id_baseline(synthetic, n_folds=5, seed=42)
    print(f"[self-test] Synthetic AUROC: {res['auroc']} {res['auroc_ci95']}")
    assert res["auroc"] > 0.85, f"Expected AUROC > 0.85 on confounded synthetic data, got {res['auroc']}"
    assert "a" in res["per_dataset"]
    print("[self-test] PASSED.")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Evaluate Dataset-Identity Shortcut Baseline")
    parser.add_argument("--manifest", help="Path to combined manifest JSONL")
    parser.add_argument("--out", default="reports/dataset_id_baseline.json", help="Output ledger JSON")
    parser.add_argument("--sample-rate", type=int, default=1, help="Subsample rate (e.g. 10 for every 10th line)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--self-test", action="store_true", help="Run self-test on synthetic data")
    args = parser.parse_args()

    if args.self_test:
        return run_self_test()

    if not args.manifest:
        parser.error("--manifest is required when not running --self-test")

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    print(f"Reading manifest: {manifest_path} (sampling every {args.sample_rate} lines)...")
    records = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if args.sample_rate > 1 and idx % args.sample_rate != 0:
                continue
            line = line.strip()
            if line:
                records.append(json.loads(line))

    print(f"Evaluating {len(records)} records with 5-fold cross-validation...")
    results = evaluate_dataset_id_baseline(records, n_folds=5, seed=args.seed)

    script_path = Path(__file__).resolve()
    manifest_sha = sha256_file(manifest_path)
    script_sha = sha256_file(script_path)

    ledger = {
        "args": vars(args),
        "script_sha256": script_sha,
        "manifest_sha256": manifest_sha,
        "results": results,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(ledger, indent=2), encoding="utf-8")

    # Re-stamp script sha256 in case file self-reference
    current_sha = sha256_file(script_path)
    if current_sha != script_sha:
        ledger["script_sha256"] = current_sha
        out_path.write_text(json.dumps(ledger, indent=2), encoding="utf-8")

    print(f"Results written to: {out_path}")
    print(f"AUROC: {results['auroc']} {results['auroc_ci95']}")
    print(f"Balanced Accuracy: {results['balanced_accuracy']}")
    print(f"Brier Score: {results['brier_score']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
