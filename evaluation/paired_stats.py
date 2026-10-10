"""
evaluation/paired_stats.py
Paired diagnostic and equivalence testing suite for deepfake real-vs-fake pairings.

Pre-registration tolerances (PREREG.md):
- P-S1: Paired MAD of face crop centers <= 4.0 px
- Background equivalence margin: +/- 1.0 gray level (TOST)
- Ratio equivalence bounds: 0.05, 0.03, 0.02
- Equivalence decision: TOST paired Wilcoxon / t-test at alpha=0.05

Usage:
  python -m evaluation.paired_stats --fakes data/ff_native_fakes.jsonl --reals data/ff_native_reals.jsonl --out reports/paired_stats.json
Self-test:
  python -m evaluation.paired_stats --self-test
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
from scipy.stats import t as student_t, wilcoxon

SIZE, BORDER, MAX_SHIFT = 224, 8, 4.0


def _ellipse(frac=0.85):
    m = Image.new("L", (SIZE, SIZE), 0)
    r, c = frac * SIZE / 2, SIZE / 2
    ImageDraw.Draw(m).ellipse([c - r, c - r, c + r, c + r], fill=255)
    return np.asarray(m) > 0


ELL = _ellipse()
VALID = ~ELL
VALID[:BORDER] = VALID[-BORDER:] = False
VALID[:, :BORDER] = VALID[-BORDER:] = False
WIN = cv2.createHanningWindow((SIZE, SIZE), cv2.CV_32F)


def sha256_file(p):
    p = Path(p)
    if not p.exists():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while c := f.read(8 << 20):
            h.update(c)
    return h.hexdigest()


def read_jsonl(p):
    with open(p, "r", encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def load_gray(path):
    im = Image.open(path).convert("L")
    if im.size != (SIZE, SIZE):
        im = im.resize((SIZE, SIZE), Image.LANCZOS)
    return np.asarray(im, dtype=np.float32)


def _masked(a):
    b = a.copy()
    b[ELL] = a[~ELL].mean()
    return b


def bg_mad_and_shift(pa, pb, align=True):
    """
    Mean absolute grey-level difference outside the face ellipse,
    plus global (dx, dy) translation shift.
    """
    a, b = load_gray(pa), load_gray(pb)
    dx, dy = 0.0, 0.0
    if align:
        (dx, dy), _ = cv2.phaseCorrelate(_masked(a), _masked(b), WIN)
        if abs(dx) > MAX_SHIFT or abs(dy) > MAX_SHIFT:
            return None, (dx, dy)
        b = cv2.warpAffine(
            b,
            np.float32([[1, 0, -dx], [0, 1, -dy]]),
            (SIZE, SIZE),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )
    mad = float(np.abs(a - b)[VALID].mean())
    return mad, (float(dx), float(dy))


def tost_equivalence(diffs: np.ndarray, bound: float = 1.0, alpha: float = 0.05):
    """
    Two One-Sided Tests (TOST) for paired difference equivalence within [-bound, +bound].
    Tests H0: |mean(diffs)| >= bound  vs  H1: |mean(diffs)| < bound.
    """
    diffs = np.asarray(diffs, dtype=float)
    n = len(diffs)
    if n < 5:
        return {"equivalent": False, "p_upper": 1.0, "p_lower": 1.0, "p_tost": 1.0}
    mean_d = float(np.mean(diffs))
    std_d = float(np.std(diffs, ddof=1))
    se = std_d / math.sqrt(n) if std_d > 1e-12 else 1e-12
    df = n - 1

    # Test 1: mean > -bound  (t1 = (mean - (-bound)) / se)
    t1 = (mean_d - (-bound)) / se
    p1 = float(1.0 - student_t.cdf(t1, df))

    # Test 2: mean < bound   (t2 = (bound - mean) / se)
    t2 = (bound - mean_d) / se
    p2 = float(1.0 - student_t.cdf(t2, df))

    p_tost = max(p1, p2)
    return {
        "bound": bound,
        "mean_diff": mean_d,
        "std_diff": std_d,
        "se": se,
        "t1": float(t1),
        "p_lower": p1,
        "t2": float(t2),
        "p_upper": p2,
        "p_tost": p_tost,
        "equivalent": bool(p_tost < alpha),
    }


def q(v):
    v = np.asarray(v, dtype=float)
    return [float(np.median(v)), float(np.percentile(v, 25)), float(np.percentile(v, 75))]


def detect_parent_part(fakes, real_idx, rng, n=250):
    out = {}
    for part in (0, 1):
        vals = []
        for r in rng.permutation(len(fakes))[: n * 4]:
            f = fakes[int(r)]
            key = (f["video_id"].split("_")[part], f["frame_idx"])
            if key in real_idx:
                m, _ = bg_mad_and_shift(f["source_path"], real_idx[key])
                if m is not None:
                    vals.append(m)
            if len(vals) >= n:
                break
        out[part] = {"n": len(vals), "median_bg_mad": float(np.median(vals)) if vals else None}
    ok = [p for p in (0, 1) if out[p]["median_bg_mad"] is not None]
    return (min(ok, key=lambda p: out[p]["median_bg_mad"]) if ok else None), out


def run(a):
    rng = np.random.default_rng(a.seed)
    fakes, reals = read_jsonl(a.fakes), read_jsonl(a.reals)
    real_idx = {(r["video_id"], r["frame_idx"]): r["crop_path"] for r in reals if Path(r["crop_path"]).exists()}
    fakes = [f for f in fakes if Path(f["source_path"]).exists()]
    print(f"{len(real_idx)} real crops, {len(fakes)} fake crops on disk")
    part, table = detect_parent_part(fakes, real_idx, rng) if a.parent_part == "auto" else (int(a.parent_part), {})
    print("parent-id detection (median background MAD, lower = right parent):", table, "-> using id part", part)
    if part is None:
        sys.exit("no (parent, frame) overlap between fakes and reals; check frame_idx sampling in the manifests")

    groups = defaultdict(dict)
    for f in fakes:
        groups[(f["video_id"], f["frame_idx"])][f["method"]] = f["source_path"]
    keys = [k for k, v in groups.items() if (k[0].split("_")[part], k[1]) in real_idx and len(v) >= 2]
    rng.shuffle(keys)
    keys = keys[: a.n_groups]
    print(f"{len(keys)} (pair, frame) groups with a real frame and >=2 fake methods")

    per_parent = defaultdict(lambda: {"rf": [], "ff": [], "shift_mag": []})
    per_method = defaultdict(list)
    skipped = 0

    for k in keys:
        parent = k[0].split("_")[part]
        rp, ms = real_idx[(parent, k[1])], groups[k]

        rf_vals, rf_shifts = [], []
        for m, p in ms.items():
            mad, (dx, dy) = bg_mad_and_shift(p, rp)
            if mad is not None:
                rf_vals.append(mad)
                rf_shifts.append(math.hypot(dx, dy))
                per_method[m].append(mad)

        ff_vals = []
        for i, j in itertools.combinations(sorted(ms), 2):
            mad, _ = bg_mad_and_shift(ms[i], ms[j])
            if mad is not None:
                ff_vals.append(mad)

        if not rf_vals or not ff_vals:
            skipped += 1
            continue

        per_parent[parent]["rf"].append(float(np.mean(rf_vals)))
        per_parent[parent]["ff"].append(float(np.mean(ff_vals)))
        per_parent[parent]["shift_mag"].append(float(np.mean(rf_shifts)))

    rf_p = np.array([np.median(v["rf"]) for v in per_parent.values()])
    ff_p = np.array([np.median(v["ff"]) for v in per_parent.values()])
    shift_p = np.array([np.median(v["shift_mag"]) for v in per_parent.values()])

    if len(rf_p) < 20:
        sys.exit(f"only {len(rf_p)} parents usable (skipped {skipped} groups); not enough for statistical verdict")

    diffs = rf_p - ff_p
    ratio = float(np.median(rf_p) / np.median(ff_p))
    wilc_p = float(wilcoxon(diffs).pvalue) if np.any(diffs != 0) else 1.0

    # TOST equivalence testing
    tost_1gray = tost_equivalence(diffs, bound=1.0, alpha=0.05)
    tost_bounds = {
        "margin_0.05": bool(abs(ratio - 1.0) <= 0.05),
        "margin_0.03": bool(abs(ratio - 1.0) <= 0.03),
        "margin_0.02": bool(abs(ratio - 1.0) <= 0.02),
    }

    # P-S1 check: center MAD <= 4 px
    center_mad = float(np.median(shift_p))
    p_s1_passed = bool(center_mad <= 4.0)

    # Verdict
    if tost_1gray["equivalent"] and p_s1_passed:
        verdict = "MATCHED: background difference statistically equivalent to fake-vs-fake noise floor within +/- 1 gray level"
    elif ratio > 1.10 or not tost_1gray["equivalent"]:
        verdict = "CONFOUNDED: background difference significantly exceeds noise floor (pipeline shortcut present)"
    else:
        verdict = "BORDERLINE: within 10% ratio but fails formal 1 gray level equivalence"

    res = {
        "parent_id_part": part,
        "parent_detection": table,
        "n_groups_used": len(keys) - skipped,
        "n_parents": int(len(rf_p)),
        "skipped_groups": skipped,
        "real_vs_fake_bg_mad_median_q25_q75": q(rf_p),
        "fake_vs_fake_bg_mad_median_q25_q75": q(ff_p),
        "ratio_of_medians": ratio,
        "wilcoxon_p": wilc_p,
        "frac_parents_rf_gt_ff": float((rf_p > ff_p).mean()),
        "p_s1_center_mad_px": center_mad,
        "p_s1_passed": p_s1_passed,
        "tost_1_gray_level": tost_1gray,
        "ratio_equivalence_tolerances": tost_bounds,
        "per_method_real_vs_fake_median": {m: float(np.median(v)) for m, v in per_method.items()},
        "verdict": verdict,
    }
    print(json.dumps(res, indent=1))

    ledger = {
        "args": {k: str(v) for k, v in vars(a).items()},
        "script_sha256": sha256_file(__file__),
        "fakes_sha256": sha256_file(a.fakes),
        "reals_sha256": sha256_file(a.reals),
        "result": res,
    }
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(ledger, indent=1))
        print("ledger ->", a.out)
    return res


def self_test():
    rng = np.random.default_rng(0)
    yy, xx = np.indices((SIZE, SIZE))

    def build(real_shift, root):
        reals, fakes = [], []
        for pidx in range(40):
            for fr in (0, 25):
                base = 120 + 50 * np.sin((xx + pidx * 5) / 11.0) * np.cos((yy + fr) / 13.0)

                def save(arr, p):
                    Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).save(p, quality=95)

                rp = root / f"r_{pidx}_{fr}.jpg"
                save(base + real_shift + rng.normal(0, 2, base.shape), rp)
                reals.append({
                    "label": 0,
                    "crop_path": str(rp),
                    "source_path": f"{pidx:03d}.mp4",
                    "video_id": f"{pidx:03d}",
                    "frame_idx": fr,
                    "split": "train",
                })
                for m in ("Deepfakes", "FaceSwap", "Face2Face"):
                    arr = base + rng.normal(0, 2, base.shape)
                    arr[ELL] += 25  # manipulation inside face only
                    fp = root / f"f_{pidx}_{fr}_{m}.jpg"
                    save(arr, fp)
                    fakes.append({
                        "label": 1,
                        "source_path": str(fp),
                        "method": m,
                        "video_id": f"{pidx:03d}_{pidx + 100:03d}",
                        "frame_idx": fr,
                    })
        rf = root / f"r_{real_shift}.jsonl"
        ff = root / f"f_{real_shift}.jsonl"
        rf.write_text("\n".join(json.dumps(x) for x in reals))
        ff.write_text("\n".join(json.dumps(x) for x in fakes))
        return argparse.Namespace(
            fakes=str(ff),
            reals=str(rf),
            parent_part="auto",
            n_groups=200,
            seed=0,
            out=str(root / f"out_{real_shift}.json"),
        )

    tmp = Path(tempfile.mkdtemp())
    clean = run(build(0.0, tmp))
    conf = run(build(9.0, tmp))

    ok_clean = clean["verdict"].startswith("MATCHED") and clean["p_s1_passed"]
    ok_conf = conf["verdict"].startswith("CONFOUNDED")
    ok = ok_clean and ok_conf
    print("paired_stats self-test", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description="Paired statistical diagnostic and equivalence test")
    ap.add_argument("--fakes")
    ap.add_argument("--reals")
    ap.add_argument("--parent-part", default="auto", choices=["auto", "0", "1"])
    ap.add_argument("--n-groups", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="reports/paired_stats.json")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    if not (a.fakes and a.reals):
        ap.error("--fakes and --reals are required")
    run(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
