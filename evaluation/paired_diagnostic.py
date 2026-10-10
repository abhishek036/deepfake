"""
paired_diagnostic.py  --  put in evaluation/   (numpy, Pillow, opencv-python, scipy)

FF++ native pairing check. For one parent video and one frame there is a crop of the REAL frame and crops of several FAKE
methods, all produced by the same pipeline (same detector, alignment, JPEG). Outside the face the manipulation touched nothing, so

    background difference(real, fake)   should be as small as   background difference(fake_i, fake_j)

(the second is the noise floor: independent re-encodings + landmark jitter). If real-vs-fake is clearly larger, the real frames differ
from the fakes in the BACKGROUND, which is a pipeline confound by definition. Also decides which id of '000_003' is the scene video.

Statistics are computed per PARENT VIDEO (median over its frames/pairs) and tested with a paired Wilcoxon across parents.

One line:
  python -m evaluation.paired_diagnostic --fakes data/ff_native_fakes.jsonl --reals data/ff_native_reals.jsonl --out reports/paired_diagnostic.json
Self-test:  python -m evaluation.paired_diagnostic --self-test
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
from scipy.stats import wilcoxon

SIZE, BORDER, MAX_SHIFT = 224, 8, 4.0


def _ellipse(frac=0.85):
    m = Image.new("L", (SIZE, SIZE), 0)
    r, c = frac * SIZE / 2, SIZE / 2
    ImageDraw.Draw(m).ellipse([c - r, c - r, c + r, c + r], fill=255)
    return np.asarray(m) > 0


ELL = _ellipse()
VALID = ~ELL
VALID[:BORDER] = VALID[-BORDER:] = False
VALID[:, :BORDER] = VALID[:, -BORDER:] = False
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


def bg_mad(pa, pb, align=True):
    """Mean absolute grey-level difference outside the face ellipse, after removing a small global translation."""
    a, b = load_gray(pa), load_gray(pb)
    if align:
        (dx, dy), _ = cv2.phaseCorrelate(_masked(a), _masked(b), WIN)
        if abs(dx) > MAX_SHIFT or abs(dy) > MAX_SHIFT:
            return None
        b = cv2.warpAffine(b, np.float32([[1, 0, -dx], [0, 1, -dy]]), (SIZE, SIZE), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return float(np.abs(a - b)[VALID].mean())


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
                m = bg_mad(f["source_path"], real_idx[key])
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

    groups = defaultdict(dict)                    # (pair, frame) -> method -> path
    for f in fakes:
        groups[(f["video_id"], f["frame_idx"])][f["method"]] = f["source_path"]
    keys = [k for k, v in groups.items() if (k[0].split("_")[part], k[1]) in real_idx and len(v) >= 2]
    rng.shuffle(keys)
    keys = keys[: a.n_groups]
    print(f"{len(keys)} (pair, frame) groups with a real frame and >=2 fake methods")

    per_parent, per_method, skipped = defaultdict(lambda: {"rf": [], "ff": []}), defaultdict(list), 0
    for k in keys:
        parent = k[0].split("_")[part]
        rp, ms = real_idx[(parent, k[1])], groups[k]
        rf = {m: bg_mad(p, rp) for m, p in ms.items()}
        ff = [bg_mad(ms[i], ms[j]) for i, j in itertools.combinations(sorted(ms), 2)]
        rf_ok = [v for v in rf.values() if v is not None]
        ff_ok = [v for v in ff if v is not None]
        if not rf_ok or not ff_ok:
            skipped += 1
            continue
        for m, v in rf.items():
            if v is not None:
                per_method[m].append(v)
        per_parent[parent]["rf"].append(float(np.mean(rf_ok)))
        per_parent[parent]["ff"].append(float(np.mean(ff_ok)))
    rf_p = np.array([np.median(v["rf"]) for v in per_parent.values()])
    ff_p = np.array([np.median(v["ff"]) for v in per_parent.values()])
    if len(rf_p) < 20:
        sys.exit(f"only {len(rf_p)} parents usable (skipped {skipped} groups, mostly misaligned); not enough for a verdict")
    ratio = float(np.median(rf_p) / np.median(ff_p))
    p = float(wilcoxon(rf_p - ff_p).pvalue)
    verdict = ("MATCHED: real-vs-fake background difference is at the fake-vs-fake noise floor" if abs(ratio - 1) <= 0.10 else
               "REAL DIFFERS from fakes outside the face (pipeline confound even in the native pairing)" if ratio > 1.10 else
               "real is closer to the fakes than fakes are to each other (no confound)")
    res = {"parent_id_part": part, "parent_detection": table, "n_groups_used": len(keys) - skipped, "n_parents": int(len(rf_p)),
           "skipped_groups": skipped, "real_vs_fake_bg_mad_median_q25_q75": q(rf_p), "fake_vs_fake_bg_mad_median_q25_q75": q(ff_p),
           "ratio_of_medians": ratio, "wilcoxon_p": p, "frac_parents_rf_gt_ff": float((rf_p > ff_p).mean()),
           "per_method_real_vs_fake_median": {m: float(np.median(v)) for m, v in per_method.items()}, "verdict": verdict}
    print(json.dumps(res, indent=1))
    ledger = {"args": {k: str(v) for k, v in vars(a).items()}, "script_sha256": sha256_file(__file__),
              "fakes_sha256": sha256_file(a.fakes), "reals_sha256": sha256_file(a.reals), "result": res}
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(ledger, indent=1))
        print("ledger ->", a.out)
    return res


def self_test():
    rng = np.random.default_rng(0)
    yy, xx = np.indices((SIZE, SIZE))

    def build(real_shift):
        root = Path(tempfile.mkdtemp())
        reals, fakes = [], []
        for pidx in range(40):
            for fr in (0, 25):
                base = 120 + 50 * np.sin((xx + pidx * 5) / 11.0) * np.cos((yy + fr) / 13.0)
                def save(arr, p):
                    Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).save(p, quality=95)
                rp = root / f"r_{pidx}_{fr}.jpg"
                save(base + real_shift + rng.normal(0, 2, base.shape), rp)
                reals.append({"label": 0, "crop_path": str(rp), "source_path": f"{pidx:03d}.mp4", "video_id": f"{pidx:03d}", "frame_idx": fr, "split": "train"})
                for m in ("Deepfakes", "FaceSwap", "Face2Face"):
                    arr = base + rng.normal(0, 2, base.shape)
                    arr[ELL] += 25                       # manipulation only inside the face
                    fp = root / f"f_{pidx}_{fr}_{m}.jpg"
                    save(arr, fp)
                    fakes.append({"label": 1, "source_path": str(fp), "method": m, "video_id": f"{pidx:03d}_{pidx + 100:03d}", "frame_idx": fr})
        (root / "r.jsonl").write_text("\n".join(json.dumps(x) for x in reals))
        (root / "f.jsonl").write_text("\n".join(json.dumps(x) for x in fakes))
        return argparse.Namespace(fakes=str(root / "f.jsonl"), reals=str(root / "r.jsonl"), parent_part="auto", n_groups=200, seed=0, out=None)

    clean = run(build(0.0))
    conf = run(build(9.0))
    ok = clean["verdict"].startswith("MATCHED") and conf["verdict"].startswith("REAL DIFFERS") and clean["parent_id_part"] == 0
    print("self-test", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fakes")
    ap.add_argument("--reals")
    ap.add_argument("--parent-part", default="auto", choices=["auto", "0", "1"])
    ap.add_argument("--n-groups", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="reports/paired_diagnostic.json")
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
