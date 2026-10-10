"""
region_probe.py  --  put in evaluation/   (standalone: needs numpy, Pillow, scikit-learn)

Question it answers: is real-vs-fake separability coming from the MANIPULATED region, or from places the
manipulation never touched (background corners)?  For face-swap / reenactment fakes the corners are an
unmanipulated negative control: any separability there is confound (compression history, scene, crop), not forgery.

Regions of the canonical 224x224 image (same pipeline for both classes: centre square crop -> LANCZOS 224):
  full        whole image
  face        central 112x112 square (inside the 85% face ellipse)
  background  four 40x40 corner patches (all outside the ellipse), features averaged

Features per region (8): log JPEG-90 bytes/pixel, log Laplacian variance, high-frequency FFT ratio,
gray std, gray mean, R/G/B means.  Probe = grouped-CV logistic regression (groups = source video),
pooled out-of-fold AUROC with group-bootstrap 95% CI.

Example (one line):
  python -m evaluation.region_probe --real-manifest data/ffpp_matched_reals.jsonl --real-split any --fake-manifest data/df40_ff_subset.jsonl --families composited reenactment --out reports/region_probe.json
"""
from __future__ import annotations

import argparse
import io
import json
import math
import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SIZE, PATCH = 224, 40
FEATS = ["log_jpeg90_bpp", "log_lap_var", "hf_ratio", "gray_std", "gray_mean", "r_mean", "g_mean", "b_mean"]
NEG_CONTROL_DEFAULT = ("composited", "reenactment")


def read_jsonl(p):
    with open(p, "r", encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def canonicalize(path):
    im = Image.open(path).convert("RGB")
    w, h = im.size
    s = min(w, h)
    l, t = (w - s) // 2, (h - s) // 2
    return np.asarray(im.crop((l, t, l + s, t + s)).resize((SIZE, SIZE), Image.LANCZOS))


def region_features(a):
    h, w = a.shape[:2]
    buf = io.BytesIO()
    Image.fromarray(a).save(buf, format="JPEG", quality=90)
    bpp = buf.tell() / (h * w)
    g = np.asarray(Image.fromarray(a).convert("L"), dtype=np.float64)
    lap = -4 * g[1:-1, 1:-1] + g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:]
    p = np.abs(np.fft.fftshift(np.fft.fft2(g - g.mean()))) ** 2
    yy, xx = np.indices(p.shape)
    r = np.hypot(yy - h / 2, xx - w / 2) / h
    hf = p[r > 0.25].sum() / (p.sum() + 1e-9)
    m = a.reshape(-1, 3).mean(0)
    return np.array([math.log(bpp), math.log(lap.var() + 1e-6), hf, g.std(), g.mean(), m[0], m[1], m[2]])


def image_regions(a):
    c = SIZE // 2
    corners = [a[:PATCH, :PATCH], a[:PATCH, -PATCH:], a[-PATCH:, :PATCH], a[-PATCH:, -PATCH:]]
    return {"full": region_features(a),
            "face": region_features(a[c - 56:c + 56, c - 56:c + 56]),
            "background": np.mean([region_features(np.ascontiguousarray(q)) for q in corners], axis=0)}


def load_real(manifests, split):
    out = []
    for m in manifests:
        for o in read_jsonl(m):
            if o.get("label", 0) != 0 or (split != "any" and o.get("split") != split):
                continue
            p = o.get("crop_path") or o.get("source_path") or o.get("path")
            if p and Path(p).exists():
                grp = o.get("source_path") if o.get("crop_path") else (o.get("video_id") or str(Path(p).parent))
                out.append({"path": str(p), "group": f"0:{grp}"})
    return out


def load_fake(manifest):
    out = []
    for o in read_jsonl(manifest):
        p = o.get("source_path") or o.get("path")
        if o.get("label", 1) == 1 and p and Path(p).exists():
            out.append({"path": str(p), "group": f"1:{o.get('method')}/{o.get('video_id') or Path(p).parent}",
                        "method": o.get("method", "?"), "family": o.get("forgery_type", "?")})
    return out


def one_per_group(recs, rng):
    by = {}
    for r in recs:
        by.setdefault(r["group"], []).append(r)
    return [by[g][rng.integers(len(by[g]))] for g in sorted(by)]


def round_robin(recs, n, rng):
    by = {}
    for r in recs:
        by.setdefault(r["method"], []).append(r)
    for m in by:
        rng.shuffle(by[m])
    out, i = [], 0
    while len(out) < n and any(i < len(v) for v in by.values()):
        for m in sorted(by):
            if i < len(by[m]) and len(out) < n:
                out.append(by[m][i])
        i += 1
    return out


def cv_auc(X, y, groups, rng, n_boot=300):
    k = int(min(5, len(set(groups))))
    oof = np.full(len(y), np.nan)
    for tr, te in GroupKFold(n_splits=k).split(X, y, groups):
        if len(set(y[tr])) < 2:
            continue
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        clf.fit(X[tr], y[tr])
        oof[te] = clf.predict_proba(X[te])[:, 1]
    ok = ~np.isnan(oof)
    y, oof, g = y[ok], oof[ok], np.asarray(groups)[ok]
    auc = float(roc_auc_score(y, oof))
    ug = np.unique(g)
    idx = {k: np.where(g == k)[0] for k in ug}
    boots = []
    for _ in range(n_boot):
        pick = rng.choice(ug, len(ug), replace=True)
        ii = np.concatenate([idx[p] for p in pick])
        if len(set(y[ii])) == 2:
            boots.append(roc_auc_score(y[ii], oof[ii]))
    lo, hi = (np.percentile(boots, [2.5, 97.5]) if boots else (float("nan"), float("nan")))
    return auc, float(lo), float(hi)


def folded_uni(X, y):
    return {n: float(max(roc_auc_score(y, X[:, j]), 1 - roc_auc_score(y, X[:, j]))) for j, n in enumerate(FEATS)}


def run(a):
    rng = np.random.default_rng(a.seed)
    reals = one_per_group(load_real(a.real_manifest, a.real_split), rng)
    rng.shuffle(reals)
    reals = reals[: a.n]
    fakes = load_fake(a.fake_manifest)
    fam_counts = Counter(f["family"] for f in fakes)
    print(f"reals (1 frame/video): {len(reals)}   fake families on disk: {dict(fam_counts)}")
    if len(reals) < 60:
        print("WARNING: few reals; extract from more videos (--n-videos) before trusting any CI")
    rf = [image_regions(canonicalize(r["path"])) for r in reals]
    ledger = {"args": {k: str(v) for k, v in vars(a).items()}, "n_real": len(reals), "families": {}}
    fams = a.families or sorted(fam_counts)
    print(f"\n{'family':16s}{'n_f':>5s}  {'full':>22s}  {'face':>22s}  {'background':>22s}  verdict")
    for fam in fams:
        pool = [f for f in fakes if f["family"] == fam]
        if not pool:
            print(f"{fam:16s} no fakes found")
            continue
        pick = round_robin(one_per_group(pool, rng) if a.one_per_group else pool, a.n, rng)
        ff = [image_regions(canonicalize(p["path"])) for p in pick]
        y = np.r_[np.zeros(len(rf)), np.ones(len(ff))]
        grp = np.array([r["group"] for r in reals] + [p["group"] for p in pick])
        res, cells = {}, []
        for reg in ("full", "face", "background"):
            X = np.stack([d[reg] for d in rf + ff])
            auc, lo, hi = cv_auc(X, y, grp, rng)
            res[reg] = {"auroc": auc, "ci95": [lo, hi], "univariate": folded_uni(X, y)}
            cells.append(f"{auc:.3f} [{lo:.2f},{hi:.2f}]")
        bg = res["background"]
        if fam in a.neg_control_families:
            verdict = ("CONFOUNDED (corners separate)" if bg["ci95"][0] > 0.55 else
                       "clean control; any signal is in the manipulated region" if bg["ci95"][1] <= 0.60 else "borderline")
        else:
            verdict = "no negative control (manipulation may touch whole image)"
        print(f"{fam:16s}{len(ff):5d}  {cells[0]:>22s}  {cells[1]:>22s}  {cells[2]:>22s}  {verdict}")
        res["verdict"], res["n_fake"] = verdict, len(ff)
        ledger["families"][fam] = res
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(ledger, indent=1))
        print("\nledger ->", a.out)
    return ledger


def self_test():
    rng = np.random.default_rng(0)
    root = Path(tempfile.mkdtemp())
    yy, xx = np.indices((256, 256))

    def img(noise, smooth_face):
        base = 128 + 40 * np.sin(xx / 17.0) * np.cos(yy / 23.0)
        a = base[..., None] + rng.normal(0, noise, (256, 256, 3))
        if smooth_face:
            c = a[64:192, 64:192].copy()
            a[64:192, 64:192] = (c + np.roll(c, 1, 0) + np.roll(c, 1, 1) + np.roll(c, -1, 0)) / 4
        return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))

    reals, fakes = [], []
    for i in range(80):
        p = root / f"r{i}.png"
        img(10, False).save(p)
        reals.append({"label": 0, "split": "any", "crop_path": str(p), "source_path": f"v{i}.mp4"})
    for i in range(80):
        p = root / f"a{i}.png"
        img(10, True).save(p)      # swap-like: only the face region differs
        fakes.append({"label": 1, "method": f"m{i % 4}", "forgery_type": "composited", "video_id": f"a{i}", "source_path": str(p)})
        p = root / f"b{i}.png"
        img(4, False).save(p)      # confounded: whole image statistics differ
        fakes.append({"label": 1, "method": f"n{i % 4}", "forgery_type": "reenactment", "video_id": f"b{i}", "source_path": str(p)})
    (root / "real.jsonl").write_text("\n".join(json.dumps(x) for x in reals))
    (root / "fake.jsonl").write_text("\n".join(json.dumps(x) for x in fakes))
    a = argparse.Namespace(real_manifest=[str(root / "real.jsonl")], real_split="any", fake_manifest=str(root / "fake.jsonl"),
                           families=["composited", "reenactment"], neg_control_families=NEG_CONTROL_DEFAULT, n=80, seed=0,
                           one_per_group=True, out=None)
    led = run(a)
    ok = (led["families"]["composited"]["background"]["auroc"] < 0.60 and led["families"]["composited"]["face"]["auroc"] > 0.80
          and led["families"]["reenactment"]["background"]["auroc"] > 0.80)
    print("self-test", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real-manifest", action="append", default=[])
    ap.add_argument("--real-split", default="any")
    ap.add_argument("--fake-manifest")
    ap.add_argument("--families", nargs="*", default=None, help="forgery_type values; default = all on disk")
    ap.add_argument("--neg-control-families", nargs="*", default=list(NEG_CONTROL_DEFAULT),
                    help="families where corners are unmanipulated (composited/reenactment)")
    ap.add_argument("--n", type=int, default=300, help="max samples per class per family")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-one-per-group", dest="one_per_group", action="store_false")
    ap.add_argument("--out", default="reports/region_probe.json")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    if not a.real_manifest or not a.fake_manifest:
        ap.error("--real-manifest and --fake-manifest are required")
    run(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
