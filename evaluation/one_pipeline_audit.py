"""
one_pipeline_audit.py  --  drop into evaluation/ and run as:  python -m evaluation.one_pipeline_audit ...

Purpose
-------
Test whether a real-vs-fake pairing is separable by low-level image statistics
(sharpness, size, spectrum, blockiness) BEFORE trusting any model AUROC on it,
then measure how the model reacts to symmetric degradations, mask controls and
a sharpness intervention.

What it does
------------
0. Pre-flight : raw extension / mode / size statistics per class (format confound check)
1. One pipeline: every image -> RGB -> centre square crop -> LANCZOS resize to 224.
                 No class-conditional processing anywhere except the labelled diagnostic in step 5.
2. Probe       : grouped-CV logistic probe on low-level statistics (target: AUROC <= 0.55)
                 + univariate AUROCs + overlap/matching coverage.
3. Model       : AUROC (grouped bootstrap CI) and TPR/FPR (Wilson CI) on a held-out half split BY VIDEO,
                 under: canonical, common_jpeg90, hard_jpeg50, hard_down50_jpeg50,
                 mask A (central blackout), mask B (face-only), mask C (random patches, same area as A).
                 The threshold is chosen once on the val half (canonical) and never re-tuned.
4. AURC        : risk-coverage on the hard conditions (only meaningful with >= 20 errors).
5. Diagnostic  : blur ONLY the fakes until their sharpness matches the real median (labelled diagnostic).
6. Ledger      : JSON with args, git commit + dirty flag, sha256 of script / checkpoint / manifests / sample list.

Self-test (no data, no GPU):  python -m evaluation.one_pipeline_audit --self-test
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import subprocess
import sys
import tempfile
import zlib
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SIZE = 224
FEATURE_NAMES = ["log_jpeg90_kb", "log_lap_var", "gray_mean", "gray_std", "hf_ratio", "blockiness"]


# ----------------------------------------------------------------------------- utils
def sha256_file(path) -> str | None:
    p = Path(path)
    if not p.exists():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(8 << 20):
            h.update(chunk)
    return h.hexdigest()


def git_info() -> dict:
    def run(*a):
        try:
            return subprocess.run(["git", *a], capture_output=True, text=True, timeout=20).stdout.strip()
        except Exception:
            return ""
    return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain"))}


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def read_jsonl(path):
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


# ----------------------------------------------------------------------------- data
def load_real(manifests, split):
    recs = []
    for m in manifests:
        for o in read_jsonl(m):
            if o.get("label", 0) != 0:
                continue
            if split != "any" and o.get("split") != split:
                continue
            p = o.get("crop_path") or o.get("source_path") or o.get("path")
            if not p or not Path(p).exists():
                continue
            grp = o.get("source_path") if o.get("crop_path") else (o.get("video_id") or str(Path(p).parent))
            recs.append({"path": str(p), "label": 0, "group": f"0:{grp}", "generator": "real"})
    return recs


def load_fake(manifest):
    recs = []
    for o in read_jsonl(manifest):
        if o.get("label", 1) != 1:
            continue
        p = o.get("source_path") or o.get("path")
        if not p or not Path(p).exists():
            continue
        gen = o.get("method", "unknown")
        grp = o.get("video_id") or str(Path(p).parent)
        recs.append({"path": str(p), "label": 1, "group": f"1:{gen}/{grp}", "generator": gen})
    return recs


def sample_balanced(reals, fakes, n_per_class, gen_cap, rng):
    def one_per_group(recs):
        by = {}
        for r in recs:
            by.setdefault(r["group"], []).append(r)
        return [by[g][rng.integers(len(by[g]))] for g in sorted(by)]

    reals = one_per_group(reals)
    fakes = one_per_group(fakes)
    rng.shuffle(reals)
    reals = reals[:n_per_class]
    by_gen = {}
    for r in fakes:
        by_gen.setdefault(r["generator"], []).append(r)
    for g in by_gen:
        rng.shuffle(by_gen[g])
        by_gen[g] = by_gen[g][:gen_cap]
    picked, i = [], 0
    while len(picked) < n_per_class and any(i < len(v) for v in by_gen.values()):
        for g in sorted(by_gen):
            if i < len(by_gen[g]) and len(picked) < n_per_class:
                picked.append(by_gen[g][i])
        i += 1
    return reals + picked


def preflight(samples):
    stats = {}
    for lab in (0, 1):
        ext, mode, w, h = Counter(), Counter(), [], []
        for s in samples:
            if s["label"] != lab:
                continue
            with Image.open(s["path"]) as im:
                ext[Path(s["path"]).suffix.lower()] += 1
                mode[im.mode] += 1
                w.append(im.size[0])
                h.append(im.size[1])
        stats["fake" if lab else "real"] = {
            "n": len(w), "ext": dict(ext), "mode": dict(mode),
            "width_mean_std": [float(np.mean(w)), float(np.std(w))],
            "height_mean_std": [float(np.mean(h)), float(np.std(h))],
        }
    return stats


def canonicalize(path) -> Image.Image:
    im = Image.open(path).convert("RGB")
    w, h = im.size
    s = min(w, h)
    l, t = (w - s) // 2, (h - s) // 2
    return im.crop((l, t, l + s, t + s)).resize((SIZE, SIZE), Image.LANCZOS)


# ----------------------------------------------------------------------------- low-level features
def gray_arr(img):
    return np.asarray(img.convert("L"), dtype=np.float64)


def lap_var(g):
    lap = -4 * g[1:-1, 1:-1] + g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:]
    return float(lap.var())


def probe_features(img):
    g = gray_arr(img)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    kb = buf.tell() / 1024.0
    f = np.fft.fftshift(np.fft.fft2(g - g.mean()))
    p = np.abs(f) ** 2
    yy, xx = np.indices(p.shape)
    r = np.hypot(yy - p.shape[0] / 2, xx - p.shape[1] / 2) / p.shape[0]
    hf = float(p[r > 0.25].sum() / (p.sum() + 1e-9))
    dx = np.abs(np.diff(g, axis=1))
    edge = dx[:, 7::8].mean()
    inner = np.delete(dx, np.s_[7::8], axis=1).mean()
    return np.array([math.log(kb), math.log(lap_var(g) + 1e-6), g.mean(), g.std(), hf, edge / (inner + 1e-9)])


def grouped_cv_auc(X, y, groups, seed):
    n_splits = int(min(5, len(set(groups))))
    aucs = []
    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups):
        if len(set(y[tr])) < 2 or len(set(y[te])) < 2:
            continue
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        clf.fit(X[tr], y[tr])
        aucs.append(roc_auc_score(y[te], clf.predict_proba(X[te])[:, 1]))
    return float(np.mean(aucs)) if aucs else float("nan")


def univariate_auc(X, y):
    out = {}
    for j, n in enumerate(FEATURE_NAMES):
        a = roc_auc_score(y, X[:, j])
        out[n] = float(max(a, 1 - a))
    return out


def matched_indices(F, y, cols, bins, rng):
    cell = np.zeros(len(y), dtype=int)
    for c in cols:
        edges = np.quantile(F[:, c], np.linspace(0, 1, bins + 1)[1:-1])
        cell = cell * bins + np.digitize(F[:, c], edges)
    keep = []
    for c in np.unique(cell):
        i0 = np.where((cell == c) & (y == 0))[0]
        i1 = np.where((cell == c) & (y == 1))[0]
        k = min(len(i0), len(i1))
        if k:
            keep += list(rng.choice(i0, k, replace=False)) + list(rng.choice(i1, k, replace=False))
    return np.array(sorted(keep), dtype=int)


# ----------------------------------------------------------------------------- conditions
def jpeg(img, q):
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=q)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def down_up(img, f):
    s = max(8, int(SIZE * f))
    return img.resize((s, s), Image.BICUBIC).resize((SIZE, SIZE), Image.BICUBIC)


def ellipse_mask(frac=0.85):
    m = Image.new("L", (SIZE, SIZE), 0)
    r, c = frac * SIZE / 2, SIZE / 2
    ImageDraw.Draw(m).ellipse([c - r, c - r, c + r, c + r], fill=255)
    return np.asarray(m) > 0


ELL = ellipse_mask(0.85)


def apply_mask(img, mode, rng):
    a = np.array(img)
    if mode == "A":          # central blackout
        a[ELL] = 0
    elif mode == "B":        # face-only (background blacked out)
        a[~ELL] = 0
    elif mode == "C":        # random 16x16 patches, same total area as A
        n = SIZE // 16
        k = int(round(ELL.mean() * n * n))
        for i in rng.choice(n * n, k, replace=False):
            r, c = divmod(int(i), n)
            a[r * 16:(r + 1) * 16, c * 16:(c + 1) * 16] = 0
    return Image.fromarray(a)


CONDITIONS = {
    "canonical": lambda im, rng: im,
    "common_jpeg90": lambda im, rng: jpeg(im, 90),
    "hard_jpeg50": lambda im, rng: jpeg(im, 50),
    "hard_down50_jpeg50": lambda im, rng: jpeg(down_up(im, 0.5), 50),
    "maskA_central_blackout": lambda im, rng: apply_mask(im, "A", rng),
    "maskB_face_only": lambda im, rng: apply_mask(im, "B", rng),
    "maskC_random_patches_same_area": lambda im, rng: apply_mask(im, "C", rng),
}


def blur_to_target(img, target, iters=8):
    if lap_var(gray_arr(img)) <= target:
        return img
    lo, hi = 0.0, 4.0
    for _ in range(iters):
        mid = (lo + hi) / 2
        if lap_var(gray_arr(img.filter(ImageFilter.GaussianBlur(mid)))) > target:
            lo = mid
        else:
            hi = mid
    return img.filter(ImageFilter.GaussianBlur(hi))


# ----------------------------------------------------------------------------- metrics
def grouped_boot_auc(y, s, g, n_boot, rng):
    groups = np.unique(g)
    idx_by = {k: np.where(g == k)[0] for k in groups}
    aucs = []
    for _ in range(n_boot):
        pick = rng.choice(groups, size=len(groups), replace=True)
        idx = np.concatenate([idx_by[p] for p in pick])
        if len(set(y[idx])) == 2:
            aucs.append(roc_auc_score(y[idx], s[idx]))
    return [float(x) for x in np.percentile(aucs, [2.5, 97.5])] if aucs else [float("nan")] * 2


def youden_threshold(y, s):
    best, tau = -1.0, 0.0
    for t in np.unique(s):
        pred = s >= t
        j = pred[y == 1].mean() + (~pred)[y == 0].mean() - 1
        if j > best:
            best, tau = j, float(t)
    return tau


def aurc(y, z, tau):
    pred = (z >= tau).astype(int)
    err = (pred != y).astype(float)
    n, e = len(y), int(err.sum())
    order = np.argsort(-np.abs(z - tau))
    risk = np.cumsum(err[order]) / np.arange(1, n + 1)
    k = np.arange(1, n + 1)
    opt = np.maximum(0, k - (n - e)) / k
    return {"n_errors": e, "overall_risk": e / n, "aurc": float(risk.mean()),
            "aurc_optimal": float(opt.mean()), "e_aurc": float(risk.mean() - opt.mean()),
            "meaningful": e >= 20}


def summarize(y, z, g, tau, n_boot, rng):
    auc = float(roc_auc_score(y, z)) if len(set(y)) == 2 else float("nan")
    pred = z >= tau
    tp, fn = int(pred[y == 1].sum()), int((~pred)[y == 1].sum())
    fp, tn = int(pred[y == 0].sum()), int((~pred)[y == 0].sum())
    return {"n": int(len(y)), "auroc": auc, "auroc_ci95_grouped_boot": grouped_boot_auc(y, z, g, n_boot, rng),
            "tpr": tp / max(1, tp + fn), "tpr_wilson": wilson(tp, tp + fn),
            "fpr": fp / max(1, fp + tn), "fpr_wilson": wilson(fp, fp + tn)}


# ----------------------------------------------------------------------------- model
def load_scorer(ckpt, device, batch, allowed_missing):
    import torch
    from torchvision import transforms
    from models.forge_lite import FORGELite

    dev = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = FORGELite().to(dev)
    blob = torch.load(ckpt, map_location=dev, weights_only=False)
    res = model.load_state_dict(blob.get("model_state_dict", blob), strict=False)
    missing, unexpected = set(res.missing_keys), list(res.unexpected_keys)
    assert not unexpected, f"unexpected keys in checkpoint: {unexpected[:5]}"
    assert missing <= set(allowed_missing), f"unexpected missing keys: {sorted(missing - set(allowed_missing))[:5]}"
    model.eval()
    tf = transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def score(images):
        out = []
        with torch.no_grad():
            for i in range(0, len(images), batch):
                x = torch.stack([tf(im) for im in images[i:i + batch]]).unsqueeze(1).to(dev)  # [B,1,3,224,224]
                out += model(x)["fake_logit"].reshape(-1).float().cpu().tolist()
        return np.array(out)

    return score


# ----------------------------------------------------------------------------- main audit
def run_audit(samples, scorer, args):
    rng = np.random.default_rng(args.seed)
    y = np.array([s["label"] for s in samples])
    g = np.array([s["group"] for s in samples])
    ledger = {"args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
              "git": git_info(), "script_sha256": sha256_file(__file__),
              "n_samples": int(len(samples)), "generators": dict(Counter(s["generator"] for s in samples))}

    print("== 0. pre-flight (raw files, before any processing) ==")
    ledger["preflight"] = preflight(samples)
    print(json.dumps(ledger["preflight"], indent=1))

    imgs = [canonicalize(s["path"]) for s in samples]
    F = np.stack([probe_features(im) for im in imgs])

    print("\n== 2. low-level probe (grouped CV by video) ==")
    probe_auc = grouped_cv_auc(F, y, g, args.seed)
    uni = univariate_auc(F, y)
    keep = matched_indices(F, y, cols=[0, 1], bins=args.match_bins, rng=rng)
    coverage = len(keep) / len(y)
    probe_matched = grouped_cv_auc(F[keep], y[keep], g[keep], args.seed) if len(keep) > 20 and len(set(y[keep])) == 2 else float("nan")
    ledger["probe"] = {"cv_auroc_all_features": probe_auc, "univariate_auroc": uni,
                       "matched_coverage": coverage, "matched_kept": {"real": int((y[keep] == 0).sum()), "fake": int((y[keep] == 1).sum())},
                       "cv_auroc_after_matching_on_size_and_sharpness": probe_matched,
                       "pass_threshold": 0.55}
    print(f"probe CV-AUROC (all)        : {probe_auc:.4f}   (pass <= 0.55)")
    print("univariate (folded)         :", {k: round(v, 3) for k, v in uni.items()})
    print(f"matched coverage            : {coverage:.1%}   probe after matching: {probe_matched:.4f}")
    if probe_auc > 0.65:
        print(">> CONFOUNDED: low-level statistics alone separate the classes. Do NOT report model AUROC on this pairing as forensic performance.")
    if coverage < 0.30:
        print(">> classes barely overlap in sharpness/size: sampling cannot repair this pairing. Change the real source.")

    if scorer is None:
        ledger["note"] = "probe-only run (--skip-model)"
        return ledger

    print("\n== 3. model under conditions ==")
    rngs = {}
    groups_u = {lab: sorted(set(g[y == lab])) for lab in (0, 1)}
    val_groups = set()
    for lab in (0, 1):
        gl = list(groups_u[lab])
        rng.shuffle(gl)
        val_groups |= set(gl[: len(gl) // 2])
    is_val = np.array([x in val_groups for x in g])
    ledger["split"] = {"val_n": int(is_val.sum()), "test_n": int((~is_val).sum()), "by": "video group"}

    logits = {}
    for name, fn in CONDITIONS.items():
        r = np.random.default_rng([args.seed, zlib.crc32(name.encode())])
        logits[name] = scorer([fn(im, r) for im in imgs])
    tau = youden_threshold(y[is_val], logits["canonical"][is_val])
    ledger["threshold_raw_logit_from_val_canonical"] = tau
    print(f"threshold (val, canonical, raw logit): {tau:.4f}  (frozen for every condition)")

    te = ~is_val
    ledger["conditions"] = {}
    for name, z in logits.items():
        m = summarize(y[te], z[te], g[te], tau, args.boot, rng)
        ledger["conditions"][name] = m
        print(f"{name:34s} AUROC {m['auroc']:.4f} {np.round(m['auroc_ci95_grouped_boot'], 4)}  "
              f"TPR {m['tpr']:.3f}  FPR {m['fpr']:.3f}  (n={m['n']})")
    mm = np.zeros(len(y), dtype=bool)
    mm[keep] = True
    sel = te & mm
    if sel.sum() > 20 and len(set(y[sel])) == 2:
        ledger["conditions"]["canonical@matched_subset"] = summarize(y[sel], logits["canonical"][sel], g[sel], tau, args.boot, rng)
        print(f"{'canonical@matched_subset':34s} AUROC {ledger['conditions']['canonical@matched_subset']['auroc']:.4f}  (n={int(sel.sum())})")

    print("\n== 4. AURC on hard conditions (uncertainty = |logit - tau|) ==")
    ledger["aurc"] = {}
    for name in ("hard_jpeg50", "hard_down50_jpeg50"):
        a = aurc(y[te], logits[name][te], tau)
        ledger["aurc"][name] = a
        print(f"{name:24s} errors={a['n_errors']:3d} risk={a['overall_risk']:.3f} AURC={a['aurc']:.4f} "
              f"optimal={a['aurc_optimal']:.4f} E-AURC={a['e_aurc']:.4f}" + ("" if a["meaningful"] else "   [<20 errors: not meaningful]"))

    print("\n== 5. diagnostic: blur ONLY fakes to real-median sharpness ==")
    target = float(np.median([lap_var(gray_arr(im)) for im, l in zip(imgs, y) if l == 0]))
    new = [blur_to_target(im, target) if l == 1 else im for im, l in zip(imgs, y)]
    z = scorer(new)
    d = summarize(y[te], z[te], g[te], tau, args.boot, rng)
    ledger["diagnostic_blur_fakes_to_real"] = d
    print(f"AUROC {d['auroc']:.4f} {np.round(d['auroc_ci95_grouped_boot'], 4)}  fake TPR {d['tpr']:.3f}  (canonical AUROC {ledger['conditions']['canonical']['auroc']:.4f})")
    print("A large drop = the model is reading sharpness, not forgery evidence.")
    return ledger


# ----------------------------------------------------------------------------- self-test
def _make_synth(root: Path, n=60):
    rng = np.random.default_rng(0)
    root.mkdir(parents=True, exist_ok=True)
    real_m, fake_m = [], []
    yy, xx = np.indices((256, 256))
    for i in range(n):
        base = 128 + 60 * np.sin((xx + rng.integers(50)) / 20.0) * np.cos((yy + rng.integers(50)) / 25.0)
        im = np.stack([base + rng.normal(0, 4, base.shape) for _ in range(3)], -1)
        real = Image.fromarray(np.clip(im, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.5))
        p = root / f"r{i}.jpg"
        real.save(p, quality=75)
        real_m.append({"label": 0, "split": "test", "crop_path": str(p), "source_path": f"vid_r{i}.mp4"})
        fake = Image.fromarray(np.clip(im + rng.normal(0, 8, im.shape), 0, 255).astype(np.uint8))
        p = root / f"f{i}.png"
        fake.save(p)
        fake_m.append({"label": 1, "method": f"gen{i % 4}", "video_id": f"v{i}", "source_path": str(p)})
    for name, m in (("real.jsonl", real_m), ("fake.jsonl", fake_m)):
        with open(root / name, "w") as f:
            f.write("\n".join(json.dumps(x) for x in m))
    return root / "real.jsonl", root / "fake.jsonl"


def stub_scorer(images):  # deliberately a sharpness-only "detector"
    return np.array([math.log(lap_var(gray_arr(im)) + 1e-6) - 4.0 for im in images])


# ----------------------------------------------------------------------------- cli
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real-manifest", action="append", default=[])
    ap.add_argument("--real-split", default="test")
    ap.add_argument("--fake-manifest")
    ap.add_argument("--ckpt", default="outputs/checkpoints/best.ckpt")
    ap.add_argument("--n-per-class", type=int, default=300)
    ap.add_argument("--per-generator-cap", type=int, default=12)
    ap.add_argument("--match-bins", type=int, default=6)
    ap.add_argument("--boot", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--allowed-missing", nargs="*", default=["calibration.temperature_scaling.bias"])
    ap.add_argument("--skip-model", action="store_true", help="probe only (no GPU / checkpoint)")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--out-dir", default="reports")
    args = ap.parse_args()

    scorer = None
    if args.self_test:
        tmp = Path(tempfile.mkdtemp())
        rm, fm = _make_synth(tmp)
        args.real_manifest, args.fake_manifest, args.real_split = [str(rm)], str(fm), "test"
        args.n_per_class, args.per_generator_cap, args.boot, args.match_bins = 50, 20, 50, 4
        scorer = stub_scorer
    elif not args.skip_model:
        scorer = load_scorer(args.ckpt, args.device, args.batch_size, args.allowed_missing)

    rng = np.random.default_rng(args.seed)
    reals, fakes = load_real(args.real_manifest, args.real_split), load_fake(args.fake_manifest)
    print(f"available: {len(reals)} real frames, {len(fakes)} fake frames")
    samples = sample_balanced(reals, fakes, args.n_per_class, args.per_generator_cap, rng)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sl = out / "one_pipeline_sample_list.json"
    sl.write_text(json.dumps(samples, indent=1))

    ledger = run_audit(samples, scorer, args)
    ledger["sample_list_sha256"] = sha256_file(sl)
    ledger["checkpoint_sha256"] = None if (args.skip_model or args.self_test) else sha256_file(args.ckpt)
    ledger["manifest_sha256"] = {str(m): sha256_file(m) for m in [*args.real_manifest, args.fake_manifest]}
    (out / "one_pipeline_audit.json").write_text(json.dumps(ledger, indent=1, default=float))
    print(f"\nledger -> {out / 'one_pipeline_audit.json'}")


if __name__ == "__main__":
    sys.exit(main())
