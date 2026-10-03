"""
one_pipeline_audit.py  --  drop into evaluation/ and run as:  python -m evaluation.one_pipeline_audit ...

Purpose
-------
Test whether a real-vs-fake pairing is separable by low-level image statistics
(sharpness, size, spectrum, blockiness, COLOR, thumbnail) BEFORE trusting any model AUROC on it,
then measure how the model reacts to symmetric degradations, mask controls and
a sharpness intervention.

What it does
------------
0. Pre-flight : raw extension / mode / size statistics per class (format confound check)
1. One pipeline: every image -> RGB -> centre square crop -> LANCZOS resize to 224.
                 No class-conditional processing anywhere except the labelled diagnostic in step 5.
2. Probe       : grouped-CV logistic probe on low-level statistics (target: AUROC <= 0.55)
                 Features: grayscale (6) + color (6) + 8x8 thumbnail (192) = expanded feature set.
                 Univariate AUROCs + overlap/matching coverage reported for ALL feature groups.
3. Model       : AUROC (grouped bootstrap CI) and TPR/FPR (Wilson CI) on a held-out half split BY VIDEO,
                 under: canonical, common_jpeg90, hard_jpeg50, hard_down50_jpeg50,
                 mask A (central mean-color fill), mask B (face-only, mean-color bkg),
                 mask C (random patches same area, mean-color fill), mask D (patch shuffle).
                 The threshold is chosen ONCE on the val half (canonical) via corrected Youden J
                 (median of plateau) and never re-tuned.
4. AURC        : risk-coverage on the hard conditions (only meaningful with >= 20 errors).
5. Diagnostic  : blur ONLY the fakes until their sharpness matches the real median (labelled diagnostic).
6. Cross-probe : optional per-real-source probe sweep, scripted into ledger.
7. Ledger      : JSON with args, git commit + dirty flag, sha256 of script / checkpoint / manifests /
                 sample list. Script writes its own sha256 AFTER the run.

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
THUMB = 8   # 8x8 thumbnail probe dimensions

GRAY_FEATURE_NAMES = [
    "log_jpeg90_kb", "log_lap_var", "gray_mean", "gray_std", "hf_ratio", "blockiness"
]
COLOR_FEATURE_NAMES = [
    "r_mean", "g_mean", "b_mean", "r_std", "g_std", "b_std"
]
THUMB_FEATURE_NAMES = [f"thumb_{i}" for i in range(THUMB * THUMB * 3)]
ALL_FEATURE_NAMES = GRAY_FEATURE_NAMES + COLOR_FEATURE_NAMES + THUMB_FEATURE_NAMES


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
    """
    Returns a feature vector concatenating:
      - 6 grayscale statistics  (gray_features)
      - 6 color channel statistics  (color_features)
      - THUMB*THUMB*3 thumbnail pixel values  (thumbnail_features)

    Total = 6 + 6 + 192 = 204 features.
    """
    arr = np.asarray(img, dtype=np.float64)   # H x W x 3, [0..255]
    g = arr.mean(axis=2)                        # grayscale approximation

    # -- grayscale statistics --
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
    gray_feat = np.array([
        math.log(kb),
        math.log(lap_var(g) + 1e-6),
        g.mean(), g.std(),
        hf,
        edge / (inner + 1e-9)
    ])

    # -- color channel statistics (R, G, B mean + std) --
    r_ch = arr[:, :, 0]
    g_ch = arr[:, :, 1]
    b_ch = arr[:, :, 2]
    color_feat = np.array([
        r_ch.mean(), g_ch.mean(), b_ch.mean(),
        r_ch.std(),  g_ch.std(),  b_ch.std()
    ])

    # -- 8x8 thumbnail (flattened RGB) --
    thumb = np.asarray(img.resize((THUMB, THUMB), Image.LANCZOS), dtype=np.float32).flatten() / 255.0

    return np.concatenate([gray_feat, color_feat, thumb])


def n_gray_feats():
    return len(GRAY_FEATURE_NAMES)


def n_color_feats():
    return len(COLOR_FEATURE_NAMES)


def grouped_cv_auc(X, y, groups, seed, max_iter=1000):
    n_splits = int(min(5, len(set(groups))))
    aucs = []
    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups):
        if len(set(y[tr])) < 2 or len(set(y[te])) < 2:
            continue
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=max_iter))
        clf.fit(X[tr], y[tr])
        aucs.append(roc_auc_score(y[te], clf.predict_proba(X[te])[:, 1]))
    return float(np.mean(aucs)) if aucs else float("nan")


def univariate_auc(X, y, names):
    """Return folded AUROC (always >= 0.5) for each feature."""
    out = {}
    for j, n in enumerate(names):
        if j >= X.shape[1]:
            break
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


def _mean_fill_color(img: Image.Image) -> tuple[int, int, int]:
    """Return the per-channel mean of the whole image as an integer RGB tuple."""
    arr = np.asarray(img, dtype=np.float32)
    return (
        int(round(arr[:, :, 0].mean())),
        int(round(arr[:, :, 1].mean())),
        int(round(arr[:, :, 2].mean())),
    )


def _blurred_fill(img: Image.Image, radius: float = 16.0) -> np.ndarray:
    """Return a heavily blurred version of the image as an uint8 array."""
    return np.asarray(img.filter(ImageFilter.GaussianBlur(radius)), dtype=np.uint8)


def apply_mask(img, mode, rng):
    """
    Mask modes — fill with per-image mean color (not zeros) to stay in-distribution.

    A : central 85% ellipse filled with image mean color  (tests background cue)
    B : background filled with image mean color, face left  (tests face cue)
    C : random 16x16 patches, same total area as A, mean-color fill
    D : random 16x16 patches shuffled in-place (no color change, tests spatial layout)
    """
    a = np.array(img)
    fill_color = _mean_fill_color(img)          # (R, G, B)

    if mode == "A":        # central face → mean color; background kept
        for ch, val in enumerate(fill_color):
            a[ELL, ch] = val

    elif mode == "B":      # background → mean color; face kept
        for ch, val in enumerate(fill_color):
            a[~ELL, ch] = val

    elif mode == "C":      # random patches → mean color; equal area to A
        n = SIZE // 16
        k = int(round(ELL.mean() * n * n))
        patch_ids = rng.choice(n * n, k, replace=False)
        for i in patch_ids:
            r, c = divmod(int(i), n)
            for ch, val in enumerate(fill_color):
                a[r * 16:(r + 1) * 16, c * 16:(c + 1) * 16, ch] = val

    elif mode == "D":      # patch shuffle — same pixels, different spatial layout
        n = SIZE // 16
        patches = []
        for idx in range(n * n):
            r, c = divmod(idx, n)
            patches.append(a[r * 16:(r + 1) * 16, c * 16:(c + 1) * 16].copy())
        order = rng.permutation(n * n)
        for new_idx, old_idx in enumerate(order):
            r, c = divmod(new_idx, n)
            a[r * 16:(r + 1) * 16, c * 16:(c + 1) * 16] = patches[old_idx]

    return Image.fromarray(a)


CONDITIONS = {
    "canonical":                    lambda im, rng: im,
    "common_jpeg90":                lambda im, rng: jpeg(im, 90),
    "hard_jpeg50":                  lambda im, rng: jpeg(im, 50),
    "hard_down50_jpeg50":           lambda im, rng: jpeg(down_up(im, 0.5), 50),
    "maskA_face_meanfill":          lambda im, rng: apply_mask(im, "A", rng),
    "maskB_bkg_meanfill":           lambda im, rng: apply_mask(im, "B", rng),
    "maskC_randpatch_meanfill":     lambda im, rng: apply_mask(im, "C", rng),
    "maskD_patchshuffle":           lambda im, rng: apply_mask(im, "D", rng),
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
    """
    Corrected Youden-J threshold selection.

    Problem with argmax: when the score distribution is separable (no overlap) the
    Youden surface is a step function with a plateau of identical J-values spanning
    the entire gap between the two clusters. argmax returns the *lowest* edge of that
    plateau, which sits just above the lowest fake score. Any small downward shift of
    the fake cluster (e.g., JPEG Q=90) then drops scores below tau and recall
    collapses, even though AUROC is unchanged.

    Fix: return the MEDIAN of all candidate thresholds that achieve J >= J_max - eps.
    This places tau in the centre of the gap and is robust to cluster drift.
    """
    u = np.unique(s)
    # midpoints between adjacent unique scores + sentinels
    cand = np.concatenate([[u[0] - 1], (u[:-1] + u[1:]) / 2, [u[-1] + 1]])
    J = np.array([((s >= t)[y == 1].mean() + (s < t)[y == 0].mean() - 1) for t in cand])
    plateau = cand[J >= J.max() - 1e-12]
    return float(np.median(plateau))


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
              "git": git_info(),
              "n_samples": int(len(samples)), "generators": dict(Counter(s["generator"] for s in samples))}

    print("== 0. pre-flight (raw files, before any processing) ==")
    ledger["preflight"] = preflight(samples)
    print(json.dumps(ledger["preflight"], indent=1))

    imgs = [canonicalize(s["path"]) for s in samples]
    F_all = np.stack([probe_features(im) for im in imgs])

    # Feature slices
    ng = len(GRAY_FEATURE_NAMES)
    nc = len(COLOR_FEATURE_NAMES)
    F_gray  = F_all[:, :ng]
    F_color = F_all[:, ng:ng + nc]
    F_thumb = F_all[:, ng + nc:]
    F_gray_color = F_all[:, :ng + nc]

    print("\n== 2. low-level probe (grouped CV by video) ==")

    # (a) grayscale-only probe (legacy 6-feature)
    probe_auc_gray  = grouped_cv_auc(F_gray, y, g, args.seed)
    # (b) gray + color (12 features)
    probe_auc_color = grouped_cv_auc(F_gray_color, y, g, args.seed)
    # (c) full probe including 8x8 thumbnail (204 features)
    probe_auc_full  = grouped_cv_auc(F_all, y, g, args.seed)
    # (d) thumbnail alone
    probe_auc_thumb = grouped_cv_auc(F_thumb, y, g, args.seed)

    # Univariate AUROCs (gray + color only; thumb has 192 features, report top-5)
    uni_gray  = univariate_auc(F_gray,  y, GRAY_FEATURE_NAMES)
    uni_color = univariate_auc(F_color, y, COLOR_FEATURE_NAMES)
    thumb_col_aucs = {}
    for j in range(F_thumb.shape[1]):
        a = roc_auc_score(y, F_thumb[:, j])
        thumb_col_aucs[j] = float(max(a, 1 - a))
    top5_thumb = dict(sorted(thumb_col_aucs.items(), key=lambda x: -x[1])[:5])

    # Matching coverage (size + sharpness: cols 0,1 of F_gray)
    keep = matched_indices(F_gray, y, cols=[0, 1], bins=args.match_bins, rng=rng)
    coverage = len(keep) / len(y)
    probe_matched = (grouped_cv_auc(F_gray_color[keep], y[keep], g[keep], args.seed)
                     if len(keep) > 20 and len(set(y[keep])) == 2 else float("nan"))

    ledger["probe"] = {
        "cv_auroc_grayscale_6feat":           probe_auc_gray,
        "cv_auroc_gray_plus_color_12feat":    probe_auc_color,
        "cv_auroc_full_204feat":              probe_auc_full,
        "cv_auroc_thumbnail_8x8_192feat":     probe_auc_thumb,
        "univariate_auroc_gray":              uni_gray,
        "univariate_auroc_color":             uni_color,
        "top5_univariate_thumbnail_pixel":    top5_thumb,
        "matched_coverage":                   coverage,
        "matched_kept": {"real": int((y[keep] == 0).sum()), "fake": int((y[keep] == 1).sum())},
        "cv_auroc_gray_color_after_size_sharpness_match": probe_matched,
        "pass_threshold": 0.55,
    }

    print(f"probe CV-AUROC (gray 6-feat)  : {probe_auc_gray:.4f}   (pass <= 0.55)")
    print(f"probe CV-AUROC (gray+color)   : {probe_auc_color:.4f}")
    print(f"probe CV-AUROC (full 204-feat): {probe_auc_full:.4f}")
    print(f"probe CV-AUROC (thumbnail 8x8): {probe_auc_thumb:.4f}")
    print("univariate gray  :", {k: round(v, 3) for k, v in uni_gray.items()})
    print("univariate color :", {k: round(v, 3) for k, v in uni_color.items()})
    print(f"matched coverage : {coverage:.1%}   probe (gray+color) after matching: {probe_matched:.4f}")

    if probe_auc_gray > 0.65 or probe_auc_color > 0.65:
        print(">> CONFOUNDED: low-level statistics alone separate the classes. "
              "Do NOT report model AUROC on this pairing as forensic performance.")
    if coverage < 0.30:
        print(">> classes barely overlap in sharpness/size: sampling cannot repair this pairing. "
              "Change the real source.")

    # Per-source cross-probe sweep (optional: extra manifests passed via --cross-probe-manifests)
    cross_results = {}
    if hasattr(args, "cross_probe_manifests") and args.cross_probe_manifests:
        print("\n== 2b. cross-source probe sweep ==")
        for entry in args.cross_probe_manifests:
            # entry format: "name:real_manifest[:split]"
            parts = entry.split(":")
            name = parts[0]
            rm = parts[1]
            sp = parts[2] if len(parts) > 2 else "any"
            try:
                cross_reals = load_real([rm], sp)
                if not cross_reals:
                    cross_results[name] = {"error": "no real samples found"}
                    continue
                cross_fakes = [s for s in samples if s["label"] == 1]
                cs = sample_balanced(cross_reals, cross_fakes,
                                     min(len(cross_reals), args.n_per_class),
                                     args.per_generator_cap, np.random.default_rng(args.seed + 1))
                cy = np.array([s["label"] for s in cs])
                cg = np.array([s["group"] for s in cs])
                cimgs = [canonicalize(s["path"]) for s in cs]
                CF = np.stack([probe_features(im) for im in cimgs])
                cg_auc  = grouped_cv_auc(CF[:, :ng],         cy, cg, args.seed)
                cgc_auc = grouped_cv_auc(CF[:, :ng + nc],    cy, cg, args.seed)
                cth_auc = grouped_cv_auc(CF[:, ng + nc:],    cy, cg, args.seed)
                cu_gray = univariate_auc(CF[:, :ng],  cy, GRAY_FEATURE_NAMES)
                cu_col  = univariate_auc(CF[:, ng:ng + nc], cy, COLOR_FEATURE_NAMES)
                cross_results[name] = {
                    "n_real": int((cy == 0).sum()), "n_fake": int((cy == 1).sum()),
                    "cv_auroc_gray": cg_auc, "cv_auroc_gray_color": cgc_auc,
                    "cv_auroc_thumbnail": cth_auc,
                    "univariate_gray": cu_gray, "univariate_color": cu_col,
                    "manifest": rm, "split": sp,
                }
                print(f"  {name:24s}  gray={cg_auc:.4f}  gray+color={cgc_auc:.4f}  thumb={cth_auc:.4f}")
            except Exception as exc:
                cross_results[name] = {"error": str(exc)}
                print(f"  {name}: ERROR {exc}")
    ledger["cross_probe_sweep"] = cross_results

    if scorer is None:
        ledger["note"] = "probe-only run (--skip-model)"
        return ledger

    print("\n== 3. model under conditions ==")
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
    ledger["threshold_method"] = "Youden-J plateau median (robust to cluster drift)"
    print(f"threshold (val, canonical, Youden-median): {tau:.4f}  (frozen for every condition)")

    te = ~is_val
    ledger["conditions"] = {}
    for name, z in logits.items():
        m = summarize(y[te], z[te], g[te], tau, args.boot, rng)
        ledger["conditions"][name] = m
        print(f"{name:36s} AUROC {m['auroc']:.4f} {np.round(m['auroc_ci95_grouped_boot'], 4)}  "
              f"TPR {m['tpr']:.3f}  FPR {m['fpr']:.3f}  (n={m['n']})")

    mm = np.zeros(len(y), dtype=bool)
    mm[keep] = True
    sel = te & mm
    if sel.sum() > 20 and len(set(y[sel])) == 2:
        ledger["conditions"]["canonical@matched_subset"] = summarize(
            y[sel], logits["canonical"][sel], g[sel], tau, args.boot, rng)
        print(f"{'canonical@matched_subset':36s} AUROC "
              f"{ledger['conditions']['canonical@matched_subset']['auroc']:.4f}  (n={int(sel.sum())})")

    print("\n== 4. AURC on hard conditions (uncertainty = |logit - tau|) ==")
    ledger["aurc"] = {}
    for name in ("hard_jpeg50", "hard_down50_jpeg50"):
        a = aurc(y[te], logits[name][te], tau)
        ledger["aurc"][name] = a
        print(f"{name:24s} errors={a['n_errors']:3d} risk={a['overall_risk']:.3f} AURC={a['aurc']:.4f} "
              f"optimal={a['aurc_optimal']:.4f} E-AURC={a['e_aurc']:.4f}"
              + ("" if a["meaningful"] else "   [<20 errors: not meaningful]"))

    print("\n== 5. diagnostic: blur ONLY fakes to real-median sharpness ==")
    target = float(np.median([lap_var(gray_arr(im)) for im, l in zip(imgs, y) if l == 0]))
    new = [blur_to_target(im, target) if l == 1 else im for im, l in zip(imgs, y)]
    z = scorer(new)
    d = summarize(y[te], z[te], g[te], tau, args.boot, rng)
    ledger["diagnostic_blur_fakes_to_real"] = d
    print(f"AUROC {d['auroc']:.4f} {np.round(d['auroc_ci95_grouped_boot'], 4)}  "
          f"fake TPR {d['tpr']:.3f}  "
          f"(canonical AUROC {ledger['conditions']['canonical']['auroc']:.4f})")
    print("A large AUROC drop = the model is reading sharpness, not forgery evidence.")
    return ledger


# ----------------------------------------------------------------------------- self-test
def _make_synth(root: Path, n=60):
    rng = np.random.default_rng(0)
    root.mkdir(parents=True, exist_ok=True)
    real_m, fake_m = [], []
    yy, xx = np.indices((256, 256))
    for i in range(n):
        base = 128 + 60 * np.sin((xx + rng.integers(50)) / 20.0) * np.cos((yy + rng.integers(50)) / 25.0)
        # Reals: slightly warm-tinted, blurred, saved as JPEG
        im_r = np.stack([
            np.clip(base + rng.normal(0, 4, base.shape) + 8, 0, 255),   # R boost
            np.clip(base + rng.normal(0, 4, base.shape),      0, 255),   # G neutral
            np.clip(base + rng.normal(0, 4, base.shape) - 8, 0, 255),   # B reduced
        ], -1).astype(np.uint8)
        real = Image.fromarray(im_r).filter(ImageFilter.GaussianBlur(1.5))
        p = root / f"r{i}.jpg"
        real.save(p, quality=75)
        real_m.append({"label": 0, "split": "test", "crop_path": str(p), "source_path": f"vid_r{i}.mp4"})

        # Fakes: slightly cool-tinted, sharp, saved as PNG
        im_f = np.stack([
            np.clip(base + rng.normal(0, 8, base.shape) - 8, 0, 255),   # R reduced
            np.clip(base + rng.normal(0, 8, base.shape),      0, 255),   # G neutral
            np.clip(base + rng.normal(0, 8, base.shape) + 8, 0, 255),   # B boost
        ], -1).astype(np.uint8)
        fake = Image.fromarray(im_f)
        p = root / f"f{i}.png"
        fake.save(p)
        fake_m.append({"label": 1, "method": f"gen{i % 4}", "video_id": f"v{i}", "source_path": str(p)})

    for name, m in (("real.jsonl", real_m), ("fake.jsonl", fake_m)):
        with open(root / name, "w") as f:
            f.write("\n".join(json.dumps(x) for x in m))
    return root / "real.jsonl", root / "fake.jsonl"


def stub_scorer(images):  # deliberately sharpness-only; color-aware audit should flag the color gap
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
    ap.add_argument("--cross-probe-manifests", nargs="*", default=[],
                    help="Extra per-source probes. Format: 'name:real_manifest[:split]'. "
                         "Fakes are reused from --fake-manifest. "
                         "Example: 'ffpp:data/ffpp_real.jsonl:test'")
    args = ap.parse_args()

    scorer = None
    if args.self_test:
        tmp = Path(tempfile.mkdtemp())
        rm, fm = _make_synth(tmp)
        args.real_manifest = [str(rm)]
        args.fake_manifest = str(fm)
        args.real_split = "test"
        args.n_per_class = 50
        args.per_generator_cap = 20
        args.boot = 50
        args.match_bins = 4
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

    out_path = out / "one_pipeline_audit.json"
    out_path.write_text(json.dumps(ledger, indent=1, default=float))

    # Write script sha256 AFTER the run so the ledger's own hash is correct
    ledger["script_sha256"] = sha256_file(__file__)
    out_path.write_text(json.dumps(ledger, indent=1, default=float))

    print(f"\nledger -> {out_path}")


if __name__ == "__main__":
    sys.exit(main())
