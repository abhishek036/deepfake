"""
linear_probe_beforeafter.py  --  put in evaluation/

Cheap, fully controlled before/after: logistic regression on FROZEN DINOv2-small embeddings
(CLS + mean patch = 768-d), trained and tested on two pairings that share the SAME fake images:

  C  confounded pairing : reals = your old aligned `data/crops` frames  (e.g. data/manifests/faceforensics.jsonl)
  M  matched pairing    : reals = geometry-matched PNG crops           (e.g. data/ffpp_matched_reals_fsfr.jsonl)

Splits are by SCENE (FF++ parent video id, shared by reals and fakes) so no scene is in both train and test.
Protocols:
  P1  scene-disjoint split (all generators seen in training)
  P2  scene- AND generator-disjoint (true leave-one-generator-out; feasible because only a linear head is retrained)
Experiments per protocol: C->C, C->M, M->M, M->C, for two models: `dino` (embeddings) and `lowlevel`
(8 hand-made image statistics; tells you whether the embedding finds anything beyond low-level cues).

Extra controls:
  --mask face_out   zero out the central 112x112 face square (background-only probe)
  --mask face_only  zero out the four 40x40 background corners (face-only probe)
  --identity-disjoint  partition fake source-identity pairs so no identity appears on both sides of train/test

One-line example:
  python -m evaluation.linear_probe_beforeafter --confounded-real-manifest data/manifests/faceforensics.jsonl --clean-real-manifest data/ffpp_matched_reals_fsfr.jsonl --fake-manifest data/df40_fsfr_subset.jsonl --families composited --out reports/linear_probe_beforeafter.json
Self-test (no data, no weights):  python -m evaluation.linear_probe_beforeafter --self-test
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import pickle
import re
import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SIZE = 224
FACE_HALF = SIZE // 4          # 56px: central 112x112 face square half-width
CORNER_PAT = 40                # background corner patch size
FF_ID = re.compile(r"(\d{3}_\d{3})")
PAIRS = [("C", "C"), ("C", "M"), ("M", "M"), ("M", "C")]


# ------------------------------------------------------------------ io / features
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


def canonical(path):
    im = Image.open(path).convert("RGB")
    w, h = im.size
    s = min(w, h)
    l, t = (w - s) // 2, (h - s) // 2
    return np.asarray(im.crop((l, t, l + s, t + s)).resize((SIZE, SIZE), Image.LANCZOS))


def apply_mask(arr, mask):
    """Apply spatial mask to a (SIZE, SIZE, 3) array.  Applied identically to both classes.
    face_out  : blank the central 112x112 face square with per-image mean colour (background-only probe).
    face_only : blank the four 40x40 background corners with per-image mean colour (face-only probe).
    None      : no masking.
    """
    if mask is None:
        return arr
    out = arr.copy()
    c = SIZE // 2
    fill = arr.reshape(-1, 3).mean(0).astype(arr.dtype)
    if mask == "face_out":
        out[c - FACE_HALF:c + FACE_HALF, c - FACE_HALF:c + FACE_HALF] = fill
    elif mask == "face_only":
        out[:CORNER_PAT, :CORNER_PAT] = fill
        out[:CORNER_PAT, -CORNER_PAT:] = fill
        out[-CORNER_PAT:, :CORNER_PAT] = fill
        out[-CORNER_PAT:, -CORNER_PAT:] = fill
    else:
        raise ValueError(f"Unknown mask={mask!r}; expected 'face_out' or 'face_only'")
    return out


def lowlevel(a):
    buf = io.BytesIO()
    Image.fromarray(a).save(buf, format="JPEG", quality=90)
    g = np.asarray(Image.fromarray(a).convert("L"), dtype=np.float64)
    lap = -4 * g[1:-1, 1:-1] + g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:]
    p = np.abs(np.fft.fftshift(np.fft.fft2(g - g.mean()))) ** 2
    yy, xx = np.indices(p.shape)
    hf = p[np.hypot(yy - 112, xx - 112) / SIZE > 0.25].sum() / (p.sum() + 1e-9)
    m = a.reshape(-1, 3).mean(0)
    return np.array([math.log(buf.tell() / (SIZE * SIZE)), math.log(lap.var() + 1e-6), hf, g.std(), g.mean(), *m])


def stub_embedder(arrs):
    out = []
    for a in arrs:
        g = np.asarray(Image.fromarray(a).convert("L"), dtype=np.float64)
        cells = []
        for i in range(4):
            for j in range(4):
                c = g[i * 56:(i + 1) * 56, j * 56:(j + 1) * 56]
                cells.append(math.log(np.abs(np.diff(c, axis=0)).mean() + 1e-3))
        out.append(cells + a.reshape(-1, 3).mean(0).tolist() + a.reshape(-1, 3).std(0).tolist())
    return np.array(out)


def hf_embedder(name, device, batch):
    import torch
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)

    if "dinov2" in name.lower():
        try:
            model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").to(dev).eval()
            def embed_hub(arrs):
                out = []
                with torch.no_grad():
                    for i in range(0, len(arrs), batch):
                        x = torch.from_numpy(np.stack(arrs[i:i + batch])).to(dev).permute(0, 3, 1, 2).float() / 255
                        feat = model.forward_features((x - mean) / std)
                        rep = torch.cat([feat["x_norm_clstoken"], feat["x_norm_patchtokens"].mean(1)], 1)
                        out.append(rep.float().cpu().numpy())
                return np.concatenate(out)
            return embed_hub
        except Exception as e:
            pass

    from transformers import AutoModel
    model = AutoModel.from_pretrained(name).to(dev).eval()

    def embed(arrs):
        out = []
        with torch.no_grad():
            for i in range(0, len(arrs), batch):
                x = torch.from_numpy(np.stack(arrs[i:i + batch])).to(dev).permute(0, 3, 1, 2).float() / 255
                h = model(pixel_values=(x - mean) / std).last_hidden_state
                out.append(torch.cat([h[:, 0], h[:, 1:].mean(1)], 1).float().cpu().numpy())
        return np.concatenate(out)
    return embed


def compute_reps(paths, embedder, backbone, cache_path, chunk=64):
    """Cache holds raw (unmasked) representations; masking is applied at build() time."""
    cache = {}
    if cache_path and Path(cache_path).exists():
        blob = pickle.loads(Path(cache_path).read_bytes())
        if blob.get("backbone") == backbone:
            cache = blob["reps"]
    todo = [p for p in dict.fromkeys(paths) if p not in cache]
    print(f"representations: {len(paths) - len(todo)} cached, {len(todo)} to compute")
    for i in range(0, len(todo), chunk):
        ps = todo[i:i + chunk]
        arrs = [canonical(p) for p in ps]
        emb = embedder(arrs)
        for p, a, e in zip(ps, arrs, emb):
            cache[p] = (e, lowlevel(a), a)           # store raw array for re-embed with mask
        if (i // chunk) % 10 == 0:
            print(f"  {min(i + chunk, len(todo))}/{len(todo)}")
    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        Path(cache_path).write_bytes(pickle.dumps({"backbone": backbone, "reps": cache}))
    return cache


def masked_reps(reps, mask, embedder, chunk=64, cache_path=None):
    """Re-embed cached images with the requested spatial mask applied.
    Returns a dict path -> (dino_feat, lowlevel_feat) on masked images.
    Backwards-compatible: handles both old 2-tuple caches (no raw array stored)
    and new 3-tuple caches. Old entries fall back to reading the image from disk.
    If cache_path is provided, caches masked embeddings to <stem>_<mask_name>.pkl.
    """
    if mask is None:
        # Return views into existing cache (no copy needed)
        return {p: (v[0], v[1]) for p, v in reps.items()}
    mask_cache_file = None
    if cache_path:
        cp = Path(cache_path)
        mask_cache_file = cp.with_name(cp.stem + f"_{mask}.pkl")
        if mask_cache_file.exists():
            try:
                blob = pickle.loads(mask_cache_file.read_bytes())
                if isinstance(blob, dict) and "reps" in blob:
                    cached = blob["reps"]
                    if all(p in cached for p in reps):
                        print(f"Loaded {len(cached)} masked ({mask}) representations from {mask_cache_file.name}")
                        return cached
            except Exception:
                pass
    paths = sorted(reps)
    out = {}
    print(f"Computing masked ({mask}) representations for {len(paths)} images...")
    for i in range(0, len(paths), chunk):
        ps = paths[i:i + chunk]
        arrs = []
        for p in ps:
            v = reps[p]
            if len(v) >= 3 and v[2] is not None:
                arr = v[2]           # new cache: raw array stored
            else:
                arr = canonical(p)   # old cache: reload from disk
            arrs.append(apply_mask(arr, mask))
        emb = embedder(arrs)
        for p, a, e in zip(ps, arrs, emb):
            out[p] = (e, lowlevel(a))
        if (i // chunk) % 10 == 0:
            print(f"  {min(i + chunk, len(paths))}/{len(paths)}")
    if mask_cache_file:
        try:
            mask_cache_file.parent.mkdir(parents=True, exist_ok=True)
            mask_cache_file.write_bytes(pickle.dumps({"mask": mask, "reps": out}))
            print(f"Saved masked ({mask}) representations to {mask_cache_file.name}")
        except Exception:
            pass
    return out


# ------------------------------------------------------------------ data selection
def load_reals(manifest, k, rng):
    by = {}
    for o in read_jsonl(manifest):
        if o.get("label", 0) != 0:
            continue
        stem = Path(str(o.get("source_path", ""))).stem
        p = o.get("crop_path") or o.get("source_path")
        if p and re.fullmatch(r"\d{3}", stem) and Path(p).exists():
            by.setdefault(stem, []).append(str(p))
    return {s: [v[i] for i in rng.permutation(len(v))[:k]] for s, v in by.items()}


def load_fakes(manifest, families, part, per_scene, rng, identity_disjoint=False):
    """Load fake records grouped by FF++ parent scene.

    identity_disjoint=True: a video_id is a pair 'A_B' where A=scene, B=source-identity.
    Scenes with the same B appearing in both train and test half would leak identity.
    We assign each B to either the train or test pool upfront so B never crosses the split.
    This only partitions at call time; the actual train/test split is done in run().
    We encode the assignment in a 'side' field (0=train-only, 1=test-only) on each record.
    build() then filters accordingly.
    """
    by = {}            # scene -> {method -> [path, ...]}
    scene_id_map = {}  # scene -> source-identity (the other part of the pair)
    for o in read_jsonl(manifest):
        if o.get("label", 1) != 1 or (families and o.get("forgery_type") not in families):
            continue
        m = FF_ID.search(str(o.get("video_id", "")) + " " + str(o.get("source_path", "")))
        if m and o.get("source_path"):
            pair = m.group(1)                          # e.g. '001_870'
            scene = pair.split("_")[part]              # parent video id
            identity = pair.split("_")[1 - part]      # source identity id
            scene_id_map[scene] = identity
            by.setdefault(scene, {}).setdefault(o.get("method", "?"), []).append(o["source_path"])

    # Assign each source-identity to a side (0=train, 1=test) to prevent leakage
    identity_side = {}
    if identity_disjoint:
        ids = sorted({v for v in scene_id_map.values()})
        rng.shuffle(ids)  # shuffle deterministically
        for j, identity in enumerate(ids):
            identity_side[identity] = j % 2

    out = {}
    for scene, meths in by.items():
        for m in meths:
            meths[m] = [meths[m][i] for i in rng.permutation(len(meths[m]))[:per_scene]]
        picked, i = [], 0
        while len(picked) < per_scene and any(i < len(v) for v in meths.values()):
            for m in sorted(meths):
                if i < len(meths[m]) and len(picked) < per_scene and Path(meths[m][i]).exists():
                    side = identity_side.get(scene_id_map.get(scene, ""), None)
                    picked.append((meths[m][i], m, side))
            i += 1
        if picked:
            out[scene] = picked
    return out


# ------------------------------------------------------------------ fit / eval
def boot_auc(y, s, g, rng, n=300):
    ug = np.unique(g)
    idx = {k: np.where(g == k)[0] for k in ug}
    b = []
    for _ in range(n):
        ii = np.concatenate([idx[k] for k in rng.choice(ug, len(ug), replace=True)])
        if len(set(y[ii])) == 2:
            b.append(roc_auc_score(y[ii], s[ii]))
    return [float(x) for x in np.percentile(b, [2.5, 97.5])] if b else [float("nan")] * 2


def fit_eval(Xtr, ytr, Xte, yte, gte, C, rng, ci=True):
    if len(set(ytr)) < 2 or len(set(yte)) < 2:
        return None
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, class_weight="balanced", max_iter=3000))
    clf.fit(Xtr, ytr)
    s = clf.decision_function(Xte)
    pred = s >= 0
    return {"auroc": float(roc_auc_score(yte, s)), "ci95": boot_auc(yte, s, gte, rng) if ci else None,
            "tpr": float(pred[yte == 1].mean()), "fpr": float(pred[yte == 0].mean()),
            "bal_acc": float((pred[yte == 1].mean() + (~pred)[yte == 0].mean()) / 2),
            "n_real": int((yte == 0).sum()), "n_fake": int((yte == 1).sum())}


def build(pair, scenes, reals, fakes, reps, which, gen_in=None, gen_out=None, id_side=None):
    """id_side: if not None, only include fakes whose identity-side tag equals id_side (0=train, 1=test)."""
    X, y, g = [], [], []
    for s in scenes:
        for p in reals[pair].get(s, []):
            if p in reps:
                X.append(reps[p][which]); y.append(0); g.append(s)
        for entry in fakes.get(s, []):
            p, m = entry[0], entry[1]
            side = entry[2] if len(entry) > 2 else None
            if (gen_in is not None and m not in gen_in) or (gen_out is not None and m in gen_out):
                continue
            if id_side is not None and side != id_side:
                continue
            if p in reps:
                X.append(reps[p][which]); y.append(1); g.append(s)
    return np.array(X), np.array(y), np.array(g)


def run(a, embedder, backbone):
    seeds_list = getattr(a, "seeds", None) or [a.seed]
    mask = getattr(a, "mask", None)
    identity_disjoint = getattr(a, "identity_disjoint", False)

    seed_scenes = {}
    seed_reals = {}
    seed_fakes = {}
    all_paths = set()
    for s in seeds_list:
        s_rng = np.random.default_rng(s)
        rc = load_reals(a.confounded_real_manifest, a.reals_per_scene, s_rng)
        rm = load_reals(a.clean_real_manifest, a.reals_per_scene, s_rng)
        fk = load_fakes(a.fake_manifest, a.families, a.parent_part, a.fakes_per_scene, s_rng,
                        identity_disjoint=identity_disjoint)
        sc = sorted(set(rc) & set(rm) & set(fk))
        seed_scenes[s] = sc
        seed_reals[s] = {"C": rc, "M": rm}
        seed_fakes[s] = fk
        for scene in sc:
            for pr in "CM":
                all_paths.update(seed_reals[s][pr].get(scene, []))
            all_paths.update(entry[0] for entry in fk.get(scene, []))

    all_scenes = seed_scenes[seeds_list[0]]
    print(f"scenes with C reals, M reals and fakes: {len(all_scenes)} "
          f"(C {len(seed_reals[seeds_list[0]]['C'])}, M {len(seed_reals[seeds_list[0]]['M'])}, fakes {len(seed_fakes[seeds_list[0]])})")
    if mask:
        print(f"Spatial mask: {mask}")
    if identity_disjoint:
        print("Identity-disjoint mode: source identities are partitioned across train/test")
    if len(all_scenes) < 40:
        sys.exit("too few shared scenes; check --parent-part and that both real manifests cover the same FF++ videos")

    tr_set, te_set = None, None
    if getattr(a, "split_dir", None):
        sdir = Path(a.split_dir)
        tr_p, te_p = sdir / "train.json", sdir / "test.json"
        if tr_p.exists() and te_p.exists():
            tr_set = set(json.loads(tr_p.read_text(encoding="utf-8")))
            te_set = set(json.loads(te_p.read_text(encoding="utf-8")))
            print(f"Using official FF++ split from {sdir}")

    # Collect all unique paths across all seeds
    paths = sorted(all_paths)
    reps_raw = compute_reps(paths, embedder, backbone, a.cache)
    reps = masked_reps(reps_raw, mask, embedder, cache_path=a.cache)

    models = {"dino": 0, "lowlevel": 1}
    tr_id_side = 0 if identity_disjoint else None
    te_id_side = 1 if identity_disjoint else None

    seed_ledgers = {}
    print(f"\nRunning evaluation across {len(seeds_list)} seed(s): {seeds_list}")

    for seed in seeds_list:
        rng = np.random.default_rng(seed)
        reals = seed_reals[seed]
        fakes = seed_fakes[seed]
        scenes = seed_scenes[seed]

        if tr_set is not None and te_set is not None:
            tr_sc = [s for s in scenes if s in tr_set][: a.max_scenes]
            te_sc = [s for s in scenes if s in te_set][: a.max_scenes]
        else:
            rng.shuffle(scenes)
            scenes = scenes[: a.max_scenes]
            half = len(scenes) // 2
            tr_sc, te_sc = scenes[:half], scenes[half:]

        p1_res = {}
        for mname, w in models.items():
            for a_, b_ in PAIRS:
                Xtr, ytr, _ = build(a_, tr_sc, reals, fakes, reps, w, id_side=tr_id_side)
                Xte, yte, gte = build(b_, te_sc, reals, fakes, reps, w, id_side=te_id_side)
                r = fit_eval(Xtr, ytr, Xte, yte, gte, a.C, rng)
                p1_res[f"{mname}:{a_}->{b_}"] = r

        meth = Counter(entry[1] for s in te_sc for entry in fakes[s])
        gens = a.logo_generators or [m for m, _ in meth.most_common(a.logo_k)]
        p2_res = {}
        for mname, w in models.items():
            for a_, b_ in PAIRS:
                per = {}
                for gname in gens:
                    Xtr, ytr, _ = build(a_, tr_sc, reals, fakes, reps, w, gen_out={gname}, id_side=tr_id_side)
                    Xte, yte, gte = build(b_, te_sc, reals, fakes, reps, w, gen_in={gname}, id_side=te_id_side)
                    if (yte == 1).sum() < 10:
                        continue
                    r = fit_eval(Xtr, ytr, Xte, yte, gte, a.C, rng, ci=False)
                    if r:
                        per[gname] = r["auroc"]
                p2_res[f"{mname}:{a_}->{b_}"] = per

        seed_ledgers[seed] = {"P1": p1_res, "P2": p2_res, "n_scenes": {"train": len(tr_sc), "test": len(te_sc)}}

    primary_seed = seeds_list[0]
    ledger = {
        "args": {k: str(v) for k, v in vars(a).items()},
        "backbone": backbone,
        "mask": mask,
        "identity_disjoint": identity_disjoint,
        "script_sha256": sha256_file(__file__),
        "manifest_sha256": {k: sha256_file(getattr(a, k)) for k in
                            ("confounded_real_manifest", "clean_real_manifest", "fake_manifest")},
        "n_scenes": seed_ledgers[primary_seed]["n_scenes"],
        "seeds": seeds_list,
        "P1": seed_ledgers[primary_seed]["P1"],
        "P2": seed_ledgers[primary_seed]["P2"],
    }

    if len(seeds_list) > 1:
        ledger["per_seed"] = {str(k): v for k, v in seed_ledgers.items()}
        summary_p1 = {}
        first_p1 = seed_ledgers[primary_seed]["P1"]
        for k in first_p1:
            aucs = [seed_ledgers[s]["P1"][k]["auroc"] for s in seeds_list if seed_ledgers[s]["P1"].get(k)]
            baccs = [seed_ledgers[s]["P1"][k]["bal_acc"] for s in seeds_list if seed_ledgers[s]["P1"].get(k)]
            fprs = [seed_ledgers[s]["P1"][k]["fpr"] for s in seeds_list if seed_ledgers[s]["P1"].get(k)]
            tprs = [seed_ledgers[s]["P1"][k]["tpr"] for s in seeds_list if seed_ledgers[s]["P1"].get(k)]
            if aucs:
                mean_auc = float(np.mean(aucs))
                std_auc = float(np.std(aucs, ddof=1)) if len(aucs) > 1 else 0.0
                summary_p1[k] = {
                    "auroc_mean": mean_auc,
                    "auroc_std": std_auc,
                    "auroc_min": float(np.min(aucs)),
                    "auroc_max": float(np.max(aucs)),
                    "auroc_seeds": [float(x) for x in aucs],
                    "bal_acc_mean": float(np.mean(baccs)),
                    "fpr_mean": float(np.mean(fprs)),
                    "tpr_mean": float(np.mean(tprs)),
                }
        ledger["summary_P1"] = summary_p1

        summary_p2 = {}
        first_p2 = seed_ledgers[primary_seed]["P2"]
        for k in first_p2:
            all_folds = []
            for s in seeds_list:
                per = seed_ledgers[s]["P2"].get(k, {})
                if per:
                    all_folds.append(np.mean(list(per.values())))
            if all_folds:
                summary_p2[k] = {
                    "mean_auroc_mean": float(np.mean(all_folds)),
                    "mean_auroc_std": float(np.std(all_folds, ddof=1)) if len(all_folds) > 1 else 0.0,
                }
        ledger["summary_P2"] = summary_p2

    print(f"\n== P1: scene-disjoint summary ({len(seeds_list)} seed(s)) ==")
    print(f"{'model':9s}{'train->test':>12s}  {'AUROC':>20s}  {'[min, max]':>15s}  balAcc")
    for mname in models:
        for a_, b_ in PAIRS:
            k = f"{mname}:{a_}->{b_}"
            if len(seeds_list) > 1 and "summary_P1" in ledger and k in ledger["summary_P1"]:
                s_info = ledger["summary_P1"][k]
                print(f"{mname:9s}{a_ + '->' + b_:>12s}  {s_info['auroc_mean']:.3f} ± {s_info['auroc_std']:.3f}  "
                      f"[{s_info['auroc_min']:.3f}, {s_info['auroc_max']:.3f}]  {s_info['bal_acc_mean']:.3f}")
            else:
                r = ledger["P1"].get(k)
                if r:
                    print(f"{mname:9s}{a_ + '->' + b_:>12s}  {r['auroc']:.3f} [{r['ci95'][0]:.3f},{r['ci95'][1]:.3f}]  "
                          f"{'-':>15s}  {r['bal_acc']:.3f}")

    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(ledger, indent=1))
        print("ledger ->", a.out)
    return ledger


# ------------------------------------------------------------------ self-test
def self_test():
    rng = np.random.default_rng(0)
    root = Path(tempfile.mkdtemp())
    yy, xx = np.indices((256, 256))
    rc, rm, fk = [], [], []

    def base(scene):
        return 128 + 50 * np.sin((xx + scene * 7) / 19.0) * np.cos((yy + scene * 3) / 29.0)

    def save(arr, p, jpeg=False):
        im = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
        im.save(p, quality=75) if jpeg else im.save(p)

    for s in range(80):
        for k in range(2):
            b = base(s)[..., None] + rng.normal(0, 10, (256, 256, 3))
            p = root / f"c_{s}_{k}.jpg"
            im = Image.fromarray(np.clip(b, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.6))
            im.save(p, quality=75)
            rc.append({"label": 0, "split": "x", "crop_path": str(p), "source_path": f"{s:03d}.mp4"})
            b = base(s)[..., None] + rng.normal(0, 10, (256, 256, 3))
            p = root / f"m_{s}_{k}.png"
            save(b, p)
            rm.append({"label": 0, "split": "any", "crop_path": str(p), "source_path": f"{s:03d}.mp4"})
        for k in range(4):
            b = base(s)[..., None] + rng.normal(0, 10, (256, 256, 3))
            c = b[64:192, 64:192].copy()
            b[64:192, 64:192] = (c + np.roll(c, 1, 0) + np.roll(c, 1, 1) + np.roll(c, -1, 0)) / 4
            p = root / f"f_{s}_{k}.png"
            save(b, p)
            fk.append({"label": 1, "method": f"g{k % 4}", "forgery_type": "composited",
                       "video_id": f"df40_g{k % 4}_{s:03d}_{(s + 5) % 90 + 100:03d}", "source_path": str(p)})
    for n, rows in (("c.jsonl", rc), ("m.jsonl", rm), ("f.jsonl", fk)):
        (root / n).write_text("\n".join(json.dumps(r) for r in rows))
    ns_base = argparse.Namespace(confounded_real_manifest=str(root / "c.jsonl"), clean_real_manifest=str(root / "m.jsonl"),
                                 fake_manifest=str(root / "f.jsonl"), families=["composited"], parent_part=0, reals_per_scene=2,
                                 fakes_per_scene=4, max_scenes=80, C=1.0, seed=0, logo_generators=None, logo_k=4, cache=None, out=None,
                                 mask=None, identity_disjoint=False)
    led = run(ns_base, stub_embedder, "stub")
    p1 = led["P1"]
    ok = (p1["dino:C->C"]["auroc"] > 0.95 and p1["dino:C->M"]["fpr"] > 0.5 and p1["dino:M->M"]["auroc"] > 0.8)
    print("baseline self-test", "PASSED" if ok else "FAILED")

    # Test face_out mask: background-only should be near chance for our synthetic data
    import copy
    ns_fo = copy.copy(ns_base); ns_fo.mask = "face_out"
    led_fo = run(ns_fo, stub_embedder, "stub")
    # In synthetic data face_out blanks manipulation; M->M should collapse
    ok_fo = led_fo["P1"]["dino:M->M"]["auroc"] < led["P1"]["dino:M->M"]["auroc"]
    print("face_out self-test (M->M drops)", "PASSED" if ok_fo else "FAILED")

    # Test identity_disjoint: should run without error
    ns_id = copy.copy(ns_base); ns_id.identity_disjoint = True
    led_id = run(ns_id, stub_embedder, "stub")
    ok_id = "P1" in led_id and "dino:M->M" in led_id["P1"]
    print("identity_disjoint self-test", "PASSED" if ok_id else "FAILED")

    all_ok = ok and ok_fo and ok_id
    print("self-test", "PASSED" if all_ok else "FAILED")
    return 0 if all_ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--confounded-real-manifest")
    ap.add_argument("--clean-real-manifest")
    ap.add_argument("--fake-manifest")
    ap.add_argument("--families", nargs="*", default=["composited"])
    ap.add_argument("--parent-part", type=int, default=0, choices=[0, 1], help="same value you used in extract")
    ap.add_argument("--reals-per-scene", type=int, default=3)
    ap.add_argument("--fakes-per-scene", type=int, default=6)
    ap.add_argument("--max-scenes", type=int, default=600)
    ap.add_argument("--split-dir", default=None,
                    help="Directory containing official train.json and test.json split files")
    ap.add_argument("--backbone", default="facebook/dinov2-small",
                    help="HF model name, or 'torchhub:dinov2_vits14' for torch.hub variant")
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--C", type=float, default=0.1, help="inverse L2 strength of the logistic head")
    ap.add_argument("--logo-generators", nargs="*", default=None)
    ap.add_argument("--logo-k", type=int, default=6,
                    help="how many of the most frequent generators to hold out, one at a time")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seeds", type=int, nargs="*", default=None,
                    help="Optional list of seeds to run and aggregate (e.g. --seeds 42 7 13 99 2024)")
    ap.add_argument("--cache", default="reports/emb_cache.pkl")
    ap.add_argument("--out", default="reports/linear_probe_beforeafter.json")
    ap.add_argument("--mask", choices=["face_out", "face_only"], default=None,
                    help="face_out=blank face region (background probe); face_only=blank corners (face probe)")
    ap.add_argument("--identity-disjoint", action="store_true",
                    help="Prevent source-identity leakage: assign each fake source-identity to train or test only")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    if not (a.confounded_real_manifest and a.clean_real_manifest and a.fake_manifest):
        ap.error("need --confounded-real-manifest, --clean-real-manifest and --fake-manifest")
    # torchhub alias maps to the same model hf_embedder already handles via torch.hub
    backbone_name = "facebook/dinov2-small" if a.backbone == "torchhub:dinov2_vits14" else a.backbone
    run(a, hf_embedder(backbone_name, a.device, a.batch_size), a.backbone)
    return 0


if __name__ == "__main__":
    sys.exit(main())
