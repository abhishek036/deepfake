"""
extract_matched_png_reals.py  --  put in scripts/

Builds REAL face crops whose geometry (face scale, placement, roll, output size) is drawn from the
joint distribution measured on the FAKE crops, saved as lossless PNG, so that crop scale / scene context
stop being the thing a probe or a model can separate on.

Commands (run each as ONE line in PowerShell)
  calibrate   measure landmark geometry of fakes -> targets json (+ optional filtered fake manifest)
  extract     decode real videos, detect face, render crops with sampled target geometry -> PNG + manifest
  verify      re-measure the written reals with the same measurement as the fakes; KS tests + pass/fail
  self-test   synthetic end-to-end check (stub detector, no data, no insightface)

Measurement (same function for both classes): pad the crop 50% with its mean colour, run the detector,
take the largest face, and read the 5 insightface keypoints:
  ipd_ratio = inter-ocular distance / sqrt(W*H)     ex, ey = eye-midpoint / (W, H)
  roll = angle of the eye line (deg)                mouth_drop = |mouth_mid - eye_mid| / ipd
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import tempfile
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

DET_CONFIDENCE = 0.70   # same thresholds as preprocessing/preprocess_all.py
MIN_FACE_PX = 60
PAD_CONF = 0.40         # looser threshold for tight crops that are padded before detection
FF_ID = re.compile(r"(\d{3}_\d{3})")


# ------------------------------------------------------------------ helpers
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


def make_detector():
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name="buffalo_sc", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(640, 640))
    return app.get


def largest(faces, min_conf, min_px=0):
    ok = []
    for f in faces:
        b = np.asarray(f.bbox, dtype=np.float32)
        if float(f.det_score) >= min_conf and (b[2] - b[0]) >= min_px and (b[3] - b[1]) >= min_px:
            ok.append(f)
    return max(ok, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])) if ok else None


def geometry(kps):
    k = np.asarray(kps, dtype=np.float64)          # [left_eye, right_eye, nose, left_mouth, right_mouth]
    d = k[1] - k[0]
    ipd = float(np.hypot(*d))
    mid = (k[0] + k[1]) / 2
    mouth = (k[3] + k[4]) / 2
    return {"ipd": ipd, "mid": mid, "roll": math.degrees(math.atan2(d[1], d[0])),
            "mouth_drop": float(np.hypot(*(mouth - mid)) / max(ipd, 1e-9))}


def measure_crop(img, det):
    """Same measurement for fakes (already crops) and for written real crops."""
    h, w = img.shape[:2]
    pad = int(0.5 * max(h, w))
    fill = [int(v) for v in img.reshape(-1, 3).mean(0)]
    big = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=fill)
    f = largest(det(big), PAD_CONF)
    if f is None:
        return None
    g = geometry(np.asarray(f.kps) - pad)
    side = math.sqrt(w * h)
    return {"ipd_ratio": g["ipd"] / side, "ex": g["mid"][0] / w, "ey": g["mid"][1] / h,
            "roll": g["roll"], "mouth_drop": g["mouth_drop"], "size": int(round(side))}


def filter_fakes(recs, forgery_types, ff_domain_only):
    out = []
    for r in recs:
        if r.get("label", 1) != 1 or not r.get("source_path"):
            continue
        if forgery_types and r.get("forgery_type") not in forgery_types:
            continue
        if ff_domain_only:
            vid = str(r.get("video_id", ""))
            sp = str(r.get("source_path", ""))
            if not (FF_ID.search(vid) or FF_ID.search(sp)):
                continue
        if not Path(r["source_path"]).exists():
            continue
        out.append(r)
    return out


# ------------------------------------------------------------------ calibrate
def cmd_calibrate(a, det=None):
    det = det or make_detector()
    recs = read_jsonl(a.fake_manifest)
    print("forgery_type counts (all records):", dict(Counter(r.get("forgery_type") for r in recs)))
    sel = filter_fakes(recs, a.forgery_types, a.ff_domain_only)
    print(f"{len(sel)} fake records pass filters; methods: {dict(Counter(r.get('method') for r in sel))}")
    if a.write_fake_subset:
        with open(a.write_fake_subset, "w", encoding="utf-8") as f_out:
            for r in sel:
                f_out.write(json.dumps(r) + "\n")
        print("filtered fake manifest ->", a.write_fake_subset)
    rng = np.random.default_rng(a.seed)
    by = {}
    for r in sel:
        by.setdefault(r.get("method", "?"), []).append(r)
    for m in by:
        rng.shuffle(by[m])
    picked, i = [], 0
    while len(picked) < a.n and any(i < len(v) for v in by.values()):
        for m in sorted(by):
            if i < len(by[m]) and len(picked) < a.n:
                picked.append(by[m][i])
        i += 1
    rows = []
    for r in picked:
        img = cv2.imread(r["source_path"])
        if img is None:
            continue
        m = measure_crop(img, det)
        if m:
            rows.append(m)
    fail = 1 - len(rows) / max(1, len(picked))
    print(f"measured {len(rows)}/{len(picked)} fakes (detection failure {fail:.0%})")
    if fail > 0.3:
        print("WARNING: >30% failures; targets may be biased toward easy-to-detect fakes")
    if len(rows) < 20:
        sys.exit("too few measurements to calibrate")
    keys = ["ipd_ratio", "ex", "ey", "roll", "mouth_drop", "size"]
    out = {"n_attempted": len(picked), "n_ok": len(rows), "rows": rows,
           "summary": {k: {"median": float(np.median([r[k] for r in rows])),
                           "q25": float(np.percentile([r[k] for r in rows], 25)),
                           "q75": float(np.percentile([r[k] for r in rows], 75))} for k in keys},
           "fake_manifest_sha256": sha256_file(a.fake_manifest),
           "filters": {"forgery_types": a.forgery_types, "ff_domain_only": a.ff_domain_only}}
    Path(a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out["summary"], indent=1))
    print("targets ->", a.out)


# ------------------------------------------------------------------ extract
def render(frame, g, tgt, out):
    s = tgt["ipd_ratio"] * out / g["ipd"]
    th = math.radians(tgt["roll"] - g["roll"])
    if s < 1.0:                                   # downscale with INTER_AREA first (no aliasing), then rotate/translate
        frame = cv2.resize(frame, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        mid, s2 = g["mid"] * s, 1.0
    else:
        mid, s2 = g["mid"], s
    c, sn = math.cos(th) * s2, math.sin(th) * s2
    M = np.array([[c, -sn, 0.0], [sn, c, 0.0]], dtype=np.float64)
    M[:, 2] = np.array([tgt["ex"] * out, tgt["ey"] * out]) - M[:, :2] @ mid
    inv = cv2.invertAffineTransform(M)
    corners = np.array([[0, 0, 1], [out, 0, 1], [0, out, 1], [out, out, 1]], dtype=np.float64) @ inv.T
    h, w = frame.shape[:2]
    if (corners[:, 0] < 0).any() or (corners[:, 1] < 0).any() or (corners[:, 0] > w).any() or (corners[:, 1] > h).any():
        return None                               # crop would leave the frame: skip instead of padding
    return cv2.warpAffine(frame, M, (out, out), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)


def cmd_extract(a, det=None):
    det = det or make_detector()
    T = json.loads(Path(a.targets).read_text(encoding="utf-8"))["rows"]
    rng = np.random.default_rng(a.seed)
    if a.split_json:
        raw_split = json.loads(Path(a.split_json).read_text(encoding="utf-8"))
        if raw_split and isinstance(raw_split[0], list):
            ids = sorted({str(x) for pair in raw_split for x in pair})
        else:
            ids = sorted({str(x) for x in raw_split})
        split = "test"
    else:
        ids = set()
        with open(a.fake_subset, "r", encoding="utf-8") as f_sub:
            for line in f_sub:
                if not line.strip():
                    continue
                r = json.loads(line)
                vid = str(r.get("video_id", ""))
                sp = str(r.get("source_path", ""))
                m = FF_ID.search(vid) or FF_ID.search(sp)
                if m:
                    parts = m.group(1).split("_")
                    ids.add(parts[a.parent_part])
        ids = sorted(ids)
        split = "parent"
    rng.shuffle(ids)
    ids = ids[: a.n_videos]
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, skipped = [], Counter()
    print(f"Extracting matched crops from {len(ids)} videos ({a.frames_per_video} frames/video)...", flush=True)
    for i_vid, vid in enumerate(ids):
        if (i_vid + 1) % 50 == 0 or (i_vid + 1) == len(ids):
            print(f"Progress: [{i_vid + 1}/{len(ids)}] videos processed, {len(rows)} crops written...", flush=True)
        vp = Path(a.video_dir) / f"{vid}.mp4"
        cap = cv2.VideoCapture(str(vp))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not cap.isOpened() or n < 5:
            skipped["video_unreadable"] += 1
            continue
        for idx in np.linspace(0.1 * n, 0.9 * n, a.frames_per_video).astype(int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if not ok:
                skipped["frame_unreadable"] += 1
                continue
            f = largest(det(frame), DET_CONFIDENCE, MIN_FACE_PX)
            if f is None:
                skipped["no_face"] += 1
                continue
            tgt = T[int(rng.integers(len(T)))]       # joint resampling of the fakes' geometry
            out = int(np.clip(tgt["size"], 128, 1024))
            crop = render(frame, geometry(f.kps), tgt, out)
            if crop is None:
                skipped["crop_out_of_frame"] += 1
                continue
            p = out_dir / f"{vid}_{int(idx):05d}.png"
            cv2.imwrite(str(p), crop)
            rows.append({"label": 0, "split": split, "crop_path": str(p), "source_path": str(vp),
                         "video_id": vid, "frame_idx": int(idx), "target_size": out})
    with open(a.manifest, "w", encoding="utf-8") as f_m:
        for r in rows:
            f_m.write(json.dumps(r) + "\n")
    meta = {"script_sha256": sha256_file(__file__), "targets_sha256": sha256_file(a.targets),
            "args": {k: str(v) for k, v in vars(a).items()}, "n_crops": len(rows),
            "n_videos": len({r["video_id"] for r in rows}), "skipped": dict(skipped), "split_label": split}
    Path(str(a.manifest) + ".meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"wrote {len(rows)} crops from {meta['n_videos']} videos; skipped: {dict(skipped)}; manifest -> {a.manifest}")


# ------------------------------------------------------------------ verify
def cmd_verify(a, det=None):
    from scipy.stats import ks_2samp
    det = det or make_detector()
    T = json.loads(Path(a.targets).read_text(encoding="utf-8"))["rows"]
    rng = np.random.default_rng(a.seed)
    reals = read_jsonl(a.manifest)
    rng.shuffle(reals)
    R = []
    for r in reals[: a.n]:
        img = cv2.imread(r["crop_path"])
        m = measure_crop(img, det) if img is not None else None
        if m:
            R.append(m)
    print(f"re-measured {len(R)}/{min(a.n, len(reals))} real crops")
    ok = len(R) >= 20
    print(f"{'metric':12s} {'fake med [q25,q75]':26s} {'real med [q25,q75]':26s} KS-stat   p")
    for k in ["ipd_ratio", "ex", "ey", "roll", "mouth_drop"]:
        f = np.array([t[k] for t in T])
        r = np.array([t[k] for t in R]) if R else np.array([np.nan])
        ks = ks_2samp(f, r) if R else None
        fs = f"{np.median(f):.3f} [{np.percentile(f, 25):.3f},{np.percentile(f, 75):.3f}]"
        rs = f"{np.median(r):.3f} [{np.percentile(r, 25):.3f},{np.percentile(r, 75):.3f}]"
        print(f"{k:12s} {fs:26s} {rs:26s} {ks.statistic:.3f}   {ks.pvalue:.3f}" if ks else k)
        if ks and k in ("ipd_ratio", "ey") and (ks.statistic > 0.25 or abs(np.median(r) - np.median(f)) > 0.05 * abs(np.median(f))):
            ok = False
    print("PARITY", "PASS" if ok else "FAIL (scale/placement still differs: do not trust a probe on this pairing)")
    return 0 if ok else 1


# ------------------------------------------------------------------ self-test (synthetic, stub detector)
class _Face:
    def __init__(self, kps):
        self.kps = np.asarray(kps, dtype=np.float32)
        lo, hi = self.kps.min(0) - 8, self.kps.max(0) + 8
        self.bbox = np.array([lo[0], lo[1], hi[0], hi[1]], dtype=np.float32)
        self.det_score = 0.99


_COLORS = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255)]   # BGR dots = 5 keypoints


def _stub_det(img):
    pts = []
    for c in _COLORS:
        m = (np.abs(img.astype(np.int16) - np.array(c, dtype=np.int16)).sum(2) < 120)
        ys, xs = np.nonzero(m)
        if len(xs) < 4:
            return []
        pts.append((xs.mean(), ys.mean()))
    return [_Face(pts)]


def _draw(img, mid, ipd, roll_deg, drop):
    th = math.radians(roll_deg)
    u, v = np.array([math.cos(th), math.sin(th)]), np.array([-math.sin(th), math.cos(th)])
    mm = mid + drop * ipd * v
    kps = [mid - 0.5 * ipd * u, mid + 0.5 * ipd * u, mid + 0.45 * ipd * v, mm - 0.3 * ipd * u, mm + 0.3 * ipd * u]
    r = max(3, int(ipd * 0.07))
    for p, c in zip(kps, _COLORS):
        cv2.circle(img, (int(round(p[0])), int(round(p[1]))), r, c, -1)
    return img


def cmd_selftest(_a):
    rng = np.random.default_rng(1)
    root = Path(tempfile.mkdtemp())
    fakes = []
    for i in range(60):
        img = rng.integers(90, 160, (256, 256, 3), dtype=np.uint8)
        _draw(img, np.array([0.5 * 256, 0.40 * 256]) + rng.normal(0, 2, 2), 0.34 * 256 * rng.uniform(0.95, 1.05), rng.normal(0, 3), 1.0)
        p = root / f"f{i}.png"
        cv2.imwrite(str(p), img)
        fakes.append({"label": 1, "method": f"g{i % 3}", "forgery_type": "FS", "video_id": f"{i % 6:03d}_{i + 100:03d}", "source_path": str(p)})
    (root / "fake.jsonl").write_text("\n".join(json.dumps(r) for r in fakes))
    vd = root / "vids"
    vd.mkdir()
    ok_video = True
    for v in range(6):
        vw = cv2.VideoWriter(str(vd / f"{v:03d}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 25, (640, 360))
        ok_video &= vw.isOpened()
        for t in range(30):
            fr = rng.integers(60, 200, (360, 640, 3), dtype=np.uint8)
            _draw(fr, np.array([320.0 + rng.normal(0, 40), 150.0 + rng.normal(0, 15)]), rng.uniform(40, 60), rng.uniform(-12, 12), 1.0)
            vw.write(fr)
        vw.release()
    if not ok_video:
        print("self-test: mp4 writer unavailable in this OpenCV build; skipping video stage")
        return 0
    ns = argparse.Namespace
    cmd_calibrate(ns(fake_manifest=str(root / "fake.jsonl"), forgery_types=["FS"], ff_domain_only=True,
                     write_fake_subset=str(root / "subset.jsonl"), n=60, seed=0, out=str(root / "targets.json")), _stub_det)
    cmd_extract(ns(targets=str(root / "targets.json"), split_json=None, fake_subset=str(root / "subset.jsonl"), parent_part=0,
                   video_dir=str(vd), n_videos=6, frames_per_video=6, seed=0, out_dir=str(root / "png"),
                   manifest=str(root / "reals.jsonl")), _stub_det)
    rc = cmd_verify(ns(targets=str(root / "targets.json"), manifest=str(root / "reals.jsonl"), n=60, seed=0), _stub_det)
    print("self-test", "PASSED" if rc == 0 else "FAILED")
    return rc


# ------------------------------------------------------------------ cli
def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    c = sp.add_parser("calibrate")
    c.add_argument("--fake-manifest", required=True)
    c.add_argument("--forgery-types", nargs="*", default=None, help="exact forgery_type strings to keep (first run prints them)")
    c.add_argument("--ff-domain-only", action="store_true", help="keep fakes whose video_id looks like 123_456 (FF++ parents)")
    c.add_argument("--write-fake-subset", default=None)
    c.add_argument("--n", type=int, default=300)
    c.add_argument("--seed", type=int, default=42)
    c.add_argument("--out", default="data/crop_targets.json")
    e = sp.add_parser("extract")
    e.add_argument("--video-dir", required=True)
    e.add_argument("--targets", required=True)
    e.add_argument("--split-json", default=None, help="official FF++ test.json (use for any model run)")
    e.add_argument("--fake-subset", default=None, help="with no --split-json: take PARENT videos of these fakes (probe-only)")
    e.add_argument("--parent-part", type=int, default=0, choices=[0, 1], help="which id in <a>_<b> is the scene video")
    e.add_argument("--n-videos", type=int, default=140)
    e.add_argument("--frames-per-video", type=int, default=6)
    e.add_argument("--seed", type=int, default=42)
    e.add_argument("--out-dir", required=True)
    e.add_argument("--manifest", required=True)
    v = sp.add_parser("verify")
    v.add_argument("--manifest", required=True)
    v.add_argument("--targets", required=True)
    v.add_argument("--n", type=int, default=150)
    v.add_argument("--seed", type=int, default=42)
    sp.add_parser("self-test")
    a = ap.parse_args()
    if a.cmd == "extract" and not a.split_json and not a.fake_subset:
        ap.error("extract needs --split-json (model runs) or --fake-subset (probe-only, parent videos)")
    return {"calibrate": cmd_calibrate, "extract": cmd_extract, "verify": cmd_verify, "self-test": cmd_selftest}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main() or 0)
