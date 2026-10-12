"""
scripts/extract_shared_transform_pairs.py
Extracts shared-transform real-vs-fake image pairs from FaceForensics++ C23 videos.

Key concept:
For each pair of videos (real parent video '000.mp4' and manipulated video '000_003.mp4'):
1. Sample identical frame timestamps.
2. Estimate the facial bounding box on the real frame.
3. Apply the IDENTICAL spatial crop transform to BOTH the real and fake frames.
4. Save both crops as lossless PNG at 224x224.

This guarantees:
- Crop center MAD = 0 px (P-S1 <= 4 px satisfied).
- Spatial scale, aspect ratio, and face coordinate equalized across classes.
- Container format equalized (both lossless PNG).
- Background difference outside face ellipse is at the video codec noise floor.

Usage:
  python scripts/extract_shared_transform_pairs.py --ffpp-dir datasets/FaceForensics++_C23 --n-pairs 600 --n-frames 4 --seed 42
Self-test:
  python scripts/extract_shared_transform_pairs.py --self-test
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

METHODS = ["Deepfakes", "Face2Face", "FaceSwap", "FaceShifter", "NeuralTextures"]
COMPOSITED_METHODS = {"Deepfakes", "FaceSwap", "FaceShifter"}
TARGET_SIZE = 224


def sha256_file(p):
    p = Path(p)
    if not p.exists():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while c := f.read(8 << 20):
            h.update(c)
    return h.hexdigest()


def load_splits(splits_dir: Path):
    splits = {}
    for name in ("train", "val", "test"):
        p = splits_dir / f"{name}.json"
        if p.exists():
            vids = json.loads(p.read_text(encoding="utf-8"))
            for v in vids:
                splits[str(v)] = name
    return splits


def get_face_box(frame: np.ndarray, app=None):
    """Detect face bbox [x1, y1, x2, y2] or fallback to central crop."""
    h, w = frame.shape[:2]
    if app is not None:
        try:
            faces = app.get(frame)
            if faces:
                # pick largest face
                f = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
                x1, y1, x2, y2 = f.bbox
                return float(x1), float(y1), float(x2), float(y2)
        except Exception:
            pass
    # Fallback to standard central 50% box
    s = min(w, h) * 0.5
    cx, cy = w / 2.0, h / 2.0
    return cx - s / 2.0, cy - s / 2.0, cx + s / 2.0, cy + s / 2.0


def crop_with_transform(frame: np.ndarray, bbox: tuple[float, float, float, float], size: int = TARGET_SIZE):
    """Crop square region around bbox with 1.3 margin and resize to size."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    s = max(bw, bh) * 1.3
    s = max(s, 64.0)

    half = s / 2.0
    src_pts = np.float32([
        [cx - half, cy - half],
        [cx + half, cy - half],
        [cx + half, cy + half],
    ])
    dst_pts = np.float32([
        [0, 0],
        [size - 1, 0],
        [size - 1, size - 1],
    ])
    M = cv2.getAffineTransform(src_pts, dst_pts)
    crop = cv2.warpAffine(frame, M, (size, size), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT)
    return crop


def run_extraction(args, app=None):
    root = Path(args.ffpp_dir)
    splits = load_splits(Path(args.splits_dir) if args.splits_dir else root)
    rng = np.random.default_rng(args.seed)

    orig_dir = root / "original"
    if not orig_dir.exists():
        sys.exit(f"Error: original video directory not found at {orig_dir}")

    # Discover pairs across methods
    pairs = []  # list of (parent_id, fake_video_name, method, fake_path, real_path, split)
    for m in METHODS:
        mdir = root / m
        if not mdir.exists():
            continue
        for fpath in sorted(mdir.glob("*.mp4")):
            stem = fpath.stem
            parts = stem.split("_")
            if len(parts) == 2:
                parent_id = parts[0]
                rpath = orig_dir / f"{parent_id}.mp4"
                if rpath.exists():
                    sp = splits.get(parent_id, "train")
                    pairs.append((parent_id, stem, m, fpath, rpath, sp))

    print(f"Discovered {len(pairs)} available real-fake video pairs across {len(METHODS)} methods.")
    if not pairs:
        sys.exit("No pairs found. Check directory paths.")

    rng.shuffle(pairs)
    if args.n_pairs and args.n_pairs < len(pairs):
        pairs = pairs[: args.n_pairs]
    print(f"Selected {len(pairs)} pairs for extraction ({args.n_frames} frames per pair).")

    out_crops = Path(args.out_crops_dir)
    real_out_dir = out_crops / "reals"
    fake_out_dir = out_crops / "fakes"
    real_out_dir.mkdir(parents=True, exist_ok=True)
    fake_out_dir.mkdir(parents=True, exist_ok=True)

    reals_manifest = []
    fakes_manifest = []
    extracted_frames = 0
    saved_reals = set()

    for p_idx, (parent_id, fake_id, method, fpath, rpath, split) in enumerate(pairs):
        cap_r = cv2.VideoCapture(str(rpath))
        cap_f = cv2.VideoCapture(str(fpath))

        count_r = int(cap_r.get(cv2.CAP_PROP_FRAME_COUNT))
        count_f = int(cap_f.get(cv2.CAP_PROP_FRAME_COUNT))
        n_frames_avail = min(count_r, count_f)

        if n_frames_avail < 25:
            cap_r.release()
            cap_f.release()
            continue

        # Sample frame indices avoiding first/last 10 frames
        lo, hi = 10, max(11, n_frames_avail - 10)
        indices = np.linspace(lo, hi - 1, args.n_frames, dtype=int)

        for fi in indices:
            cap_r.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
            cap_f.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
            ret_r, frame_r = cap_r.read()
            ret_f, frame_f = cap_f.read()

            if not ret_r or not ret_f:
                continue

            if getattr(args, "align_mode", "shared") == "independent":
                bbox_r = get_face_box(frame_r, app=app)
                bbox_f = get_face_box(frame_f, app=app)
                crop_r = crop_with_transform(frame_r, bbox_r, TARGET_SIZE)
                crop_f = crop_with_transform(frame_f, bbox_f, TARGET_SIZE)
            else:
                # Shared transform estimated on real frame and applied identically to both
                bbox = get_face_box(frame_r, app=app)
                crop_r = crop_with_transform(frame_r, bbox, TARGET_SIZE)
                crop_f = crop_with_transform(frame_f, bbox, TARGET_SIZE)

            r_name = f"{parent_id}_f{fi:05d}.png"
            f_name = f"{method}_{fake_id}_f{fi:05d}.png"

            r_crop_path = real_out_dir / r_name
            f_crop_path = fake_out_dir / f_name

            if not r_crop_path.exists():
                cv2.imwrite(str(r_crop_path), crop_r, [cv2.IMWRITE_PNG_COMPRESSION, 0])
                reals_manifest.append({
                    "label": 0,
                    "crop_path": str(r_crop_path).replace("\\", "/"),
                    "source_path": str(rpath).replace("\\", "/"),
                    "video_id": parent_id,
                    "frame_idx": int(fi),
                    "split": split,
                })
                saved_reals.add(r_name)

            cv2.imwrite(str(f_crop_path), crop_f, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            fakes_manifest.append({
                "label": 1,
                "source_path": str(f_crop_path).replace("\\", "/"),
                "method": method,
                "forgery_type": "composited" if method in COMPOSITED_METHODS else "reenactment",
                "video_id": fake_id,
                "frame_idx": int(fi),
                "split": split,
                "video_path": str(fpath).replace("\\", "/"),
            })
            extracted_frames += 1

        cap_r.release()
        cap_f.release()

        if (p_idx + 1) % 50 == 0 or (p_idx + 1) == len(pairs):
            print(f"Progress: {p_idx + 1}/{len(pairs)} pairs processed ({extracted_frames} frame pairs).")

    # Write manifests
    out_reals_p = Path(args.out_reals)
    out_fakes_p = Path(args.out_fakes)
    out_reals_p.parent.mkdir(parents=True, exist_ok=True)
    out_fakes_p.parent.mkdir(parents=True, exist_ok=True)

    with open(out_reals_p, "w", encoding="utf-8") as f:
        for r in reals_manifest:
            f.write(json.dumps(r) + "\n")
    with open(out_fakes_p, "w", encoding="utf-8") as f:
        for r in fakes_manifest:
            f.write(json.dumps(r) + "\n")

    print(f"\nExtraction complete:")
    print(f"Reals: {len(reals_manifest)} crops -> {out_reals_p}")
    print(f"Fakes: {len(fakes_manifest)} crops -> {out_fakes_p}")
    return len(reals_manifest), len(fakes_manifest)


def self_test():
    tmp = Path(tempfile.mkdtemp())
    orig = tmp / "original"
    df_dir = tmp / "Deepfakes"
    orig.mkdir(parents=True)
    df_dir.mkdir(parents=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    for vid, fn in [("001", orig / "001.mp4"), ("001_002", df_dir / "001_002.mp4")]:
        out = cv2.VideoWriter(str(fn), fourcc, 25.0, (320, 240))
        for i in range(30):
            img = np.zeros((240, 320, 3), dtype=np.uint8)
            cv2.circle(img, (160, 120), 40, (200, 200, 200), -1)
            out.write(img)
        out.release()

    (tmp / "train.json").write_text(json.dumps(["001"]))
    args = argparse.Namespace(
        ffpp_dir=str(tmp),
        splits_dir=str(tmp),
        out_crops_dir=str(tmp / "crops"),
        out_reals=str(tmp / "reals.jsonl"),
        out_fakes=str(tmp / "fakes.jsonl"),
        n_pairs=1,
        n_frames=2,
        seed=42,
        align_mode="shared",
    )
    nr, nf = run_extraction(args, app=None)
    ok_shared = nr == 2 and nf == 2 and Path(args.out_reals).exists() and Path(args.out_fakes).exists()

    args_indep = argparse.Namespace(
        ffpp_dir=str(tmp),
        splits_dir=str(tmp),
        out_crops_dir=str(tmp / "crops_indep"),
        out_reals=str(tmp / "reals_indep.jsonl"),
        out_fakes=str(tmp / "fakes_indep.jsonl"),
        n_pairs=1,
        n_frames=2,
        seed=42,
        align_mode="independent",
    )
    nr_i, nf_i = run_extraction(args_indep, app=None)
    ok_indep = nr_i == 2 and nf_i == 2 and Path(args_indep.out_reals).exists() and Path(args_indep.out_fakes).exists()

    ok = ok_shared and ok_indep
    print("extract_shared_transform_pairs self-test", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description="Extract shared-transform real-vs-fake image pairs from FF++ C23")
    ap.add_argument("--ffpp-dir", default="datasets/FaceForensics++_C23")
    ap.add_argument("--splits-dir", default=None)
    ap.add_argument("--out-crops-dir", default="datasets/FaceForensics++_C23/shared_transform_crops")
    ap.add_argument("--out-reals", default="data/ff_shared_reals.jsonl")
    ap.add_argument("--out-fakes", default="data/ff_shared_fakes.jsonl")
    ap.add_argument("--align-mode", choices=["shared", "independent"], default="shared",
                    help="shared=identical crop on real and fake; independent=detect face box on each frame independently")
    ap.add_argument("--n-pairs", type=int, default=600)
    ap.add_argument("--n-frames", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()

    if a.self_test:
        return self_test()

    app = None
    try:
        from insightface.app import FaceAnalysis
        app = FaceAnalysis(name="buffalo_sc", providers=["CPUExecutionProvider"])
        app.prepare(ctx_id=-1, det_size=(640, 640))
        print("Initialized InsightFace buffalo_sc detector for face tracking.")
    except Exception as e:
        print(f"Warning: InsightFace not available ({e}), using geometric face fallback.")

    run_extraction(a, app=app)
    return 0


if __name__ == "__main__":
    sys.exit(main())
