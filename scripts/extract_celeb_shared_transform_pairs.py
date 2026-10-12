"""
scripts/extract_celeb_shared_transform_pairs.py
Extracts shared-transform real-vs-fake pairs from Celeb-DF-v2.

Celeb-DF-v2 pairing: each fake video 'id0_id16_0000.mp4' maps to real
'id0_0000.mp4' via the target_video field. We:
  1. Sample identical frame indices from both videos.
  2. Detect face on the real frame only.
  3. Apply the IDENTICAL spatial crop transform to both real and fake frames.
  4. Save both as lossless PNG at 224x224.

Same shared-transform protocol used on FF++ (scripts/extract_shared_transform_pairs.py),
adapted for Celeb-DF-v2 naming conventions.

Usage:
  python scripts/extract_celeb_shared_transform_pairs.py
      --celeb-dir data/raw/Celeb-DF-v2
      --n-pairs 300 --n-frames 4 --seed 42

Self-test:
  python scripts/extract_celeb_shared_transform_pairs.py --self-test
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

TARGET_SIZE = 224


def sha256_file(p: Path):
    if not p.exists():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(8 << 20):
            h.update(chunk)
    return h.hexdigest()


def get_face_box(frame: np.ndarray, app=None):
    h, w = frame.shape[:2]
    if app is not None:
        try:
            faces = app.get(frame)
            if faces:
                f = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
                x1, y1, x2, y2 = f.bbox
                return float(x1), float(y1), float(x2), float(y2)
        except Exception:
            pass
    s = min(w, h) * 0.5
    cx, cy = w / 2.0, h / 2.0
    return cx - s / 2.0, cy - s / 2.0, cx + s / 2.0, cy + s / 2.0


def crop_with_transform(frame: np.ndarray, bbox: tuple, size: int = TARGET_SIZE) -> np.ndarray:
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
    return cv2.warpAffine(frame, M, (size, size),
                          flags=cv2.INTER_LANCZOS4,
                          borderMode=cv2.BORDER_REFLECT)


def build_celeb_pairs(celeb_dir: Path):
    real_dir = celeb_dir / "Celeb-real"
    fake_dir = celeb_dir / "Celeb-synthesis"
    real_index = {rp.stem: rp for rp in sorted(real_dir.glob("*.mp4"))}
    pairs = []
    for fp in sorted(fake_dir.glob("*.mp4")):
        parts = fp.stem.split("_")
        if len(parts) >= 3:
            parent_id = parts[0] + "_" + parts[-1]  # id0_id16_0000 -> id0_0000
            if parent_id in real_index:
                pairs.append((real_index[parent_id], fp, parent_id, fp.stem))
    return pairs


def run_extraction(args, app=None):
    celeb_dir = Path(args.celeb_dir)
    pairs = build_celeb_pairs(celeb_dir)
    print(f"Discovered {len(pairs)} real-fake video pairs in Celeb-DF-v2.")
    if not pairs:
        sys.exit("No pairs found. Check --celeb-dir path.")

    rng = np.random.default_rng(args.seed)
    indices_arr = rng.permutation(len(pairs))
    pairs = [pairs[i] for i in indices_arr]
    if args.n_pairs and args.n_pairs < len(pairs):
        pairs = pairs[: args.n_pairs]
    print(f"Selected {len(pairs)} pairs ({args.n_frames} frames each).")

    out_crops = Path(args.out_crops_dir)
    real_out_dir = out_crops / "reals"
    fake_out_dir = out_crops / "fakes"
    real_out_dir.mkdir(parents=True, exist_ok=True)
    fake_out_dir.mkdir(parents=True, exist_ok=True)

    reals_manifest = []
    fakes_manifest = []
    extracted = 0
    saved_reals = set()

    for p_idx, (rpath, fpath, parent_id, fake_id) in enumerate(pairs):
        cap_r = cv2.VideoCapture(str(rpath))
        cap_f = cv2.VideoCapture(str(fpath))
        count_r = int(cap_r.get(cv2.CAP_PROP_FRAME_COUNT))
        count_f = int(cap_f.get(cv2.CAP_PROP_FRAME_COUNT))
        n_avail = min(count_r, count_f)
        if n_avail < 25:
            cap_r.release()
            cap_f.release()
            continue
        lo, hi = 10, max(11, n_avail - 10)
        frame_indices = np.linspace(lo, hi - 1, args.n_frames, dtype=int)
        for fi in frame_indices:
            cap_r.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
            cap_f.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
            ret_r, frame_r = cap_r.read()
            ret_f, frame_f = cap_f.read()
            if not ret_r or not ret_f:
                continue
            bbox = get_face_box(frame_r, app=app)
            crop_r = crop_with_transform(frame_r, bbox)
            crop_f = crop_with_transform(frame_f, bbox)
            r_name = f"{parent_id}_f{fi:05d}.png"
            f_name = f"{fake_id}_f{fi:05d}.png"
            r_crop_path = real_out_dir / r_name
            f_crop_path = fake_out_dir / f_name
            if r_name not in saved_reals:
                cv2.imwrite(str(r_crop_path), crop_r, [cv2.IMWRITE_PNG_COMPRESSION, 0])
                reals_manifest.append({
                    "label": 0,
                    "crop_path": str(r_crop_path).replace("\\", "/"),
                    "source_path": str(r_crop_path).replace("\\", "/"),
                    "video_path": str(rpath).replace("\\", "/"),
                    "video_id": parent_id,
                    "frame_idx": int(fi),
                    "split": "test",
                })
                saved_reals.add(r_name)
            cv2.imwrite(str(f_crop_path), crop_f, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            fakes_manifest.append({
                "label": 1,
                "crop_path": str(f_crop_path).replace("\\", "/"),
                "source_path": str(f_crop_path).replace("\\", "/"),
                "video_path": str(fpath).replace("\\", "/"),
                "method": "Celeb-synthesis",
                "forgery_type": "face_swap",
                "video_id": fake_id,
                "target_video": parent_id,
                "frame_idx": int(fi),
                "split": "test",
            })
            extracted += 1
        cap_r.release()
        cap_f.release()
        if (p_idx + 1) % 50 == 0 or (p_idx + 1) == len(pairs):
            print(f"  {p_idx + 1}/{len(pairs)} pairs, {extracted} frame pairs extracted.")

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
    print(f"\nDone: {len(reals_manifest)} reals, {len(fakes_manifest)} fakes.")
    return len(reals_manifest), len(fakes_manifest)


def self_test():
    tmp = Path(tempfile.mkdtemp())
    real_dir = tmp / "Celeb-DF-v2" / "Celeb-real"
    fake_dir = tmp / "Celeb-DF-v2" / "Celeb-synthesis"
    real_dir.mkdir(parents=True)
    fake_dir.mkdir(parents=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    for fn in [real_dir / "id0_0000.mp4", fake_dir / "id0_id16_0000.mp4"]:
        out = cv2.VideoWriter(str(fn), fourcc, 25.0, (320, 240))
        for _ in range(30):
            img = np.zeros((240, 320, 3), dtype=np.uint8)
            cv2.circle(img, (160, 120), 40, (200, 200, 200), -1)
            out.write(img)
        out.release()
    args = argparse.Namespace(
        celeb_dir=str(tmp / "Celeb-DF-v2"),
        out_crops_dir=str(tmp / "crops"),
        out_reals=str(tmp / "celeb_shared_reals.jsonl"),
        out_fakes=str(tmp / "celeb_shared_fakes.jsonl"),
        n_pairs=1, n_frames=2, seed=42,
    )
    nr, nf = run_extraction(args, app=None)
    ok = nr >= 1 and nf >= 1
    print("self-test", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description="Shared-transform extractor for Celeb-DF-v2")
    ap.add_argument("--celeb-dir", default="data/raw/Celeb-DF-v2")
    ap.add_argument("--out-crops-dir", default="data/crops/celeb_shared_transform")
    ap.add_argument("--out-reals", default="data/celeb_shared_reals.jsonl")
    ap.add_argument("--out-fakes", default="data/celeb_shared_fakes.jsonl")
    ap.add_argument("--n-pairs", type=int, default=300)
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
        print("InsightFace buffalo_sc initialized.")
    except Exception as e:
        print(f"Warning: InsightFace not available ({e}), using geometric fallback.")
    run_extraction(a, app=app)
    return 0


if __name__ == "__main__":
    sys.exit(main())
