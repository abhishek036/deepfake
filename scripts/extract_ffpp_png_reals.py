"""
extract_ffpp_png_reals.py
=========================
Extract FaceForensics++ c23 original frames as PNG using the IDENTICAL spatial
pipeline as the DF40 fake crops (centre-crop -> LANCZOS 224), so that the two
sides of the audit pairing share the same container format, resolution, and
compression history.

Goals
-----
- Eliminate the .jpg vs .png container confound (pre-flight audit flag).
- Eliminate the pre-existing JPEG DCT history in the real crops.
- Give the low-level probe a real test: if sharpness and file-size are now
  matched, probe CV-AUROC should drop from ~0.87 toward 0.55.

Usage
-----
    python scripts/extract_ffpp_png_reals.py \
        --video-dir    datasets/FaceForensics++_C23/original \
        --split-csv    datasets/FaceForensics++_C23/csv/test.csv \
        --out-dir      datasets/FaceForensics++_C23/original_png_224 \
        --manifest     data/ffpp_png_reals.jsonl \
        --fps-limit    1 \
        --max-per-video 5 \
        --n-videos      200 \
        --seed          42

    The resulting manifest can be passed to one_pipeline_audit.py as
        --real-manifest data/ffpp_png_reals.jsonl --real-split any

Dependencies
------------
    pip install opencv-python pillow

Notes
-----
- We extract at most --max-per-video frames per video (default 5), chosen at
  uniform frame intervals, to limit temporal autocorrelation.
- Frame index is deterministic (seed-controlled) for reproducibility.
- The extracted PNG uses lossless compression (cv2.IMWRITE_PNG_COMPRESSION=0)
  to eliminate any new compression history from this step.
- A sha256 of this script is written into the manifest header so the ledger
  can verify the exact extraction code used.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


def sha256_file(path) -> str | None:
    p = Path(path)
    if not p.exists():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(8 << 20):
            h.update(chunk)
    return h.hexdigest()


def centre_crop_resize(frame_bgr, size: int = 224):
    """Replicate canonicalize() from one_pipeline_audit.py in OpenCV."""
    import cv2
    h, w = frame_bgr.shape[:2]
    s = min(h, w)
    t, l = (h - s) // 2, (w - s) // 2
    crop = frame_bgr[t:t + s, l:l + s]
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_LANCZOS4)


def load_split_ids(split_csv: Path | None) -> set | None:
    """Load video IDs from a FF++ split CSV (columns: video_id OR first column)."""
    if split_csv is None or not split_csv.exists():
        return None
    ids = set()
    with open(split_csv, newline="") as f:
        for row in csv.reader(f):
            if row:
                ids.add(row[0].strip())
    return ids


def extract_frames(video_path: Path, out_dir: Path, video_id: str,
                   max_per_video: int, fps_limit: float, rng) -> list:
    """
    Extract up to max_per_video frames from video_path, save as PNG lossless,
    and return a list of manifest records.
    """
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [WARN] cannot open {video_path}")
        return []

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, int(src_fps / fps_limit)) if fps_limit > 0 else 1

    candidates = list(range(0, total, step))
    if not candidates:
        cap.release()
        return []

    if len(candidates) > max_per_video:
        chosen_idx = np.linspace(0, len(candidates) - 1, max_per_video, dtype=int)
        candidates = [candidates[i] for i in chosen_idx]

    out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for fi in candidates:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        if not ret:
            continue
        processed = centre_crop_resize(frame, 224)
        out_path = out_dir / f"{video_id}_f{fi:06d}.png"
        cv2.imwrite(str(out_path), processed,
                    [cv2.IMWRITE_PNG_COMPRESSION, 0])   # lossless, no DCT history
        records.append({
            "label": 0,
            "split": "any",
            "path": str(out_path),
            "video_id": video_id,
            "source_video": str(video_path),
            "frame_index": fi,
            "extraction_size": 224,
            "format": "PNG_lossless_LANCZOS",
        })

    cap.release()
    return records


def main():
    ap = argparse.ArgumentParser(description="Extract FF++ c23 original frames as PNG (audit-clean)")
    ap.add_argument("--video-dir",     required=True, help="Directory of .mp4 original videos")
    ap.add_argument("--split-csv",     default=None,  help="FF++ split CSV (optional; filters video IDs)")
    ap.add_argument("--out-dir",       required=True, help="Output directory for PNG frames")
    ap.add_argument("--manifest",      required=True, help="Output JSONL manifest path")
    ap.add_argument("--fps-limit",     type=float, default=1.0,  help="Max frames per second to sample")
    ap.add_argument("--max-per-video", type=int,   default=5,    help="Max frames per video")
    ap.add_argument("--n-videos",      type=int,   default=None, help="Limit number of videos processed")
    ap.add_argument("--seed",          type=int,   default=42)
    args = ap.parse_args()

    try:
        import cv2  # noqa
    except ImportError:
        print("ERROR: opencv-python is required. Run: pip install opencv-python", file=sys.stderr)
        sys.exit(1)

    rng = np.random.default_rng(args.seed)
    video_dir = Path(args.video_dir)
    out_dir   = Path(args.out_dir)
    manifest  = Path(args.manifest)
    manifest.parent.mkdir(parents=True, exist_ok=True)

    split_ids = load_split_ids(Path(args.split_csv) if args.split_csv else None)
    videos = sorted(video_dir.glob("*.mp4"))

    if split_ids is not None:
        videos = [v for v in videos if v.stem in split_ids]
        print(f"After split filter: {len(videos)} videos")

    if args.n_videos:
        idxs = rng.choice(len(videos), min(args.n_videos, len(videos)), replace=False)
        videos = sorted([videos[i] for i in idxs])

    print(f"Extracting from {len(videos)} videos -> {out_dir}")

    all_records = []
    meta = {
        "_type": "manifest_header",
        "extraction_script_sha256": sha256_file(__file__),
        "fps_limit": args.fps_limit,
        "max_per_video": args.max_per_video,
        "pipeline": "centre_crop_LANCZOS_224_PNG_lossless",
        "seed": args.seed,
        "n_videos_requested": args.n_videos,
        "split_csv": args.split_csv,
    }

    for v in videos:
        video_id = v.stem
        recs = extract_frames(v, out_dir, video_id, args.max_per_video, args.fps_limit, rng)
        all_records.extend(recs)
        if len(all_records) % 100 == 0 and all_records:
            print(f"  ... {len(all_records)} frames so far")

    print(f"Total extracted: {len(all_records)} frames from {len(videos)} videos")

    with open(manifest, "w", encoding="utf-8") as f:
        f.write(json.dumps(meta) + "\n")
        for r in all_records:
            f.write(json.dumps(r) + "\n")

    print(f"Manifest written to: {manifest}")
    print("\nNext step -- run the audit on the clean pairing:")
    print(f"  python -m evaluation.one_pipeline_audit \\")
    print(f"    --real-manifest {manifest} \\")
    print(f"    --real-split any \\")
    print(f"    --fake-manifest <your_df40_fake_manifest.jsonl> \\")
    print(f"    --out-dir reports/clean_pairing_audit \\")
    print(f"    --skip-model   # run probe first to verify AUROC <= 0.55")


if __name__ == "__main__":
    main()
