"""
data/setup_df40_from_disk.py
=============================
Build data/manifests/df40_disk.jsonl by DIRECTLY SCANNING the DF40 directories
on disk at data/raw/DF40/.

This is the authoritative manifest builder for DF40 — it walks the actual
file tree rather than relying on JSON config paths (which may be relative to
a different machine).

DF40 disk layout (discovered by inspection):
  data/raw/DF40/
    DF40_train/
      <method>/         (e.g. blendface/, DiT/, wav2lip/, ...)
        <method>/
          frames/
            <video_id>/
              000.png, 001.png, ...
          landmarks/
    inswap/             (root-level, different structure)
      frames/<video_id>/*.png
    simswap/
      frames/<video_id>/*.png
    sd2.1/
      <batch_id>/<video_id>.png   (image-level, not video frames)
    RDDM/
      <batch_id>/<filename>.png   (image-level)

Usage:
    python data/setup_df40_from_disk.py
    python data/setup_df40_from_disk.py --dry-run
    python data/setup_df40_from_disk.py --output data/manifests/df40_disk.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Forgery type mapping (MASTER_PLAN §5B)
# ─────────────────────────────────────────────────────────────────────────────
FORGERY_TYPE: dict[str, str] = {
    # Fully synthetic
    "DiT": "fully_synthetic", "SiT": "fully_synthetic", "sd2.1": "fully_synthetic",
    "pixart": "fully_synthetic", "ddim": "fully_synthetic", "RDDM": "fully_synthetic",
    "rddm": "fully_synthetic", "CollabDiff": "fully_synthetic",
    "StyleGAN2": "fully_synthetic", "StyleGAN3": "fully_synthetic",
    "StyleGANXL": "fully_synthetic", "VQGAN": "fully_synthetic",
    "MidJourney": "fully_synthetic", "stargan": "fully_synthetic",
    "starganv2": "fully_synthetic", "styleclip": "fully_synthetic",
    "whichisreal": "fully_synthetic", "e4e": "fully_synthetic", "e4s": "fully_synthetic",
    # Face Swap
    "faceswap": "composited", "fsgan": "composited", "simswap": "composited",
    "blendface": "composited", "facedancer": "composited", "inswap": "composited",
    "mobileswap": "composited", "uniface": "composited", "deepfacelab": "composited",
    "danet": "composited",
    # Reenactment
    "fomm": "reenactment", "facevid2vid": "reenactment", "hyperreenact": "reenactment",
    "pirender": "reenactment", "lia": "reenactment", "mcnet": "reenactment",
    "tpsm": "reenactment", "one_shot_free": "reenactment", "MRAA": "reenactment",
    # Talking head
    "sadtalker": "talking_head", "wav2lip": "talking_head", "heygen": "talking_head",
}

MANIPULATION_FAMILY: dict[str, str] = {
    "faceswap": "face_swap", "fsgan": "face_swap", "simswap": "face_swap",
    "blendface": "face_swap", "facedancer": "face_swap", "inswap": "face_swap",
    "mobileswap": "face_swap", "uniface": "face_swap", "deepfacelab": "face_swap",
    "danet": "face_swap",
    "fomm": "reenactment", "facevid2vid": "reenactment", "hyperreenact": "reenactment",
    "pirender": "reenactment", "lia": "reenactment", "mcnet": "reenactment",
    "tpsm": "reenactment", "one_shot_free": "reenactment", "MRAA": "reenactment",
    "sadtalker": "talking_head", "wav2lip": "talking_head", "heygen": "talking_head",
}

SYNTHESIS_METHODS = {
    "DiT", "SiT", "sd2.1", "pixart", "ddim", "RDDM", "rddm", "CollabDiff",
    "StyleGAN2", "StyleGAN3", "StyleGANXL", "VQGAN", "MidJourney",
    "stargan", "starganv2", "styleclip", "whichisreal", "e4e", "e4s",
}

# Stable hash split matching data/splits.py
def _stable_hash(key: str, seed: int = 42) -> float:
    h = hashlib.md5(f"{seed}:{key}".encode()).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF

def assign_split(video_id: str, seed: int = 42) -> str:
    h = _stable_hash(video_id, seed)
    if h < 0.70:
        return "train"
    elif h < 0.85:
        return "val"
    else:
        return "test"

def make_record_id(path: str) -> str:
    return hashlib.md5(path.encode()).hexdigest()[:16]

def make_record(
    frame_path: Path,
    method: str,
    video_id: str,
    frame_idx: int,
    df40_root: Path,
) -> dict:
    project_root = df40_root.parent.parent.parent
    rel_path = frame_path.relative_to(project_root).as_posix()
    source_path = rel_path  # e.g. data/raw/DF40/DF40_train/blendface/blendface/frames/001_870/000.png

    forgery_type      = FORGERY_TYPE.get(method, "unknown")
    manip_family      = MANIPULATION_FAMILY.get(method, "synthesis" if method in SYNTHESIS_METHODS else "unknown")
    split             = assign_split(f"{method}_{video_id}")

    return {
        "record_id":             make_record_id(source_path),
        "video_id":              f"df40_{method}_{video_id}",
        "dataset":               "DF40",
        "split":                 split,
        "source_group_id":       f"df40_{method}_{video_id}",
        "source_path":           source_path,
        "frame_idx":             frame_idx,
        "label":                 1,
        "label_int":             1,
        "method":                method,
        "manipulation":          method,
        "forgery_type":          forgery_type,
        "manipulation_family":   manip_family,
        "generator_family":      method,
        "audio_available":       False,
        "sufficient_for_rppg":   False,
        "sufficient_for_analysis": True,
        "skin_tone_tag":         "unknown",
        "source": "disk_scan",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Scanners per disk layout style
# ─────────────────────────────────────────────────────────────────────────────

def scan_df40_train_method(method_dir: Path, method: str, df40_root: Path) -> list[dict]:
    """
    Scan DF40_train/<method>/ — handles two layouts:

    Layout A (video-level, face-swap/reenactment methods):
      <method>/<method>/frames/<video_id>/*.png

    Layout B (image-level batch, EFS/diffusion methods):
      <method>/<method>/<batch_id>/<image>.png
      (no 'frames' subdir — images are direct children of batch dirs)
    """
    records = []

    # Check for Layout A first (has frames/ subdir)
    frames_dirs = list(method_dir.rglob("frames"))
    if frames_dirs:
        # Layout A — video-level frames
        for frames_dir in frames_dirs:
            for video_dir in sorted(frames_dir.iterdir()):
                if not video_dir.is_dir():
                    continue
                video_id = video_dir.name
                for idx, frame_file in enumerate(sorted(video_dir.glob("*.png"))):
                    records.append(make_record(frame_file, method, video_id, idx, df40_root))
        return records

    # Layout B — image-level batch (DiT, SiT, StyleGAN*, VQGAN, ddim, etc.)
    # Structure: <method>/<method>/<batch_id>/<image>.png
    # Walk: method_dir -> inner_dir (same name) -> batch_dirs -> images
    inner_dirs = [d for d in method_dir.iterdir() if d.is_dir()]
    if not inner_dirs:
        logger.warning("Empty method dir: %s", method_dir)
        return records

    found_images = False
    for inner_dir in sorted(inner_dirs):
        # Each inner_dir is a batch folder or another level
        batch_dirs = [d for d in inner_dir.iterdir() if d.is_dir()]
        if batch_dirs:
            # inner_dir contains batch subdirs
            for batch_dir in sorted(batch_dirs):
                for idx, img_file in enumerate(sorted(batch_dir.glob("*.png"))):
                    video_id = f"{batch_dir.name}_{img_file.stem}"
                    records.append(make_record(img_file, method, video_id, 0, df40_root))
                    found_images = True
        else:
            # inner_dir directly contains images (one level flatter)
            for idx, img_file in enumerate(sorted(inner_dir.glob("*.png"))):
                video_id = f"{inner_dir.name}_{img_file.stem}"
                records.append(make_record(img_file, method, video_id, 0, df40_root))
                found_images = True

    if not found_images:
        logger.warning("No PNG frames found under %s (Layout B search)", method_dir)

    return records


def scan_root_method_flat(method_dir: Path, method: str, df40_root: Path) -> list[dict]:
    """
    Scan <method>/frames/<video_id>/*.png  (inswap, simswap style)
    """
    records = []
    frames_dir = method_dir / "frames"
    if not frames_dir.exists():
        logger.warning("No frames/ dir in %s", method_dir)
        return records

    for video_dir in sorted(frames_dir.iterdir()):
        if not video_dir.is_dir():
            continue
        video_id = video_dir.name
        for idx, frame_file in enumerate(sorted(video_dir.glob("*.png"))):
            records.append(make_record(frame_file, method, video_id, idx, df40_root))

    return records


def scan_root_method_images(method_dir: Path, method: str, df40_root: Path) -> list[dict]:
    """
    Scan <method>/<batch>/<image>.png  (sd2.1, RDDM style — image-level, not video frames)
    """
    records = []
    for batch_dir in sorted(method_dir.iterdir()):
        if not batch_dir.is_dir():
            continue
        for idx, img_file in enumerate(sorted(batch_dir.glob("*.png"))):
            video_id = f"{batch_dir.name}_{img_file.stem}"
            records.append(make_record(img_file, method, video_id, 0, df40_root))

    return records


# ─────────────────────────────────────────────────────────────────────────────
# Main scanner
# ─────────────────────────────────────────────────────────────────────────────

def build_manifest(
    df40_root: Path,
    output_path: Path,
    dry_run: bool = False,
) -> dict:
    """
    Walk data/raw/DF40/ and build complete manifest from actual files on disk.
    """
    if not df40_root.exists():
        raise FileNotFoundError(f"DF40 root not found: {df40_root}")

    all_records: list[dict] = []
    seen_ids: set[str] = set()
    method_counts: dict[str, int] = defaultdict(int)
    forgery_type_counts: dict[str, int] = defaultdict(int)

    def add_records(recs: list[dict], method: str) -> None:
        added = 0
        for rec in recs:
            rid = rec["record_id"]
            if rid in seen_ids:
                continue
            seen_ids.add(rid)
            all_records.append(rec)
            method_counts[method] += 1
            forgery_type_counts[rec["forgery_type"]] += 1
            added += 1
        logger.info("  %-20s → %d records added", method + ":", added)

    # ── DF40_train/<method>/ ──────────────────────────────────────────────
    train_dir = df40_root / "DF40_train"
    if train_dir.exists():
        logger.info("Scanning DF40_train/ ...")
        for method_dir in sorted(train_dir.iterdir()):
            if not method_dir.is_dir():
                continue
            method = method_dir.name
            recs = scan_df40_train_method(method_dir, method, df40_root)
            add_records(recs, method)
    else:
        logger.warning("DF40_train/ not found at %s", train_dir)

    # ── Root-level extracted dirs (inswap, simswap — have frames/ subdir) ──
    flat_methods = ["inswap", "simswap"]
    for method in flat_methods:
        d = df40_root / method
        if d.exists():
            logger.info("Scanning root-level flat: %s ...", method)
            recs = scan_root_method_flat(d, method, df40_root)
            add_records(recs, method)

    # ── Root-level image-level dirs (pixart, sd2.1, RDDM) ─────────────────
    image_methods = [("pixart", "pixart"), ("sd2.1", "sd2.1"), ("RDDM", "rddm")]
    for dirname, method_name in image_methods:
        d = df40_root / dirname
        if d.exists():
            logger.info("Scanning root-level image-style: %s ...", dirname)
            recs = scan_root_method_images(d, method_name, df40_root)
            add_records(recs, method_name)

    # ── Statistics ────────────────────────────────────────────────────────
    split_counts: dict[str, int] = defaultdict(int)
    for rec in all_records:
        split_counts[rec["split"]] += 1

    logger.info("=" * 70)
    logger.info("DF40 DISK SCAN MANIFEST STATISTICS")
    logger.info("=" * 70)
    logger.info("Total records:  %d", len(all_records))
    logger.info("Split breakdown:")
    for split in ["train", "val", "test"]:
        logger.info("  %-12s %d", split + ":", split_counts.get(split, 0))
    logger.info("\nForgery-type breakdown:")
    for ft, n in sorted(forgery_type_counts.items()):
        logger.info("  %-30s %d", ft + ":", n)
    logger.info("\nMethod breakdown:")
    for method, n in sorted(method_counts.items(), key=lambda x: -x[1]):
        logger.info("  %-25s %d", method + ":", n)
    logger.info("=" * 70)

    if dry_run:
        logger.info("DRY RUN — no files written.")
        return {"total": len(all_records), "forgery_types": dict(forgery_type_counts)}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for rec in all_records:
            f.write(json.dumps(rec) + "\n")

    logger.info("Written %d records → %s", len(all_records), output_path)
    return {"total": len(all_records), "forgery_types": dict(forgery_type_counts)}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build DF40 manifest by scanning data/raw/DF40/ directly"
    )
    parser.add_argument(
        "--df40-root",
        type=Path,
        default=Path("data/raw/DF40"),
        help="Root directory of DF40 data (default: data/raw/DF40)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/manifests/df40_disk.jsonl"),
        help="Output JSONL (default: data/manifests/df40_disk.jsonl)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print statistics without writing",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    df40_root  = args.df40_root if args.df40_root.is_absolute() else project_root / args.df40_root
    output_path = args.output if args.output.is_absolute() else project_root / args.output

    stats = build_manifest(df40_root=df40_root, output_path=output_path, dry_run=args.dry_run)
    logger.info("Done: total=%d", stats["total"])


if __name__ == "__main__":
    main()
