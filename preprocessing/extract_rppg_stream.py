"""
preprocessing/extract_rppg_stream.py
=====================================
Extract a dedicated 25fps face-crop stream for the rPPG branch.

MASTER_PLAN RISK 2 fix:
  Standard frame sampling at 8fps violates Nyquist for bpm_max=210.
  rPPG requires >= 25fps (Nyquist margin for 3.5Hz = 210 BPM).

  This script runs SEPARATELY from the main preprocessing pipeline and
  outputs a high-fps face-crop sequence for rPPG-only use.

Output structure:
  data/rppg_crops/<dataset>/<video_id>/
      frames/  — 224x224 face crops at 25fps
      meta.json — fps, frame_count, quality flags

Usage:
    python preprocessing/extract_rppg_stream.py --manifest data/manifests/fakeavceleb.jsonl
    python preprocessing/extract_rppg_stream.py --manifest data/manifests/faceforensics.jsonl
    python preprocessing/extract_rppg_stream.py --all
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# MASTER_PLAN rppg_sampling config defaults (mirrored from data.yaml)
RPPG_FPS      = 25       # Must be >= 2 * (bpm_max / 60) = 7.0; use 25 for headroom
RPPG_MAX_FRAMES = 750    # 30s @ 25fps
RPPG_MIN_SEC  = 8.0      # minimum clip duration for reliable PPG


def check_ffmpeg() -> bool:
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True, check=True,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def get_video_duration(video_path: Path) -> float | None:
    """Get video duration in seconds using ffprobe."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(video_path),
            ],
            capture_output=True, text=True, check=True,
        )
        return float(result.stdout.strip())
    except Exception:
        return None


def extract_rppg_frames(
    video_path: Path,
    output_dir: Path,
    fps: int = RPPG_FPS,
    max_frames: int = RPPG_MAX_FRAMES,
    face_size: int = 224,
) -> dict:
    """
    Extract face crops at target fps from video.

    Returns meta dict with extraction results.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_pattern = output_dir / "frame_%06d.jpg"

    # ffmpeg command: extract at target fps, scale to face_size
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vf", f"fps={fps},scale={face_size}:{face_size}:flags=lanczos",
        "-vframes", str(max_frames),
        "-q:v", "2",             # JPEG quality
        "-f", "image2",
        str(frame_pattern),
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            return {
                "success": False,
                "error": result.stderr[:500],
                "frame_count": 0,
            }
    except FileNotFoundError:
        return {"success": False, "error": "ffmpeg not found", "frame_count": 0}

    # Count extracted frames
    frames = sorted(output_dir.glob("frame_*.jpg"))
    frame_count = len(frames)
    duration_sec = frame_count / fps

    meta = {
        "success": True,
        "fps": fps,
        "frame_count": frame_count,
        "duration_sec": duration_sec,
        "sufficient_for_rppg": duration_sec >= RPPG_MIN_SEC,
        "video_path": str(video_path),
        "output_dir": str(output_dir),
        "face_size": face_size,
        "max_frames": max_frames,
    }
    return meta


def process_manifest(
    manifest_path: Path,
    output_root: Path,
    project_root: Path,
    dry_run: bool = False,
    max_videos: int | None = None,
) -> dict[str, int]:
    """
    Process all videos in a manifest that are sufficient for rPPG.

    Only extracts rPPG stream for:
    - Videos with sufficient_for_rppg=True
    - Videos with label=0 (real) OR label=1 where rPPG signal is expected
    - Videos not already processed

    Returns stats dict.
    """
    logger.info("Loading manifest: %s", manifest_path)
    with open(manifest_path, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    logger.info("Loaded %d records", len(records))

    # Group by video_id to get unique videos (records are frame-level)
    video_records: dict[str, dict] = {}
    for rec in records:
        vid = rec.get("video_id", rec.get("record_id", ""))
        if vid not in video_records:
            video_records[vid] = rec

    logger.info("Unique videos: %d", len(video_records))

    # Filter to rPPG-eligible videos
    eligible = {
        vid: rec for vid, rec in video_records.items()
        if rec.get("sufficient_for_rppg", True)  # default True if not set
    }
    logger.info("rPPG-eligible videos: %d", len(eligible))

    if max_videos:
        eligible_items = list(eligible.items())[:max_videos]
        eligible = dict(eligible_items)
        logger.info("Processing at most %d videos (--max-videos)", max_videos)

    stats = defaultdict(int)

    if dry_run:
        logger.info("DRY RUN — no extraction will happen.")
        logger.info("Would process %d videos from %s", len(eligible), manifest_path.name)
        return dict(stats)

    dataset_name = records[0].get("dataset", manifest_path.stem) if records else manifest_path.stem

    for video_id, rec in eligible.items():
        source_path_rel = rec.get("source_path", "")
        if not source_path_rel:
            stats["skipped_no_path"] += 1
            continue

        # Resolve absolute video path
        video_path_norm = source_path_rel.replace("\\", "/")
        video_path = project_root / video_path_norm

        if not video_path.exists():
            logger.debug("Video not found on disk: %s", video_path)
            stats["skipped_missing"] += 1
            continue

        # Check duration
        duration = get_video_duration(video_path)
        if duration is not None and duration < RPPG_MIN_SEC:
            logger.debug("Video too short (%.1fs < %.1fs): %s", duration, RPPG_MIN_SEC, video_id)
            stats["skipped_too_short"] += 1
            continue

        # Output directory
        out_dir = output_root / dataset_name / video_id
        meta_file = out_dir / "meta.json"

        # Skip if already processed
        if meta_file.exists():
            stats["already_done"] += 1
            continue

        # Extract
        meta = extract_rppg_frames(video_path, out_dir)
        if meta["success"]:
            meta_file.write_text(json.dumps(meta, indent=2))
            if meta["sufficient_for_rppg"]:
                stats["success"] += 1
            else:
                stats["success_short"] += 1
            logger.info(
                "✓ %s | frames=%d | duration=%.1fs | sufficient=%s",
                video_id, meta["frame_count"], meta["duration_sec"],
                meta["sufficient_for_rppg"],
            )
        else:
            stats["failed"] += 1
            logger.warning("✗ %s | %s", video_id, meta.get("error", "unknown error"))

    logger.info("=" * 60)
    logger.info("rPPG EXTRACTION COMPLETE (%s)", manifest_path.name)
    for k, v in sorted(stats.items()):
        logger.info("  %-25s %d", k + ":", v)
    logger.info("=" * 60)

    return dict(stats)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract 25fps face-crop stream for rPPG branch (MASTER_PLAN RISK 2 fix)"
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Path to a single JSONL manifest",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all JSONL manifests in data/manifests/",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/rppg_crops"),
        help="Root directory for rPPG crop output (default: data/rppg_crops)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=RPPG_FPS,
        help=f"Target fps for rPPG extraction (default: {RPPG_FPS})",
    )
    parser.add_argument(
        "--max-videos",
        type=int,
        default=None,
        help="Limit number of videos to process (for testing)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be processed without extracting",
    )
    args = parser.parse_args()

    if not args.manifest and not args.all:
        parser.error("Specify --manifest <path> or --all")

    project_root = Path(__file__).resolve().parent.parent
    output_root = args.output_root if args.output_root.is_absolute() else project_root / args.output_root

    # Check ffmpeg
    if not args.dry_run and not check_ffmpeg():
        logger.error("ffmpeg not found. Install with: winget install ffmpeg")
        sys.exit(1)

    manifests_dir = project_root / "data" / "manifests"

    if args.all:
        manifests = sorted(manifests_dir.glob("*.jsonl"))
        skip_names = {"combined.jsonl", "final_dataset_train.jsonl",
                      "final_dataset_val.jsonl", "final_dataset_test.jsonl"}
        manifests = [m for m in manifests if m.name not in skip_names]
    else:
        target = args.manifest if args.manifest.is_absolute() else project_root / args.manifest
        manifests = [target]

    for manifest in manifests:
        if not manifest.exists():
            logger.warning("Manifest not found: %s", manifest)
            continue
        process_manifest(
            manifest_path=manifest,
            output_root=output_root,
            project_root=project_root,
            dry_run=args.dry_run,
            max_videos=args.max_videos,
        )

    logger.info("All done.")


if __name__ == "__main__":
    main()
