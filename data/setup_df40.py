"""
data/setup_df40.py
==================
Build data/manifests/df40_test.jsonl from all JSON configs in
preprocessing/dataset_json/. Maps each entry to the unified schema with
forgery_type from MASTER_PLAN §5B.

Real DF40 JSON structure (discovered by inspection):
{
  "<method>_<base>": {           # e.g. "blendface_ff"
    "<method>_Real": {           # real source frames
      "train": {
        "<video_id>": {
          "label": "<method>_Real",
          "frames": ["path/to/frame.png", ...]
        }
      },
      "test": { ... },
      "val":  { ... }
    },
    "<method>_Fake": {           # manipulated frames
      "train": { "<video_id>": {"label": "<method>_Fake", "frames": [...]} },
      "test":  { ... },
      "val":   { ... }
    }
  }
}

This script can run while DF40 images download — it builds the manifest from
config files alone. Evaluation scripts check file existence at runtime.

Usage:
    python data/setup_df40.py
    python data/setup_df40.py --json-dir preprocessing/dataset_json
    python data/setup_df40.py --dry-run
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
# MASTER_PLAN §5B — Method-to-forgery-type mapping for DF40
# ─────────────────────────────────────────────────────────────────────────────
FORGERY_TYPE: dict[str, str] = {
    # Fully synthetic / EFS
    "DiT": "fully_synthetic", "SiT": "fully_synthetic", "sd2.1": "fully_synthetic",
    "pixart": "fully_synthetic", "ddim": "fully_synthetic", "rddm": "fully_synthetic",
    "CollabDiff": "fully_synthetic", "StyleGAN2": "fully_synthetic",
    "StyleGAN3": "fully_synthetic", "StyleGANXL": "fully_synthetic",
    "VQGAN": "fully_synthetic", "MidJourney": "fully_synthetic",
    "stargan": "fully_synthetic", "starganv2": "fully_synthetic",
    "styleclip": "fully_synthetic", "whichisreal": "fully_synthetic",
    "e4e": "fully_synthetic", "e4s": "fully_synthetic",
    # Face Swap / FS
    "faceswap": "composited", "fsgan": "composited", "simswap": "composited",
    "blendface": "composited", "facedancer": "composited", "inswap": "composited",
    "mobileswap": "composited", "uniface": "composited", "deepfacelab": "composited",
    "danet": "composited",
    # Reenactment / FR
    "fomm": "reenactment", "facevid2vid": "reenactment", "hyperreenact": "reenactment",
    "pirender": "reenactment", "lia": "reenactment", "mcnet": "reenactment",
    "tpsm": "reenactment", "one_shot_free": "reenactment", "MRAA": "reenactment",
    # Talking Head
    "sadtalker": "talking_head", "wav2lip": "talking_head", "heygen": "talking_head",
    # Real
    "original": "real", "real": "real",
}

MANIPULATION_FAMILY: dict[str, str] = {
    "real": "real", "original": "real",
    "faceswap": "face_swap", "fsgan": "face_swap", "simswap": "face_swap",
    "blendface": "face_swap", "facedancer": "face_swap", "inswap": "face_swap",
    "mobileswap": "face_swap", "uniface": "face_swap", "deepfacelab": "face_swap",
    "danet": "face_swap",
    "fomm": "reenactment", "facevid2vid": "reenactment", "hyperreenact": "reenactment",
    "pirender": "reenactment", "lia": "reenactment", "mcnet": "reenactment",
    "tpsm": "reenactment", "one_shot_free": "reenactment", "MRAA": "reenactment",
    "sadtalker": "talking_head", "wav2lip": "talking_head", "heygen": "talking_head",
}


def make_record_id(path: str) -> str:
    return hashlib.md5(path.encode()).hexdigest()[:16]


def extract_method_from_role_key(role_key: str) -> str:
    """
    Extract clean method name from role key like 'blendface_Fake' or 'blendface_Real'.
    Strips _Fake/_Real suffix.
    """
    for suffix in ("_Fake", "_Real", "_fake", "_real"):
        if role_key.endswith(suffix):
            return role_key[: -len(suffix)]
    return role_key


def infer_method_from_json_filename(stem: str) -> str:
    """Extract method from filename like 'blendface_ff' → 'blendface'."""
    # Strip _cdf, _ff, _ori suffixes
    for suffix in ("_cdf", "_ff", "_ori"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def is_real_role(role_key: str) -> bool:
    """Returns True if this role key represents real (not manipulated) frames."""
    k = role_key.lower()
    return k.endswith("_real")


def process_df40_json(json_path: Path) -> list[dict]:
    """
    Parse one DF40 JSON config → list of manifest records.

    Actual DF40 JSON structure:
    {
      "<method>_<base>": {
        "<method>_Real": {"train": {<vid_id>: {"label": ..., "frames": [...]}, ...}, "test": ..., "val": ...},
        "<method>_Fake": {"train": {<vid_id>: {"label": ..., "frames": [...]}, ...}, "test": ..., "val": ...},
      }
    }
    """
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        logger.warning("Unexpected top-level type %s in %s", type(data), json_path.name)
        return []

    file_stem = Path(json_path).stem
    inferred_method = infer_method_from_json_filename(file_stem)

    records: list[dict] = []

    # Iterate over top-level keys (usually one per file, e.g. "blendface_ff")
    for top_key, top_val in data.items():
        if not isinstance(top_val, dict):
            continue

        # top_val has role keys like "<method>_Real", "<method>_Fake"
        for role_key, role_val in top_val.items():
            if not isinstance(role_val, dict):
                continue

            method = extract_method_from_role_key(role_key)
            is_real = is_real_role(role_key)
            label_int = 0 if is_real else 1

            forgery_type = "real" if is_real else FORGERY_TYPE.get(method, "unknown")
            manip_family = "real" if is_real else MANIPULATION_FAMILY.get(method, "synthesis")
            gen_family   = "real" if is_real else method

            # role_val: {"train": {vid_id: {...}, ...}, "test": {...}, "val": {...}}
            for split_key, split_val in role_val.items():
                if not isinstance(split_val, dict):
                    continue

                # Normalise split name
                split = split_key if split_key in ("train", "val", "test") else "test"

                for video_id_key, video_entry in split_val.items():
                    if not isinstance(video_entry, dict):
                        continue

                    frames = video_entry.get("frames", [])
                    if not isinstance(frames, list):
                        frames = []

                    # Each frame becomes a record
                    for frame_path in frames:
                        frame_path_norm = frame_path.replace("\\", "/")
                        # Remap deepfakes_detection_datasets/DF40/ → datasets/DF40/
                        if frame_path_norm.startswith("deepfakes_detection_datasets/DF40/"):
                            frame_path_clean = "datasets/DF40/" + frame_path_norm[len("deepfakes_detection_datasets/DF40/"):]
                        elif frame_path_norm.startswith("deepfakes_detection_datasets/"):
                            frame_path_clean = "datasets/" + frame_path_norm[len("deepfakes_detection_datasets/"):]
                        else:
                            frame_path_clean = frame_path_norm

                        rec = {
                            "record_id":             make_record_id(frame_path_clean),
                            "video_id":              f"df40_{method}_{video_id_key}",
                            "dataset":               "DF40",
                            "split":                 split,
                            "source_group_id":       f"df40_{method}_{video_id_key}",
                            "source_path":           frame_path_clean,
                            "label":                 label_int,
                            "label_int":             label_int,
                            "method":                method,
                            "manipulation":          method if not is_real else "original",
                            "forgery_type":          forgery_type,
                            "manipulation_family":   manip_family,
                            "generator_family":      gen_family,
                            "audio_available":       False,
                            "sufficient_for_rppg":   False,
                            "sufficient_for_analysis": True,
                            "skin_tone_tag":         "unknown",
                            "config_source":         json_path.name,
                        }
                        records.append(rec)

    return records


def build_manifest(
    json_dir: Path,
    output_path: Path,
    dry_run: bool = False,
    skip_combined: bool = True,
) -> dict:
    """Parse all DF40 JSON configs → df40_test.jsonl."""
    json_files = sorted(json_dir.glob("*.json"))
    logger.info("Found %d JSON config files in %s", len(json_files), json_dir)

    # Skip the giant combined/aggregate files to avoid duplicates
    skip_files = {
        "DF40_all.json", "EFSAll_cdf.json", "EFSAll_ff.json",
        "FRAll_cdf.json", "FRAll_ff.json", "FSAll_cdf.json", "FSAll_ff.json",
    }
    if skip_combined:
        json_files = [j for j in json_files if j.name not in skip_files]
        logger.info("After skipping combined files: %d JSON configs to process", len(json_files))

    all_records: list[dict] = []
    seen_record_ids: set[str] = set()
    method_counts: dict[str, int] = defaultdict(int)
    forgery_type_counts: dict[str, int] = defaultdict(int)
    split_counts: dict[str, int] = defaultdict(int)
    files_parsed = 0
    files_failed = 0

    for json_path in json_files:
        try:
            recs = process_df40_json(json_path)
            added = 0
            for rec in recs:
                rid = rec["record_id"]
                if rid in seen_record_ids:
                    continue
                seen_record_ids.add(rid)
                all_records.append(rec)
                method_counts[rec["method"]] += 1
                forgery_type_counts[rec["forgery_type"]] += 1
                split_counts[rec["split"]] += 1
                added += 1
            logger.debug("  %-40s → %d records", json_path.name, added)
            if recs:
                files_parsed += 1
            else:
                logger.warning("  Zero records parsed from: %s", json_path.name)
        except Exception as e:
            logger.warning("Failed to parse %s: %s", json_path.name, e)
            files_failed += 1

    logger.info("=" * 60)
    logger.info("DF40 MANIFEST STATISTICS")
    logger.info("=" * 60)
    logger.info("Files parsed:   %d / %d (failed: %d)", files_parsed, len(json_files), files_failed)
    logger.info("Total records:  %d", len(all_records))
    logger.info("Split breakdown:")
    for split in ["train", "val", "test"]:
        logger.info("  %-12s %d", split + ":", split_counts.get(split, 0))
    logger.info("\nForgery-type breakdown:")
    for ft, n in sorted(forgery_type_counts.items()):
        logger.info("  %-30s %d", ft + ":", n)
    logger.info("\nMethod breakdown (top 30):")
    for method, n in sorted(method_counts.items(), key=lambda x: -x[1])[:30]:
        logger.info("  %-30s %d", method + ":", n)
    logger.info("=" * 60)

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
        description="Build DF40 manifest (df40_test.jsonl) from dataset JSON configs"
    )
    parser.add_argument(
        "--json-dir", type=Path, default=Path("preprocessing/dataset_json"),
        help="Directory with DF40 JSON configs (default: preprocessing/dataset_json)",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/manifests/df40_test.jsonl"),
        help="Output JSONL path (default: data/manifests/df40_test.jsonl)",
    )
    parser.add_argument(
        "--include-combined", action="store_true",
        help="Also parse combined JSON files (EFSAll, FRAll, FSAll). Warning: creates duplicates.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print statistics but do not write any files",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    json_dir     = args.json_dir if args.json_dir.is_absolute() else project_root / args.json_dir
    output_path  = args.output if args.output.is_absolute() else project_root / args.output

    if not json_dir.exists():
        logger.error("JSON dir not found: %s", json_dir)
        sys.exit(1)

    stats = build_manifest(
        json_dir=json_dir,
        output_path=output_path,
        dry_run=args.dry_run,
        skip_combined=not args.include_combined,
    )
    logger.info("Done: total=%d", stats["total"])


if __name__ == "__main__":
    main()
