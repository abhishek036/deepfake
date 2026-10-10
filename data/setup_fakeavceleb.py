"""
data/setup_fakeavceleb.py
=========================
Build data/manifests/fakeavceleb.jsonl from FakeAVCeleb v1.2 meta_data.csv.

Schema per MASTER_PLAN §5A:
  - Identity-disjoint 80/10/10 train/val/test split by `source` identity
  - forgery_type, manipulation_family, generator_family from MASTER_PLAN §5E mapping
  - race, gender fairness fields from meta_data.csv
  - video_id, record_id, source_path constructed from dir+path columns
  - audio_available, sufficient_for_rppg flags

Usage:
    python data/setup_fakeavceleb.py
    python data/setup_fakeavceleb.py --dataset-root datasets/FakeAVCeleb_v1.2
    python data/setup_fakeavceleb.py --dry-run
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
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
# MASTER_PLAN §5A + §5E  — Method-to-forgery-type mapping
# ─────────────────────────────────────────────────────────────────────────────
METHOD_TO_FORGERY_TYPE: dict[str, str] = {
    "real":           "real",
    "rtvc":           "audio_only",
    "faceswap":       "composited",
    "fsgan":          "composited",
    "wav2lip":        "talking_head",
    "fsgan-wav2lip":  "composited+audio",
    "faceswap-wav2lip": "composited+audio",
}

METHOD_TO_MANIPULATION_FAMILY: dict[str, str] = {
    "real":           "real",
    "rtvc":           "voice_clone",
    "faceswap":       "face_swap",
    "fsgan":          "face_swap",
    "wav2lip":        "reenactment",
    "fsgan-wav2lip":  "face_swap",
    "faceswap-wav2lip": "face_swap",
}

METHOD_TO_GENERATOR_FAMILY: dict[str, str] = {
    "real":           "real",
    "rtvc":           "rtvc",
    "faceswap":       "faceswap",
    "fsgan":          "fsgan",
    "wav2lip":        "wav2lip",
    "fsgan-wav2lip":  "fsgan-wav2lip",
    "faceswap-wav2lip": "faceswap-wav2lip",
}

# Category → label
CATEGORY_TO_LABEL: dict[str, int] = {
    "A": 0,   # RealVideo-RealAudio
    "B": 1,   # RealVideo-FakeAudio
    "C": 1,   # FakeVideo-RealAudio
    "D": 1,   # FakeVideo-FakeAudio
}

# Category → audio/rppg flags
CATEGORY_AUDIO_AVAILABLE: dict[str, bool] = {
    "A": True,
    "B": True,
    "C": True,
    "D": True,
}

# rPPG is only reliable if video is real (Cat A/C have real video)
# Cat B has real video → rPPG usable
# Cat C has fake video → rPPG should NOT be used as ground truth
# Marked False to be conservative — rPPG branch can't rely on Cat C/D being real faces
CATEGORY_RPPG_AVAILABLE: dict[str, bool] = {
    "A": True,   # RealVideo-RealAudio — real face, real PPG
    "B": True,   # RealVideo-FakeAudio — real face, audio is fake
    "C": False,  # FakeVideo-RealAudio — fake face, PPG unreliable
    "D": False,  # FakeVideo-FakeAudio — fake face + fake audio
}

# ─────────────────────────────────────────────────────────────────────────────
# Stable hash split — matches data/splits.py implementation
# ─────────────────────────────────────────────────────────────────────────────
def _stable_hash(key: str, seed: int = 42) -> float:
    """Hash a string key to a stable float in [0, 1)."""
    h = hashlib.md5(f"{seed}:{key}".encode()).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


def assign_split(source_id: str, seed: int = 42) -> str:
    """Identity-disjoint 80/10/10 split by source identity."""
    h = _stable_hash(source_id, seed)
    if h < 0.80:
        return "train"
    elif h < 0.90:
        return "val"
    else:
        return "test"


# ─────────────────────────────────────────────────────────────────────────────
# Record ID
# ─────────────────────────────────────────────────────────────────────────────
def make_record_id(source_path: str) -> str:
    return hashlib.md5(source_path.encode()).hexdigest()[:16]


# ─────────────────────────────────────────────────────────────────────────────
# Main builder
# ─────────────────────────────────────────────────────────────────────────────
def build_manifest(
    dataset_root: Path,
    output_path: Path,
    seed: int = 42,
    dry_run: bool = False,
) -> dict[str, int]:
    """
    Parse meta_data.csv → fakeavceleb.jsonl.

    Returns dict of split → count statistics.
    """
    meta_csv = dataset_root / "meta_data.csv"
    if not meta_csv.exists():
        raise FileNotFoundError(f"meta_data.csv not found: {meta_csv}")

    logger.info("Reading %s", meta_csv)
    with open(meta_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    logger.info("Loaded %d rows from meta_data.csv", len(rows))

    # The CSV has an unnamed last column that contains the directory path
    # Columns: source, target1, target2, method, category, type, race, gender, path, <dir>
    # We need to handle the unnamed column
    columns = list(rows[0].keys())
    dir_col = columns[-1]  # last column — unnamed, contains dir like "FakeAVCeleb/..."
    logger.info("CSV columns: %s", columns)
    logger.info("Directory column key: %r (last column)", dir_col)

    records: list[dict] = []
    split_counts: dict[str, int] = defaultdict(int)
    missing_files: list[str] = []
    unknown_methods: set[str] = set()

    for row in rows:
        source   = row["source"].strip()
        method   = row["method"].strip()
        category = row["category"].strip()
        race     = row["race"].strip()
        gender   = row["gender"].strip()
        filename = row["path"].strip()
        rel_dir  = row[dir_col].strip()   # e.g. "FakeAVCeleb/RealVideo-RealAudio/African/men/id00076"

        if not rel_dir or not filename:
            continue

        # Build path: strip leading "FakeAVCeleb/" since dataset_root IS FakeAVCeleb_v1.2
        # rel_dir looks like: FakeAVCeleb/RealVideo-RealAudio/African/men/id00076
        # Strip prefix "FakeAVCeleb/"
        if rel_dir.startswith("FakeAVCeleb/"):
            rel_dir_clean = rel_dir[len("FakeAVCeleb/"):]
        else:
            rel_dir_clean = rel_dir

        source_path_rel = f"datasets/FakeAVCeleb_v1.2/{rel_dir_clean}/{filename}"
        source_path_abs = dataset_root / rel_dir_clean / filename

        # File existence check (warn only, don't skip — some files may not be extracted yet)
        if not source_path_abs.exists():
            missing_files.append(str(source_path_abs))

        # Forgery type mapping
        forgery_type = METHOD_TO_FORGERY_TYPE.get(method)
        if forgery_type is None:
            unknown_methods.add(method)
            forgery_type = "unknown"

        manipulation_family = METHOD_TO_MANIPULATION_FAMILY.get(method, "unknown")
        generator_family    = METHOD_TO_GENERATOR_FAMILY.get(method, method)
        label               = CATEGORY_TO_LABEL[category]
        split               = assign_split(source, seed)
        record_id           = make_record_id(source_path_rel)
        video_id            = f"fakeav_{source}_{filename.replace('.mp4', '')}"

        rec = {
            "record_id":              record_id,
            "video_id":               video_id,
            "dataset":                "FakeAVCeleb",
            "split":                  split,
            "source_group_id":        source,          # for source-disjoint bootstrap CI
            "source_path":            source_path_rel,
            "audio_path":             source_path_rel,  # audio is embedded in same MP4
            "audio_available":        CATEGORY_AUDIO_AVAILABLE[category],
            "sufficient_for_rppg":    CATEGORY_RPPG_AVAILABLE[category],
            "sufficient_for_analysis": True,
            "label":                  label,
            "label_int":              label,
            "forgery_type":           forgery_type,
            "manipulation_family":    manipulation_family,
            "generator_family":       generator_family,
            "method":                 method,
            "category":               category,
            "type":                   row["type"].strip(),
            "source":                 source,
            "target1":                row["target1"].strip(),
            "target2":                row["target2"].strip(),
            "race":                   race,
            "gender":                 gender,
        }
        records.append(rec)
        split_counts[split] += 1

    if unknown_methods:
        logger.warning("Unknown methods (no forgery_type mapping): %s", unknown_methods)

    if missing_files:
        sample = missing_files[:5]
        logger.warning(
            "%d source files not found on disk (first 5): %s",
            len(missing_files), sample,
        )

    # ── Statistics ────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("FAKEAVCELEB MANIFEST STATISTICS")
    logger.info("=" * 60)
    logger.info("Total records:  %d", len(records))
    for split in ["train", "val", "test"]:
        n = split_counts.get(split, 0)
        logger.info("  %-12s %d", split + ":", n)

    # Per-category breakdown
    cat_counts: dict[str, int] = defaultdict(int)
    for r in records:
        cat_counts[r["category"]] += 1
    logger.info("Category breakdown:")
    for cat in ["A", "B", "C", "D"]:
        logger.info("  Cat %-2s %d records", cat + ":", cat_counts[cat])

    # Per-forgery-type
    ft_counts: dict[str, int] = defaultdict(int)
    for r in records:
        ft_counts[r["forgery_type"]] += 1
    logger.info("Forgery-type breakdown:")
    for ft, n in sorted(ft_counts.items()):
        logger.info("  %-25s %d", ft + ":", n)

    # Per-race
    race_counts: dict[str, int] = defaultdict(int)
    for r in records:
        race_counts[r["race"]] += 1
    logger.info("Race breakdown:")
    for race, n in sorted(race_counts.items()):
        logger.info("  %-30s %d", race + ":", n)

    logger.info("=" * 60)

    if dry_run:
        logger.info("DRY RUN — no files written.")
        return dict(split_counts)

    # ── Write JSONL ───────────────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    logger.info("Written %d records → %s", len(records), output_path)
    return dict(split_counts)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build FakeAVCeleb v1.2 manifest (fakeavceleb.jsonl)"
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("datasets/FakeAVCeleb_v1.2"),
        help="Path to FakeAVCeleb_v1.2 directory (default: datasets/FakeAVCeleb_v1.2)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/manifests/fakeavceleb.jsonl"),
        help="Output JSONL path (default: data/manifests/fakeavceleb.jsonl)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for split assignment (default: 42)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print statistics but do not write any files",
    )
    args = parser.parse_args()

    # Resolve relative to project root
    project_root = Path(__file__).resolve().parent.parent
    dataset_root = args.dataset_root if args.dataset_root.is_absolute() else project_root / args.dataset_root
    output_path  = args.output if args.output.is_absolute() else project_root / args.output

    if not dataset_root.exists():
        logger.error("Dataset root does not exist: %s", dataset_root)
        sys.exit(1)

    stats = build_manifest(
        dataset_root=dataset_root,
        output_path=output_path,
        seed=args.seed,
        dry_run=args.dry_run,
    )
    logger.info("Done. Split counts: %s", stats)


if __name__ == "__main__":
    main()
