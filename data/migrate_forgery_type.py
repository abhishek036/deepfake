"""
data/migrate_forgery_type.py
============================
Add `forgery_type`, `manipulation_family`, `generator_family`, and `skin_tone_tag`
fields to all existing JSONL manifests.

MASTER_PLAN §5E mapping is applied. All manifests in data/manifests/ are
updated in-place (originals backed up to data/manifests/backup/).

Also adds `source_group_id` for bootstrap CI grouping (fixes celeb_df frame-
level bootstrap: must group by video then by source identity).

Usage:
    python data/migrate_forgery_type.py
    python data/migrate_forgery_type.py --dry-run
    python data/migrate_forgery_type.py --manifest data/manifests/celeb_df.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# MASTER_PLAN §5E — forgery_type mapping
# ─────────────────────────────────────────────────────────────────────────────
FORGERY_TYPE_MAP: dict[str, str] = {
    # FaceForensics++
    "Deepfakes":          "composited",
    "FaceSwap":           "composited",
    "Face2Face":          "reenactment",
    "NeuralTextures":     "reenactment",
    "original":           "real",
    # FakeAVCeleb
    "synthesis":          "composited",
    "faceswap":           "composited",
    "fsgan":              "composited",
    "wav2lip":            "talking_head",
    "fsgan-wav2lip":      "composited+audio",
    "faceswap-wav2lip":   "composited+audio",
    "rtvc":               "audio_only",
    "real":               "real",
    # DF40 — Face Swap
    "simswap":            "composited",
    "inswap":             "composited",
    "mobileswap":         "composited",
    "facedancer":         "composited",
    "blendface":          "composited",
    "uniface":            "composited",
    "deepfacelab":        "composited",
    "danet":              "composited",
    # DF40 — Reenactment
    "fomm":               "reenactment",
    "facevid2vid":        "reenactment",
    "hyperreenact":       "reenactment",
    "pirender":           "reenactment",
    "lia":                "reenactment",
    "mcnet":              "reenactment",
    "tpsm":               "reenactment",
    "one_shot_free":      "reenactment",
    "MRAA":               "reenactment",
    # DF40 — Talking Head
    "sadtalker":          "talking_head",
    "heygen":             "talking_head",
    # DF40 — Fully Synthetic / EFS (Entire Face Synthesis)
    "DiT":                "fully_synthetic",
    "SiT":                "fully_synthetic",
    "sd2.1":              "fully_synthetic",
    "pixart":             "fully_synthetic",
    "ddim":               "fully_synthetic",
    "rddm":               "fully_synthetic",
    "CollabDiff":         "fully_synthetic",
    "StyleGAN2":          "fully_synthetic",
    "StyleGAN3":          "fully_synthetic",
    "StyleGANXL":         "fully_synthetic",
    "VQGAN":              "fully_synthetic",
    "MidJourney":         "fully_synthetic",
    "stargan":            "fully_synthetic",
    "starganv2":          "fully_synthetic",
    "styleclip":          "fully_synthetic",
    "whichisreal":        "fully_synthetic",
    "e4e":                "fully_synthetic",
    "e4s":                "fully_synthetic",
    # WildDeepfake / generic
    "wilddeepfake":       "composited",
}

MANIPULATION_FAMILY_MAP: dict[str, str] = {
    "real":               "real",
    "original":           "real",
    "Deepfakes":          "face_swap",
    "FaceSwap":           "face_swap",
    "Face2Face":          "reenactment",
    "NeuralTextures":     "reenactment",
    "faceswap":           "face_swap",
    "fsgan":              "face_swap",
    "wav2lip":            "reenactment",
    "fsgan-wav2lip":      "face_swap",
    "faceswap-wav2lip":   "face_swap",
    "rtvc":               "voice_clone",
    "simswap":            "face_swap",
    "inswap":             "face_swap",
    "mobileswap":         "face_swap",
    "facedancer":         "face_swap",
    "blendface":          "face_swap",
    "uniface":            "face_swap",
    "deepfacelab":        "face_swap",
    "danet":              "face_swap",
    "fomm":               "reenactment",
    "facevid2vid":        "reenactment",
    "hyperreenact":       "reenactment",
    "pirender":           "reenactment",
    "lia":                "reenactment",
    "mcnet":              "reenactment",
    "tpsm":               "reenactment",
    "one_shot_free":      "reenactment",
    "MRAA":               "reenactment",
    "sadtalker":          "talking_head",
    "heygen":             "talking_head",
}

# All diffusion/GAN synthesis → generator_family = method name
SYNTHESIS_METHODS = {
    "DiT", "SiT", "sd2.1", "pixart", "ddim", "rddm", "CollabDiff",
    "StyleGAN2", "StyleGAN3", "StyleGANXL", "VQGAN", "MidJourney",
    "stargan", "starganv2", "styleclip", "whichisreal", "e4e", "e4s",
}


def infer_forgery_type(record: dict) -> str:
    """Infer forgery_type from available fields."""
    # Try manipulation field first
    manip = record.get("manipulation", "")
    if manip in FORGERY_TYPE_MAP:
        return FORGERY_TYPE_MAP[manip]

    # Try method
    method = record.get("method", "")
    if method in FORGERY_TYPE_MAP:
        return FORGERY_TYPE_MAP[method]

    # Try generator_family
    gen = record.get("generator_family", "")
    if gen in FORGERY_TYPE_MAP:
        return FORGERY_TYPE_MAP[gen]

    # Infer from label
    label = record.get("label", record.get("label_int", -1))
    if label == 0:
        return "real"

    return "unknown"


def infer_manipulation_family(record: dict, forgery_type: str) -> str:
    manip = record.get("manipulation", record.get("method", ""))
    if manip in MANIPULATION_FAMILY_MAP:
        return MANIPULATION_FAMILY_MAP[manip]
    if forgery_type == "real":
        return "real"
    if forgery_type == "fully_synthetic":
        return "synthesis"
    if forgery_type in ("composited", "composited+audio"):
        return "face_swap"
    if forgery_type == "reenactment":
        return "reenactment"
    if forgery_type == "talking_head":
        return "talking_head"
    if forgery_type == "audio_only":
        return "voice_clone"
    return "unknown"


def infer_generator_family(record: dict, forgery_type: str) -> str:
    method = record.get("manipulation", record.get("method", record.get("generator_family", "")))
    if method:
        return method
    if forgery_type == "real":
        return "real"
    return "unknown"


def extract_video_id_from_source_path(source_path: str) -> str:
    """
    For bootstrap CI grouping, derive a video_id from source_path.
    Groups all frames from the same video under the same ID.
    """
    p = Path(source_path.replace("\\", "/"))
    # e.g. data/raw/FaceForensics++/original/000.mp4 → ff_original_000
    return p.stem


def migrate_manifest(
    manifest_path: Path,
    backup_dir: Path,
    dry_run: bool = False,
) -> dict[str, int]:
    """
    Add forgery_type + related fields to all records in a JSONL manifest.
    Returns {forgery_type: count} stats.
    """
    logger.info("Processing: %s", manifest_path.name)

    with open(manifest_path, encoding="utf-8") as f:
        lines = f.readlines()

    records = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        records.append(json.loads(line))

    if not records:
        logger.warning("Empty manifest — skipping: %s", manifest_path)
        return {}

    # Back up
    if not dry_run:
        backup_dir.mkdir(parents=True, exist_ok=True)
        backup_path = backup_dir / manifest_path.name
        if not backup_path.exists():
            shutil.copy2(manifest_path, backup_path)
            logger.info("Backed up → %s", backup_path)

    ft_counts: dict[str, int] = {}
    updated = 0

    for rec in records:
        # Only add fields that are missing
        needs_update = any(
            k not in rec for k in ["forgery_type", "manipulation_family", "generator_family", "source_group_id"]
        )
        if not needs_update:
            continue

        updated += 1

        ft = rec.get("forgery_type") or infer_forgery_type(rec)
        rec["forgery_type"]       = ft
        rec["manipulation_family"] = rec.get("manipulation_family") or infer_manipulation_family(rec, ft)
        rec["generator_family"]   = rec.get("generator_family") or infer_generator_family(rec, ft)

        # source_group_id for bootstrap CI (video-level grouping)
        # If source_group_id already set, keep it. Otherwise derive from source_path.
        if "source_group_id" not in rec:
            sp = rec.get("source_path", "")
            rec["source_group_id"] = extract_video_id_from_source_path(sp)

        # skin_tone_tag: use race if available, else "unknown"
        if "skin_tone_tag" not in rec:
            rec["skin_tone_tag"] = rec.get("race", "unknown")

        ft_counts[ft] = ft_counts.get(ft, 0) + 1

    logger.info("  Records: %d | Updated: %d", len(records), updated)
    for ft, n in sorted(ft_counts.items()):
        logger.info("    %-30s %d", ft + ":", n)

    if dry_run:
        return ft_counts

    # Write back in-place
    with open(manifest_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    return ft_counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add forgery_type and related fields to all JSONL manifests"
    )
    parser.add_argument(
        "--manifests-dir",
        type=Path,
        default=Path("data/manifests"),
        help="Directory containing JSONL manifests (default: data/manifests)",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Migrate a single manifest file instead of all",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would change without writing",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent

    if args.manifest:
        target = args.manifest if args.manifest.is_absolute() else project_root / args.manifest
        manifests = [target]
    else:
        mdir = args.manifests_dir if args.manifests_dir.is_absolute() else project_root / args.manifests_dir
        manifests = sorted(mdir.glob("*.jsonl"))

    backup_dir = project_root / "data" / "manifests" / "backup"

    # Skip fakeavceleb (it's built with all fields already)
    # Skip combined (will need separate rebuild)
    skip_names = {"combined.jsonl", "final_dataset_train.jsonl",
                  "final_dataset_val.jsonl", "final_dataset_test.jsonl",
                  "fakeavceleb.jsonl"}

    total_updated = 0
    for mp in manifests:
        if mp.name in skip_names:
            logger.info("Skipping: %s", mp.name)
            continue
        counts = migrate_manifest(mp, backup_dir, dry_run=args.dry_run)
        total_updated += sum(counts.values())

    logger.info("=" * 60)
    logger.info("Migration complete. Total records updated: %d", total_updated)
    if args.dry_run:
        logger.info("DRY RUN — no files written.")


if __name__ == "__main__":
    main()
