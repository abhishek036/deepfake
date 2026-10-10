"""
scripts/download_deepfake_60k.py
Download and extract prithivMLmods/Deepfake-vs-Real-60K from Hugging Face.
Supports resume, progress display, automatic extraction, and manifest building.
"""

import os
import sys
import zipfile
import json
import logging
from pathlib import Path
from huggingface_hub import hf_hub_download
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("download_60k")

REPO_ROOT = Path(__file__).resolve().parent.parent
TARGET_DIR = REPO_ROOT / "datasets" / "Deepfake-vs-Real-60K"
MANIFEST_DIR = REPO_ROOT / "data" / "manifests"


def download_and_extract():
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading deepfake-vs-real-60k.zip (22.39 GB) from Hugging Face...")
    logger.info("Target directory: %s", TARGET_DIR)

    # Download with local_dir to avoid duplicating space in HF cache
    zip_path = hf_hub_download(
        repo_id="prithivMLmods/Deepfake-vs-Real-60K",
        filename="deepfake-vs-real-60k.zip",
        repo_type="dataset",
        local_dir=str(TARGET_DIR),
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    logger.info("Download finished! File saved to: %s", zip_path)

    zip_file = Path(zip_path)
    logger.info("Extracting %s (size: %.2f GB)...", zip_file.name, zip_file.stat().st_size / (1024**3))

    with zipfile.ZipFile(zip_file, "r") as z:
        members = z.namelist()
        logger.info("Zip contains %d items. Extracting to %s...", len(members), TARGET_DIR)
        with tqdm(total=len(members), desc="Extracting", unit="file") as pbar:
            for m in members:
                z.extract(m, TARGET_DIR)
                pbar.update(1)

    logger.info("Extraction complete!")

    # Build manifest
    build_manifest()


def build_manifest():
    logger.info("Scanning extracted images to generate FORGE-Lite manifest...")
    valid_exts = {".jpg", ".jpeg", ".png", ".webp"}
    records = []

    for root, dirs, files in os.walk(TARGET_DIR):
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in valid_exts and not f.endswith(".tmp"):
                full_path = Path(root) / f
                rel_path = full_path.relative_to(REPO_ROOT)

                # Determine label from directory or path
                lower_p = str(full_path).lower()
                if "fake" in lower_p or "deepfake" in lower_p or "0-fake" in lower_p:
                    label = 1  # FORGE-Lite: 1 = Fake
                    cat = "fake"
                elif "real" in lower_p or "1-real" in lower_p:
                    label = 0  # FORGE-Lite: 0 = Real
                    cat = "real"
                else:
                    label = 1
                    cat = "unknown"

                records.append({
                    "record_id": f"dfr60k_{len(records)}",
                    "dataset": "Deepfake-vs-Real-60K",
                    "split": "test",
                    "source_path": str(rel_path),
                    "crop_path": str(rel_path),
                    "frame_idx": 0,
                    "timestamp_sec": 0.0,
                    "label": label,
                    "manipulation": "deepfake" if label == 1 else "original",
                    "category": cat,
                })

    manifest_path = MANIFEST_DIR / "deepfake_vs_real_60k.jsonl"
    with open(manifest_path, "w", encoding="utf-8") as out:
        for r in records:
            out.write(json.dumps(r) + "\n")

    logger.info("Manifest created with %d total images -> %s", len(records), manifest_path)


if __name__ == "__main__":
    download_and_extract()
