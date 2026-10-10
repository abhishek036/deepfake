"""
data/setup_final_dataset.py
Setup and prepare FINAL_DATASET.csv for FORGE-Lite evaluation and training.

Features:
- Fast multi-threaded concurrent downloader with custom headers and retry logic
- Automatic image integrity verification (detects corrupted/truncated files)
- Local CSV mapping generation (data/final_dataset/final_dataset_local.csv)
- FORGE-Lite manifest generation (train, val, test jsonl manifests)
- Resumable: skips already downloaded images
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import urllib.request
import urllib.error

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import pandas as pd
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("setup_final_dataset")

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
DEFAULT_IMAGES_DIR = REPO_ROOT / "data" / "final_dataset" / "images"
DEFAULT_MANIFESTS_DIR = REPO_ROOT / "data" / "manifests"


def download_single_image(
    image_id: int | str,
    url: str,
    save_path: Path,
    timeout: int = 15,
    max_retries: int = 3,
) -> Tuple[int | str, bool, Optional[str]]:
    """Downloads a single image with retry logic and browser headers."""
    if save_path.exists() and save_path.stat().st_size > 1024:
        # Quick validation check: try opening with cv2
        img = cv2.imread(str(save_path))
        if img is not None and img.size > 0:
            return image_id, True, None

    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        },
    )

    save_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = save_path.with_suffix(".tmp")

    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                if response.status == 200:
                    with open(temp_path, "wb") as out_file:
                        out_file.write(response.read())

                    # Verify image validity
                    img = cv2.imread(str(temp_path))
                    if img is None or img.size == 0:
                        if temp_path.exists():
                            temp_path.unlink()
                        return image_id, False, "Corrupted image content"

                    # Atomically move temp file to destination
                    if temp_path.exists():
                        temp_path.replace(save_path)
                    return image_id, True, None
                else:
                    time.sleep(0.5 * (attempt + 1))
        except Exception as e:
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except OSError:
                    pass
            if attempt == max_retries - 1:
                return image_id, False, str(e)
            time.sleep(0.5 * (attempt + 1))

    return image_id, False, "Exceeded retry attempts"


def download_dataset(
    df: pd.DataFrame,
    images_dir: Path,
    max_workers: int = 16,
    limit: int = 0,
) -> pd.DataFrame:
    """Download images in parallel using ThreadPoolExecutor."""
    if limit > 0:
        df = df.head(limit).copy()
    else:
        df = df.copy()

    images_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Setting up downloads for %d images with %d workers...", len(df), max_workers)

    tasks = []
    for idx, row in df.iterrows():
        img_id = row["image_id"]
        url = row["image_url"]
        dest = images_dir / f"{img_id}.jpg"
        tasks.append((img_id, url, dest))

    download_results: Dict[int | str, Tuple[bool, Optional[str]]] = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_id = {
            executor.submit(download_single_image, item[0], item[1], item[2]): item[0]
            for item in tasks
        }

        with tqdm(total=len(tasks), desc="Downloading images", unit="img") as pbar:
            for future in concurrent.futures.as_completed(future_to_id):
                img_id = future_to_id[future]
                try:
                    res_id, success, error_msg = future.result()
                    download_results[res_id] = (success, error_msg)
                except Exception as e:
                    download_results[img_id] = (False, str(e))
                pbar.update(1)

    # Attach local paths and download status
    df["local_path"] = df["image_id"].apply(lambda x: str((images_dir / f"{x}.jpg").resolve()))
    df["download_success"] = df["image_id"].apply(lambda x: download_results.get(x, (False, "Not attempted"))[0])
    df["download_error"] = df["image_id"].apply(lambda x: download_results.get(x, (False, "Not attempted"))[1])

    succeeded = df["download_success"].sum()
    logger.info("Download completed: %d / %d images successfully saved (%.1f%%).", succeeded, len(df), (succeeded / max(1, len(df))) * 100)
    return df


def build_manifests(
    df: pd.DataFrame,
    manifests_dir: Path,
) -> Dict[str, Path]:
    """Build JSONL manifests compatible with FORGE-Lite training and evaluation."""
    manifests_dir.mkdir(parents=True, exist_ok=True)
    out_files = {}

    for split in ["train", "val", "test"]:
        split_df = df[(df["dataset_split"] == split) & (df["download_success"] == True)]
        if len(split_df) == 0:
            continue

        manifest_path = manifests_dir / f"final_dataset_{split}.jsonl"
        with open(manifest_path, "w", encoding="utf-8") as f:
            for _, row in split_df.iterrows():
                # Label in FORGE-Lite: 0 = Real, 1 = Fake
                label_val = 0 if str(row["label"]).strip().upper() == "REAL" else 1
                record = {
                    "record_id": f"fd_{row['image_id']}",
                    "dataset": "final_dataset",
                    "split": split,
                    "source_path": row["local_path"],
                    "frame_idx": 0,
                    "timestamp_sec": 0.0,
                    "label": label_val,
                    "manipulation": str(row.get("fake_method", "original") if label_val == 1 else "original"),
                    "crop_path": row["local_path"],
                    "category": str(row.get("category", "")),
                    "difficulty": str(row.get("detection_difficulty", "medium")),
                    "quality": str(row.get("image_quality", "high")),
                }
                f.write(json.dumps(record) + "\n")

        logger.info("Created %s manifest with %d records -> %s", split, len(split_df), manifest_path)
        out_files[split] = manifest_path

    return out_files


def main():
    parser = argparse.ArgumentParser(description="Setup and download images for FINAL_DATASET.csv")
    parser.add_argument("--csv", type=str, default="FINAL_DATASET.csv", help="Path to FINAL_DATASET.csv")
    parser.add_argument("--split", type=str, default="test", choices=["test", "val", "train", "all"], help="Split to download (default: test)")
    parser.add_argument("--images-dir", type=str, default="data/final_dataset/images", help="Target images directory")
    parser.add_argument("--manifests-dir", type=str, default="data/manifests", help="Target manifests directory")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of images to download (0 = all in split)")
    parser.add_argument("--workers", type=int, default=16, help="Concurrent download threads (default: 16)")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        logger.error("CSV not found at: %s", csv_path)
        sys.exit(1)

    df = pd.read_csv(csv_path)
    logger.info("Loaded CSV '%s' with %d total records.", csv_path, len(df))

    if args.split != "all":
        df = df[df["dataset_split"] == args.split].reset_index(drop=True)
        logger.info("Filtered to split '%s' (%d records).", args.split, len(df))

    images_dir = Path(args.images_dir)
    df_processed = download_dataset(df, images_dir=images_dir, max_workers=args.workers, limit=args.limit)

    # Save local mapping CSV
    local_csv_path = images_dir.parent / "final_dataset_local.csv"
    df_processed.to_csv(local_csv_path, index=False)
    logger.info("Saved updated dataset mapping to %s", local_csv_path)

    # Build manifests
    build_manifests(df_processed, manifests_dir=Path(args.manifests_dir))
    logger.info("All setup complete! You are ready to run tests.")


if __name__ == "__main__":
    main()
