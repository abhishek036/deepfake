"""
data/download_datasets.py
Automated CLI downloader for FORGE-Lite datasets:
- FFHQ (thumbnails 128x128, 70k images) via Hugging Face
- WildDeepfake (real-world deepfake benchmark) via Hugging Face
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
import tarfile
import zipfile
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def download_ffhq(output_dir: str = "data/raw/FFHQ", extract: bool = True) -> Path:
    """Download FFHQ 128x128 thumbnails (70,000 face images) from Hugging Face."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading FFHQ thumbnails (approx 1.97 GB) from nuwandaa/ffhq128...")
    zip_path = hf_hub_download(
        repo_id="nuwandaa/ffhq128",
        filename="thumbnails128x128.zip",
        repo_type="dataset",
        local_dir=str(out_path),
    )
    logger.info("FFHQ zip downloaded to: %s", zip_path)

    if extract:
        images_dir = out_path / "images128x128"
        if not images_dir.exists() or len(list(images_dir.glob("*.png"))) < 100:
            logger.info("Extracting FFHQ zip archive...")
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(path=out_path)
            logger.info("FFHQ extracted successfully to %s", out_path)
        else:
            logger.info("FFHQ already extracted.")

    return out_path


def download_wilddeepfake(
    output_dir: str = "data/raw/WildDeepfake",
    split: str = "test",
    extract: bool = True,
) -> Path:
    """
    Download WildDeepfake from Hugging Face (xingjunm/WildDeepfake).
    
    Args:
        output_dir: Target directory
        split: 'test' (real_test + fake_test, ~9.25 GB) or 'all' (~67.8 GB)
        extract: Whether to unpack .tar.gz archives
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    api = HfApi()
    logger.info("Fetching file list for xingjunm/WildDeepfake from Hugging Face...")
    info = api.dataset_info("xingjunm/WildDeepfake", files_metadata=True)

    if split == "test":
        target_splits = ["real_test", "fake_test"]
    else:
        target_splits = ["real_test", "fake_test", "real_train", "fake_train"]

    selected_files = [
        s.rfilename for s in info.siblings
        if s.rfilename and any(f"/{ts}/" in s.rfilename for ts in target_splits)
    ]
    logger.info("Found %d files for split(s): %s", len(selected_files), target_splits)

    for i, rfilename in enumerate(selected_files, start=1):
        filename = Path(rfilename).name
        category = Path(rfilename).parent.name  # real_test / fake_test ...
        cat_dir = out_path / category
        cat_dir.mkdir(parents=True, exist_ok=True)

        dest_file = cat_dir / filename
        if dest_file.exists() and dest_file.stat().st_size > 0:
            logger.debug("[%d/%d] Skipping existing: %s", i, len(selected_files), filename)
        else:
            logger.info("[%d/%d] Downloading %s/%s ...", i, len(selected_files), category, filename)
            downloaded = hf_hub_download(
                repo_id="xingjunm/WildDeepfake",
                filename=rfilename,
                repo_type="dataset",
                local_dir=str(out_path / "_raw"),
            )
            shutil.move(downloaded, dest_file)

        # Optional extraction of tar.gz
        if extract and dest_file.name.endswith(".tar.gz"):
            seq_dir = cat_dir / dest_file.stem.replace(".tar", "")
            if not seq_dir.exists():
                try:
                    with tarfile.open(dest_file, "r:gz") as tar:
                        tar.extractall(path=cat_dir)
                except Exception as e:
                    logger.warning("Failed to extract %s: %s", dest_file, e)

    logger.info("WildDeepfake download and processing complete in: %s", out_path)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Download datasets for FORGE-Lite")
    parser.add_argument(
        "--dataset",
        choices=["ffhq", "wilddeepfake", "all"],
        default="all",
        help="Dataset to download",
    )
    parser.add_argument(
        "--wilddeepfake-split",
        choices=["test", "all"],
        default="test",
        help="WildDeepfake split to download ('test' ~9.25GB for cross-domain eval, 'all' ~68GB)",
    )
    parser.add_argument(
        "--no-extract",
        action="store_true",
        help="Do not unpack archives after downloading",
    )
    args = parser.parse_args()

    extract = not args.no_extract

    if args.dataset in ("ffhq", "all"):
        download_ffhq(extract=extract)

    if args.dataset in ("wilddeepfake", "all"):
        download_wilddeepfake(split=args.wilddeepfake_split, extract=extract)


if __name__ == "__main__":
    main()
