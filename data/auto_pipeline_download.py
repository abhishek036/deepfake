"""
data/auto_pipeline_download.py
Orchestrates downloading and extraction of:
1. FFHQ (monitors and unpacks thumbnails128x128.zip)
2. WildDeepfake (downloads test split with auto-resume)
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

FFHQ_EXPECTED_SIZE = 2114632904  # bytes
FFHQ_ZIP = Path("data/raw/FFHQ/thumbnails128x128.zip")
FFHQ_DIR = Path("data/raw/FFHQ")


def wait_and_extract_ffhq():
    """Wait for FFHQ zip download to reach expected size and extract it."""
    logger.info("Monitoring FFHQ download at %s...", FFHQ_ZIP)
    while True:
        if FFHQ_ZIP.exists():
            size = FFHQ_ZIP.stat().st_size
            pct = (size / FFHQ_EXPECTED_SIZE) * 100
            logger.info("FFHQ progress: %.2f MB / %.2f MB (%.1f%%)", size / (1024 * 1024), FFHQ_EXPECTED_SIZE / (1024 * 1024), pct)
            if size >= FFHQ_EXPECTED_SIZE:
                logger.info("FFHQ download complete! Extracting archive...")
                try:
                    with zipfile.ZipFile(FFHQ_ZIP, "r") as zf:
                        zf.extractall(path=FFHQ_DIR)
                    logger.info("FFHQ extracted successfully into %s", FFHQ_DIR)
                except Exception as e:
                    logger.error("Extraction error: %s", e)
                break
        time.sleep(15)


def download_wilddeepfake_curl():
    """Download WildDeepfake test split using curl with auto-resume."""
    from huggingface_hub import HfApi
    api = HfApi()
    logger.info("Querying WildDeepfake test split files from Hugging Face...")
    info = api.dataset_info("xingjunm/WildDeepfake", files_metadata=True)
    target_files = [
        s.rfilename for s in info.siblings
        if s.rfilename and any(cat in s.rfilename for cat in ["/real_test/", "/fake_test/"])
    ]
    logger.info("Found %d test archives to download.", len(target_files))

    base_dir = Path("data/raw/WildDeepfake")
    base_dir.mkdir(parents=True, exist_ok=True)

    for i, rpath in enumerate(target_files, start=1):
        category = "real_test" if "real_test" in rpath else "fake_test"
        filename = Path(rpath).name
        dest_dir = base_dir / category
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_file = dest_dir / filename

        url = f"https://huggingface.co/datasets/xingjunm/WildDeepfake/resolve/main/{rpath}"
        logger.info("[%d/%d] Downloading %s/%s ...", i, len(target_files), category, filename)
        cmd = [
            "curl.exe", "-L", "-C", "-",
            "--retry", "10", "--retry-delay", "3",
            "-o", str(dest_file),
            url,
        ]
        res = subprocess.run(cmd)
        if res.returncode != 0:
            logger.warning("Download failed for %s, returncode: %d", filename, res.returncode)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "wilddeepfake":
        download_wilddeepfake_curl()
    else:
        wait_and_extract_ffhq()
        download_wilddeepfake_curl()
