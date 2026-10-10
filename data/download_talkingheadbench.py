"""
data/download_talkingheadbench.py
=================================
Automated Downloader & Manifest Builder for TalkingHeadBench (2024 SOTA Talking Head Deepfakes).

Source: luchaoqi/TalkingHeadBench on Hugging Face (100% public, direct access).
Generators:
- LivePortrait (KwaiVGI — implicit keypoints)
- Hallo2 (Fudan/Baidu — hierarchical audio-driven portraits)
- AniPortraitAudio (Tencent — diffusion-based audio-driven animation)
- EmoPortrait (emotion-conditioned expressive portraits)
- Hallo (audio-driven talking face)

Features:
- Direct HTTP stream / Hugging Face download with resume and progress bars
- Audio extraction (16kHz mono) and 80-band log-Mel spectrogram computation
- Pairing with real talking-head videos (LAV-DF & FaceForensics++)
- FORGE-Lite manifest generation (data/manifests/talkingheadbench.jsonl)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from huggingface_hub import HfApi, hf_hub_download
from tqdm import tqdm

from data.download_av_datasets import extract_audio_mel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("download_thb")
RAW_DIR = REPO_ROOT / "data" / "raw" / "TalkingHeadBench"
AUDIO_DIR = REPO_ROOT / "data" / "audio"
MEL_DIR = REPO_ROOT / "data" / "audio" / "mel"
MANIFEST_DIR = REPO_ROOT / "data" / "manifests"

REPO_ID = "luchaoqi/TalkingHeadBench"


def download_talkingheadbench(
    generators: Optional[List[str]] = None,
    split: str = "train",
    max_per_gen: Optional[int] = None,
) -> List[Path]:
    """
    Downloads TalkingHeadBench video files (.mp4) from Hugging Face.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    MEL_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

    api = HfApi()
    logger.info("Querying file list from %s...", REPO_ID)
    info = api.dataset_info(REPO_ID, files_metadata=True)

    available_files = [s.rfilename for s in info.siblings if s.rfilename.endswith(".mp4")]

    gen_filter = [g.lower() for g in generators] if generators else None

    # Group files by generator and split
    selected_files = []
    by_gen: Dict[str, List[str]] = {}

    for f in available_files:
        # Expected path: fake/<generator>/<split>/<filename>.mp4
        parts = f.split("/")
        if len(parts) >= 4 and parts[0] == "fake":
            gen_name = parts[1]
            file_split = parts[2]

            if gen_filter and gen_name.lower() not in gen_filter:
                continue

            if split != "all" and file_split != split:
                continue

            by_gen.setdefault(gen_name, []).append(f)

    logger.info("Found generators: %s", list(by_gen.keys()))
    for gen_name, g_files in by_gen.items():
        if max_per_gen and max_per_gen > 0:
            picked = g_files[:max_per_gen]
        else:
            picked = g_files
        selected_files.extend(picked)
        logger.info("  %s [%s]: %d / %d selected", gen_name, split, len(picked), len(g_files))

    logger.info("Total files to download: %d", len(selected_files))

    downloaded_paths = []
    for i, rpath in enumerate(tqdm(selected_files, desc="Downloading THB", unit="video"), start=1):
        target_local = RAW_DIR / rpath
        if target_local.exists() and target_local.stat().st_size > 1024:
            downloaded_paths.append(target_local)
            continue

        target_local.parent.mkdir(parents=True, exist_ok=True)
        try:
            downloaded = hf_hub_download(
                repo_id=REPO_ID,
                filename=rpath,
                repo_type="dataset",
                local_dir=str(RAW_DIR),
                local_dir_use_symlinks=False,
            )
            downloaded_paths.append(Path(downloaded))
        except Exception as e:
            logger.warning("Failed to download %s: %s", rpath, e)

    logger.info("Downloaded %d videos into %s", len(downloaded_paths), RAW_DIR)
    return downloaded_paths


HDTF_REPO_ID = "saeed-5959/hdtf"
HDTF_DIR = REPO_ROOT / "data" / "raw" / "HDTF"


def download_hdtf_reals(count: int) -> List[Path]:
    """
    Downloads real high-definition talking face videos from HDTF on Hugging Face.
    Provides ground-truth genuine audio-visual synchronization controls.
    """
    HDTF_DIR.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    logger.info("Querying HDTF real video files from %s...", HDTF_REPO_ID)
    info = api.dataset_info(HDTF_REPO_ID, files_metadata=True)
    hdtf_files = [s.rfilename for s in info.siblings if s.rfilename.endswith(".mp4")]

    selected = hdtf_files[:count]
    logger.info("Downloading %d real HDTF control videos...", len(selected))
    downloaded_paths = []

    for rpath in tqdm(selected, desc="Downloading HDTF Reals", unit="video"):
        target_local = HDTF_DIR / rpath
        if target_local.exists() and target_local.stat().st_size > 1024:
            downloaded_paths.append(target_local)
            continue

        try:
            downloaded = hf_hub_download(
                repo_id=HDTF_REPO_ID,
                filename=rpath,
                repo_type="dataset",
                local_dir=str(HDTF_DIR),
            )
            downloaded_paths.append(Path(downloaded))
        except Exception as e:
            logger.warning("Failed to download HDTF video %s: %s", rpath, e)

    logger.info("Downloaded %d real HDTF videos into %s", len(downloaded_paths), HDTF_DIR)
    return downloaded_paths


def build_talkingheadbench_manifest(
    downloaded_videos: List[Path],
    include_real_hdtf: bool = True,
) -> Path:
    """
    Extracts Mel spectrograms from downloaded videos, pairs 1:1 with real HDTF videos,
    and builds the FORGE-Lite manifest: data/manifests/talkingheadbench.jsonl.
    """
    logger.info("Building manifest for TalkingHeadBench...")
    records = []

    # 1. Process fake talking-head videos
    for vid_path in tqdm(downloaded_videos, desc="Processing THB Fakes"):
        rel_path = vid_path.relative_to(REPO_ROOT)
        stem = vid_path.stem
        # Extract generator from path
        parts = vid_path.parts
        gen = "TalkingHead"
        for idx, part in enumerate(parts):
            if part == "fake" and idx + 2 < len(parts):
                gen = parts[idx + 1]

        # Stratified deterministic split (70% train, 15% val, 15% test)
        h = abs(hash(stem)) % 100
        if h < 70:
            split = "train"
        elif h < 85:
            split = "val"
        else:
            split = "test"

        mel_file = MEL_DIR / f"THB_{gen}_{stem}_mel.pt"
        has_mel = mel_file.exists()
        if not has_mel:
            has_mel = extract_audio_mel(vid_path, mel_file)

        rel_mel = str(mel_file.relative_to(REPO_ROOT)) if has_mel else None

        records.append({
            "record_id": f"thb_fake_{len(records)}",
            "video_id": stem,
            "dataset": "TalkingHeadBench",
            "split": split,
            "source_path": str(rel_path),
            "audio_path": str(rel_path),
            "mel_path": rel_mel,
            "audio_available": has_mel,
            "label": 1,
            "label_int": 1,
            "manipulation_family": f"talking_head_{gen.lower()}",
            "generator_family": gen,
            "sufficient_for_analysis": True,
            "sufficient_for_rppg": False,
        })

    # 2. Balance with real talking-head videos from HDTF
    fake_count = len(records)
    if include_real_hdtf and fake_count > 0:
        real_videos = download_hdtf_reals(fake_count)
        real_count = 0
        for vid_path in tqdm(real_videos, desc="Processing HDTF Reals"):
            rel_path = vid_path.relative_to(REPO_ROOT)
            stem = vid_path.stem

            mel_file = MEL_DIR / f"HDTF_{stem}_mel.pt"
            has_mel = mel_file.exists()
            if not has_mel:
                has_mel = extract_audio_mel(vid_path, mel_file)

            rel_mel = str(mel_file.relative_to(REPO_ROOT)) if has_mel else None

            # Split: 70% train, 15% val, 15% test (deterministic by hash)
            h = abs(hash(stem)) % 100
            if h < 70:
                split = "train"
            elif h < 85:
                split = "val"
            else:
                split = "test"

            records.append({
                "record_id": f"hdtf_real_{real_count}",
                "video_id": stem,
                "dataset": "HDTF",
                "split": split,
                "source_path": str(rel_path),
                "audio_path": str(rel_path),
                "mel_path": rel_mel,
                "audio_available": has_mel,
                "label": 0,
                "label_int": 0,
                "manipulation_family": "original",
                "generator_family": "HDTF_real",
                "sufficient_for_analysis": True,
                "sufficient_for_rppg": True,
            })
            real_count += 1

        logger.info("Added %d real audio-visual controls from HDTF (Total records: %d)", real_count, len(records))

    out_manifest = MANIFEST_DIR / "talkingheadbench.jsonl"
    with open(out_manifest, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    logger.info("Created manifest: %s with %d balanced records", out_manifest, len(records))
    return out_manifest


def main():
    parser = argparse.ArgumentParser(description="Download & Setup TalkingHeadBench (2024 SOTA Talking Heads)")
    parser.add_argument(
        "--generators",
        nargs="+",
        default=["LivePortrait", "Hallo2", "AniPortraitAudio", "EmoPortrait", "Hallo"],
        help="Generators to download",
    )
    parser.add_argument(
        "--split",
        default="all",
        choices=["train", "val", "test", "all"],
        help="Split to download",
    )
    parser.add_argument(
        "--max-per-gen",
        type=int,
        default=50,
        help="Max videos per generator (default 50 for fast progressive start, 0 or -1 for all)",
    )
    parser.add_argument(
        "--no-real",
        action="store_true",
        help="Do not download HDTF real controls",
    )
    args = parser.parse_args()

    max_p = None if args.max_per_gen <= 0 else args.max_per_gen
    vids = download_talkingheadbench(
        generators=args.generators,
        split=args.split,
        max_per_gen=max_p,
    )
    build_talkingheadbench_manifest(vids, include_real_hdtf=not args.no_real)


if __name__ == "__main__":
    main()
