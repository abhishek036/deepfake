"""
data/codec_bank.py
Offline codec re-encoding pipeline for FORGE-Lite.

Creates codec bank: multiple compressed versions of training videos at different
H.264 CRF levels, with optional resize and blur. Used for consistency training
to teach the model that compression ≠ forgery.

This runs ONCE offline. Never during training epochs.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class CodecConfig:
    name: str
    codec: str = "libx264"
    crf: Optional[int] = None
    scale: Optional[float] = None        # resize scale factor (e.g. 0.75)
    blur_sigma: Optional[float] = None   # pre-blur sigma
    extra_ffmpeg_args: list[str] = None  # additional ffmpeg arguments

    def __post_init__(self):
        if self.extra_ffmpeg_args is None:
            self.extra_ffmpeg_args = []


# Standard codec bank configurations (mirroring configs/data.yaml)
STANDARD_CONFIGS = [
    CodecConfig(name="clean", codec="copy"),
    CodecConfig(name="h264_crf23", codec="libx264", crf=23),
    CodecConfig(name="h264_crf30", codec="libx264", crf=30),
    CodecConfig(name="h264_crf40", codec="libx264", crf=40),
    CodecConfig(name="resize_compress", codec="libx264", crf=28, scale=0.75),
    CodecConfig(name="blur_compress", codec="libx264", crf=30, blur_sigma=1.0),
]


class CodecBank:
    """
    Builds and manages the offline codec bank for training.

    For each video in the dataset, creates N re-encoded versions under:
        codec_bank_dir/{dataset}/{video_id}/{config_name}/faces/{frame}.jpg

    The ForgeDataset then randomly selects one variant during training.
    """

    def __init__(
        self,
        codec_bank_dir: str = "data/codec_bank",
        configs: Optional[list[CodecConfig]] = None,
    ) -> None:
        self.bank_dir = Path(codec_bank_dir)
        self.configs = configs or STANDARD_CONFIGS
        self.bank_dir.mkdir(parents=True, exist_ok=True)

    def build(
        self,
        video_paths: list[tuple[str, str, str]],  # (video_path, dataset, video_id)
        num_workers: int = 2,
        skip_existing: bool = True,
    ) -> None:
        """
        Build codec bank for all given videos.

        Args:
            video_paths: List of (video_path, dataset_name, video_id)
            num_workers: Parallel ffmpeg processes (I/O bound)
            skip_existing: Skip videos already in the bank
        """
        logger.info("Building codec bank: %d videos × %d configs", len(video_paths), len(self.configs))

        tasks = []
        for video_path, dataset, video_id in video_paths:
            for cfg in self.configs:
                if cfg.name == "clean":
                    continue  # "clean" = original faces, no re-encoding needed
                out_dir = self.bank_dir / dataset / video_id / cfg.name
                if skip_existing and out_dir.exists() and any(out_dir.iterdir()):
                    continue
                tasks.append((video_path, dataset, video_id, cfg, out_dir))

        logger.info("Codec bank tasks: %d (skipped existing)", len(tasks))

        for task in tqdm(tasks, desc="Codec bank"):
            video_path, dataset, video_id, cfg, out_dir = task
            try:
                self._encode_video(video_path, out_dir, cfg)
            except Exception as e:
                logger.error("Failed to encode %s/%s (%s): %s", dataset, video_id, cfg.name, e)

    def _encode_video(self, input_path: str, output_dir: Path, cfg: CodecConfig) -> None:
        """
        Re-encode a video using ffmpeg and extract face crops.
        For simplicity: re-encode to temp video, then the preprocessing pipeline
        reads it normally. The codec bank stores re-encoded temp videos.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        temp_video = output_dir / "encoded.mp4"

        if cfg.codec == "copy":
            shutil.copy2(input_path, temp_video)
            return

        # Build ffmpeg filter chain
        vf_filters = []
        if cfg.blur_sigma:
            ksize = max(3, int(cfg.blur_sigma * 3) * 2 + 1)
            vf_filters.append(f"gblur=sigma={cfg.blur_sigma}")
        if cfg.scale:
            vf_filters.append(f"scale=iw*{cfg.scale}:ih*{cfg.scale}")

        cmd = ["ffmpeg", "-y", "-i", input_path]

        if vf_filters:
            cmd += ["-vf", ",".join(vf_filters)]

        if cfg.crf is not None:
            cmd += ["-c:v", cfg.codec, "-crf", str(cfg.crf), "-preset", "fast"]
        else:
            cmd += ["-c:v", cfg.codec]

        cmd += cfg.extra_ffmpeg_args
        cmd += ["-an", str(temp_video)]  # no audio in codec bank

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {result.stderr[-500:]}")

    def get_variant_path(self, dataset: str, video_id: str, config_name: str) -> Optional[Path]:
        """Return the path to a codec bank variant, or None if not present."""
        path = self.bank_dir / dataset / video_id / config_name
        return path if path.exists() else None

    def list_variants(self, dataset: str, video_id: str) -> list[str]:
        """List all available codec variants for a video."""
        base = self.bank_dir / dataset / video_id
        if not base.exists():
            return []
        return [d.name for d in base.iterdir() if d.is_dir()]
