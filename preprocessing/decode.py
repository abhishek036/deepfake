"""
preprocessing/decode.py
Video decoding, frame sampling, and metadata extraction for FORGE-Lite.

Supports uniform sampling, keyframe sampling, and combined strategies.
Never decodes every frame — always samples at a target fps.
"""

from __future__ import annotations

import subprocess
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class VideoMetadata:
    """Metadata extracted from a video file before frame decoding."""
    path: str
    fps: float
    duration_seconds: float
    total_frames: int
    width: int
    height: int
    codec: str
    audio_available: bool
    file_size_bytes: int
    # Derived
    sampled_fps: float = 0.0
    sampled_frame_indices: list[int] = field(default_factory=list)

    @property
    def resolution(self) -> tuple[int, int]:
        return (self.width, self.height)

    @property
    def megapixels(self) -> float:
        return (self.width * self.height) / 1_000_000

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "fps": self.fps,
            "duration_seconds": self.duration_seconds,
            "total_frames": self.total_frames,
            "width": self.width,
            "height": self.height,
            "codec": self.codec,
            "audio_available": self.audio_available,
            "file_size_bytes": self.file_size_bytes,
            "sampled_fps": self.sampled_fps,
            "sampled_frame_indices": self.sampled_frame_indices,
        }


@dataclass
class DecodedFrame:
    """A single decoded video frame with its metadata."""
    frame_idx: int          # original frame index in the video
    timestamp_sec: float    # time position in the video
    image: np.ndarray       # HWC BGR uint8


# ─────────────────────────────────────────────────────────────────────────────
# VideoDecoder
# ─────────────────────────────────────────────────────────────────────────────

class VideoDecoder:
    """
    Efficient video decoder that samples frames at a target fps.

    Strategy:
    - 'uniform': evenly spaced frames at fps_target
    - 'keyframe': only I-frames (fast, lower density)
    - 'uniform+keyframe': union of both (more coverage)

    All frames are returned as BGR numpy arrays (OpenCV convention).
    """

    def __init__(
        self,
        fps_target: float = 8.0,
        max_frames: int = 300,
        min_frames: int = 8,
        strategy: str = "uniform",
    ) -> None:
        self.fps_target = fps_target
        self.max_frames = max_frames
        self.min_frames = min_frames
        self.strategy = strategy

    # ── Public API ────────────────────────────────────────────────────────────

    def get_metadata(self, video_path: str | Path) -> VideoMetadata:
        """Extract metadata from a video file using ffprobe (fast, no decode)."""
        path = str(video_path)
        cmd = [
            "ffprobe", "-v", "quiet", "-print_format", "json",
            "-show_streams", "-show_format", path
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            info = json.loads(result.stdout)
        except (subprocess.TimeoutExpired, json.JSONDecodeError, FileNotFoundError) as e:
            logger.warning("ffprobe failed for %s: %s — falling back to OpenCV", path, e)
            return self._get_metadata_opencv(path)

        streams = info.get("streams", [])
        video_stream = next((s for s in streams if s.get("codec_type") == "video"), None)
        audio_stream = next((s for s in streams if s.get("codec_type") == "audio"), None)
        fmt = info.get("format", {})

        if video_stream is None:
            raise ValueError(f"No video stream found in {path}")

        # Parse fps fraction (e.g., "25/1" or "30000/1001")
        fps_str = video_stream.get("avg_frame_rate", "25/1")
        try:
            num, den = fps_str.split("/")
            fps = float(num) / float(den) if float(den) > 0 else 25.0
        except ValueError:
            fps = 25.0

        duration = float(fmt.get("duration", 0.0))
        total_frames = int(video_stream.get("nb_frames", duration * fps))
        width = int(video_stream.get("width", 0))
        height = int(video_stream.get("height", 0))
        codec = video_stream.get("codec_name", "unknown")
        file_size = int(fmt.get("size", 0))

        meta = VideoMetadata(
            path=path,
            fps=fps,
            duration_seconds=duration,
            total_frames=total_frames,
            width=width,
            height=height,
            codec=codec,
            audio_available=audio_stream is not None,
            file_size_bytes=file_size,
        )
        meta.sampled_frame_indices = self._compute_frame_indices(meta)
        meta.sampled_fps = self.fps_target
        return meta

    def decode_frames(
        self, video_path: str | Path, metadata: Optional[VideoMetadata] = None
    ) -> Generator[DecodedFrame, None, None]:
        """
        Yield sampled frames from the video as DecodedFrame objects.
        Uses pre-computed frame indices from metadata for efficiency.
        """
        path = str(video_path)
        if metadata is None:
            metadata = self.get_metadata(path)

        indices_set = set(metadata.sampled_frame_indices)
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {path}")

        frame_idx = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx in indices_set:
                    timestamp = frame_idx / max(metadata.fps, 1e-6)
                    yield DecodedFrame(
                        frame_idx=frame_idx,
                        timestamp_sec=timestamp,
                        image=frame,
                    )
                frame_idx += 1
                if frame_idx > max(indices_set, default=0) + 1:
                    break
        finally:
            cap.release()

    def decode_all(
        self, video_path: str | Path, metadata: Optional[VideoMetadata] = None
    ) -> tuple[list[DecodedFrame], VideoMetadata]:
        """Decode all sampled frames and return as a list."""
        if metadata is None:
            metadata = self.get_metadata(video_path)
        frames = list(self.decode_frames(video_path, metadata))
        return frames, metadata

    # ── Private helpers ───────────────────────────────────────────────────────

    def _compute_frame_indices(self, meta: VideoMetadata) -> list[int]:
        """Compute the set of frame indices to sample."""
        total = meta.total_frames
        fps = meta.fps

        if self.strategy == "uniform":
            indices = self._uniform_indices(total, fps)
        elif self.strategy == "keyframe":
            indices = list(range(0, total, max(1, int(fps))))  # approx keyframe spacing
        elif self.strategy == "uniform+keyframe":
            uniform = set(self._uniform_indices(total, fps))
            keyframe = set(range(0, total, max(1, int(fps * 2))))
            indices = sorted(uniform | keyframe)
        else:
            indices = self._uniform_indices(total, fps)

        # Clamp to max_frames
        if len(indices) > self.max_frames:
            step = len(indices) // self.max_frames
            indices = indices[::step][: self.max_frames]

        return indices

    def _uniform_indices(self, total_frames: int, fps: float) -> list[int]:
        """Compute uniformly spaced frame indices at fps_target."""
        if total_frames <= 0 or fps <= 0:
            return []
        frame_step = max(1, int(round(fps / self.fps_target)))
        return list(range(0, total_frames, frame_step))

    def _get_metadata_opencv(self, path: str) -> VideoMetadata:
        """Fallback metadata extraction using OpenCV."""
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        meta = VideoMetadata(
            path=path,
            fps=fps,
            duration_seconds=total_frames / fps,
            total_frames=total_frames,
            width=width,
            height=height,
            codec="unknown",
            audio_available=False,
            file_size_bytes=Path(path).stat().st_size if Path(path).exists() else 0,
        )
        meta.sampled_frame_indices = self._compute_frame_indices(meta)
        meta.sampled_fps = self.fps_target
        return meta
