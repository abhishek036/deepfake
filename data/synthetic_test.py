"""
data/synthetic_test.py
Synthetic test dataset generator for FORGE-Lite pipeline smoke-testing.

Creates a minimal synthetic dataset without needing real deepfake videos.
"Fake" videos = real faces with simulated manipulation artifacts.
"Real" videos = clean synthetic face sequences.

This allows full end-to-end pipeline validation (preprocessing → model → eval)
without downloading any datasets.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class SyntheticDatasetGenerator:
    """
    Generates a small synthetic test dataset for smoke-testing FORGE-Lite.

    Each "video" is a sequence of generated face-like images saved as
    pre-aligned 224×224 JPEG crops (bypassing face detection entirely).

    Real samples: smooth Gaussian noise faces with natural color variation.
    Fake samples: same + simulated GAN artifacts (checkerboard pattern, boundary artifacts).
    """

    RESOLUTION = 224
    FPS = 25

    def __init__(
        self,
        output_dir: str = "data/synthetic_test",
        num_real: int = 20,
        num_fake: int = 20,
        frames_per_video: int = 32,
        seed: int = 42,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.num_real = num_real
        self.num_fake = num_fake
        self.frames_per_video = frames_per_video
        np.random.seed(seed)

    def generate(self, overwrite: bool = False) -> str:
        """
        Generate the synthetic dataset.
        Returns path to manifest JSONL file.
        """
        manifest_path = self.output_dir / "manifest.jsonl"
        if manifest_path.exists() and not overwrite:
            logger.info("Synthetic dataset already exists at %s", manifest_path)
            return str(manifest_path)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        records = []

        logger.info("Generating %d real + %d fake synthetic videos...", self.num_real, self.num_fake)

        for i in range(self.num_real):
            record = self._generate_video(
                video_id=f"real_{i:04d}",
                label="real",
                split=self._assign_split(i, self.num_real),
            )
            records.append(record)

        for i in range(self.num_fake):
            record = self._generate_video(
                video_id=f"fake_{i:04d}",
                label="fake",
                split=self._assign_split(i, self.num_fake),
                add_artifacts=True,
            )
            records.append(record)

        with open(manifest_path, "w") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")

        logger.info("Synthetic dataset ready: %d records → %s", len(records), manifest_path)
        return str(manifest_path)

    def _generate_video(
        self,
        video_id: str,
        label: str,
        split: str,
        add_artifacts: bool = False,
    ) -> dict:
        """Generate one synthetic video and save its face crops."""
        faces_dir = self.output_dir / "faces" / video_id
        faces_dir.mkdir(parents=True, exist_ok=True)

        # Generate base "face" — synthetic skin-tone gradient + noise
        base_face = self._generate_base_face()
        landmarks = self._generate_landmarks()

        quality_frames = []
        for frame_idx in range(self.frames_per_video):
            frame = self._animate_frame(base_face, frame_idx, add_artifacts)
            out_path = faces_dir / f"{frame_idx:06d}.jpg"
            cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            quality_frames.append(self._compute_simple_quality(frame))

        mean_blur = float(np.mean([q["blur"] for q in quality_frames]))
        mask_available = add_artifacts  # synthetic masks available for fakes

        if mask_available:
            mask_dir = self.output_dir / "masks" / video_id
            mask_dir.mkdir(parents=True, exist_ok=True)
            for frame_idx in range(self.frames_per_video):
                mask = self._generate_mask(frame_idx)
                cv2.imwrite(str(mask_dir / f"{frame_idx:06d}.png"), mask)

        return {
            "dataset": "synthetic_test",
            "video_id": video_id,
            "source_id": f"src_{video_id}",
            "identity_id": f"id_{video_id}",
            "label": label,
            "label_int": 1 if label == "fake" else 0,
            "manipulation_scope": "face_localized" if add_artifacts else "none",
            "manipulation_family": "simulated_artifacts" if add_artifacts else "real",
            "generator_family": "synthetic" if add_artifacts else "real",
            "split": split,
            "fps": float(self.FPS),
            "duration_seconds": float(self.frames_per_video / self.FPS),
            "total_frames": self.frames_per_video,
            "sampled_frames": self.frames_per_video,
            "width": self.RESOLUTION,
            "height": self.RESOLUTION,
            "codec": "synthetic",
            "audio_available": False,
            "mask_available": mask_available,
            "mask_path": str(self.output_dir / "masks" / video_id) if mask_available else None,
            "faces_dir": str(faces_dir),
            "track_id": 0,
            "track_length": self.frames_per_video,
            "track_continuity": 1.0,
            "track_duration_sec": float(self.frames_per_video / self.FPS),
            "track_quality": {
                "track_id": 0,
                "num_frames": self.frames_per_video,
                "continuity": 1.0,
                "mean_blur": float(mean_blur),
                "mean_brightness": 0.5,
                "mean_face_area_ratio": 0.8,
                "max_pose_yaw": 0.0,
                "mean_occlusion": 0.0,
                "mean_compression": 0.9,
                "landmark_jitter": 0.01,
                "track_duration_sec": float(self.frames_per_video / self.FPS),
                "sufficient_for_rppg": True,
                "sufficient_for_analysis": True,
            },
            "sufficient_for_analysis": True,
            "sufficient_for_rppg": True,
        }

    def _generate_base_face(self) -> np.ndarray:
        """Generate a synthetic skin-tone face image."""
        h = w = self.RESOLUTION
        # Skin tone base: warm orange-ish gradient
        face = np.zeros((h, w, 3), dtype=np.float32)
        cx, cy = w // 2, h // 2
        Y, X = np.ogrid[:h, :w]
        dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
        # Elliptical face mask
        face_mask = (dist < min(cx, cy) * 0.85).astype(np.float32)
        # Skin tone [B, G, R] approximately
        skin = np.array([140, 170, 210], dtype=np.float32)
        for c in range(3):
            face[:, :, c] = skin[c] * face_mask + 40 * (1 - face_mask)
        # Add random texture
        noise = np.random.normal(0, 15, (h, w, 3)).astype(np.float32)
        face = np.clip(face + noise, 0, 255).astype(np.uint8)
        return face

    def _animate_frame(
        self, base: np.ndarray, frame_idx: int, add_artifacts: bool
    ) -> np.ndarray:
        """Animate the base face with temporal variation."""
        frame = base.copy().astype(np.float32)
        # Subtle brightness oscillation (simulates rPPG signal in real)
        osc = np.sin(frame_idx * 0.3) * 5.0
        frame = np.clip(frame + osc, 0, 255)
        # Add random per-frame noise
        noise = np.random.normal(0, 3, frame.shape)
        frame = np.clip(frame + noise, 0, 255)

        if add_artifacts:
            # Simulated GAN checkerboard artifact
            frame = self._add_checkerboard(frame, frame_idx)
            # Simulated boundary artifact at face edge
            frame = self._add_boundary_artifact(frame)

        return frame.astype(np.uint8)

    @staticmethod
    def _add_checkerboard(frame: np.ndarray, frame_idx: int) -> np.ndarray:
        """Add subtle checkerboard pattern (simulating GAN upsampling artifact)."""
        h, w = frame.shape[:2]
        artifact = np.zeros_like(frame, dtype=np.float32)
        # 8×8 checkerboard with phase variation
        for y in range(0, h, 8):
            for x in range(0, w, 8):
                if (y // 8 + x // 8 + frame_idx) % 2 == 0:
                    artifact[y:y+8, x:x+8] = 4.0
        return np.clip(frame + artifact, 0, 255)

    @staticmethod
    def _add_boundary_artifact(frame: np.ndarray) -> np.ndarray:
        """Add a bright ring artifact at the face boundary (simulates paste boundary)."""
        h, w = frame.shape[:2]
        mask = np.zeros((h, w), dtype=np.float32)
        cx, cy = w // 2, h // 2
        Y, X = np.ogrid[:h, :w]
        dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
        r = min(cx, cy) * 0.85
        ring = ((dist > r - 4) & (dist < r + 4)).astype(np.float32)
        for c in range(3):
            frame[:, :, c] = np.clip(frame[:, :, c] + ring * 20.0, 0, 255)
        return frame

    @staticmethod
    def _generate_landmarks() -> np.ndarray:
        """Generate plausible 5-point landmarks for a 224×224 face."""
        return np.array([
            [74.0, 82.0],    # left eye
            [150.0, 82.0],   # right eye
            [112.0, 118.0],  # nose
            [76.0, 150.0],   # left mouth
            [148.0, 150.0],  # right mouth
        ], dtype=np.float32)

    @staticmethod
    def _generate_mask(frame_idx: int) -> np.ndarray:
        """Generate a synthetic manipulation mask (lower half of face)."""
        mask = np.zeros((224, 224), dtype=np.uint8)
        mask[112:, 56:168] = 255  # lower face region
        return mask

    @staticmethod
    def _compute_simple_quality(frame: np.ndarray) -> dict:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        return {"blur": min(blur / 2000.0, 1.0)}

    @staticmethod
    def _assign_split(i: int, total: int) -> str:
        if i < int(total * 0.72):
            return "train"
        elif i < int(total * 0.86):
            return "val"
        else:
            return "test"
