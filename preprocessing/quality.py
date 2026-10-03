"""
preprocessing/quality.py
Per-frame and per-track quality feature extraction for FORGE-Lite.

Quality features are used by:
- The reliability-aware fusion head (branch weighting)
- The abstention layer (INSUFFICIENT_EVIDENCE decisions)
- The rPPG branch gate (minimum quality requirements)

All features are normalized to [0, 1] unless otherwise noted.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FrameQuality:
    """Quality metrics for a single aligned face frame."""
    # Core quality scores (all [0,1] unless noted)
    blur_score: float          # higher = sharper (Laplacian variance, normalized)
    brightness: float          # mean luminance [0,1]
    contrast: float            # std of luminance [0,1]
    face_area_ratio: float     # face area / frame area [0,1]
    # Pose estimates (in degrees, not normalized)
    pose_yaw_deg: float
    pose_pitch_deg: float
    pose_roll_deg: float
    # Occlusion
    occlusion_estimate: float  # 0=no occlusion, 1=fully occluded
    # Compression
    compression_score: float   # 0=heavy compression, 1=clean (blockiness measure)
    # Landmark quality
    landmark_confidence: float # 0=bad, 1=confident landmarks

    def to_vector(self) -> np.ndarray:
        """Return 10-dimensional quality feature vector (normalized)."""
        yaw_norm = min(abs(self.pose_yaw_deg) / 90.0, 1.0)
        pitch_norm = min(abs(self.pose_pitch_deg) / 60.0, 1.0)
        return np.array([
            self.blur_score,
            self.brightness,
            self.contrast,
            self.face_area_ratio,
            yaw_norm,
            pitch_norm,
            self.occlusion_estimate,
            self.compression_score,
            self.landmark_confidence,
            # Overall quality composite
            self._composite(),
        ], dtype=np.float32)

    @property
    def composite_quality(self) -> float:
        """Overall composite quality score in [0, 1]."""
        return self._composite()

    def _composite(self) -> float:
        scaled_blur = min(1.0, self.blur_score * 3.0)
        scaled_area = min(1.0, self.face_area_ratio * 3.0) if self.face_area_ratio > 0 else 0.5
        return float(np.mean([
            scaled_blur,
            1.0 - self.occlusion_estimate,
            scaled_area,
            self.compression_score,
        ]))

    def to_dict(self) -> dict:
        return {
            "blur_score": float(self.blur_score),
            "brightness": float(self.brightness),
            "contrast": float(self.contrast),
            "face_area_ratio": float(self.face_area_ratio),
            "pose_yaw_deg": float(self.pose_yaw_deg),
            "pose_pitch_deg": float(self.pose_pitch_deg),
            "pose_roll_deg": float(self.pose_roll_deg),
            "occlusion_estimate": float(self.occlusion_estimate),
            "compression_score": float(self.compression_score),
            "landmark_confidence": float(self.landmark_confidence),
            "composite": float(self._composite()),
        }


@dataclass
class TrackQuality:
    """Aggregated quality metrics for a face track (sequence of frames)."""
    track_id: int
    num_frames: int
    continuity: float
    mean_blur: float
    mean_brightness: float
    mean_face_area_ratio: float
    max_pose_yaw: float
    mean_occlusion: float
    mean_compression: float
    landmark_jitter: float     # std of landmark displacement across frames
    track_duration_sec: float
    # Derived flags
    sufficient_for_rppg: bool
    sufficient_for_analysis: bool

    def to_vector(self) -> np.ndarray:
        """5-dimensional video-level quality vector."""
        return np.array([
            self.continuity,
            self.mean_blur,
            self.mean_face_area_ratio,
            1.0 - self.mean_occlusion,
            self.mean_compression,
        ], dtype=np.float32)

    @property
    def composite_quality(self) -> float:
        """
        Overall quality score for quality-routing decision in inference.

        Weighted mean of the four factors that most affect all branch reliability:
          - blur (sharpness):      35% — blurry frames hurt DINOv2 + forensic + rPPG equally
          - continuity (tracking): 35% — discontinuous tracks break temporal GRU
          - face_visibility:       20% — small faces reduce forensic and rPPG SNR
          - non-occlusion:         10% — partial occlusion degrades landmark geometry only

        Returns float in [0, 1]. Below 0.35 → conservative path (DINOv2 disabled).
        """
        return float(
            0.35 * self.mean_blur
            + 0.35 * self.continuity
            + 0.20 * self.mean_face_area_ratio
            + 0.10 * (1.0 - self.mean_occlusion)
        )

    def to_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "num_frames": self.num_frames,
            "continuity": float(self.continuity),
            "mean_blur": float(self.mean_blur),
            "mean_brightness": float(self.mean_brightness),
            "mean_face_area_ratio": float(self.mean_face_area_ratio),
            "max_pose_yaw": float(self.max_pose_yaw),
            "mean_occlusion": float(self.mean_occlusion),
            "mean_compression": float(self.mean_compression),
            "landmark_jitter": float(self.landmark_jitter),
            "track_duration_sec": float(self.track_duration_sec),
            "composite_quality": round(self.composite_quality, 4),
            "sufficient_for_rppg": self.sufficient_for_rppg,
            "sufficient_for_analysis": self.sufficient_for_analysis,
        }


# ─────────────────────────────────────────────────────────────────────────────
# QualityAnalyzer
# ─────────────────────────────────────────────────────────────────────────────

class QualityAnalyzer:
    """
    Computes per-frame and per-track quality features for FORGE-Lite.

    All quality scores are normalized to [0, 1]:
    - blur_score: higher = sharper (better)
    - face_area_ratio: higher = larger face (better)
    - occlusion_estimate: lower = less occluded (better)
    - compression_score: higher = less compressed (better)
    """

    def __init__(
        self,
        min_face_area_ratio: float = 0.05,
        min_blur_score: float = 0.1,
        min_track_continuity: float = 0.6,
        min_rppg_duration_sec: float = 8.0,
    ) -> None:
        self.min_face_area_ratio = min_face_area_ratio
        self.min_blur_score = min_blur_score
        self.min_track_continuity = min_track_continuity
        self.min_rppg_duration_sec = min_rppg_duration_sec

    def analyze_frame(
        self,
        face_crop_bgr: np.ndarray,
        full_frame_bgr: Optional[np.ndarray] = None,
        landmarks_5pt: Optional[np.ndarray] = None,
        bbox: Optional[np.ndarray] = None,
        frame_wh: Optional[tuple[int, int]] = None,
    ) -> FrameQuality:
        """
        Compute quality features for a single aligned 224×224 face crop.

        Args:
            face_crop_bgr: Aligned 224×224 face crop
            full_frame_bgr: Full video frame (for face area ratio)
            landmarks_5pt: 5-point landmarks in frame space
            bbox: Face bounding box [x1,y1,x2,y2] in frame space
            frame_wh: (width, height) of the original frame
        """
        h, w = face_crop_bgr.shape[:2]

        # 1. Blur — Laplacian variance on grayscale crop
        gray = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2GRAY)
        blur_raw = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        blur_score = self._normalize_blur(blur_raw)

        # 2. Brightness & contrast
        lum = gray.astype(np.float32) / 255.0
        brightness = float(lum.mean())
        contrast = float(lum.std())

        # 3. Face area ratio
        face_area_ratio = 0.0
        if bbox is not None and frame_wh is not None:
            fw, fh = frame_wh
            face_area = max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1])
            frame_area = fw * fh
            face_area_ratio = float(face_area / frame_area) if frame_area > 0 else 0.0

        # 4. Pose estimation (from landmarks if available)
        yaw, pitch, roll = self._estimate_pose(landmarks_5pt)

        # 5. Occlusion estimate (heuristic: low face coverage in crop)
        occlusion = self._estimate_occlusion(gray)

        # 6. Compression score (DCT block artifacts)
        compression_score = self._estimate_compression(gray)

        # 7. Landmark confidence
        lm_conf = self._landmark_confidence(landmarks_5pt)

        return FrameQuality(
            blur_score=blur_score,
            brightness=brightness,
            contrast=min(contrast * 4.0, 1.0),  # normalize std to [0,1]
            face_area_ratio=face_area_ratio,
            pose_yaw_deg=yaw,
            pose_pitch_deg=pitch,
            pose_roll_deg=roll,
            occlusion_estimate=occlusion,
            compression_score=compression_score,
            landmark_confidence=lm_conf,
        )

    def analyze_track(
        self,
        frame_qualities: list[FrameQuality],
        track_id: int,
        continuity: float,
        track_duration_sec: float,
        all_landmarks: Optional[list[np.ndarray]] = None,
    ) -> TrackQuality:
        """Aggregate per-frame quality metrics over a full face track."""
        if not frame_qualities:
            return TrackQuality(
                track_id=track_id, num_frames=0, continuity=0.0,
                mean_blur=0.0, mean_brightness=0.0, mean_face_area_ratio=0.0,
                max_pose_yaw=0.0, mean_occlusion=1.0, mean_compression=0.0,
                landmark_jitter=1.0, track_duration_sec=0.0,
                sufficient_for_rppg=False, sufficient_for_analysis=False,
            )

        mean_blur = float(np.mean([q.blur_score for q in frame_qualities]))
        mean_brightness = float(np.mean([q.brightness for q in frame_qualities]))
        mean_area = float(np.mean([q.face_area_ratio for q in frame_qualities]))
        max_yaw = float(np.max([abs(q.pose_yaw_deg) for q in frame_qualities]))
        mean_occlusion = float(np.mean([q.occlusion_estimate for q in frame_qualities]))
        mean_compression = float(np.mean([q.compression_score for q in frame_qualities]))

        # Landmark jitter — std of landmark displacement between frames
        lm_jitter = self._compute_landmark_jitter(all_landmarks)

        sufficient_rppg = (
            continuity >= self.min_track_continuity
            and mean_area >= self.min_face_area_ratio
            and track_duration_sec >= self.min_rppg_duration_sec
            and mean_blur >= self.min_blur_score
        )
        sufficient_analysis = (
            continuity >= 0.4
            and mean_area >= self.min_face_area_ratio * 0.5
        )

        return TrackQuality(
            track_id=track_id,
            num_frames=len(frame_qualities),
            continuity=continuity,
            mean_blur=mean_blur,
            mean_brightness=mean_brightness,
            mean_face_area_ratio=mean_area,
            max_pose_yaw=max_yaw,
            mean_occlusion=mean_occlusion,
            mean_compression=mean_compression,
            landmark_jitter=lm_jitter,
            track_duration_sec=track_duration_sec,
            sufficient_for_rppg=sufficient_rppg,
            sufficient_for_analysis=sufficient_analysis,
        )

    # ── Private quality estimators ────────────────────────────────────────────

    @staticmethod
    def _normalize_blur(laplacian_var: float, max_val: float = 500.0) -> float:
        """Normalize Laplacian variance to [0,1]. Values above max_val -> 1.0."""
        return float(min(laplacian_var / max_val, 1.0))

    @staticmethod
    def _estimate_pose(landmarks: Optional[np.ndarray]) -> tuple[float, float, float]:
        """Estimate head pose (yaw, pitch, roll) from 5 landmarks."""
        if landmarks is None or len(landmarks) < 5:
            return 0.0, 0.0, 0.0
        left_eye, right_eye = landmarks[0], landmarks[1]
        nose = landmarks[2]
        left_mouth, right_mouth = landmarks[3], landmarks[4]

        # Roll: angle of eye-to-eye vector
        eye_vec = right_eye - left_eye
        roll = float(np.degrees(np.arctan2(eye_vec[1], eye_vec[0])))

        # Yaw: asymmetry of eye-to-nose distances
        d_left = float(np.linalg.norm(nose - left_eye))
        d_right = float(np.linalg.norm(nose - right_eye))
        eye_width = float(np.linalg.norm(right_eye - left_eye)) + 1e-6
        yaw = float(np.degrees(np.arctan2(d_right - d_left, eye_width))) * 1.5

        # Pitch: vertical displacement of nose relative to eye midpoint
        eye_mid_y = float((left_eye[1] + right_eye[1]) / 2)
        mouth_mid_y = float((left_mouth[1] + right_mouth[1]) / 2)
        face_height = abs(mouth_mid_y - eye_mid_y) + 1e-6
        pitch_raw = (nose[1] - eye_mid_y) / face_height
        pitch = float(np.degrees(np.arctan(pitch_raw - 0.5)))

        return yaw, pitch, roll

    @staticmethod
    def _estimate_occlusion(gray: np.ndarray) -> float:
        """
        Heuristic occlusion estimate: fraction of very dark pixels.
        Very rough — a proper occlusion detector would use a trained model.
        """
        dark_fraction = float((gray < 30).mean())
        return min(dark_fraction * 5.0, 1.0)

    @staticmethod
    def _estimate_compression(gray: np.ndarray) -> float:
        """
        Estimate image compression quality via block DCT artifact measure.
        Returns score in [0,1] where 1 = minimal compression artifacts.
        """
        # Compute variance within 8×8 DCT blocks and at block boundaries
        h, w = gray.shape
        block_size = 8
        boundary_diffs = []
        for y in range(block_size, h - block_size, block_size):
            row_diff = float(np.abs(gray[y, :].astype(float) - gray[y-1, :].astype(float)).mean())
            boundary_diffs.append(row_diff)
        for x in range(block_size, w - block_size, block_size):
            col_diff = float(np.abs(gray[:, x].astype(float) - gray[:, x-1].astype(float)).mean())
            boundary_diffs.append(col_diff)

        if not boundary_diffs:
            return 1.0
        mean_boundary = float(np.mean(boundary_diffs))
        # High boundary diff = strong block artifacts = low compression quality
        # Normalize inversely: 0 artifact → score=1.0, heavy artifact → score≈0
        score = 1.0 - min(mean_boundary / 20.0, 1.0)
        return float(score)

    @staticmethod
    def _landmark_confidence(landmarks: Optional[np.ndarray]) -> float:
        """Heuristic landmark confidence based on geometry plausibility."""
        if landmarks is None or len(landmarks) < 5:
            return 0.0
        # Check that landmarks form a plausible face arrangement
        left_eye, right_eye, nose, left_mouth, right_mouth = landmarks[:5]
        eye_dist = float(np.linalg.norm(right_eye - left_eye))
        if eye_dist < 5.0:
            return 0.1  # landmarks collapsed
        nose_y_valid = left_eye[1] < nose[1] < left_mouth[1]
        mouth_y_valid = nose[1] < left_mouth[1]
        # Eye horizontal ordering
        eye_x_valid = left_eye[0] < right_eye[0]
        score = (float(nose_y_valid) + float(mouth_y_valid) + float(eye_x_valid)) / 3.0
        return score

    @staticmethod
    def _compute_landmark_jitter(landmarks_seq: Optional[list[np.ndarray]]) -> float:
        """
        Compute landmark jitter as mean std of landmark positions across frames.
        Normalized to [0, 1].
        """
        if landmarks_seq is None or len(landmarks_seq) < 2:
            return 0.0
        valid = [lm for lm in landmarks_seq if lm is not None and len(lm) == 5]
        if len(valid) < 2:
            return 0.0
        stack = np.stack(valid, axis=0)  # (T, 5, 2)
        displacements = np.diff(stack, axis=0)  # (T-1, 5, 2)
        jitter = float(np.std(np.linalg.norm(displacements, axis=-1)))
        return min(jitter / 50.0, 1.0)  # 50px normalized
