"""
models/geometry_branch.py
Geometric facial landmark feature extractor for FORGE-Lite.

Replaces raw landmark velocity zeros with semantically meaningful per-frame geometry
descriptors. Deepfakes frequently exhibit subtle geometric anomalies: unnatural eye
blink rates, rigid jaw motion, abnormal inter-ocular distances, mouth aspect ratio
discontinuities.

Input : 5-point landmarks per frame  (B, T, 5, 2)  — [left_eye, right_eye, nose, mouth_left, mouth_right]
Output: (B, T, 16) geometry feature sequence fed to TemporalEncoder

Features (16-dim per frame):
  0  — inter-ocular distance (normalized by face scale)
  1  — eye aspect ratio proxy (vertical spread)
  2  — mouth width (normalized)
  3  — mouth height proxy (left–right vs nose–mouth distance)
  4  — jaw asymmetry: |left_mouth_x - nose_x| vs |right_mouth_x - nose_x|
  5  — nose-to-chin proxy (nose to mouth-center vertical)
  6  — face width proxy (right_eye_x - left_eye_x)
  7  — pose_roll proxy: angle of eye-line from horizontal
  8  — pose_yaw proxy: eye midpoint offset from nose x
  9  — left eye–nose distance normalized
  10 — right eye–nose distance normalized
  11 — mouth_left–left_eye distance normalized
  12 — mouth_right–right_eye distance normalized
  13 — inter-frame Δ inter-ocular distance (temporal first-derivative)
  14 — inter-frame Δ pose_roll (temporal first-derivative)
  15 — inter-frame Δ mouth_width (temporal first-derivative)
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

GEOMETRY_DIM = 16  # Output feature dimension per frame


def compute_geometry_from_landmarks(
    landmarks: np.ndarray,  # (5, 2) — left_eye, right_eye, nose, mouth_left, mouth_right
    prev_landmarks: Optional[np.ndarray] = None,  # previous frame for temporal deltas
) -> np.ndarray:
    """
    Compute 16-dim geometry feature vector from a single frame's 5-pt landmarks.

    All spatial features are normalized by the inter-ocular distance (IOD) to be
    scale-invariant. This is the standard face normalization used in AU analysis.

    Args:
        landmarks:      (5, 2) float32 — [left_eye, right_eye, nose, mouth_left, mouth_right]
        prev_landmarks: (5, 2) optional — previous frame landmarks for delta features

    Returns:
        (16,) float32 feature vector, all values in approximately [-2, 2]
    """
    lm = landmarks.astype(np.float32)
    left_eye    = lm[0]
    right_eye   = lm[1]
    nose        = lm[2]
    mouth_left  = lm[3]
    mouth_right = lm[4]
    mouth_center = (mouth_left + mouth_right) * 0.5
    eye_center   = (left_eye + right_eye) * 0.5

    # Scale reference: inter-ocular distance
    iod = float(np.linalg.norm(right_eye - left_eye)) + 1e-6

    # ── Spatial features (0–12) ───────────────────────────────────────────────
    f0_iod_norm       = iod / (224.0 * 0.5)                                     # [0,1] relative face scale
    f1_eye_vert_ratio = float(abs(right_eye[1] - left_eye[1])) / iod             # eye-line non-horizontal
    f2_mouth_width    = float(np.linalg.norm(mouth_right - mouth_left)) / iod    # mouth width / IOD
    f3_mouth_nose_v   = float(abs(mouth_center[1] - nose[1])) / iod              # vertical mouth-nose dist
    f4_jaw_asymm      = abs(
        float(abs(mouth_left[0] - nose[0])) - float(abs(mouth_right[0] - nose[0]))
    ) / iod
    f5_nose_mouth_v   = float(abs(nose[1] - mouth_center[1])) / iod
    f6_face_width_px  = float(right_eye[0] - left_eye[0]) / 224.0
    f7_pose_roll      = float(np.arctan2(
        right_eye[1] - left_eye[1], right_eye[0] - left_eye[0]
    ))  # radians, approx [-π/6, π/6] in normal conditions
    f8_pose_yaw_proxy = float(eye_center[0] - nose[0]) / iod                     # +left, -right
    f9_le_nose_dist   = float(np.linalg.norm(left_eye - nose)) / iod
    f10_re_nose_dist  = float(np.linalg.norm(right_eye - nose)) / iod
    f11_ml_le_dist    = float(np.linalg.norm(mouth_left - left_eye)) / iod
    f12_mr_re_dist    = float(np.linalg.norm(mouth_right - right_eye)) / iod

    # ── Temporal delta features (13–15) ──────────────────────────────────────
    if prev_landmarks is not None:
        prev_lm = prev_landmarks.astype(np.float32)
        prev_iod = float(np.linalg.norm(prev_lm[1] - prev_lm[0])) + 1e-6
        prev_mw  = float(np.linalg.norm(prev_lm[4] - prev_lm[3])) / prev_iod
        prev_roll = float(np.arctan2(
            prev_lm[1][1] - prev_lm[0][1], prev_lm[1][0] - prev_lm[0][0]
        ))
        f13_d_iod  = (f0_iod_norm - prev_iod / (224.0 * 0.5))
        f14_d_roll = float(f7_pose_roll - prev_roll)
        f15_d_mw   = float(f2_mouth_width - prev_mw)
    else:
        f13_d_iod = f14_d_roll = f15_d_mw = 0.0

    return np.array([
        f0_iod_norm, f1_eye_vert_ratio, f2_mouth_width, f3_mouth_nose_v,
        f4_jaw_asymm, f5_nose_mouth_v, f6_face_width_px, f7_pose_roll,
        f8_pose_yaw_proxy, f9_le_nose_dist, f10_re_nose_dist,
        f11_ml_le_dist, f12_mr_re_dist,
        f13_d_iod, f14_d_roll, f15_d_mw,
    ], dtype=np.float32)


def landmarks_sequence_to_geometry(
    landmarks_list: list[np.ndarray],  # list of (5, 2) arrays, one per frame
) -> np.ndarray:
    """
    Convert a temporal sequence of 5-pt landmark arrays to a (T, 16) geometry matrix.

    Handles:
    - Missing frames (None entries) → zero-padded geometry
    - Temporal delta computation using previous valid frame
    - Returns (T, GEOMETRY_DIM) float32

    Args:
        landmarks_list: T-length list, each element is (5, 2) or None

    Returns:
        (T, 16) float32 geometry feature matrix
    """
    T = len(landmarks_list)
    geometry = np.zeros((T, GEOMETRY_DIM), dtype=np.float32)
    prev_lm = None

    for t, lm in enumerate(landmarks_list):
        if lm is None:
            # Zero-pad missing frames, keep prev_lm unchanged for next delta
            continue
        try:
            geometry[t] = compute_geometry_from_landmarks(lm, prev_landmarks=prev_lm)
            prev_lm = lm
        except Exception:
            pass  # malformed landmark → zero row

    return geometry


class GeometryFeatureExtractor(nn.Module):
    """
    Lightweight learnable projector on top of raw geometry features.

    Raw geometry (16-dim) → projected geometry (output_dim) with LayerNorm.
    This makes the geometry features learnable and matches the expected dim of
    the TemporalEncoder's GEOMETRY_DIM slot.

    Used as a thin adapter so the GRU can learn geometry salience per-task.
    """

    def __init__(
        self,
        input_dim: int = GEOMETRY_DIM,
        output_dim: int = GEOMETRY_DIM,
        hidden_dim: int = 32,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        if input_dim == output_dim:
            self.proj = nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, output_dim),
                nn.LayerNorm(output_dim),
            )
        else:
            self.proj = nn.Sequential(
                nn.Linear(input_dim, output_dim),
                nn.LayerNorm(output_dim),
                nn.GELU(),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, input_dim) raw geometry features
        Returns:
            (B, T, output_dim) projected geometry features
        """
        return self.proj(x)


# ─────────────────────────────────────────────────────────────────────────────
# Utility: convert inference landmarks to tensor for model input
# ─────────────────────────────────────────────────────────────────────────────

def build_geometry_tensor(
    landmarks_list: list[np.ndarray],
    device: torch.device,
) -> torch.Tensor:
    """
    Build a (1, T, 16) geometry tensor from a list of per-frame landmarks.
    Ready for direct input to TemporalEncoder.

    Args:
        landmarks_list: T-length list of (5, 2) or None arrays
        device: target torch device

    Returns:
        (1, T, 16) float32 tensor
    """
    geo_np = landmarks_sequence_to_geometry(landmarks_list)  # (T, 16)
    return torch.from_numpy(geo_np).unsqueeze(0).to(device)  # (1, T, 16)
