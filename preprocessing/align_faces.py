"""
preprocessing/align_faces.py
5-point affine face alignment to 224×224 canonical template for FORGE-Lite.

Uses OpenCV's estimateAffinePartial2D with fixed reference landmarks
derived from the FFHQ alignment convention. Deterministic and reproducible.
"""

from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Canonical landmark template (FFHQ convention, normalized to [0,1])
# Scaled to 224×224 with a margin applied
# ─────────────────────────────────────────────────────────────────────────────

# Reference 5-point landmarks for a 224×224 aligned face image
# Points: left_eye, right_eye, nose_tip, left_mouth, right_mouth
_REFERENCE_LANDMARKS_224 = np.array([
    [74.07997,  82.35223],   # left eye
    [149.92003, 82.35223],   # right eye
    [112.00011, 117.56964],  # nose tip
    [76.36537,  150.48108],  # left mouth corner
    [147.63463, 150.48108],  # right mouth corner
], dtype=np.float32)


class FaceAligner:
    """
    Aligns face crops to a canonical 224×224 template using 5 landmarks.

    Uses estimateAffinePartial2D (similarity transform: scale + rotate + translate)
    for robust alignment that handles yaw/roll but not extreme perspective.

    Output: 224×224 BGR uint8 numpy array.
    If alignment fails (bad landmarks), falls back to bbox crop + resize.
    """

    def __init__(
        self,
        output_size: int = 224,
        margin: float = 0.4,
    ) -> None:
        self.output_size = output_size
        self.margin = margin
        # Scale reference landmarks to output_size (already at 224)
        scale = output_size / 224.0
        self._ref_landmarks = _REFERENCE_LANDMARKS_224 * scale

    def align(
        self,
        image_bgr: np.ndarray,
        landmarks_5pt: np.ndarray,
        bbox: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Align a face crop.

        Args:
            image_bgr: Full frame (HWC BGR uint8)
            landmarks_5pt: (5, 2) float array of detected landmarks
            bbox: Optional [x1,y1,x2,y2] for fallback if alignment fails

        Returns:
            (aligned_face, transform_matrix)
            aligned_face: (output_size, output_size, 3) BGR uint8
            transform_matrix: (2, 3) affine transform for mask alignment
        """
        if landmarks_5pt is not None and not np.any(np.isnan(landmarks_5pt)):
            return self._align_by_landmarks(image_bgr, landmarks_5pt)
        elif bbox is not None:
            return self._align_by_bbox(image_bgr, bbox), np.eye(2, 3, dtype=np.float32)
        else:
            # No bbox, no landmarks — return center crop
            return self._center_crop(image_bgr), np.eye(2, 3, dtype=np.float32)

    def align_mask(
        self,
        mask: np.ndarray,
        transform_matrix: np.ndarray,
    ) -> np.ndarray:
        """
        Apply the same affine transform to a binary mask.
        Used to warp ground-truth manipulation masks into aligned space.
        """
        h, w = mask.shape[:2]
        aligned = cv2.warpAffine(
            mask, transform_matrix,
            (self.output_size, self.output_size),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        return aligned

    # ── Private helpers ───────────────────────────────────────────────────────

    def _align_by_landmarks(
        self, image_bgr: np.ndarray, landmarks: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute similarity transform from landmarks and warp."""
        # estimateAffinePartial2D: robust to outliers, returns 2×3 matrix
        transform, inliers = cv2.estimateAffinePartial2D(
            landmarks.astype(np.float32),
            self._ref_landmarks,
            method=cv2.LMEDS,
        )
        if transform is None:
            # Fallback: least-squares solve
            transform = self._least_squares_affine(landmarks, self._ref_landmarks)

        aligned = cv2.warpAffine(
            image_bgr,
            transform,
            (self.output_size, self.output_size),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )
        return aligned, transform

    def _align_by_bbox(self, image_bgr: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        """Fallback: crop with margin and resize."""
        h, w = image_bgr.shape[:2]
        x1, y1, x2, y2 = bbox.astype(int)
        bw, bh = x2 - x1, y2 - y1
        mx = int(bw * self.margin)
        my = int(bh * self.margin)
        x1c = max(0, x1 - mx)
        y1c = max(0, y1 - my)
        x2c = min(w, x2 + mx)
        y2c = min(h, y2 + my)
        crop = image_bgr[y1c:y2c, x1c:x2c]
        if crop.size == 0:
            return np.zeros((self.output_size, self.output_size, 3), dtype=np.uint8)
        return cv2.resize(crop, (self.output_size, self.output_size), interpolation=cv2.INTER_LINEAR)

    def _center_crop(self, image_bgr: np.ndarray) -> np.ndarray:
        """Return center square crop resized to output_size."""
        h, w = image_bgr.shape[:2]
        s = min(h, w)
        y0 = (h - s) // 2
        x0 = (w - s) // 2
        crop = image_bgr[y0:y0+s, x0:x0+s]
        return cv2.resize(crop, (self.output_size, self.output_size), interpolation=cv2.INTER_LINEAR)

    @staticmethod
    def _least_squares_affine(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
        """Fit a full affine transform via least-squares (6 DOF)."""
        n = len(src)
        A = np.zeros((2 * n, 6), dtype=np.float64)
        b = np.zeros(2 * n, dtype=np.float64)
        for i, ((sx, sy), (dx, dy)) in enumerate(zip(src, dst)):
            A[2*i]   = [sx, sy, 1, 0,  0,  0]
            A[2*i+1] = [0,  0,  0, sx, sy, 1]
            b[2*i]   = dx
            b[2*i+1] = dy
        params, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        transform = np.array([
            [params[0], params[1], params[2]],
            [params[3], params[4], params[5]],
        ], dtype=np.float32)
        return transform
