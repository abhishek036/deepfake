"""
preprocessing/track_faces.py
Multi-frame face tracking for FORGE-Lite.

Uses IoU-based track association across frames to build stable face tracks.
Assigns track IDs, computes track continuity scores, and selects the
dominant (longest stable) track for single-face analysis.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from preprocessing.detect_faces import FaceDetection

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FaceTrack:
    """A temporal sequence of face detections forming a single track."""
    track_id: int
    detections: list[FaceDetection] = field(default_factory=list)
    frame_indices: list[int] = field(default_factory=list)

    @property
    def length(self) -> int:
        return len(self.detections)

    @property
    def start_frame(self) -> int:
        return self.frame_indices[0] if self.frame_indices else -1

    @property
    def end_frame(self) -> int:
        return self.frame_indices[-1] if self.frame_indices else -1

    @property
    def duration_frames(self) -> int:
        if not self.frame_indices:
            return 0
        return self.end_frame - self.start_frame + 1

    @property
    def continuity(self) -> float:
        """Fraction of frames in the track's span that have a detection."""
        if self.duration_frames == 0:
            return 0.0
        return self.length / self.duration_frames

    @property
    def mean_score(self) -> float:
        if not self.detections:
            return 0.0
        return float(np.mean([d.score for d in self.detections]))

    @property
    def mean_bbox(self) -> np.ndarray:
        """Mean bounding box across all detections in the track."""
        if not self.detections:
            return np.zeros(4)
        return np.mean([d.bbox for d in self.detections], axis=0)

    def to_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "length": self.length,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "duration_frames": self.duration_frames,
            "continuity": float(self.continuity),
            "mean_score": float(self.mean_score),
            "detections": [d.to_dict() for d in self.detections],
        }


# ─────────────────────────────────────────────────────────────────────────────
# FaceTracker
# ─────────────────────────────────────────────────────────────────────────────

class FaceTracker:
    """
    IoU-based multi-object face tracker.

    Algorithm:
    1. For each frame, detect faces.
    2. Match detections to existing active tracks via Hungarian/greedy IoU matching.
    3. Create new tracks for unmatched detections.
    4. Mark tracks as ended if no match for max_gap frames.

    Outputs a list of FaceTrack objects. The dominant track (highest
    length * continuity score) is identified for single-face pipelines.
    """

    def __init__(
        self,
        iou_threshold: float = 0.4,
        max_gap_frames: int = 5,
        min_track_length: int = 4,
    ) -> None:
        self.iou_threshold = iou_threshold
        self.max_gap_frames = max_gap_frames
        self.min_track_length = min_track_length

        self._next_id = 0
        self._active_tracks: dict[int, FaceTrack] = {}
        self._completed_tracks: list[FaceTrack] = []
        self._last_seen: dict[int, int] = {}  # track_id → last frame idx

    def reset(self) -> None:
        self._next_id = 0
        self._active_tracks = {}
        self._completed_tracks = []
        self._last_seen = {}

    def update(self, detections: list[FaceDetection], frame_idx: int) -> dict[int, FaceDetection]:
        """
        Update tracker with new detections for a frame.

        Returns mapping of track_id → FaceDetection for this frame.
        """
        if not detections:
            self._expire_tracks(frame_idx)
            return {}

        active_ids = list(self._active_tracks.keys())
        if not active_ids:
            # No existing tracks — create new for all detections
            result = {}
            for det in detections:
                tid = self._new_track(det, frame_idx)
                result[tid] = det
            return result

        # Build cost matrix (1 - IoU) for greedy matching
        active_bboxes = np.array([self._active_tracks[tid].detections[-1].bbox for tid in active_ids])
        det_bboxes = np.array([d.bbox for d in detections])
        iou_matrix = self._batch_iou(active_bboxes, det_bboxes)

        matched_tracks = set()
        matched_dets = set()
        result = {}

        # Greedy matching: highest IoU first
        flat_indices = np.argsort(-iou_matrix.ravel())
        for flat_idx in flat_indices:
            t_idx, d_idx = divmod(flat_idx, len(detections))
            if t_idx in matched_tracks or d_idx in matched_dets:
                continue
            if iou_matrix[t_idx, d_idx] < self.iou_threshold:
                break
            tid = active_ids[t_idx]
            self._active_tracks[tid].detections.append(detections[d_idx])
            self._active_tracks[tid].frame_indices.append(frame_idx)
            self._last_seen[tid] = frame_idx
            result[tid] = detections[d_idx]
            matched_tracks.add(t_idx)
            matched_dets.add(d_idx)

        # Create new tracks for unmatched detections
        for d_idx, det in enumerate(detections):
            if d_idx not in matched_dets:
                tid = self._new_track(det, frame_idx)
                result[tid] = det

        # Expire stale tracks
        self._expire_tracks(frame_idx)
        return result

    def finalize(self) -> list[FaceTrack]:
        """End all active tracks and return all completed tracks."""
        for track in self._active_tracks.values():
            if track.length >= self.min_track_length:
                self._completed_tracks.append(track)
        self._active_tracks = {}
        return self._completed_tracks

    def get_dominant_track(self, tracks: Optional[list[FaceTrack]] = None) -> Optional[FaceTrack]:
        """
        Select the dominant track (largest area * continuity score).
        Suitable for single-face deepfake scenarios.
        """
        tracks = tracks or self._completed_tracks
        if not tracks:
            return None
        return max(tracks, key=lambda t: t.length * t.continuity)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _new_track(self, det: FaceDetection, frame_idx: int) -> int:
        tid = self._next_id
        self._next_id += 1
        track = FaceTrack(track_id=tid)
        track.detections.append(det)
        track.frame_indices.append(frame_idx)
        self._active_tracks[tid] = track
        self._last_seen[tid] = frame_idx
        return tid

    def _expire_tracks(self, current_frame: int) -> None:
        """Move tracks that haven't been updated for max_gap_frames into completed."""
        to_expire = [
            tid for tid, last in self._last_seen.items()
            if current_frame - last > self.max_gap_frames
        ]
        for tid in to_expire:
            track = self._active_tracks.pop(tid)
            del self._last_seen[tid]
            if track.length >= self.min_track_length:
                self._completed_tracks.append(track)

    @staticmethod
    def _iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
        """Compute IoU between two [x1,y1,x2,y2] boxes."""
        xa1, ya1, xa2, ya2 = box_a
        xb1, yb1, xb2, yb2 = box_b
        inter_x1 = max(xa1, xb1)
        inter_y1 = max(ya1, yb1)
        inter_x2 = min(xa2, xb2)
        inter_y2 = min(ya2, yb2)
        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        area_a = max(0, xa2 - xa1) * max(0, ya2 - ya1)
        area_b = max(0, xb2 - xb1) * max(0, yb2 - yb1)
        union = area_a + area_b - inter_area
        return float(inter_area / union) if union > 0 else 0.0

    @staticmethod
    def _batch_iou(bboxes_a: np.ndarray, bboxes_b: np.ndarray) -> np.ndarray:
        """Compute IoU matrix between two sets of boxes. Shape: (N, M)."""
        N, M = len(bboxes_a), len(bboxes_b)
        iou_matrix = np.zeros((N, M), dtype=np.float32)
        for i, box_a in enumerate(bboxes_a):
            for j, box_b in enumerate(bboxes_b):
                xa1, ya1, xa2, ya2 = box_a
                xb1, yb1, xb2, yb2 = box_b
                ix1, iy1 = max(xa1, xb1), max(ya1, yb1)
                ix2, iy2 = min(xa2, xb2), min(ya2, yb2)
                inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                area_a = max(0, xa2 - xa1) * max(0, ya2 - ya1)
                area_b = max(0, xb2 - xb1) * max(0, yb2 - yb1)
                union = area_a + area_b - inter
                iou_matrix[i, j] = inter / union if union > 0 else 0.0
        return iou_matrix
