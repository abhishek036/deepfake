"""
preprocessing/detect_faces.py
RetinaFace-based face detection for FORGE-Lite.

Detects faces in individual frames, returns bounding boxes, landmarks,
and confidence scores. Filters by min face size and confidence threshold.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FaceDetection:
    """A single face detection result."""
    bbox: np.ndarray        # [x1, y1, x2, y2] in pixel coordinates
    score: float            # detection confidence
    landmarks: np.ndarray   # (5, 2) — left_eye, right_eye, nose, left_mouth, right_mouth
    frame_idx: int
    timestamp_sec: float

    @property
    def width(self) -> float:
        return float(self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> float:
        return float(self.bbox[3] - self.bbox[1])

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return (
            float((self.bbox[0] + self.bbox[2]) / 2),
            float((self.bbox[1] + self.bbox[3]) / 2),
        )

    @property
    def face_area_ratio(self) -> float:
        """Placeholder — computed in quality.py with image dims."""
        return 0.0

    def to_dict(self) -> dict:
        return {
            "bbox": self.bbox.tolist(),
            "score": float(self.score),
            "landmarks": self.landmarks.tolist(),
            "frame_idx": self.frame_idx,
            "timestamp_sec": float(self.timestamp_sec),
        }


# ─────────────────────────────────────────────────────────────────────────────
# FaceDetector
# ─────────────────────────────────────────────────────────────────────────────

class FaceDetector:
    """
    RetinaFace-based face detector wrapping facenet-pytorch's MTCNN
    as a reliable cross-platform fallback, with RetinaFace as primary backend.

    Backend priority:
    1. RetinaFace (insightface) — most accurate
    2. MTCNN (facenet-pytorch) — reliable fallback

    Always initialized lazily to avoid GPU memory at import time.
    """

    def __init__(
        self,
        backend: str = "retinaface",
        min_face_size: int = 30,
        confidence_threshold: float = 0.5,
        max_faces_per_frame: int = 3,
        device: Optional[str] = None,
    ) -> None:
        self.backend = backend
        self.min_face_size = min_face_size
        self.confidence_threshold = confidence_threshold
        self.max_faces_per_frame = max_faces_per_frame
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._detector = None
        self._mtcnn_fallback = None

    def _ensure_loaded(self) -> None:
        if self._detector is not None:
            return
        if self.backend == "retinaface":
            self._detector = self._load_retinaface()
        else:
            self._detector = self._load_mtcnn()

    def _load_retinaface(self):
        """Load RetinaFace via insightface."""
        try:
            import insightface
            from insightface.app import FaceAnalysis
            app = FaceAnalysis(name="buffalo_sc", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
            app.prepare(ctx_id=0 if self.device == "cuda" else -1, det_size=(640, 640))
            logger.info("RetinaFace loaded via insightface")
            return ("retinaface_insightface", app)
        except ImportError:
            logger.warning("insightface not available, falling back to MTCNN")
            return self._load_mtcnn()

    def _load_mtcnn(self):
        """Load MTCNN via facenet-pytorch."""
        from facenet_pytorch import MTCNN
        mtcnn = MTCNN(
            min_face_size=self.min_face_size,
            thresholds=[0.6, 0.7, 0.8],
            keep_all=True,
            device=self.device,
            post_process=False,
        )
        logger.info("MTCNN face detector loaded on %s", self.device)
        return ("mtcnn", mtcnn)

    def detect(
        self, image_bgr: np.ndarray, frame_idx: int = 0, timestamp_sec: float = 0.0
    ) -> list[FaceDetection]:
        """
        Detect faces in a single BGR frame.

        Returns list of FaceDetection objects, sorted by score descending,
        filtered by confidence and minimum face size, capped at max_faces_per_frame.
        """
        self._ensure_loaded()
        h, w = image_bgr.shape[:2]

        backend_name, detector = self._detector
        if backend_name == "retinaface_insightface":
            detections = self._detect_insightface(detector, image_bgr, frame_idx, timestamp_sec)
        else:
            detections = self._detect_mtcnn(detector, image_bgr, frame_idx, timestamp_sec, h, w)

        # If primary backend found nothing, attempt MTCNN fallback
        if not detections and backend_name == "retinaface_insightface":
            try:
                if self._mtcnn_fallback is None:
                    from facenet_pytorch import MTCNN
                    self._mtcnn_fallback = MTCNN(min_face_size=20, keep_all=True, device=self.device)
                detections = self._detect_mtcnn(self._mtcnn_fallback, image_bgr, frame_idx, timestamp_sec, h, w)
            except Exception as e:
                logger.debug("MTCNN fallback failed: %s", e)

        # Filter by confidence and min size
        detections = [
            d for d in detections
            if d.score >= self.confidence_threshold
            and d.width >= self.min_face_size
            and d.height >= self.min_face_size
        ]

        # Sort by confidence, cap at max_faces_per_frame
        detections.sort(key=lambda d: d.score, reverse=True)
        return detections[: self.max_faces_per_frame]

    def detect_dominant(
        self, image_bgr: np.ndarray, frame_idx: int = 0, timestamp_sec: float = 0.0
    ) -> Optional[FaceDetection]:
        """Return the single largest-area face (dominant face) or None."""
        faces = self.detect(image_bgr, frame_idx, timestamp_sec)
        if faces:
            return max(faces, key=lambda d: d.area)

        # Fallback for pre-cropped face videos/images (e.g. 124x152, where face fills the whole frame)
        h, w = image_bgr.shape[:2]
        if min(h, w) <= 320 and 0.5 <= (h / max(w, 1)) <= 2.0:
            landmarks = np.array([
                [w * 0.35, h * 0.38],  # left eye
                [w * 0.65, h * 0.38],  # right eye
                [w * 0.50, h * 0.55],  # nose
                [w * 0.38, h * 0.75],  # mouth left
                [w * 0.62, h * 0.75],  # mouth right
            ], dtype=np.float32)
            return FaceDetection(
                bbox=np.array([0, 0, w, h], dtype=np.float32),
                score=0.95,
                landmarks=landmarks,
                frame_idx=frame_idx,
                timestamp_sec=timestamp_sec,
            )
        return None

    # ── Backend-specific detection ────────────────────────────────────────────

    def _detect_insightface(self, app, image_bgr: np.ndarray, frame_idx: int, ts: float) -> list[FaceDetection]:
        faces = app.get(image_bgr)
        results = []
        for face in faces:
            bbox = face.bbox.astype(np.float32)
            score = float(face.det_score)
            kps = face.kps.astype(np.float32)  # (5, 2)
            results.append(FaceDetection(
                bbox=bbox, score=score, landmarks=kps,
                frame_idx=frame_idx, timestamp_sec=ts
            ))

        # If no faces detected on raw frame (e.g. tight crops or low-res images), try reflection padding
        if not results:
            h, w = image_bgr.shape[:2]
            pad_h, pad_w = int(h * 0.4), int(w * 0.4)
            padded = cv2.copyMakeBorder(image_bgr, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_REFLECT)
            padded_faces = app.get(padded)
            for face in padded_faces:
                bbox = face.bbox.astype(np.float32)
                bbox[0] = np.clip(bbox[0] - pad_w, 0, w)
                bbox[1] = np.clip(bbox[1] - pad_h, 0, h)
                bbox[2] = np.clip(bbox[2] - pad_w, 0, w)
                bbox[3] = np.clip(bbox[3] - pad_h, 0, h)
                kps = face.kps.astype(np.float32)
                kps[:, 0] = np.clip(kps[:, 0] - pad_w, 0, w)
                kps[:, 1] = np.clip(kps[:, 1] - pad_h, 0, h)
                score = float(face.det_score)
                results.append(FaceDetection(
                    bbox=bbox, score=score, landmarks=kps,
                    frame_idx=frame_idx, timestamp_sec=ts
                ))
        return results

    def _detect_mtcnn(self, mtcnn, image_bgr: np.ndarray, frame_idx: int, ts: float,
                       h: int, w: int) -> list[FaceDetection]:
        from PIL import Image
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(image_rgb)

        boxes, probs, points = mtcnn.detect(pil_img, landmarks=True)
        results = []
        if boxes is None:
            return results

        for box, prob, pts in zip(boxes, probs, points if points is not None else [None]*len(boxes)):
            if prob is None:
                continue
            bbox = np.array([box[0], box[1], box[2], box[3]], dtype=np.float32)
            landmarks = pts.astype(np.float32) if pts is not None else np.zeros((5, 2), dtype=np.float32)
            results.append(FaceDetection(
                bbox=bbox, score=float(prob), landmarks=landmarks,
                frame_idx=frame_idx, timestamp_sec=ts
            ))
        return results
