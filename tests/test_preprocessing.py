"""
tests/test_preprocessing.py
Unit tests for FORGE-Lite preprocessing modules:
- decode.py (VideoDecoder)
- detect_faces.py (FaceDetector)
- track_faces.py (FaceTracker)
- align_faces.py (FaceAligner)
- quality.py (QualityAnalyzer)
- build_manifest.py (Manifest generation)
"""

import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from preprocessing.align_faces import FaceAligner
from preprocessing.decode import VideoDecoder, VideoMetadata
from preprocessing.detect_faces import FaceDetection, FaceDetector
from preprocessing.quality import QualityAnalyzer
from preprocessing.track_faces import FaceTrack, FaceTracker


# ─────────────────────────────────────────────────────────────────────────────
# QualityAnalyzer Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_quality_analyzer_sharp_vs_blurred():
    qa = QualityAnalyzer()

    # Create a sharp image with high-frequency edges
    sharp_img = np.zeros((224, 224, 3), dtype=np.uint8)
    sharp_img[::4, ::4] = 255
    q_sharp = qa.analyze_frame(sharp_img, bbox=np.array([10, 10, 200, 200], dtype=np.float32), frame_wh=(640, 480))

    # Create a heavily blurred image
    blurred_img = cv2.GaussianBlur(sharp_img, (31, 31), 0)
    q_blur = qa.analyze_frame(blurred_img, bbox=np.array([10, 10, 200, 200], dtype=np.float32), frame_wh=(640, 480))

    assert q_sharp.blur_score > q_blur.blur_score
    assert 0.0 <= q_sharp.blur_score <= 1.0
    assert 0.0 <= q_blur.blur_score <= 1.0


def test_quality_track_aggregation():
    qa = QualityAnalyzer()
    frames_q = []
    landmarks_seq = []

    for i in range(10):
        img = np.random.randint(50, 200, (224, 224, 3), dtype=np.uint8)
        bbox = np.array([20, 20, 180, 180], dtype=np.float32)
        lm = np.array([[50 + i, 50], [150 + i, 50], [100 + i, 100], [70 + i, 140], [130 + i, 140]], dtype=np.float32)
        q = qa.analyze_frame(img, landmarks_5pt=lm, bbox=bbox, frame_wh=(640, 480))
        frames_q.append(q)
        landmarks_seq.append(lm)

    track_q = qa.analyze_track(
        frame_qualities=frames_q,
        track_id=1,
        continuity=0.95,
        track_duration_sec=1.25,
        all_landmarks=landmarks_seq,
    )

    vec = track_q.to_vector()
    assert len(vec) == 5
    assert 0.0 <= track_q.mean_blur <= 1.0
    assert 0.0 <= track_q.continuity <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# FaceAligner Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_face_aligner_canonical_transformation():
    aligner = FaceAligner(output_size=224)
    canvas = np.full((480, 640, 3), 120, dtype=np.uint8)

    # Standard facial landmarks
    landmarks = np.array([
        [260.0, 200.0],  # left eye
        [380.0, 200.0],  # right eye
        [320.0, 250.0],  # nose
        [280.0, 310.0],  # left mouth
        [360.0, 310.0],  # right mouth
    ], dtype=np.float32)

    aligned, M = aligner.align(canvas, landmarks)

    assert aligned.shape == (224, 224, 3)
    assert M.shape == (2, 3)
    assert aligned.dtype == np.uint8


def test_face_aligner_fallback_on_bbox():
    aligner = FaceAligner(output_size=224)
    canvas = np.zeros((480, 640, 3), dtype=np.uint8)
    bbox = np.array([100.0, 100.0, 300.0, 300.0], dtype=np.float32)

    # Pass None for landmarks
    aligned, M = aligner.align(canvas, landmarks_5pt=None, bbox=bbox)
    assert aligned.shape == (224, 224, 3)


# ─────────────────────────────────────────────────────────────────────────────
# FaceTracker Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_face_tracker_association():
    tracker = FaceTracker(iou_threshold=0.3, max_gap_frames=3, min_track_length=2)

    # Frame 1: Detection at (100, 100, 200, 200)
    det1 = FaceDetection(
        bbox=np.array([100.0, 100.0, 200.0, 200.0]),
        score=0.95,
        landmarks=np.zeros((5, 2)),
        frame_idx=0,
        timestamp_sec=0.0,
    )
    tracker.update([det1], frame_idx=0)

    # Frame 2: Slight motion (105, 105, 205, 205)
    det2 = FaceDetection(
        bbox=np.array([105.0, 105.0, 205.0, 205.0]),
        score=0.94,
        landmarks=np.zeros((5, 2)),
        frame_idx=1,
        timestamp_sec=0.1,
    )
    tracker.update([det2], frame_idx=1)

    tracks = tracker.finalize()
    assert len(tracks) >= 1
    dom = tracker.get_dominant_track(tracks)
    assert dom is not None
    assert dom.length == 2
    assert dom.continuity == 1.0


# ─────────────────────────────────────────────────────────────────────────────
# VideoDecoder Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_video_decoder_synthetic_video():
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        video_path = tmp.name

    try:
        # Create a small valid MP4 video using OpenCV
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(video_path, fourcc, 10.0, (160, 120))
        for _ in range(15):
            f = np.random.randint(0, 255, (120, 160, 3), dtype=np.uint8)
            writer.write(f)
        writer.release()

        decoder = VideoDecoder(fps_target=8.0, max_frames=30)
        meta = decoder.get_metadata(video_path)

        assert meta.total_frames >= 15
        assert meta.width == 160
        assert meta.height == 120

        frames, timestamps = decoder.decode_all(video_path, meta)
        assert len(frames) > 0
        assert frames[0].image.shape == (120, 160, 3)
    finally:
        if Path(video_path).exists():
            Path(video_path).unlink()
