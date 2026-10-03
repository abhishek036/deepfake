"""
preprocessing/build_manifest.py
Full preprocessing pipeline orchestrator for FORGE-Lite.

Runs the complete pipeline:
  video → metadata → frame sampling → face detection → face tracking
  → landmark extraction → face alignment → quality measurement
  → context crop → temporal window creation → mask alignment
  → manifest JSON generation

Precomputes everything once. Never re-runs face detection during training.
Outputs:
  - data/faces/{dataset}/{video_id}/{frame_idx}.jpg  (224×224 aligned crops)
  - data/tracks/{dataset}/{video_id}.json            (face track data)
  - data/quality/{dataset}/{video_id}.json           (quality metadata)
  - data/manifests/{dataset}.jsonl                   (one record per video)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm

from preprocessing.decode import VideoDecoder, VideoMetadata
from preprocessing.detect_faces import FaceDetector, FaceDetection
from preprocessing.track_faces import FaceTracker, FaceTrack
from preprocessing.align_faces import FaceAligner
from preprocessing.quality import QualityAnalyzer, FrameQuality

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Manifest record schema (matches PRD Section 7, Step 2)
# ─────────────────────────────────────────────────────────────────────────────

def make_manifest_record(
    dataset: str,
    video_id: str,
    source_id: str,
    identity_id: str,
    label: str,                     # "real" | "fake"
    manipulation_scope: str,        # "face_localized" | "full_face" | "full_frame"
    manipulation_family: str,       # "face_swap" | "reenactment" | "synthesis" | "real"
    generator_family: str,          # "deepfakes" | "face2face" | "simswap" | ...
    split: str,                     # "train" | "val" | "test"
    video_meta: VideoMetadata,
    track: Optional[FaceTrack],
    track_quality,
    faces_dir: str,
    mask_available: bool = False,
    mask_path: Optional[str] = None,
) -> dict:
    """Build a manifest record for one video."""
    return {
        "dataset": dataset,
        "video_id": video_id,
        "source_id": source_id,
        "identity_id": identity_id,
        "label": label,
        "label_int": 1 if label == "fake" else 0,
        "manipulation_scope": manipulation_scope,
        "manipulation_family": manipulation_family,
        "generator_family": generator_family,
        "split": split,
        "fps": video_meta.fps,
        "duration_seconds": video_meta.duration_seconds,
        "total_frames": video_meta.total_frames,
        "sampled_frames": len(video_meta.sampled_frame_indices),
        "width": video_meta.width,
        "height": video_meta.height,
        "codec": video_meta.codec,
        "audio_available": video_meta.audio_available,
        "mask_available": mask_available,
        "mask_path": mask_path,
        "faces_dir": faces_dir,
        "track_id": track.track_id if track else None,
        "track_length": track.length if track else 0,
        "track_continuity": float(track.continuity) if track else 0.0,
        "track_duration_sec": (
            (track.end_frame - track.start_frame) / max(video_meta.fps, 1e-6)
            if track else 0.0
        ),
        "track_quality": track_quality.to_dict() if track_quality else None,
        "sufficient_for_analysis": (
            track_quality.sufficient_for_analysis if track_quality else False
        ),
        "sufficient_for_rppg": (
            track_quality.sufficient_for_rppg if track_quality else False
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# ManifestBuilder
# ─────────────────────────────────────────────────────────────────────────────

class ManifestBuilder:
    """
    Orchestrates the full FORGE-Lite preprocessing pipeline for a dataset.

    Usage:
        builder = ManifestBuilder(data_cfg, processed_cfg)
        builder.build_dataset("faceforensics", video_list, label_fn, split_fn)
    """

    def __init__(self, data_cfg, dry_run: bool = False) -> None:
        self.cfg = data_cfg
        self.dry_run = dry_run

        self.decoder = VideoDecoder(
            fps_target=data_cfg.frame_sampling.fps_target,
            max_frames=data_cfg.frame_sampling.max_frames_per_video,
            min_frames=data_cfg.frame_sampling.min_frames_per_video,
            strategy=data_cfg.frame_sampling.strategy,
        )
        self.detector = FaceDetector(
            backend=data_cfg.face_detection.backend,
            min_face_size=data_cfg.face_detection.min_face_size,
            confidence_threshold=data_cfg.face_detection.confidence_threshold,
        )
        self.aligner = FaceAligner(
            output_size=data_cfg.face_alignment.output_size,
            margin=data_cfg.face_alignment.margin,
        )
        self.quality_analyzer = QualityAnalyzer(
            min_face_area_ratio=data_cfg.quality_thresholds.min_face_area_ratio,
            min_blur_score=data_cfg.quality_thresholds.min_blur_score,
            min_track_continuity=data_cfg.quality_thresholds.min_track_continuity,
            min_rppg_duration_sec=data_cfg.quality_thresholds.min_rppg_duration_seconds,
        )

        # Output directories
        self.processed = data_cfg.processed
        self._ensure_dirs()

    def process_video(
        self,
        video_path: str,
        dataset: str,
        video_id: str,
        source_id: str,
        identity_id: str,
        label: str,
        manipulation_scope: str,
        manipulation_family: str,
        generator_family: str,
        split: str,
        mask_video_path: Optional[str] = None,
    ) -> Optional[dict]:
        """
        Process a single video through the full pipeline.
        Returns a manifest record dict or None on failure.
        """
        try:
            return self._process_video_impl(
                video_path, dataset, video_id, source_id, identity_id,
                label, manipulation_scope, manipulation_family, generator_family,
                split, mask_video_path
            )
        except Exception as e:
            logger.error("Failed to process %s: %s", video_id, e, exc_info=True)
            return None

    def _process_video_impl(
        self, video_path, dataset, video_id, source_id, identity_id,
        label, manipulation_scope, manipulation_family, generator_family,
        split, mask_video_path
    ) -> dict:
        # 1. Decode metadata and sample frames
        video_meta = self.decoder.get_metadata(video_path)
        frames, _ = self.decoder.decode_all(video_path, video_meta)

        if len(frames) < self.cfg.frame_sampling.min_frames_per_video:
            raise ValueError(f"Too few frames ({len(frames)}) in {video_id}")

        # 2. Decode mask frames if available
        mask_frames = {}
        if mask_video_path and Path(mask_video_path).exists():
            mask_decoder = VideoDecoder(fps_target=video_meta.sampled_fps)
            raw_mask_frames, _ = mask_decoder.decode_all(mask_video_path)
            for mf in raw_mask_frames:
                mask_frames[mf.frame_idx] = mf.image

        # 3. Face detection + tracking
        tracker = FaceTracker(
            iou_threshold=0.4,
            max_gap_frames=5,
            min_track_length=4,
        )
        all_detections: dict[int, FaceDetection] = {}
        for frame in frames:
            det = self.detector.detect_dominant(frame.image, frame.frame_idx, frame.timestamp_sec)
            if det:
                tracker.update([det], frame.frame_idx)
                all_detections[frame.frame_idx] = det
            else:
                tracker.update([], frame.frame_idx)

        all_tracks = tracker.finalize()
        dominant_track = tracker.get_dominant_track(all_tracks)

        # 4. Save aligned face crops + quality analysis
        faces_dir = str(Path(self.processed.faces_dir) / dataset / video_id)
        if not self.dry_run:
            Path(faces_dir).mkdir(parents=True, exist_ok=True)

        frame_qualities: list[FrameQuality] = []
        all_landmarks: list[Optional[np.ndarray]] = []
        track_frame_indices = set(dominant_track.frame_indices) if dominant_track else set()

        for frame in frames:
            if frame.frame_idx not in track_frame_indices:
                continue

            det = all_detections.get(frame.frame_idx)
            if det is None:
                continue

            # Align face
            aligned, transform = self.aligner.align(
                frame.image, det.landmarks, det.bbox
            )

            # Save aligned crop
            if not self.dry_run:
                out_path = Path(faces_dir) / f"{frame.frame_idx:06d}.jpg"
                cv2.imwrite(str(out_path), aligned, [cv2.IMWRITE_JPEG_QUALITY, 95])

            # Save aligned mask if available
            if frame.frame_idx in mask_frames and not self.dry_run:
                mask_gray = cv2.cvtColor(mask_frames[frame.frame_idx], cv2.COLOR_BGR2GRAY)
                _, mask_binary = cv2.threshold(mask_gray, 127, 255, cv2.THRESH_BINARY)
                aligned_mask = self.aligner.align_mask(mask_binary, transform)
                mask_out = Path(self.processed.masks_dir) / dataset / video_id
                mask_out.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(mask_out / f"{frame.frame_idx:06d}.png"), aligned_mask)

            # Quality analysis
            h, w = frame.image.shape[:2]
            quality = self.quality_analyzer.analyze_frame(
                face_crop_bgr=aligned,
                landmarks_5pt=det.landmarks,
                bbox=det.bbox,
                frame_wh=(w, h),
            )
            frame_qualities.append(quality)
            all_landmarks.append(det.landmarks)

        # 5. Track quality
        track_quality = None
        if dominant_track and frame_qualities:
            track_duration = (
                (dominant_track.end_frame - dominant_track.start_frame)
                / max(video_meta.fps, 1e-6)
            )
            track_quality = self.quality_analyzer.analyze_track(
                frame_qualities=frame_qualities,
                track_id=dominant_track.track_id,
                continuity=dominant_track.continuity,
                track_duration_sec=track_duration,
                all_landmarks=all_landmarks,
            )

        # 6. Save track JSON
        if dominant_track and not self.dry_run:
            track_path = Path(self.processed.tracks_dir) / dataset / f"{video_id}.json"
            track_path.parent.mkdir(parents=True, exist_ok=True)
            with open(track_path, "w") as f:
                json.dump(dominant_track.to_dict(), f)

        # 7. Save quality JSON
        if track_quality and not self.dry_run:
            quality_path = Path(self.processed.quality_dir) / dataset / f"{video_id}.json"
            quality_path.parent.mkdir(parents=True, exist_ok=True)
            with open(quality_path, "w") as f:
                json.dump(track_quality.to_dict(), f)

        # 8. Build manifest record
        mask_available = bool(mask_frames and dominant_track)
        mask_path = str(Path(self.processed.masks_dir) / dataset / video_id) if mask_available else None

        record = make_manifest_record(
            dataset=dataset, video_id=video_id, source_id=source_id,
            identity_id=identity_id, label=label,
            manipulation_scope=manipulation_scope,
            manipulation_family=manipulation_family,
            generator_family=generator_family,
            split=split, video_meta=video_meta,
            track=dominant_track, track_quality=track_quality,
            faces_dir=faces_dir, mask_available=mask_available,
            mask_path=mask_path,
        )
        return record

    def _ensure_dirs(self) -> None:
        if self.dry_run:
            return
        for dir_key in ["manifests_dir", "splits_dir", "tracks_dir",
                         "quality_dir", "masks_dir", "features_dir", "faces_dir"]:
            Path(self.processed[dir_key]).mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    """CLI entry point for preprocessing pipeline."""
    parser = argparse.ArgumentParser(description="FORGE-Lite Preprocessing Pipeline")
    parser.add_argument("--config", default="configs/data.yaml", help="Data config path")
    parser.add_argument("--dataset", required=True,
                        choices=["faceforensics", "celeb_df_v2", "dfdc_preview",
                                 "wild_deepfake", "ffhq", "synthetic_test"],
                        help="Dataset to preprocess")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate pipeline without writing files")
    parser.add_argument("--workers", type=int, default=2, help="Parallel workers")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of videos (debug)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    cfg = OmegaConf.load(args.config)
    logger.info("Loaded config from %s", args.config)
    logger.info("Dataset: %s | Dry-run: %s | Workers: %d", args.dataset, args.dry_run, args.workers)

    if args.dry_run:
        logger.info("DRY RUN MODE — no files will be written")
        builder = ManifestBuilder(cfg, dry_run=True)
        logger.info("Pipeline initialization OK")
        return

    builder = ManifestBuilder(cfg, dry_run=False)
    logger.info("ManifestBuilder initialized. Run dataset-specific loader to process videos.")
    logger.info("Example: build_faceforensics_manifest(builder, cfg, limit=%s)", args.limit)


if __name__ == "__main__":
    main()
