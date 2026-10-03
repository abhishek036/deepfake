"""
preprocessing/preprocess_all.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FORGE-Lite — Complete Preprocessing Pipeline

Processes ALL datasets in one pass:
  FF++ C23 | Celeb-DF v2 | WildDeepfake | FFHQ (real)

Per video/image this pipeline:
  1. Decodes frames at uniform FPS
  2. Detects + aligns faces (224x224)
  3. Computes quality features (10-dim)
  4. Writes JSONL manifest records
  5. Saves aligned crops to disk

After all datasets processed:
  6. Generates identity-disjoint train/val/test splits

Usage:
    python -m preprocessing.preprocess_all [--dataset all|faceforensics|celeb_df|wild_deepfake|ffhq]
                                           [--max-videos N] [--resume]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterator, Optional

import cv2
import numpy as np
from tqdm import tqdm

# -- Project root on path --
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# -- Setup logging --
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(ROOT / "preprocessing" / "preprocess.log", mode="a"),
    ],
)
logger = logging.getLogger("preprocess_all")

# -- Constants --
SAMPLE_FPS: float = 1.0
MAX_FRAMES: int   = 30
FACE_SIZE: int    = 224
MIN_FACE_PX: int  = 60           # skip detections smaller than this
DET_CONFIDENCE: float = 0.70     # buffalo_sc scores 0.78-0.85 on clear faces
QUALITY_PASS: float = 0.15       # lowered from 0.20 — FFHQ 128px thumbnails score lower
QUALITY_PASS_FFHQ: float = 0.10  # even more lenient for FFHQ (very small resolution)




# ---------------------------------------------------------------------------
# Lazy global models (loaded once per process)
# ---------------------------------------------------------------------------

_face_detector = None
_face_aligner  = None


# Simple insightface face dataclass to match existing usage
class _InsightFace:
    """Thin wrapper around insightface face object for compatibility."""
    def __init__(self, face):
        self.bbox = face.bbox.astype(np.float32)
        self.score = float(face.det_score)
        self.landmarks = face.kps.astype(np.float32) if face.kps is not None else np.zeros((5,2), np.float32)
        self.area = float((self.bbox[2]-self.bbox[0]) * (self.bbox[3]-self.bbox[1]))


def _get_detector():
    global _face_detector
    if _face_detector is None:
        import warnings
        warnings.filterwarnings('ignore')
        from insightface.app import FaceAnalysis
        app = FaceAnalysis(name='buffalo_sc', providers=['CPUExecutionProvider'])
        app.prepare(ctx_id=-1, det_size=(640, 640))
        _face_detector = app
        logger.info('InsightFace detector ready (CPUExecutionProvider)')
    return _face_detector


def _detect_dominant(app, frame_bgr: np.ndarray) -> 'optional[_InsightFace]':
    """Detect the largest-area face in a frame."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        faces = app.get(frame_bgr)
    if not faces:
        return None
    wrapped = [_InsightFace(f) for f in faces if float(f.det_score) >= DET_CONFIDENCE]
    if not wrapped:
        return None
    # Filter by minimum size
    wrapped = [w for w in wrapped if (w.bbox[2]-w.bbox[0]) >= MIN_FACE_PX and (w.bbox[3]-w.bbox[1]) >= MIN_FACE_PX]
    if not wrapped:
        return None
    return max(wrapped, key=lambda w: w.area)


def _get_aligner():
    global _face_aligner
    if _face_aligner is None:
        from preprocessing.align_faces import FaceAligner
        _face_aligner = FaceAligner(output_size=FACE_SIZE)
    return _face_aligner


# ---------------------------------------------------------------------------
# Quality estimation (lightweight)
# ---------------------------------------------------------------------------

def _quality_features(crop_bgr: np.ndarray, det_score: float, landmarks: np.ndarray) -> dict:
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h, w = gray.shape

    lap_var = float(cv2.Laplacian(gray, cv2.CV_32F).var())
    blur_score = float(min(lap_var / 1500.0, 1.0))

    brightness = float(gray.mean() / 255.0)
    contrast   = float(min(gray.std() / 128.0, 1.0))

    face_area_ratio = 1.0

    pose_yaw_deg   = 0.0
    pose_pitch_deg = 0.0
    if landmarks is not None and landmarks.shape == (5, 2):
        le, re = landmarks[0], landmarks[1]
        dx = re[0] - le[0]
        dy = re[1] - le[1]
        expected_sep = 0.44 * w
        actual_sep   = float(np.sqrt(dx ** 2 + dy ** 2))
        yaw_ratio    = actual_sep / (expected_sep + 1e-6)
        pose_yaw_deg = float((1.0 - min(yaw_ratio, 1.0)) * 45.0)

    block_diff = 0.5
    try:
        gh = gray.shape[0] - gray.shape[0] % 8
        gw = gray.shape[1] - gray.shape[1] % 8
        gy = gray[:gh, :gw]
        blocks = gy.reshape(-1, 8, 8)
        block_diff = float(min(blocks.mean(axis=(1, 2)).std() / (gray.std() + 1e-6), 1.0))
    except Exception:
        pass
    compression_score = float(1.0 - block_diff * 0.4)

    composite = float(np.mean([blur_score, brightness, compression_score]))

    return dict(
        blur_score=round(blur_score, 4),
        brightness=round(brightness, 4),
        contrast=round(contrast, 4),
        face_area_ratio=round(face_area_ratio, 4),
        pose_yaw_deg=round(pose_yaw_deg, 2),
        pose_pitch_deg=round(pose_pitch_deg, 2),
        composite_quality=round(composite, 4),
        landmark_confidence=round(float(det_score), 4),
    )


# ---------------------------------------------------------------------------
# Frame sampler
# ---------------------------------------------------------------------------

def _sample_frames(video_path: str, sample_fps: float, max_frames: int):
    """
    Yield (frame_idx, timestamp_sec, frame_bgr) tuples at ~sample_fps.

    Tries cv2.VideoCapture first (fast, GPU-friendly).
    Falls back to PyAV if cv2 cannot open the file or yields 0 frames —
    this covers H.265/HEVC videos (codec_id=61) which OpenCV's bundled
    FFmpeg on Windows does not include.
    """
    # ── cv2 attempt ──────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(video_path)
    cv2_ok = cap.isOpened()
    if cv2_ok:
        native_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames > 0:
            step = max(1, int(native_fps / sample_fps))
            indices = list(range(0, total_frames, step))[:max_frames]
            yielded = 0
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret and frame is not None:
                    yield idx, float(idx) / native_fps, frame
                    yielded += 1
            cap.release()
            if yielded > 0:
                return   # success — no fallback needed
        cap.release()

    # ── PyAV fallback (handles HEVC, AV1, VP9, etc.) ─────────────────────────
    try:
        import av
        with av.open(video_path) as container:
            stream = container.streams.video[0]
            native_fps = float(stream.average_rate) if stream.average_rate else 25.0
            step = max(1, int(native_fps / sample_fps))
            frame_idx = 0
            yielded = 0
            for packet in container.demux(stream):
                for av_frame in packet.decode():
                    if frame_idx % step == 0:
                        # Convert PyAV frame to BGR numpy array
                        bgr = av_frame.to_ndarray(format="bgr24")
                        ts = float(av_frame.pts * stream.time_base) if av_frame.pts else frame_idx / native_fps
                        yield frame_idx, ts, bgr
                        yielded += 1
                        if yielded >= max_frames:
                            return
                    frame_idx += 1
    except Exception as e:
        logger.debug("PyAV fallback failed for %s: %s", video_path, e)



# ---------------------------------------------------------------------------
# Record ID
# ---------------------------------------------------------------------------

def _make_record_id(dataset: str, rel_path: str, frame_idx: int) -> str:
    raw = f"{dataset}|{rel_path}|{frame_idx}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Single item processors
# ---------------------------------------------------------------------------

def process_video(video_path, dataset, label, manipulation, crops_root, sample_fps, max_frames):
    detector = _get_detector()
    aligner  = _get_aligner()
    project_root = ROOT
    crops_root_path = Path(crops_root)

    try:
        rel_video = str(Path(video_path).relative_to(project_root))
    except ValueError:
        rel_video = str(video_path)

    records = []
    for frame_idx, ts, frame_bgr in _sample_frames(video_path, sample_fps, max_frames):
        try:
            face = _detect_dominant(detector, frame_bgr)
        except Exception:
            continue
        if face is None:
            continue
        try:
            align_result = aligner.align(frame_bgr, face.landmarks, face.bbox)
            crop = align_result[0] if isinstance(align_result, tuple) else align_result
        except Exception:
            continue
        if crop is None or crop.shape[:2] != (FACE_SIZE, FACE_SIZE):
            continue

        qf = _quality_features(crop, face.score, face.landmarks)
        if qf["composite_quality"] < QUALITY_PASS:
            continue

        record_id = _make_record_id(dataset, rel_video, frame_idx)
        ds_dir = crops_root_path / dataset
        ds_dir.mkdir(parents=True, exist_ok=True)
        crop_path_abs = ds_dir / f"{record_id}.jpg"
        cv2.imwrite(str(crop_path_abs), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
        try:
            crop_path_rel = str(crop_path_abs.relative_to(project_root))
        except ValueError:
            crop_path_rel = str(crop_path_abs)

        records.append(dict(
            record_id=record_id,
            dataset=dataset,
            split="",
            source_path=rel_video,
            frame_idx=frame_idx,
            timestamp_sec=round(ts, 3),
            label=label,
            manipulation=manipulation,
            crop_path=crop_path_rel,
            **qf,
        ))
    return records


def process_image(image_path, dataset, label, manipulation, crops_root):
    detector = _get_detector()
    aligner  = _get_aligner()
    project_root = ROOT
    crops_root_path = Path(crops_root)

    img = cv2.imread(str(image_path))
    if img is None:
        return []

    try:
        rel_img = str(Path(image_path).relative_to(project_root))
    except ValueError:
        rel_img = str(image_path)

    try:
        face = _detect_dominant(detector, img)
    except Exception:
        return []
    if face is None:
        return []

    try:
        align_result = aligner.align(img, face.landmarks, face.bbox)
        crop = align_result[0] if isinstance(align_result, tuple) else align_result
    except Exception:
        return []
    if crop is None or crop.shape[:2] != (FACE_SIZE, FACE_SIZE):
        return []

    qf = _quality_features(crop, face.score, face.landmarks)
    q_thresh = QUALITY_PASS_FFHQ if dataset == "ffhq" else QUALITY_PASS
    if qf["composite_quality"] < q_thresh:
        return []

    record_id = _make_record_id(dataset, rel_img, 0)
    ds_dir = crops_root_path / dataset
    ds_dir.mkdir(parents=True, exist_ok=True)
    crop_path_abs = ds_dir / f"{record_id}.jpg"
    cv2.imwrite(str(crop_path_abs), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
    try:
        crop_path_rel = str(crop_path_abs.relative_to(project_root))
    except ValueError:
        crop_path_rel = str(crop_path_abs)

    return [dict(
        record_id=record_id,
        dataset=dataset,
        split="",
        source_path=rel_img,
        frame_idx=0,
        timestamp_sec=0.0,
        label=label,
        manipulation=manipulation,
        crop_path=crop_path_rel,
        **qf,
    )]


# ---------------------------------------------------------------------------
# Dataset scanners
# ---------------------------------------------------------------------------

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def _find_videos(directory: Path):
    if not directory.exists():
        return []
    return [p for p in directory.rglob("*") if p.suffix.lower() in VIDEO_EXTS and p.is_file()]


def _find_all_media(directory: Path):
    if not directory.exists():
        return []
    all_exts = VIDEO_EXTS | IMAGE_EXTS
    return [p for p in directory.rglob("*") if p.suffix.lower() in all_exts and p.is_file()]


def _scan_faceforensics(root: Path, max_videos: Optional[int]) -> list:
    """
    Scan FF++ with actual flat layout:
      root/original/          — 1000 real mp4s
      root/Deepfakes/         — fake mp4s
      root/Face2Face/
      root/FaceSwap/
      root/NeuralTextures/
      root/FaceShifter/
      root/DeepFakeDetection/
    """
    items = []

    # Real: "original" folder (flat layout confirmed)
    for real_dir_name in ["original", "original_sequences", "YouTube-real"]:
        rd = root / real_dir_name
        if rd.exists():
            for v in _find_videos(rd):
                items.append((str(v), 0, "original"))

    # Fake: method subfolders directly under root
    for method in ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures",
                   "FaceShifter", "DeepFakeDetection"]:
        md = root / method
        if not md.exists():
            # Also try under manipulated_sequences/
            md = root / "manipulated_sequences" / method
        if md.exists():
            for v in _find_videos(md):
                items.append((str(v), 1, method))

    if max_videos:
        reals = [i for i in items if i[1] == 0]
        fakes = [i for i in items if i[1] == 1]
        n = max_videos // 2
        items = reals[:n] + fakes[:n]

    logger.info("FF++: %d items (%d real, %d fake)", len(items),
                sum(1 for i in items if i[1] == 0), sum(1 for i in items if i[1] == 1))
    return items

    if max_videos:
        reals = [i for i in items if i[1] == 0]
        fakes = [i for i in items if i[1] == 1]
        n = max_videos // 2
        items = reals[:n] + fakes[:n]

    logger.info("FF++: %d items (%d real, %d fake)", len(items),
                sum(1 for i in items if i[1] == 0), sum(1 for i in items if i[1] == 1))
    return items


def _scan_celeb_df(root: Path, max_videos: Optional[int]) -> list:
    items = []
    for rd_name in ["Celeb-real", "YouTube-real"]:
        rd = root / rd_name
        if rd.exists():
            for v in _find_videos(rd):
                items.append((str(v), 0, "original"))

    fd = root / "Celeb-synthesis"
    if fd.exists():
        for v in _find_videos(fd):
            items.append((str(v), 1, "Celeb-DF-synthesis"))

    if max_videos:
        reals = [i for i in items if i[1] == 0]
        fakes = [i for i in items if i[1] == 1]
        n = max_videos // 2
        items = reals[:n] + fakes[:n]

    logger.info("Celeb-DF v2: %d items (%d real, %d fake)", len(items),
                sum(1 for i in items if i[1] == 0), sum(1 for i in items if i[1] == 1))
    return items


def _scan_wild_deepfake(root: Path, max_videos: Optional[int]) -> list:
    items = []

    # Auto-extract any tar archives not yet extracted
    # NOTE: WildDeepfake files from HuggingFace are named .tar.gz but are
    # actually plain .tar (no gzip compression). Try both modes.
    import tarfile
    for split_name in ["real_train", "real_test", "fake_train", "fake_test"]:
        split_dir = root / split_name
        if not split_dir.exists():
            continue
        archives = list(split_dir.glob("*.tar.gz")) + list(split_dir.glob("*.tar"))
        if not archives:
            continue
        logger.info("WildDeepfake: extracting %d archives in %s ...", len(archives), split_name)
        for arc in archives:
            seq_name = arc.name.replace(".tar.gz", "").replace(".tar", "")
            dest = split_dir / seq_name
            if dest.exists() and any(dest.rglob("*")):
                continue   # already extracted
            # Try gz → plain tar → zip fallback
            extracted = False
            for mode in ("r:gz", "r:", "r:bz2", "r:xz"):
                try:
                    with tarfile.open(arc, mode) as tf:
                        tf.extractall(path=split_dir)
                    extracted = True
                    break
                except Exception:
                    continue
            if not extracted:
                logger.warning("WildDeepfake: could not extract %s (tried all tar modes)", arc.name)

    # ── Collect pre-cropped PNG clip folders ──────────────────────────────────
    # Structure after extraction:
    #   fake_test/<seq_id>/<seq_id>/fake/<clip_id>/*.png
    #   real_test/<seq_id>/<seq_id>/real/<clip_id>/*.png
    # (Some archives may omit the double-seq_id nesting)

    def _collect_wdf_clips(split_dir: Path, label: int, tag: str):
        if not split_dir.exists():
            return
        # Walk every directory that contains *.png files directly
        for clip_dir in split_dir.rglob("*"):
            if not clip_dir.is_dir():
                continue
            pngs = list(clip_dir.glob("*.png"))
            if len(pngs) >= 3:   # at least 3 frames = a valid clip
                items.append((str(clip_dir), label, tag))

    for split_name in ["real_train", "real_test"]:
        _collect_wdf_clips(root / split_name, 0, "original")
    for split_name in ["fake_train", "fake_test"]:
        _collect_wdf_clips(root / split_name, 1, "WildDeepfake")

    # Deduplicate (parent/child can both match rglob)
    # Keep deepest directory only (most specific clip dir)
    seen_parents: set = set()
    deduped = []
    for path, label, tag in sorted(items, key=lambda x: -len(x[0])):
        p = Path(path)
        if not any(str(p) in str(sp) or str(sp) in str(p) for sp in seen_parents):
            deduped.append((path, label, tag))
            seen_parents.add(str(p))
    items = deduped

    if max_videos:
        reals = [i for i in items if i[1] == 0]
        fakes = [i for i in items if i[1] == 1]
        n = max_videos // 2
        items = reals[:n] + fakes[:n]

    logger.info("WildDeepfake: %d items (%d real, %d fake)", len(items),
                sum(1 for i in items if i[1] == 0), sum(1 for i in items if i[1] == 1))
    return items



def _scan_ffhq(root: Path, max_images: Optional[int]) -> list:
    items = []
    for sd in [root / "thumbnails128x128", root / "images1024x1024", root]:
        if sd.exists():
            found = list(sd.rglob("*.png")) + list(sd.rglob("*.jpg"))
            if found:
                items = [(str(p), 0, "original") for p in found]
                break
    if max_images:
        items = items[:max_images]
    logger.info("FFHQ: %d images", len(items))
    return items


# DF40 method → canonical manipulation family mapping (NeurIPS 2024 taxonomy)
# Covers all 30 confirmed methods from extraction:
#   DF40_train/: blendface, danet, ddim, DiT, e4s, facedancer, faceswap,
#                facevid2vid, fomm, fsgan, hyperreenact, lia, mcnet,
#                mobileswap, MRAA, one_shot_free, pirender, sadtalker,
#                SiT, StyleGAN2, StyleGAN3, StyleGANXL, tpsm, uniface,
#                VQGAN, wav2lip
#   root-level: inswap, RDDM, sd2.1, simswap
_DF40_METHOD_FAMILY = {
    # ── Face swap ────────────────────────────────────────────────────────────
    "DeepFakes":     "face_swap",
    "FaceSwap":      "face_swap",       # classic FF++ method
    "faceswap":      "face_swap",       # DF40 faceswap folder
    "inswap":        "face_swap",       # InsightFace inswapper
    "simswap":       "face_swap",
    "e4s":           "face_swap",       # Edit Everything in 3D Space
    "faceshifter":   "face_swap",
    "mobileswap":    "face_swap",
    "uniface":       "face_swap",
    "blendface":     "face_swap",
    "facedancer":    "face_swap",       # FaceDancer: identity-preserving
    "fsgan":         "face_swap",       # FSGAN: subject agnostic face swap
    # ── Face reenactment ─────────────────────────────────────────────────────
    "Face2Face":     "reenactment",
    "fomm":          "reenactment",     # First Order Motion Model
    "tpsm":          "reenactment",     # Thin-Plate Spline Motion
    "oneshot":       "reenactment",
    "one_shot_free": "reenactment",     # One-Shot Free-View reenactment
    "sadtalker":     "reenactment",     # audio-driven talking head
    "wav2lip":       "reenactment",     # lip-sync reenactment
    "pirender":      "reenactment",     # pixel-wise warping reenactment
    "liveportrait":  "reenactment",
    "facevid2vid":   "reenactment",     # face vid-to-vid
    "hyperreenact":  "reenactment",     # hypernetwork-driven reenactment
    "lia":           "reenactment",     # Latent Image Animator
    "mcnet":         "reenactment",     # Motion Control Network
    "MRAA":          "reenactment",     # Motion Representations Artic. Anim.
    "danet":         "reenactment",     # Dense Appearance Transfer Network
    # ── Diffusion / generative synthesis ─────────────────────────────────────
    "pixart":        "diffusion_synthesis",
    "sd2.1":         "diffusion_synthesis",   # Stable Diffusion 2.1
    "RDDM":          "diffusion_synthesis",   # Residual Denoising Diffusion
    "DiT":           "diffusion_synthesis",   # Diffusion Transformer
    "SiT":           "diffusion_synthesis",   # Scalable Interpolant Transformer
    "ddim":          "diffusion_synthesis",   # DDIM sampling
    "ddpm":          "diffusion_synthesis",
    "ldm":           "diffusion_synthesis",
    "repaint":       "diffusion_synthesis",
    # ── GAN synthesis ────────────────────────────────────────────────────────
    "StyleGAN2":     "gan_synthesis",
    "StyleGAN3":     "gan_synthesis",
    "StyleGANXL":    "gan_synthesis",
    "stylegan":      "gan_synthesis",   # generic stylegan prefix
    "VQGAN":         "gan_synthesis",   # Vector-Quantized GAN
    "pggan":         "gan_synthesis",
    "starganv2":     "gan_synthesis",
    "crgan":         "gan_synthesis",
    # ── Attribute edit / neural texture ──────────────────────────────────────
    "NeuralTextures": "attribute_edit",
    "talkinhead":    "attribute_edit",
}


def _df40_method_family(method_name: str) -> str:
    """Resolve a DF40 method folder name to its manipulation_family."""
    # Exact match first
    if method_name in _DF40_METHOD_FAMILY:
        return _DF40_METHOD_FAMILY[method_name]
    # Case-insensitive prefix match
    lname = method_name.lower()
    for key, fam in _DF40_METHOD_FAMILY.items():
        if lname.startswith(key.lower()):
            return fam
    return "unknown"  # fallback — still processed, just unclassified


def _df40_frames_root(method_dir: Path) -> Optional[Path]:
    """
    Locate the 'frames/' directory for a DF40 method.

    Confirmed layouts after extraction:
      DF40_train/<method>/<method>/frames/<video_id>/*.png   (inner zip kept method name)
      <method>/frames/<video_id>/*.png                       (root-level method zip)
    """
    # Layout A: method/method/frames/
    nested = method_dir / method_dir.name / "frames"
    if nested.exists():
        return nested
    # Layout B: method/frames/
    direct = method_dir / "frames"
    if direct.exists():
        return direct
    # Layout C: search one level deep for any 'frames' dir
    for child in method_dir.iterdir():
        if child.is_dir():
            candidate = child / "frames"
            if candidate.exists():
                return candidate
    return None


def _scan_df40(root: Path, max_videos: Optional[int]) -> list:
    """
    Scan DF40 dataset.

    Actual layout after extraction::

        DF40_train/
          <method>/
            <method>/frames/<video_id>/*.png   <- pre-aligned face crops
                    /landmarks/<video_id>/*.npy
        inswap/
          frames/<video_id>/*.png
          landmarks/<video_id>/*.npy
        RDDM/   sd2.1/   simswap/   (same layout as inswap/)

    DF40 is pre-cropped — no face detection needed.
    Returns list of (frames_dir_str, label_int, method_tag)
    where frames_dir_str points to the per-clip folder containing *.png files.
    """
    items = []
    fake_clips = 0
    real_clips = 0
    method_set: set = set()

    def _collect_clips(frames_root: Path, label: int, method_tag: str):
        nonlocal fake_clips, real_clips
        if not frames_root.exists():
            return
        for clip_dir in sorted(frames_root.iterdir()):
            if not clip_dir.is_dir():
                continue
            pngs = list(clip_dir.glob("*.png"))
            if not pngs:
                pngs = list(clip_dir.glob("*.jpg"))
            if not pngs:
                continue
            items.append((str(clip_dir), label, method_tag))
            if label == 0:
                real_clips += 1
            else:
                fake_clips += 1
                method_set.add(method_tag)

    # ── Real clips ────────────────────────────────────────────────────────────
    for real_name in ("real", "Real"):
        for candidate in [root / real_name, root / "DF40_train" / real_name]:
            frames_root = _df40_frames_root(candidate) if candidate.exists() else None
            if frames_root:
                _collect_clips(frames_root, 0, "original")

    # ── Fake clips — DF40_train/<method>/ ────────────────────────────────────
    train_dir = root / "DF40_train"
    if train_dir.exists():
        for method_dir in sorted(train_dir.iterdir()):
            if not method_dir.is_dir() or method_dir.name.lower() == "real":
                continue
            method_tag = method_dir.name
            frames_root = _df40_frames_root(method_dir)
            if frames_root:
                _collect_clips(frames_root, 1, method_tag)
            else:
                logger.debug("DF40: no frames/ found in DF40_train/%s", method_tag)

    # ── Fake clips — root-level method folders (inswap/, RDDM/, sd2.1/, ...) ─
    known_skip = {"real", "Real", "DF40_train"}
    for candidate in sorted(root.iterdir()):
        if not candidate.is_dir() or candidate.name in known_skip:
            continue
        method_tag = candidate.name.split("-")[0]   # strip any -NNN suffix
        frames_root = _df40_frames_root(candidate)
        if frames_root:
            _collect_clips(frames_root, 1, method_tag)

    if max_videos:
        reals = [i for i in items if i[1] == 0]
        fakes = [i for i in items if i[1] == 1]
        n = max_videos // 2
        items = reals[:n] + fakes[:n]

    logger.info(
        "DF40: %d clip folders (%d real, %d fake across %d methods)",
        len(items), real_clips, fake_clips, len(method_set),
    )
    return items


def _scan_df40_efs(root: Path, max_videos: Optional[int]) -> list:
    """
    Scan DF40 Entire Face Synthesis (EFS / fully synthetic) methods:
    DiT, SiT, StyleGAN2, StyleGAN3, StyleGANXL, VQGAN, ddim, pixart, sd2.1, RDDM, e4s.
    """
    items = []
    mpath = ROOT / "data" / "manifests" / "df40_efs.jsonl"
    if mpath.exists():
        with open(mpath, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    items.append((rec["source_path"], 1, rec.get("method", "fully_synthetic")))
                    if max_videos and len(items) >= max_videos:
                        break
        logger.info("DF40 EFS (from manifest): %d items across generative synthesis methods", len(items))
        return items

    train_dir = root / "DF40_train"
    efs_train_methods = ["DiT", "SiT", "StyleGAN2", "StyleGAN3", "StyleGANXL", "VQGAN", "ddim", "e4s"]
    if train_dir.exists():
        for m in efs_train_methods:
            mdir = train_dir / m
            if mdir.exists():
                for p in mdir.rglob("*.png"):
                    items.append((str(p), 1, m))
                    if max_videos and len(items) >= max_videos:
                        break
    for m in ["pixart", "sd2.1", "RDDM"]:
        mdir = root / m
        if mdir.exists():
            for p in mdir.rglob("*.png"):
                items.append((str(p), 1, m))
                if max_videos and len(items) >= max_videos:
                    break
    logger.info("DF40 EFS (disk scan): %d items across generative synthesis methods", len(items))
    return items


def process_df40_clip(clip_dir: str, dataset: str, label: int,
                      manipulation: str, crops_root: str,
                      sample_fps: float, max_frames: int) -> list:
    """
    Process one DF40 pre-cropped clip folder.

    DF40 frames are already aligned 256x256 face crops — no face detection
    needed. We just resize to 224x224, run quality gate, and write records.
    """
    aligner  = _get_aligner()
    project_root = ROOT
    crops_root_path = Path(crops_root)

    clip_path = Path(clip_dir)
    try:
        rel_clip = str(clip_path.relative_to(project_root))
    except ValueError:
        rel_clip = str(clip_path)

    # Collect PNGs sorted by numeric filename (frame index)
    pngs = sorted(clip_path.glob("*.png"), key=lambda p: int(p.stem) if p.stem.isdigit() else 0)
    if not pngs:
        pngs = sorted(clip_path.glob("*.jpg"), key=lambda p: int(p.stem) if p.stem.isdigit() else 0)
    if not pngs:
        return []

    # Uniform subsample to max_frames
    step = max(1, len(pngs) // max_frames)
    sampled = pngs[::step][:max_frames]

    records = []
    for frame_idx, png_path in enumerate(sampled):
        img = cv2.imread(str(png_path))
        if img is None:
            continue
        # Resize to 224x224 (DF40 frames are 256x256)
        crop = cv2.resize(img, (FACE_SIZE, FACE_SIZE), interpolation=cv2.INTER_AREA)

        # Lightweight quality gate (no landmarks — use dummy)
        dummy_lmk = np.zeros((5, 2), dtype=np.float32)
        qf = _quality_features(crop, det_score=0.95, landmarks=dummy_lmk)
        if qf["composite_quality"] < QUALITY_PASS:
            continue

        record_id = _make_record_id(dataset, rel_clip, frame_idx)
        ds_dir = crops_root_path / dataset
        ds_dir.mkdir(parents=True, exist_ok=True)
        crop_path_abs = ds_dir / f"{record_id}.jpg"
        cv2.imwrite(str(crop_path_abs), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
        try:
            crop_path_rel = str(crop_path_abs.relative_to(project_root))
        except ValueError:
            crop_path_rel = str(crop_path_abs)

        records.append(dict(
            record_id=record_id,
            dataset=dataset,
            split="",
            source_path=rel_clip,
            frame_idx=frame_idx,
            timestamp_sec=round(frame_idx / max(sample_fps, 1.0), 3),
            label=label,
            manipulation=manipulation,
            crop_path=crop_path_rel,
            **qf,
        ))
    return records



# ---------------------------------------------------------------------------
# Manifest writer with resume support
# ---------------------------------------------------------------------------

def _load_processed_sources(manifest_path: Path) -> set:
    """Return set of source_path already in manifest."""
    seen = set()
    if not manifest_path.exists():
        return seen
    with open(manifest_path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    seen.add(json.loads(line)["source_path"])
                except Exception:
                    pass
    return seen


def run_dataset(dataset_name, items, manifest_path, crops_root, resume,
                sample_fps, max_frames, is_image_dataset=False, is_df40=False):
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    processed_sources = _load_processed_sources(manifest_path) if resume else set()
    logger.info("[%s] %d items | %d sources already done", dataset_name, len(items), len(processed_sources))

    # Build relative path set for resume check
    def _rel(path):
        try:
            return str(Path(path).relative_to(ROOT))
        except ValueError:
            return str(path)

    pending = [(p, l, m) for p, l, m in items if _rel(p) not in processed_sources]
    logger.info("[%s] %d items to process", dataset_name, len(pending))

    # Load existing records
    all_records = []
    if resume and manifest_path.exists():
        with open(manifest_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        all_records.append(json.loads(line))
                    except Exception:
                        pass

    with open(manifest_path, "a", buffering=1) as mfile:
        pbar = tqdm(pending, desc=f"[{dataset_name}]", unit="item",
                    dynamic_ncols=True, colour="cyan")
        for path, label, manip in pbar:
            try:
                if is_df40:
                    # DF40: path is a per-clip folder of pre-cropped PNGs
                    records = process_df40_clip(
                        path, dataset_name, label, manip, str(crops_root),
                        sample_fps, max_frames,
                    )
                elif is_image_dataset:
                    records = process_image(path, dataset_name, label, manip, str(crops_root))
                else:
                    records = process_video(path, dataset_name, label, manip, str(crops_root),
                                            sample_fps, max_frames)
                for rec in records:
                    mfile.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    all_records.append(rec)
                if records:
                    pbar.set_postfix({"crops": len(records)})
            except KeyboardInterrupt:
                logger.warning("Interrupted — progress saved to %s", manifest_path)
                raise
            except Exception as e:
                logger.error("[%s] %s: %s", dataset_name, Path(path).name, e)

    logger.info("[%s] Done — %d records total", dataset_name, len(all_records))
    return all_records


# ---------------------------------------------------------------------------
# Split generation
# ---------------------------------------------------------------------------

def generate_splits(all_records, splits_dir, train_ratio=0.70, val_ratio=0.15):
    splits_dir.mkdir(parents=True, exist_ok=True)

    has_existing = bool(all_records) and all(rec.get("split") in ("train", "val", "test") for rec in all_records)
    if has_existing:
        logger.info("Preserving existing split assignments across %d records", len(all_records))
        train_ids = [r["record_id"] for r in all_records if r["split"] == "train"]
        val_ids   = [r["record_id"] for r in all_records if r["split"] == "val"]
        test_ids  = [r["record_id"] for r in all_records if r["split"] == "test"]
    else:
        video_to_records = defaultdict(list)
        for rec in all_records:
            video_to_records[rec["source_path"]].append(rec)

        videos = list(video_to_records.keys())
        random.Random(42).shuffle(videos)

        n_train = int(len(videos) * train_ratio)
        n_val   = int(len(videos) * val_ratio)

        train_v = set(videos[:n_train])
        val_v   = set(videos[n_train:n_train + n_val])
        test_v  = set(videos[n_train + n_val:])

        split_map = {**{v: "train" for v in train_v},
                     **{v: "val"   for v in val_v},
                     **{v: "test"  for v in test_v}}

        train_ids, val_ids, test_ids = [], [], []
        for rec in all_records:
            s = split_map.get(rec["source_path"], "train")
            rec["split"] = s
            (train_ids if s == "train" else val_ids if s == "val" else test_ids).append(rec["record_id"])

    for name, ids in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
        p = splits_dir / f"split_{name}.txt"
        p.write_text("\n".join(ids) + "\n")
        logger.info("Split %s: %d records", name, len(ids))

    stats = {
        "total": len(all_records),
        "train": len(train_ids), "val": len(val_ids), "test": len(test_ids),
        "real":  sum(1 for r in all_records if r["label"] == 0),
        "fake":  sum(1 for r in all_records if r["label"] == 1),
        "by_dataset":     dict(Counter(r["dataset"] for r in all_records)),
        "by_manipulation": dict(Counter(r["manipulation"] for r in all_records)),
    }
    (splits_dir / "split_stats.json").write_text(json.dumps(stats, indent=2))
    logger.info("Split stats: %s", json.dumps(stats, indent=2))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="FORGE-Lite preprocessing pipeline")
    ap.add_argument("--dataset", default="all",
                    choices=["all", "faceforensics", "celeb_df", "wild_deepfake", "ffhq", "df40", "df40_efs"])
    ap.add_argument("--max-videos", type=int, default=None,
                    help="Cap per dataset (for quick test runs)")
    ap.add_argument("--sample-fps", type=float, default=SAMPLE_FPS)
    ap.add_argument("--max-frames",  type=int,   default=MAX_FRAMES)
    ap.add_argument("--resume",    dest="resume", action="store_true", default=True)
    ap.add_argument("--no-resume", dest="resume", action="store_false")
    ap.add_argument("--skip-splits", action="store_true", default=False)
    args = ap.parse_args()

    from omegaconf import OmegaConf
    cfg = OmegaConf.load(ROOT / "configs" / "data.yaml")
    roots = cfg.dataset_roots
    proc  = cfg.processed

    manifests_dir = ROOT / proc.manifests_dir
    splits_dir    = ROOT / proc.splits_dir
    crops_root    = ROOT / "data" / "crops"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    crops_root.mkdir(parents=True, exist_ok=True)

    # Tuple: (dataset_root, manifest_path, is_image_dataset, is_df40)
    # is_df40=True means items are pre-cropped clip dirs processed via process_df40_clip()
    dataset_cfgs = {
        "faceforensics": (ROOT / roots.faceforensics_pp, manifests_dir / "faceforensics.jsonl", False, False),
        "celeb_df":      (ROOT / roots.celeb_df_v2,      manifests_dir / "celeb_df.jsonl",      False, False),
        "wild_deepfake": (ROOT / roots.wild_deepfake,    manifests_dir / "wild_deepfake.jsonl", False, True),
        "ffhq":          (ROOT / roots.ffhq,             manifests_dir / "ffhq.jsonl",          True,  False),
        "df40":          (ROOT / roots.df40,             manifests_dir / "df40.jsonl",          False, True),
        "df40_efs":      (ROOT / roots.df40,             manifests_dir / "df40_efs.jsonl",      False, False),
    }

    to_run = list(dataset_cfgs.keys()) if args.dataset == "all" else [args.dataset]

    logger.info("=" * 70)
    logger.info("FORGE-Lite Preprocessing | datasets=%s | max=%s | fps=%.1f",
                ",".join(to_run), args.max_videos or "all", args.sample_fps)
    logger.info("=" * 70)

    all_records = []

    scanner_map = {
        "faceforensics": _scan_faceforensics,
        "celeb_df":      _scan_celeb_df,
        "wild_deepfake": _scan_wild_deepfake,
        "ffhq":          _scan_ffhq,
        "df40":          _scan_df40,
        "df40_efs":      _scan_df40_efs,
    }

    for ds_name in to_run:
        ds_root, ds_manifest, is_img, is_df40 = dataset_cfgs[ds_name]
        if not ds_root.exists():
            logger.error("[%s] Root not found: %s", ds_name, ds_root)
            continue
        items = scanner_map[ds_name](ds_root, args.max_videos)
        if not items:
            logger.warning("[%s] No items found — skipping", ds_name)
            continue
        recs = run_dataset(ds_name, items, ds_manifest, crops_root, args.resume,
                           args.sample_fps, args.max_frames, is_img, is_df40)
        all_records.extend(recs)

    if all_records and not args.skip_splits:
        logger.info("Generating splits from %d total records...", len(all_records))
        generate_splits(all_records, splits_dir)

        # Rewrite manifests with split labels filled in
        if args.dataset != "all":
            _, mpath, _is_img, _is_df40 = dataset_cfgs[args.dataset]
            with open(mpath, "w", encoding="utf-8") as f:
                for rec in all_records:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        else:
            by_ds = defaultdict(list)
            for rec in all_records:
                by_ds[rec["dataset"]].append(rec)
            for ds_name, recs in by_ds.items():
                cfg_entry = dataset_cfgs.get(ds_name) or dataset_cfgs.get(ds_name.lower())
                if not cfg_entry:
                    continue
                _, mpath, _is_img, _is_df40 = cfg_entry
                with open(mpath, "w", encoding="utf-8") as f:
                    for rec in recs:
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        logger.info("Manifests updated with split labels.")

    logger.info("=" * 70)
    logger.info("PREPROCESSING COMPLETE. Manifests: %s", manifests_dir)
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
