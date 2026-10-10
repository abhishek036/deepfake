"""
data/dataset.py
PyTorch Dataset for FORGE-Lite.

Loads:
- Aligned face crops (224×224 JPEG) from manifest
- Quality vectors (per-frame + track-level)
- Optional masks (56×56 downsampled)
- Temporal windows (T=16 frames)
- Codec-augmented versions (from codec bank)

Returns batches suitable for all FORGE-Lite training phases.
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from data.augmentation import (
    PhotometricAugmentation,
    TemporalAugmentation,
    frames_to_tensor,
    to_tensor_normalized,
)

logger = logging.getLogger(__name__)


class ForgeDataset(Dataset):
    """
    FORGE-Lite dataset.

    Each __getitem__ returns a dict with:
      - frames: (T, 3, 224, 224) float tensor — temporal window
      - frame_diffs: (T-1, 3, 224, 224) float — aligned frame differences
      - label: int scalar (0=real, 1=fake)
      - face_quality: (10,) float — face quality vector
      - track_quality: (5,) float — track quality vector
      - mask: (1, 56, 56) float or None
      - mask_available: bool
      - video_id: str
      - dataset: str
      - manipulation_family: str
      - generator_family: str
      - sufficient_for_analysis: bool
    """

    TARGET_MASK_SIZE = 56

    def __init__(
        self,
        manifest_path: str,
        split: str = "train",
        window_size: int = 16,
        window_stride: int = 8,
        augment: bool = False,
        codec_aug: bool = False,
        codec_bank_dir: Optional[str] = None,
        codec_bank_p: float = 0.5,      # probability of loading from codec bank
        max_samples: Optional[int] = None,
        min_track_length: int = 4,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.split = split
        self.window_size = window_size
        self.window_stride = window_stride
        self.augment = augment
        self.codec_aug = codec_aug
        self.codec_bank_dir = codec_bank_dir
        self.codec_bank_p = codec_bank_p
        self.min_track_length = min_track_length

        # Augmentation pipeline
        self.temporal_aug = TemporalAugmentation() if augment else None

        # Load manifest
        self.records = self._load_manifest(manifest_path, split, max_samples, min_track_length)
        logger.info("ForgeDataset [%s]: %d records", split, len(self.records))

    # ── Dataset interface ─────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        record = self.records[idx]
        try:
            return self._load_sample(record)
        except Exception as e:
            logger.warning("Failed to load sample %s: %s", record["video_id"], e)
            # Return a zero sample as fallback
            return self._zero_sample(record)

    # ── Internal loading ──────────────────────────────────────────────────────

    def _load_sample(self, record: dict) -> dict:
        label = record.get("label_int", record.get("label", 0))

        # Check if record has direct frame crops (from preprocess_all.py)
        if "frame_crops" in record and record["frame_crops"]:
            frame_crops = record["frame_crops"]
            total = len(frame_crops)
            if total <= self.window_size:
                window_paths = list(frame_crops)
            elif self.split == "train":
                start = random.randint(0, total - self.window_size)
                window_paths = frame_crops[start : start + self.window_size]
            else:
                start = (total - self.window_size) // 2
                window_paths = frame_crops[start : start + self.window_size]

            root_dir = Path(__file__).resolve().parent.parent
            frames_bgr = []
            for p in window_paths:
                fp = Path(p) if Path(p).is_absolute() else root_dir / p
                im = cv2.imread(str(fp))
                if im is not None:
                    if im.shape[:2] != (224, 224):
                        im = cv2.resize(im, (224, 224))
                    frames_bgr.append(im)

            if not frames_bgr:
                return self._zero_sample(record)

            frames_bgr = self._pad_or_truncate(frames_bgr, self.window_size)
            if self.augment and self.temporal_aug:
                frames_bgr, _ = self.temporal_aug(frames_bgr, label=label)

            frames_tensor = frames_to_tensor(frames_bgr)
            frame_diffs = self._compute_frame_diffs(frames_tensor)
            face_quality = self._load_face_quality(record)
            track_quality = self._load_track_quality(record)
            audio_mel, has_audio = self._load_audio_mel(record)
            geo_features = self._load_geometry_features(record)

            return {
                "frames": frames_tensor,
                "frame_diffs": frame_diffs,
                "label": torch.tensor(label, dtype=torch.float32),
                "face_quality": face_quality,
                "track_quality": track_quality,
                "mask": None,
                "mask_available": torch.tensor(False),
                "audio_mel": audio_mel,
                "audio_available": has_audio,
                "geometry_features": geo_features,
                "video_id": record.get("video_id", "unknown"),
                "dataset": record.get("dataset", "unknown"),
                "manipulation_family": record.get("manipulation_family", "unknown"),
                "generator_family": record.get("generator_family", "unknown"),
                "sufficient_for_analysis": True,
                "sufficient_for_rppg": False,
                "track_continuity": torch.tensor(1.0, dtype=torch.float32),
            }

        if "faces_dir" in record and record["faces_dir"]:
            faces_dir = Path(record["faces_dir"])

            # Select codec variant if available and training
            if self.augment and self.codec_bank_dir and random.random() < self.codec_bank_p:
                faces_dir = self._codec_bank_path(record) or faces_dir

            # Load all available face crop paths, sorted by frame index
            face_paths = sorted(faces_dir.glob("*.jpg"), key=lambda p: int(p.stem))
            if len(face_paths) < 2:
                raise ValueError(f"Not enough face crops in {faces_dir}")

            # Sample a temporal window
            window_paths = self._sample_window(face_paths)
            frames_bgr = []
            for p in window_paths:
                f = cv2.imread(str(p))
                if f is not None:
                    if f.shape[:2] != (224, 224):
                        f = cv2.resize(f, (224, 224))
                    frames_bgr.append(f)

            if len(frames_bgr) < 2:
                raise ValueError(f"Could not read face crops from {faces_dir}")
        elif "source_path" in record and record["source_path"]:
            root_dir = Path(__file__).resolve().parent.parent
            src_path = Path(record["source_path"]) if Path(record["source_path"]).is_absolute() else root_dir / record["source_path"]
            frames_bgr = []
            if src_path.exists():
                if src_path.suffix.lower() in [".png", ".jpg", ".jpeg", ".webp"]:
                    img = cv2.imread(str(src_path))
                    if img is not None:
                        frames_bgr.append(cv2.resize(img, (224, 224)))
                else:
                    cap = cv2.VideoCapture(str(src_path))
                    total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    if total_f > self.window_size and self.split == "train":
                        start_f = random.randint(0, max(0, total_f - self.window_size))
                        cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
                    while len(frames_bgr) < self.window_size:
                        ret, f = cap.read()
                        if not ret or f is None:
                            break
                        frames_bgr.append(cv2.resize(f, (224, 224)))
                    cap.release()
            if not frames_bgr:
                return self._zero_sample(record)
            window_paths = []
        else:
            return self._zero_sample(record)

        # Pad or truncate to window_size
        frames_bgr = self._pad_or_truncate(frames_bgr, self.window_size)

        # Load mask if available
        mask_tensor = None
        if record.get("mask_available") and record.get("mask_path"):
            mask_tensor = self._load_mask(record, window_paths)

        # Augmentation
        if self.augment and self.temporal_aug:
            frames_bgr, _ = self.temporal_aug(frames_bgr, label=label)

        frames_aug_tensor = None
        if self.codec_aug:
            frames_aug_bgr = self._apply_codec_aug(frames_bgr)
            frames_aug_tensor = frames_to_tensor(frames_aug_bgr)

        # Convert to tensors
        frames_tensor = frames_to_tensor(frames_bgr)  # (T, 3, 224, 224)
        frame_diffs = self._compute_frame_diffs(frames_tensor)  # (T-1, 3, 224, 224)

        # Quality vectors
        face_quality = self._load_face_quality(record)
        track_quality = self._load_track_quality(record)
        audio_mel, has_audio = self._load_audio_mel(record)
        geo_features = self._load_geometry_features(record)

        out_dict = {
            "frames": frames_tensor,
            "frame_diffs": frame_diffs,
            "label": torch.tensor(label, dtype=torch.float32),
            "face_quality": face_quality,
            "track_quality": track_quality,
            "mask": mask_tensor,
            "mask_available": torch.tensor(record.get("mask_available", False)),
            "audio_mel": audio_mel,
            "audio_available": has_audio,
            "geometry_features": geo_features,
            "video_id": record["video_id"],
            "dataset": record["dataset"],
            "manipulation_family": record.get("manipulation_family", "unknown"),
            "generator_family": record.get("generator_family", "unknown"),
            "sufficient_for_analysis": record.get("sufficient_for_analysis", True),
            "sufficient_for_rppg": record.get("sufficient_for_rppg", False),
            "track_continuity": torch.tensor(record.get("track_continuity", 0.0), dtype=torch.float32),
        }
        
        if frames_aug_tensor is not None:
            out_dict["frames_aug"] = frames_aug_tensor
            
        return out_dict

    def _sample_window(self, face_paths: list[Path]) -> list[Path]:
        """Sample a temporal window of window_size frames."""
        total = len(face_paths)
        if total <= self.window_size:
            return face_paths

        if self.split == "train":
            # Random start for training
            max_start = total - self.window_size
            start = random.randint(0, max_start)
        else:
            # Center window for val/test
            start = (total - self.window_size) // 2

        return face_paths[start: start + self.window_size]

    def _pad_or_truncate(self, frames: list[np.ndarray], target: int) -> list[np.ndarray]:
        """Pad by repeating last frame or truncate to target length."""
        if len(frames) >= target:
            return frames[:target]
        # Pad with last frame
        while len(frames) < target:
            frames.append(frames[-1].copy())
        return frames

    def _apply_codec_aug(self, frames_bgr: list[np.ndarray]) -> list[np.ndarray]:
        """Apply V3 Codec-Invariant augmentation: JPEG (Q30-90), Resize, or Blur."""
        import io
        from PIL import Image, ImageFilter
        
        # 1 = JPEG, 2 = Resize, 3 = Blur
        aug_type = random.choice([1, 2, 3])
        aug_frames = []
        
        if aug_type == 1:
            q = random.randint(30, 90)
            for f in frames_bgr:
                img = Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=q)
                buf.seek(0)
                img_aug = Image.open(buf).convert("RGB")
                aug_frames.append(cv2.cvtColor(np.array(img_aug), cv2.COLOR_RGB2BGR))
        elif aug_type == 2:
            scale = random.uniform(0.3, 0.7)
            for f in frames_bgr:
                h, w = f.shape[:2]
                img = cv2.resize(f, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
                img = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)
                aug_frames.append(img)
        elif aug_type == 3:
            radius = random.uniform(1.0, 3.0)
            for f in frames_bgr:
                img = Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
                img_aug = img.filter(ImageFilter.GaussianBlur(radius=radius))
                aug_frames.append(cv2.cvtColor(np.array(img_aug), cv2.COLOR_RGB2BGR))
                
        return aug_frames

    @staticmethod
    def _compute_frame_diffs(frames: torch.Tensor) -> torch.Tensor:
        """Compute per-channel frame differences. Shape: (T-1, 3, H, W)."""
        return frames[1:] - frames[:-1]

    def _load_mask(self, record: dict, window_paths: list[Path]) -> Optional[torch.Tensor]:
        """Load and resize mask for the sampled window (use first frame mask)."""
        try:
            mask_dir = Path(record["mask_path"])
            if not mask_dir.exists():
                return None
            # Match mask to first frame of window
            first_stem = window_paths[0].stem
            mask_path = mask_dir / f"{first_stem}.png"
            if not mask_path.exists():
                return None
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                return None
            mask_resized = cv2.resize(mask, (self.TARGET_MASK_SIZE, self.TARGET_MASK_SIZE),
                                      interpolation=cv2.INTER_NEAREST)
            return torch.from_numpy(mask_resized / 255.0).float().unsqueeze(0)
        except Exception:
            return None

    def _codec_bank_path(self, record: dict) -> Optional[Path]:
        """Return a random codec bank variant path for a record."""
        if not self.codec_bank_dir:
            return None
        base = Path(self.codec_bank_dir) / record["dataset"] / record["video_id"]
        variants = list(base.iterdir()) if base.exists() else []
        if not variants:
            return None
        return random.choice(variants)

    def _load_face_quality(self, record: dict) -> torch.Tensor:
        """Load 10-dim face quality vector from record or defaults."""
        if "frame_records" in record and record["frame_records"]:
            frs = record["frame_records"]
            mean_blur = float(np.mean([f.get("blur_score", 0.5) for f in frs]))
            mean_bright = float(np.mean([f.get("brightness", 0.5) for f in frs]))
            mean_contrast = float(np.mean([f.get("contrast", 0.5) for f in frs]))
            mean_area = float(np.mean([f.get("face_area_ratio", 0.1) for f in frs]))
            mean_yaw = float(np.mean([abs(f.get("pose_yaw_deg", 0.0)) for f in frs])) / 90.0
            mean_pitch = float(np.mean([abs(f.get("pose_pitch_deg", 0.0)) for f in frs])) / 90.0
            mean_comp = float(np.mean([f.get("composite_quality", 0.5) for f in frs]))
            mean_lmk = float(np.mean([f.get("landmark_confidence", 0.8) for f in frs]))
            return torch.tensor([
                min(mean_blur, 1.0),
                min(mean_bright, 1.0),
                min(mean_contrast, 1.0),
                min(mean_area, 1.0),
                min(mean_yaw, 1.0),
                min(mean_pitch, 1.0),
                0.0,
                min(mean_comp, 1.0),
                min(mean_lmk, 1.0),
                1.0,
            ], dtype=torch.float32)

        tq = record.get("track_quality", {})
        if not tq:
            return torch.zeros(10, dtype=torch.float32)
        return torch.tensor([
            tq.get("mean_blur", 0.5),
            tq.get("mean_brightness", 0.5),
            0.5,  # contrast (not in track quality)
            tq.get("mean_face_area_ratio", 0.1),
            min(abs(tq.get("max_pose_yaw", 0.0)) / 90.0, 1.0),
            0.0,  # pitch
            tq.get("mean_occlusion", 0.0),
            tq.get("mean_compression", 0.8),
            1.0 - tq.get("landmark_jitter", 0.0),
            tq.get("continuity", 1.0),
        ], dtype=torch.float32)

    def _load_track_quality(self, record: dict) -> torch.Tensor:
        """Load 5-dim track quality vector."""
        if "frame_records" in record and record["frame_records"]:
            frs = record["frame_records"]
            mean_blur = float(np.mean([f.get("blur_score", 0.5) for f in frs]))
            mean_area = float(np.mean([f.get("face_area_ratio", 0.1) for f in frs]))
            mean_comp = float(np.mean([f.get("composite_quality", 0.5) for f in frs]))
            return torch.tensor([
                1.0,
                min(mean_blur, 1.0),
                min(mean_area, 1.0),
                0.0,
                min(mean_comp, 1.0),
            ], dtype=torch.float32)

        tq = record.get("track_quality", {})
        if not tq:
            return torch.zeros(5, dtype=torch.float32)
        return torch.tensor([
            tq.get("continuity", 1.0),
            tq.get("mean_blur", 0.5),
            tq.get("mean_face_area_ratio", 0.1),
            1.0 - tq.get("mean_occlusion", 0.0),
            tq.get("mean_compression", 0.8),
        ], dtype=torch.float32)

    def _load_audio_mel(self, record: dict) -> tuple[Optional[torch.Tensor], bool]:
        """Load pre-computed Mel spectrogram or extract from audio file."""
        if "mel_path" in record and record["mel_path"]:
            mp = Path(record["mel_path"])
            if not mp.is_absolute():
                mp = Path(__file__).resolve().parent.parent / mp
            if mp.exists():
                try:
                    if mp.suffix == ".pt":
                        mel = torch.load(mp, map_location="cpu", weights_only=True)
                    elif mp.suffix == ".npy":
                        mel = torch.from_numpy(np.load(mp)).float()
                    else:
                        mel = None
                    if mel is not None:
                        if mel.ndim == 2:
                            mel = mel.unsqueeze(0)
                        return mel, True
                except Exception:
                    pass

        if "audio_path" in record and record["audio_path"]:
            ap = Path(record["audio_path"])
            if not ap.is_absolute():
                ap = Path(__file__).resolve().parent.parent / ap
            if ap.exists():
                try:
                    import torchaudio
                    waveform, sr = torchaudio.load(str(ap))
                    if sr != 16000:
                        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
                        waveform = resampler(waveform)
                    if waveform.shape[0] > 1:
                        waveform = waveform.mean(dim=0, keepdim=True)
                    mel_transform = torchaudio.transforms.MelSpectrogram(
                        sample_rate=16000, n_fft=400, hop_length=160, n_mels=80
                    )
                    mel = mel_transform(waveform)
                    log_mel = torch.log(torch.clamp(mel, min=1e-5))
                    return log_mel, True
                except Exception:
                    pass

        return None, False

    def _load_geometry_features(self, record: dict) -> Optional[torch.Tensor]:
        """Load pre-computed landmark geometry features tensor (T, 16)."""
        if "geometry_path" in record and record["geometry_path"]:
            gp = Path(record["geometry_path"])
            if not gp.is_absolute():
                gp = Path(__file__).resolve().parent.parent / gp
            if gp.exists():
                try:
                    if gp.suffix == ".pt":
                        return torch.load(gp, map_location="cpu")
                    elif gp.suffix == ".npy":
                        return torch.from_numpy(np.load(gp)).float()
                except Exception:
                    pass
        return None

    def _zero_sample(self, record: dict) -> dict:
        """Return a zeroed-out sample for failed loads."""
        T = self.window_size
        out_dict = {
            "frames": torch.zeros(T, 3, 224, 224),
            "frame_diffs": torch.zeros(T - 1, 3, 224, 224),
            "label": torch.tensor(record.get("label_int", 0), dtype=torch.float32),
            "face_quality": torch.zeros(10),
            "track_quality": torch.zeros(5),
            "mask": None,
            "mask_available": torch.tensor(False),
            "audio_mel": None,
            "audio_available": False,
            "geometry_features": None,
            "video_id": record.get("video_id", "unknown"),
            "dataset": record.get("dataset", "unknown"),
            "manipulation_family": "unknown",
            "generator_family": "unknown",
            "sufficient_for_analysis": False,
            "sufficient_for_rppg": False,
            "track_continuity": torch.tensor(0.0),
        }
        if self.codec_aug:
            out_dict["frames_aug"] = torch.zeros(T, 3, 224, 224)
        return out_dict

    # ── Manifest loading ──────────────────────────────────────────────────────

    @staticmethod
    def _load_manifest(
        manifest_path: str, split: str, max_samples: Optional[int], min_track: int
    ) -> list[dict]:
        """Load and filter manifest records for the requested split."""
        from collections import defaultdict
        path = Path(manifest_path)
        raw_records = []

        if path.suffix in (".jsonl", ".json"):
            with open(path, encoding="utf-8") as f:
                content = f.read().strip()
                if content.startswith("["):
                    raw_records = json.loads(content)
                else:
                    for line in content.splitlines():
                        line = line.strip()
                        if line:
                            raw_records.append(json.loads(line))
        else:
            raise ValueError(f"Unsupported manifest format: {path.suffix}")

        # Filter by split if requested
        if split:
            records_in_split = [r for r in raw_records if r.get("split") == split]
            if records_in_split:
                raw_records = records_in_split

        # Check format: does it have direct crop_path? (from preprocess_all.py)
        if raw_records and "crop_path" in raw_records[0]:
            by_source = defaultdict(list)
            for r in raw_records:
                src_key = r.get("source_path") or r.get("record_id")
                by_source[src_key].append(r)

            grouped_records = []
            for src_key, f_list in by_source.items():
                f_list.sort(key=lambda x: x.get("frame_idx", 0))
                first = f_list[0]
                grouped_records.append({
                    "video_id": src_key,
                    "dataset": first.get("dataset", "unknown"),
                    "label_int": int(first.get("label", 0)),
                    "manipulation_family": first.get("manipulation", "unknown"),
                    "generator_family": first.get("manipulation", "unknown"),
                    "frame_crops": [f.get("crop_path") or f.get("source_path") for f in f_list],
                    "frame_records": f_list,
                    "track_length": len(f_list),
                    "split": first.get("split", split),
                    "audio_path": first.get("audio_path"),
                    "mel_path": first.get("mel_path"),
                    "audio_available": first.get("audio_available", False),
                })
            records = grouped_records
        else:
            records = raw_records
            if min_track > 1:
                records = [r for r in records if ("track_length" not in r) or r.get("track_length", 0) >= min_track]

        if max_samples and len(records) > max_samples:
            import random
            rng = random.Random(42)
            records = rng.sample(records, max_samples)

        return records


def collate_forge(batch: list[dict]) -> dict:
    """
    Custom collate function that handles:
    - Optional tensors (mask may be None)
    - String fields (video_id, dataset, etc.)
    """
    keys = batch[0].keys()
    result = {}
    for key in keys:
        vals = [b[key] for b in batch]
        if key in ("video_id", "dataset", "manipulation_family", "generator_family"):
            result[key] = vals
        elif key in ("mask",):
            # Stack masks; use zeros for None entries
            mask_list = []
            for v in vals:
                if v is None:
                    mask_list.append(torch.zeros(1, 56, 56))
                else:
                    mask_list.append(v)
            result[key] = torch.stack(mask_list, dim=0)
        elif key in ("audio_mel",):
            has_any = any(v is not None for v in vals)
            if not has_any:
                result[key] = None
            else:
                valid_mels = [v for v in vals if v is not None]
                n_mels = valid_mels[0].shape[-2]
                max_t = max(v.shape[-1] for v in valid_mels)
                mel_list = []
                for v in vals:
                    if v is None:
                        mel_list.append(torch.zeros(1, n_mels, max_t))
                    else:
                        if v.shape[-1] < max_t:
                            pad = torch.zeros(1, n_mels, max_t - v.shape[-1])
                            v = torch.cat([v, pad], dim=-1)
                        mel_list.append(v)
                result[key] = torch.stack(mel_list, dim=0)
        elif key in ("geometry_features",):
            has_any = any(v is not None for v in vals)
            if not has_any:
                result[key] = None
            else:
                geo_list = []
                for v in vals:
                    if v is None:
                        geo_list.append(torch.zeros(8, 16))
                    else:
                        geo_list.append(v)
                result[key] = torch.stack(geo_list, dim=0)
        elif key in ("sufficient_for_analysis", "sufficient_for_rppg",
                     "audio_available", "mask_available"):
            result[key] = torch.tensor([bool(v) for v in vals])
        elif isinstance(vals[0], torch.Tensor):
            result[key] = torch.stack(vals, dim=0)
        else:
            result[key] = vals
    return result


# ---------------------------------------------------------------------------
# Class-balance helpers for training
# ---------------------------------------------------------------------------

def get_pos_weight(cfg=None) -> "torch.Tensor":
    """
    Return BCEWithLogitsLoss pos_weight tensor for the real (positive) class.

    With real:fake = 1:18.4 in the combined manifest, pos_weight ≈ 18.4
    causes the loss to weight each real sample 18.4× heavier than each fake,
    compensating for the severe imbalance introduced by the large DF40 split.

    Args:
        cfg: OmegaConf config (reads cfg.class_balance.pos_weight).
             Falls back to 18.4 if not provided or not found.
    """
    weight = 18.4
    if cfg is not None:
        try:
            weight = float(cfg.class_balance.pos_weight)
        except Exception:
            pass
    return torch.tensor([weight], dtype=torch.float32)


def make_dataloader(
    dataset: "ForgeDataset",
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 2,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    cfg=None,
    is_train: bool = True,
) -> "torch.utils.data.DataLoader":
    """
    Build a DataLoader with optional WeightedRandomSampler for class balance.

    When cfg.class_balance.balanced_sampling is True **and** is_train is True,
    uses WeightedRandomSampler to draw batches with ≈50 % real / 50 % fake,
    regardless of the raw ratio in the manifest (currently 1:18 real:fake).

    Args:
        dataset:            ForgeDataset instance (already split-filtered)
        batch_size:         samples per batch
        shuffle:            only used when NOT using WeightedRandomSampler
        num_workers:        DataLoader workers
        pin_memory:         pin CUDA-mapped memory
        persistent_workers: keep workers alive between epochs
        cfg:                OmegaConf train config (reads class_balance section)
        is_train:           apply balancing only during training
    """
    from torch.utils.data import DataLoader, WeightedRandomSampler

    use_balanced = False
    if is_train and cfg is not None:
        try:
            use_balanced = bool(cfg.class_balance.balanced_sampling)
        except Exception:
            pass

    sampler = None
    if use_balanced:
        labels = [r.get("label", r.get("label_int", 0)) for r in dataset.records]
        n_real = sum(1 for lb in labels if lb == 0)
        n_fake = sum(1 for lb in labels if lb == 1)
        n_total = len(labels)

        if n_real == 0 or n_fake == 0:
            logger.warning(
                "make_dataloader: one class has 0 samples — balanced sampling disabled"
            )
        else:
            # Weight each sample inversely proportional to its class frequency
            # → expected 50 % real / 50 % fake per batch
            w_real = n_total / (2.0 * n_real)
            w_fake = n_total / (2.0 * n_fake)
            sample_weights = torch.tensor(
                [w_real if lb == 0 else w_fake for lb in labels],
                dtype=torch.float32,
            )
            n_samples = n_total
            if cfg is not None:
                try:
                    spe = cfg.class_balance.get("samples_per_epoch", None)
                    if spe:
                        n_samples = min(n_total, int(spe)) if n_total > 0 else int(spe)
                except Exception:
                    pass
            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=n_samples,
                replacement=True,
            )
            logger.info(
                "make_dataloader: WeightedRandomSampler | "
                "w_real=%.2f w_fake=%.2f | %d real / %d fake | %d samples/epoch",
                w_real, w_fake, n_real, n_fake, n_samples,
            )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(shuffle and sampler is None),
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(persistent_workers and num_workers > 0),
        collate_fn=collate_forge,
        drop_last=is_train,
    )
    return loader
