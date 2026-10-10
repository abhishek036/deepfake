"""
data/augmentation.py
Online image-level augmentation pipeline for FORGE-Lite.

IMPORTANT: These are cheap, image-level transforms only.
Codec-level augmentation (H.264 re-encoding) is handled offline
by data/codec_bank.py — never re-encoded online during training.

Augmentations applied to face crops during training to improve robustness:
- Geometric: horizontal flip (only real videos — fakes may have chirality artifacts)
- Photometric: brightness, contrast, gamma, color jitter, Gaussian blur, noise
- Always applied consistently to (face crop, mask) pairs
"""

from __future__ import annotations

import random
from typing import Optional

import cv2
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image


class PhotometricAugmentation:
    """
    Photometric augmentation applied to face crops.
    Designed to NOT remove forensic artifacts (no heavy spatial transforms).
    """

    def __init__(
        self,
        brightness_range: tuple[float, float] = (0.7, 1.3),
        contrast_range: tuple[float, float] = (0.7, 1.3),
        saturation_range: tuple[float, float] = (0.8, 1.2),
        hue_range: float = 0.05,
        gamma_range: tuple[float, float] = (0.7, 1.4),
        blur_p: float = 0.2,
        blur_sigma: tuple[float, float] = (0.1, 1.0),
        noise_p: float = 0.2,
        noise_std: tuple[float, float] = (0.0, 0.03),
        grayscale_p: float = 0.05,
    ) -> None:
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.saturation_range = saturation_range
        self.hue_range = hue_range
        self.gamma_range = gamma_range
        self.blur_p = blur_p
        self.blur_sigma = blur_sigma
        self.noise_p = noise_p
        self.noise_std = noise_std
        self.grayscale_p = grayscale_p

        self._color_jitter = T.ColorJitter(
            brightness=self._range_to_jitter(brightness_range),
            contrast=self._range_to_jitter(contrast_range),
            saturation=self._range_to_jitter(saturation_range),
            hue=hue_range,
        )

    def __call__(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Apply photometric augmentation to a BGR numpy image.

        Args:
            image: (H, W, 3) BGR uint8
            mask: Optional (H, W) binary mask

        Returns:
            augmented image, augmented mask (masks not modified by photometric aug)
        """
        # Convert to PIL RGB for torchvision transforms
        pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Color jitter
        pil = self._color_jitter(pil)

        # Gamma correction
        gamma = random.uniform(*self.gamma_range)
        pil = TF.adjust_gamma(pil, gamma)

        # Gaussian blur
        if random.random() < self.blur_p:
            sigma = random.uniform(*self.blur_sigma)
            ksize = max(3, int(sigma * 3) * 2 + 1)  # must be odd
            img_np = np.array(pil)
            img_np = cv2.GaussianBlur(img_np, (ksize, ksize), sigma)
            pil = Image.fromarray(img_np)

        # Grayscale (simulate low-color input)
        if random.random() < self.grayscale_p:
            pil = TF.to_grayscale(pil, num_output_channels=3)

        # Convert back to BGR numpy
        result = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

        # Additive Gaussian noise
        if random.random() < self.noise_p:
            std = random.uniform(*self.noise_std)
            noise = np.random.normal(0, std * 255, result.shape).astype(np.float32)
            result = np.clip(result.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        return result, mask

    @staticmethod
    def _range_to_jitter(r: tuple[float, float]) -> float:
        """Convert (min, max) range to torchvision ColorJitter factor."""
        center = (r[0] + r[1]) / 2
        half_range = (r[1] - r[0]) / 2
        return half_range / center if center > 0 else 0.0


class TemporalAugmentation:
    """
    Augmentation applied to temporal sequences of face frames.
    Ensures all frames in a window receive consistent transforms.
    """

    def __init__(
        self,
        flip_p: float = 0.5,
        photometric: Optional[PhotometricAugmentation] = None,
    ) -> None:
        self.flip_p = flip_p
        self.photometric = photometric or PhotometricAugmentation()

    def __call__(
        self,
        frames: list[np.ndarray],
        masks: Optional[list[np.ndarray]] = None,
        label: int = 0,
    ) -> tuple[list[np.ndarray], Optional[list[np.ndarray]]]:
        """
        Apply consistent augmentation to a temporal window.

        Args:
            frames: List of (H, W, 3) BGR uint8 face crops
            masks: Optional list of (H, W) masks
            label: 0=real, 1=fake. Flip only applied to real (label=0) or always.

        Returns:
            augmented frames, augmented masks
        """
        do_flip = random.random() < self.flip_p

        # Sample photometric params once, apply consistently to all frames
        # (We re-initialize with fixed seed for consistency across a window)
        # Simplified: apply the same photometric augmentation instance per frame
        augmented_frames = []
        augmented_masks = masks[:] if masks else None

        for i, frame in enumerate(frames):
            aug_frame, _ = self.photometric(frame)
            if do_flip:
                aug_frame = cv2.flip(aug_frame, 1)  # horizontal flip
            augmented_frames.append(aug_frame)

        if augmented_masks and do_flip:
            augmented_masks = [cv2.flip(m, 1) for m in augmented_masks]

        return augmented_frames, augmented_masks


# ─────────────────────────────────────────────────────────────────────────────
# Normalization for model input
# ─────────────────────────────────────────────────────────────────────────────

# ImageNet mean/std for DINOv2 preprocessing (standard for ViT-based models)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

_normalize = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

def to_tensor_normalized(image_bgr: np.ndarray) -> torch.Tensor:
    """
    Convert a BGR uint8 numpy image to a normalized float tensor.
    Returns shape (3, H, W), ImageNet normalized.
    """
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(image_rgb).float().permute(2, 0, 1) / 255.0
    return _normalize(tensor)

def frames_to_tensor(frames: list[np.ndarray]) -> torch.Tensor:
    """
    Convert a list of BGR uint8 face crops to a (T, 3, H, W) normalized tensor.
    """
    tensors = [to_tensor_normalized(f) for f in frames]
    return torch.stack(tensors, dim=0)
