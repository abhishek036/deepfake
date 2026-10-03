"""
models/srm_filters.py
Fixed SRM (Steganalysis Rich Model) filter bank and Haar wavelet
preprocessing for FORGE-Lite forensic branch.

Both are fixed (non-trainable) convolutional layers that extract
low-level forensic residuals invisible to semantic models.

References:
- Fridrich & Kodovsky, "Rich Models for Steganalysis of Digital Images" (2012)
- Chen et al., "Learning Rich Features for Image Manipulation Detection" (CVPR 2018)
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# SRM filter definitions (30 standard filters from Rich Model)
# Kernel size: 5×5
# ─────────────────────────────────────────────────────────────────────────────

def _build_srm_kernels() -> np.ndarray:
    """Build the 30 SRM high-pass filter kernels."""
    # We implement a representative subset of SRM filters.
    # Each is a 5×5 kernel designed to suppress the image signal and
    # amplify statistical deviations caused by manipulation.
    kernels = []

    # 1. Simple high-pass (Laplacian family)
    k = np.zeros((5, 5))
    k[2, 2] = 4
    k[1, 2] = k[3, 2] = k[2, 1] = k[2, 3] = -1
    kernels.append(k / 4)

    # 2. Second-order horizontal
    k = np.zeros((5, 5))
    k[2, 0] = -1; k[2, 1] = 2; k[2, 2] = -2; k[2, 3] = 2; k[2, 4] = -1
    kernels.append(k / 4)

    # 3. Second-order vertical
    k = np.zeros((5, 5))
    k[0, 2] = -1; k[1, 2] = 2; k[2, 2] = -2; k[3, 2] = 2; k[4, 2] = -1
    kernels.append(k / 4)

    # 4. Diagonal
    k = np.zeros((5, 5))
    k[0, 0] = -1; k[1, 1] = 2; k[2, 2] = -2; k[3, 3] = 2; k[4, 4] = -1
    kernels.append(k / 4)

    # 5. Anti-diagonal
    k = np.zeros((5, 5))
    k[0, 4] = -1; k[1, 3] = 2; k[2, 2] = -2; k[3, 1] = 2; k[4, 0] = -1
    kernels.append(k / 4)

    # Generate remaining 25 filters as rotations/variations of above
    # plus edge detection kernels, local mean subtraction kernels
    base_filters = kernels.copy()
    for rotation in [1, 2, 3]:
        for base in base_filters[:5]:
            rotated = np.rot90(base, rotation)
            kernels.append(rotated)
            if len(kernels) >= 30:
                break
        if len(kernels) >= 30:
            break

    # Fill remaining with Gaussian difference kernels
    while len(kernels) < 30:
        sigma1, sigma2 = np.random.uniform(0.5, 2.0, 2)
        k = _gaussian_kernel(5, sigma1) - _gaussian_kernel(5, sigma2)
        kernels.append(k)

    return np.stack(kernels[:30], axis=0)  # (30, 5, 5)


def _gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    """Create a 2D Gaussian kernel."""
    x = np.linspace(-(size // 2), size // 2, size)
    g = np.exp(-x**2 / (2 * sigma**2))
    kernel = np.outer(g, g)
    return kernel / kernel.sum()


class SRMFilterBank(nn.Module):
    """
    Fixed (non-trainable) SRM high-pass filter bank.
    Applied to the luminance channel of the input image.

    Input:  (B, 1, H, W) — luminance channel
    Output: (B, 30, H, W) — SRM residuals
    """

    def __init__(self) -> None:
        super().__init__()
        kernels = _build_srm_kernels()  # (30, 5, 5)
        # Shape: (out_channels=30, in_channels=1, kH=5, kW=5)
        weight = torch.from_numpy(kernels[:, None, :, :]).float()
        self.register_buffer("weight", weight)

    def forward(self, lum: torch.Tensor) -> torch.Tensor:
        """
        Args:
            lum: (B, 1, H, W) luminance channel in [0, 1]
        Returns:
            (B, 30, H, W) SRM residuals
        """
        return F.conv2d(lum, self.weight, padding=2)  # same spatial size


class HaarWaveletBank(nn.Module):
    """
    Fixed 1-level Haar discrete wavelet transform.
    Extracts LH (horizontal detail), HL (vertical detail), HH (diagonal detail).

    Input:  (B, 1, H, W) — luminance channel
    Output: (B, 3, H//2, W//2) — Haar subbands

    Note: Output is spatially half the input size. Upsampled to original size
    before concatenation in the forensic branch.
    """

    def __init__(self) -> None:
        super().__init__()
        # Haar wavelet filters
        # LL, LH, HL, HH — we return LH, HL, HH (detail coefficients)
        sqrt2 = np.sqrt(2)
        lp = np.array([[1, 1], [1, 1]], dtype=np.float32) / 2.0       # Low-pass
        hp = np.array([[1, -1], [1, -1]], dtype=np.float32) / 2.0     # High-pass
        lh = np.array([[1, 1], [-1, -1]], dtype=np.float32) / 2.0     # LH
        hh = np.array([[1, -1], [-1, 1]], dtype=np.float32) / 2.0     # HH

        # Stack as 3 filters: LH, HL, HH
        filters = np.stack([hp, lh, hh], axis=0)[:, None, :, :]  # (3, 1, 2, 2)
        weight = torch.from_numpy(filters).float()
        self.register_buffer("weight", weight)

    def forward(self, lum: torch.Tensor) -> torch.Tensor:
        """
        Args:
            lum: (B, 1, H, W)
        Returns:
            (B, 3, H//2, W//2)
        """
        # Stride=2 gives subband decimation
        out = F.conv2d(lum, self.weight, stride=2)
        return out


class ForensicPreprocessor(nn.Module):
    """
    Complete fixed forensic preprocessing pipeline.

    Takes an RGB image and outputs:
    - Luminance channel (1)
    - SRM residuals (30)
    - Haar subbands (3, upsampled to original size)
    - Compression noise estimate (1)
    Total: 35 channels

    All fixed — no trainable parameters.
    """

    def __init__(self) -> None:
        super().__init__()
        self.srm = SRMFilterBank()
        self.haar = HaarWaveletBank()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) normalized RGB tensor in [0, 1] approx range
        Returns:
            (B, 35, H, W) forensic feature map
        """
        # Convert RGB to luminance (BT.601)
        lum = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]

        # SRM residuals: (B, 30, H, W)
        srm_out = self.srm(lum)

        # Haar subbands: (B, 3, H//2, W//2) → upsample to (B, 3, H, W)
        haar_out = self.haar(lum)
        haar_up = F.interpolate(haar_out, size=x.shape[-2:], mode="bilinear", align_corners=False)

        # Compression noise: high-frequency residual beyond SRM
        # Simple: original lum minus low-pass smoothed version
        smoothed = F.avg_pool2d(lum, kernel_size=3, stride=1, padding=1)
        comp_noise = lum - smoothed  # (B, 1, H, W)

        return torch.cat([lum, srm_out, haar_up, comp_noise], dim=1)  # (B, 35, H, W)
