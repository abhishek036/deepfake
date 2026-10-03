"""
models/forensic_branch.py
Lightweight forensic residual CNN for FORGE-Lite.

Architecture:
- Fixed SRM + Haar preprocessing (35 input channels, no gradients)
- Depthwise-separable convolution blocks (1–6M trainable params)
- GroupNorm, GELU activations
- Global average pooling → 256-dim forensic embedding
- Auxiliary compact FFT feature vector (20-dim)

PRD Warning: Frequency features can cause generalization failure via spectral bias.
The forensic branch is retained only if ablations show cross-domain improvement.
Consistency training across codec variants is essential.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.srm_filters import ForensicPreprocessor

logger = logging.getLogger(__name__)

# Input: 35 channels from ForensicPreprocessor
# Output: 256-dim embedding + 20-dim FFT vector


class DepthwiseSeparableConv(nn.Module):
    """
    Depthwise-separable convolution block:
    depthwise conv → pointwise conv → GroupNorm → GELU
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        num_groups: int = 8,
    ) -> None:
        super().__init__()
        pad = kernel_size // 2
        # Depthwise
        self.dw = nn.Conv2d(
            in_channels, in_channels, kernel_size=kernel_size,
            stride=stride, padding=pad, groups=in_channels, bias=False
        )
        # Pointwise
        self.pw = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        # Norm + activation
        actual_groups = min(num_groups, out_channels)
        while out_channels % actual_groups != 0:
            actual_groups -= 1
        self.norm = nn.GroupNorm(actual_groups, out_channels)
        self.act = nn.GELU()

        # Residual connection
        self.residual = (
            nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False)
            if in_channels != out_channels or stride != 1
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual(x)
        out = self.dw(x)
        out = self.pw(out)
        out = self.norm(out)
        out = self.act(out)
        return out + residual


class CompactFFTFeatures(nn.Module):
    """
    Extracts compact spectral statistics from an image.
    Not a classifier — provides auxiliary forensic context.

    Output: 20-dimensional vector of spectral statistics.
    """

    def __init__(self, num_radial_bins: int = 16) -> None:
        super().__init__()
        self.num_radial_bins = num_radial_bins
        # Output: num_radial_bins + 4 scalar stats = 20
        self.output_dim = num_radial_bins + 4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) or (B, 1, H, W) normalized image
        Returns:
            (B, 20) FFT feature vector
        """
        if x.shape[1] > 1:
            # Convert to luminance
            lum = 0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]  # (B, H, W)
        else:
            lum = x[:, 0]

        # 2D FFT
        fft = torch.fft.rfft2(lum)
        magnitude = torch.abs(fft)  # (B, H, W//2+1)

        # Log magnitude for better dynamic range
        log_mag = torch.log1p(magnitude)

        # Flatten and compute statistics
        B = x.shape[0]
        flat = log_mag.view(B, -1)  # (B, H*(W//2+1))

        # Radial frequency bins (approximated as percentile-based bins)
        radial_features = self._radial_bins(log_mag)  # (B, num_radial_bins)

        # Scalar spectral statistics
        total_energy = flat.sum(dim=1, keepdim=True)
        h_half = log_mag.shape[1] // 2
        hf_energy = log_mag[:, h_half:].sum(dim=[1, 2]).unsqueeze(1)
        lf_energy = log_mag[:, :h_half].sum(dim=[1, 2]).unsqueeze(1)

        # Spectral flatness (ratio of geometric to arithmetic mean — approximated)
        epsilon = 1e-8
        geo_mean = torch.exp(torch.log(flat.clamp(min=epsilon)).mean(dim=1, keepdim=True))
        arith_mean = flat.mean(dim=1, keepdim=True).clamp(min=epsilon)
        spectral_flatness = geo_mean / arith_mean

        # Local spectral variance
        spectral_var = flat.var(dim=1, keepdim=True)

        scalars = torch.cat([
            hf_energy / (total_energy + epsilon),   # HF energy ratio
            lf_energy / (total_energy + epsilon),   # LF energy ratio
            spectral_flatness,
            spectral_var / (total_energy + epsilon),
        ], dim=1)  # (B, 4)

        return torch.cat([radial_features, scalars], dim=1)  # (B, 20)

    def _radial_bins(self, log_mag: torch.Tensor) -> torch.Tensor:
        """Compute energy in num_radial_bins concentric frequency rings."""
        B, H, W = log_mag.shape
        # Create frequency coordinate grid once (cached after first call)
        fy = torch.fft.fftfreq(H, device=log_mag.device).view(-1, 1).abs()
        fx = torch.fft.rfftfreq(H, device=log_mag.device).view(1, -1).abs()
        # Actual W dimension is H//2+1 for rfft2
        fw = log_mag.shape[2]
        fy = fy[:H, :]
        fx = fx[:, :fw]
        radius = (fy ** 2 + fx ** 2).sqrt()  # (H, fw)
        max_r = radius.max().item() + 1e-6

        bins = []
        for i in range(self.num_radial_bins):
            r_low = i / self.num_radial_bins * max_r
            r_high = (i + 1) / self.num_radial_bins * max_r
            mask = ((radius >= r_low) & (radius < r_high)).float()
            ring_energy = (log_mag * mask.unsqueeze(0)).sum(dim=[1, 2])
            count = mask.sum().clamp(min=1)
            bins.append(ring_energy / count)

        return torch.stack(bins, dim=1)  # (B, num_radial_bins)


class ForensicResidualBranch(nn.Module):
    """
    Lightweight forensic CNN branch for FORGE-Lite.

    Architecture overview:
    Input (B, 3, 224, 224) RGB
        → ForensicPreprocessor (fixed SRM+Haar): (B, 35, 224, 224)
        → DepthwiseSeparableConv stack: (B, 256, 7, 7)
        → Global average pooling: (B, 256)
        → [Forensic embedding: (B, 256)]
    + CompactFFTFeatures: (B, 20)

    Total trainable parameters: ~2–4M
    """

    def __init__(
        self,
        in_channels: int = 35,
        base_channels: int = 64,
        num_blocks: int = 4,
        output_dim: int = 256,
        fft_output_dim: int = 20,
        num_groups: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.fft_output_dim = fft_output_dim

        # Fixed forensic preprocessing (no gradients)
        self.preprocessor = ForensicPreprocessor()
        for param in self.preprocessor.parameters():
            param.requires_grad = False

        # Initial projection from 35 → base_channels
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(min(num_groups, base_channels), base_channels),
            nn.GELU(),
        )

        # Depthwise-separable blocks with progressive downsampling
        channels = [base_channels, base_channels * 2, base_channels * 4, base_channels * 4]
        strides   = [2, 2, 2, 1]

        self.blocks = nn.ModuleList()
        prev_ch = base_channels
        for i in range(num_blocks):
            out_ch = channels[i] if i < len(channels) else base_channels * 4
            stride = strides[i] if i < len(strides) else 1
            self.blocks.append(DepthwiseSeparableConv(
                prev_ch, out_ch, kernel_size=3, stride=stride, num_groups=num_groups
            ))
            prev_ch = out_ch

        final_channels = prev_ch
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=dropout)
        self.head = nn.Sequential(
            nn.Linear(final_channels, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
        )

        # Compact FFT features (auxiliary)
        self.fft_extractor = CompactFFTFeatures(num_radial_bins=fft_output_dim - 4)

        # Store intermediate feature maps for localization head
        self._intermediate_features: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            x: (B, 3, 224, 224) normalized RGB

        Returns:
            dict with:
              'embedding': (B, 256) forensic embedding
              'fft_features': (B, 20) compact FFT vector
              'feature_map': (B, 256, 14, 14) intermediate features for localization
        """
        # Fixed forensic preprocessing
        with torch.no_grad():
            forensic_input = self.preprocessor(x)  # (B, 35, 224, 224)

        # Trainable CNN
        h = self.stem(forensic_input)
        for block in self.blocks:
            h = block(h)

        # Save intermediate feature map for localization
        feature_map = h  # (B, C, H', W')

        # Global average pooling → embedding
        pooled = self.gap(h).flatten(1)  # (B, C)
        pooled = self.dropout(pooled)
        embedding = self.head(pooled)  # (B, 256)

        # Compact FFT features (from original RGB, not preprocessed)
        fft_feats = self.fft_extractor(x)  # (B, 20)

        return {
            "embedding": embedding,
            "fft_features": fft_feats,
            "feature_map": feature_map,
        }

    def count_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
