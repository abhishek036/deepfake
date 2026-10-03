"""
models/context_branch.py
Scene Context and Full-Frame Consistency Branch for FORGE-Lite.

Analyzes uncropped full frames for whole-image synthetic generation artifacts:
- Scene-level perspective inconsistencies
- Background rendering artifacts typical of diffusion video models (Sora, Gen-3, Kling)
- Lighting & shadow discrepancy between detected facial crop and background
- Global semantic coherence embedding
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class ContextBranch(nn.Module):
    """
    Full-frame scene context encoder for generative AI video detection.

    Processes downsampled full video frames (112x112) to detect global
    diffusion inconsistencies, background anomalies, and lighting mismatches.
    """

    def __init__(
        self,
        input_size: int = 112,
        output_dim: int = 384,
        base_channels: int = 48,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.output_dim = output_dim

        # Efficient multi-scale spatial context encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, base_channels, kernel_size=3, stride=2, padding=1),  # (B, 48, 56, 56)
            nn.BatchNorm2d(base_channels),
            nn.GELU(),

            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1),  # (B, 96, 28, 28)
            nn.BatchNorm2d(base_channels * 2),
            nn.GELU(),

            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, stride=2, padding=1),  # (B, 192, 14, 14)
            nn.BatchNorm2d(base_channels * 4),
            nn.GELU(),

            nn.Conv2d(base_channels * 4, base_channels * 8, kernel_size=3, stride=2, padding=1),  # (B, 384, 7, 7)
            nn.BatchNorm2d(base_channels * 8),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.proj = nn.Sequential(
            nn.Linear(base_channels * 8, output_dim),
            nn.LayerNorm(output_dim),
            nn.Dropout(dropout),
        )

        self.context_score_head = nn.Sequential(
            nn.Linear(output_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

        logger.info("ContextBranch initialized | input_size=%d output_dim=%d", input_size, output_dim)

    def forward(
        self,
        full_frames: torch.Tensor,  # (B, 3, H, W) or (B, T, 3, H, W)
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            full_frames: Full uncropped frame tensor.
        Returns:
            Dict containing:
            - 'context_embedding': (B, output_dim)
            - 'context_anomaly_score': (B, 1) probability of full-frame synthetic artifacts
        """
        if full_frames.ndim == 5:
            # (B, T, 3, H, W) -> take middle frame or pool over frames
            B, T, C, H, W = full_frames.shape
            mid_idx = T // 2
            x = full_frames[:, mid_idx]
        else:
            x = full_frames

        # Downsample to input_size if needed
        if x.shape[-2] != self.input_size or x.shape[-1] != self.input_size:
            x = F.interpolate(x, size=(self.input_size, self.input_size), mode="bilinear", align_corners=False)

        feat = self.encoder(x).flatten(1)
        emb = self.proj(feat)
        score = torch.sigmoid(self.context_score_head(emb))

        return {
            "context_embedding": emb,
            "context_anomaly_score": score,
        }
