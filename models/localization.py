"""
models/localization.py
56×56 manipulation localization head for FORGE-Lite.

Decodes DINOv2 patch tokens (256 patches, 16×16 grid) + forensic CNN feature maps
into a spatial manipulation probability mask.

PRD Critical Reporting Rules:
- Report pixel IoU only on samples with valid documented ground-truth masks
- Mark uncertain masks as weak supervision — never mix bbox and pixel masks
- Attention rollout / gradients are supporting visualizations only, NOT IoU metrics
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# DINOv2 ViT-S/14 patch grid: 224/14 = 16 → 16×16 = 256 patches
PATCH_GRID = 16  # patches per spatial dimension
PATCH_DIM = 384  # embedding dim per patch
TARGET_SIZE = 56  # output mask size


class ConvUpsampleBlock(nn.Module):
    """Bilinear upsample + depthwise conv + GroupNorm + GELU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        scale_factor: int = 2,
        num_groups: int = 8,
    ) -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=scale_factor, mode="bilinear", align_corners=False)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(min(num_groups, out_channels), out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.up(x))


class LocalizationHead(nn.Module):
    """
    Decodes DINOv2 patch tokens + forensic feature maps into a 56×56 mask.

    Pipeline:
    1. Reshape patch tokens (B, 256, 384) → (B, 384, 16, 16)
    2. Project to 128 channels
    3. Upsample 16→28, fuse forensic features
    4. Upsample 28→56 → final mask (sigmoid)

    Training loss: BCE + Dice (on samples with valid ground-truth masks only)
    """

    def __init__(
        self,
        patch_dim: int = PATCH_DIM,
        patch_grid: int = PATCH_GRID,
        forensic_channels: int = 64,
        decoder_channels: tuple[int, ...] = (128, 64, 32),
        target_size: int = TARGET_SIZE,
    ) -> None:
        super().__init__()
        self.patch_grid = patch_grid
        self.target_size = target_size

        # Patch token projection: 384 → decoder_channels[0]
        self.patch_proj = nn.Sequential(
            nn.Conv2d(patch_dim, decoder_channels[0], kernel_size=1, bias=False),
            nn.GroupNorm(min(8, decoder_channels[0]), decoder_channels[0]),
            nn.GELU(),
        )

        # Forensic feature projection: forensic_channels → decoder_channels[0]
        self.forensic_proj = nn.Sequential(
            nn.Conv2d(forensic_channels, decoder_channels[0], kernel_size=1, bias=False),
            nn.GroupNorm(min(8, decoder_channels[0]), decoder_channels[0]),
            nn.GELU(),
        )

        # Decoder blocks: 16×16 → 28×28 → 56×56
        self.decoder_blocks = nn.ModuleList()
        ch_in = decoder_channels[0] * 2  # after fusion with forensic
        for i, ch_out in enumerate(decoder_channels[1:]):
            scale = 2 if i < 1 else 2  # upsample at each block
            self.decoder_blocks.append(
                ConvUpsampleBlock(ch_in, ch_out, scale_factor=scale)
            )
            ch_in = ch_out

        # Additional upsample if needed to reach target_size from intermediate
        # 16 → 32 → 64 is too big for 56; we use adaptive final layer
        self.final_conv = nn.Sequential(
            nn.Conv2d(ch_in, 16, kernel_size=3, padding=1, bias=False),
            nn.GELU(),
            nn.Conv2d(16, 1, kernel_size=1),
        )

    def forward(
        self,
        patch_tokens: torch.Tensor,          # (B, 256, 384)
        forensic_feature_map: torch.Tensor,  # (B, C, H', W') where C=forensic_channels
    ) -> torch.Tensor:
        """
        Returns (B, 1, 56, 56) manipulation probability map (after sigmoid).
        """
        B = patch_tokens.shape[0]

        # 1. Reshape patch tokens to spatial grid
        patch_spatial = patch_tokens.transpose(1, 2).view(
            B, self.patch_dim_from_tokens(patch_tokens), self.patch_grid, self.patch_grid
        )  # (B, 384, 16, 16)

        # 2. Project patch tokens
        patch_feat = self.patch_proj(patch_spatial)  # (B, 128, 16, 16)

        # 3. Resize forensic features to same spatial size as patch grid
        forensic_resized = F.interpolate(
            forensic_feature_map[:, :64],  # use first 64 channels
            size=(self.patch_grid, self.patch_grid),
            mode="bilinear", align_corners=False,
        )
        forensic_feat = self.forensic_proj(forensic_resized)  # (B, 128, 16, 16)

        # 4. Fuse
        x = torch.cat([patch_feat, forensic_feat], dim=1)  # (B, 256, 16, 16)

        # 5. Decode
        for block in self.decoder_blocks:
            x = block(x)

        # 6. Final convolution + adaptive resize to target_size
        x = self.final_conv(x)  # (B, 1, H', W')
        x = F.interpolate(x, size=(self.target_size, self.target_size),
                          mode="bilinear", align_corners=False)
        return torch.sigmoid(x)  # (B, 1, 56, 56)

    @staticmethod
    def patch_dim_from_tokens(patch_tokens: torch.Tensor) -> int:
        return patch_tokens.shape[-1]


class LocalizationLoss(nn.Module):
    """
    BCE + Dice loss for manipulation localization.
    Applied ONLY to samples with valid ground-truth masks.
    """

    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
    ) -> None:
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(
        self,
        pred_mask: torch.Tensor,    # (B, 1, H, W) probabilities
        target_mask: torch.Tensor,  # (B, 1, H, W) binary ground truth
        mask_available: torch.Tensor,  # (B,) bool — which samples have valid masks
    ) -> torch.Tensor:
        """
        Compute loss only on samples with valid ground-truth masks.
        Returns scalar loss.
        """
        if not mask_available.any():
            return torch.tensor(0.0, device=pred_mask.device, requires_grad=True)

        # Select samples with valid masks
        valid_pred = pred_mask[mask_available]   # (N_valid, 1, H, W)
        valid_target = target_mask[mask_available]  # (N_valid, 1, H, W)

        # Resize target to match pred if needed
        if valid_pred.shape[-2:] != valid_target.shape[-2:]:
            valid_target = F.interpolate(
                valid_target.float(), size=valid_pred.shape[-2:],
                mode="nearest"
            )

        # BCE loss
        device_type = "cuda" if valid_pred.is_cuda else "cpu"
        with torch.amp.autocast(device_type=device_type, enabled=False):
            bce = F.binary_cross_entropy(
                valid_pred.float().clamp(1e-6, 1 - 1e-6),
                valid_target.float(),
            )

        # Dice loss
        smooth = 1.0
        pred_flat = valid_pred.view(valid_pred.shape[0], -1)
        target_flat = valid_target.float().view(valid_target.shape[0], -1)
        intersection = (pred_flat * target_flat).sum(dim=1)
        dice = 1.0 - (2 * intersection + smooth) / (
            pred_flat.sum(dim=1) + target_flat.sum(dim=1) + smooth
        )
        dice = dice.mean()

        return self.bce_weight * bce + self.dice_weight * dice
