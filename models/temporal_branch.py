"""
models/temporal_branch.py
Temporal encoder for FORGE-Lite using a 2-layer GRU.

Takes a sequence of per-frame features (DINO + forensic + frame diff +
geometry features + quality) and produces:
- Window-level fake score
- Frame-level suspiciousness scores
- Temporal embedding (256-dim)
- Temporal quality score

Key update: replaced raw landmark_velocity zeros (LM_VEL_DIM=10) with
semantically rich geometry_features (GEOMETRY_DIM=16) from GeometryFeatureExtractor.
Deepfake generators often exhibit subtle geometric tells — rigid jaw motion, unnatural
eye aspect ratio transitions, landmark jitter — that raw pixel/embedding branches miss.

Input dim: 384 + 256 + 20 + 64 + 16 + 10 + 1 = 751
  (previously 745 with 10-dim zero velocity)

Video-level aggregation uses multiple statistics (mean, top-10%, max, dispersion,
attention-weighted) because a fake may appear in only part of a video.
Naive mean would suppress real manipulation windows.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class TemporalAttentionPool(nn.Module):
    """
    Attention-weighted pooling over temporal windows.
    Learns which windows are most suspicious (higher weight).
    """

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.attn = nn.Linear(input_dim, 1)

    def forward(self, window_embeddings: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            window_embeddings: (B, T, D)
        Returns:
            pooled: (B, D)
            attn_weights: (B, T)
        """
        scores = self.attn(window_embeddings).squeeze(-1)  # (B, T)
        weights = torch.softmax(scores, dim=-1)
        pooled = (window_embeddings * weights.unsqueeze(-1)).sum(dim=1)  # (B, D)
        return pooled, weights


class TemporalEncoder(nn.Module):
    """
    2-layer Bidirectional GRU temporal encoder.

    Input per frame:
    - DINO CLS feature (384)
    - Forensic embedding (256)
    - FFT features (20)
    - Aligned frame difference summary (3 → pooled to 64)
    - Landmark velocity (normalized, 10)
    - Face quality vector (10)
    - Track continuity (1)
    Total per frame: 384 + 256 + 20 + 64 + 10 + 10 + 1 = 745

    The frame_diff is spatially pooled before being fed to the GRU to avoid
    the GRU having to process high-dim spatial inputs.
    """

    # Feature dimensions
    DINO_DIM = 384
    FORENSIC_DIM = 256
    FFT_DIM = 20
    FRAME_DIFF_POOL_DIM = 64
    GEOMETRY_DIM = 16   # replaces LM_VEL_DIM=10 — semantically rich geometry features
    QUALITY_DIM = 10
    CONTINUITY_DIM = 1

    def __init__(
        self,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        input_dim = (
            self.DINO_DIM
            + self.FORENSIC_DIM
            + self.FFT_DIM
            + self.FRAME_DIFF_POOL_DIM
            + self.GEOMETRY_DIM
            + self.QUALITY_DIM
            + self.CONTINUITY_DIM
        )  # 751

        # Frame difference projector: (B, T, 3*H*W) → too large;
        # instead use a small CNN pooling module
        self.diff_projector = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=7, stride=4, padding=3, bias=False),
            nn.GELU(),
            nn.Conv2d(16, 32, kernel_size=5, stride=4, padding=2, bias=False),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, self.FRAME_DIFF_POOL_DIM),
        )

        # Input normalization
        self.input_norm = nn.LayerNorm(input_dim)

        # GRU encoder
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
            batch_first=True,
        )

        gru_out_dim = hidden_dim * self.num_directions

        # Per-frame suspiciousness head
        self.frame_score_head = nn.Linear(gru_out_dim, 1)

        # Output projection to fixed temporal_dim
        self.output_proj = nn.Sequential(
            nn.Linear(gru_out_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # Temporal attention pooling
        self.attn_pool = TemporalAttentionPool(gru_out_dim)

        # Temporal quality head (how reliable is the temporal signal)
        self.quality_head = nn.Sequential(
            nn.Linear(gru_out_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        dino_features: torch.Tensor,                    # (B, T, 384)
        forensic_features: torch.Tensor,                # (B, T, 256)
        fft_features: torch.Tensor,                     # (B, T, 20)
        frame_diffs: torch.Tensor,                      # (B, T-1, 3, 224, 224)
        geometry_features: Optional[torch.Tensor] = None, # (B, T, 16) — landmark geometry
        face_quality: torch.Tensor = None,              # (B, T, 10) or (B, 10)
        track_continuity: torch.Tensor = None,          # (B,) scalar
        # Backward-compat: ignored if geometry_features provided
        landmark_velocity: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """
        Process a temporal sequence and return temporal evidence features.
        """
        B, T, _ = dino_features.shape
        device = dino_features.device

        # Process frame differences (B, T-1, 3, H, W) → (B, T-1, 64)
        if frame_diffs.shape[1] > 0:
            B_fd, T_fd = frame_diffs.shape[:2]
            diffs_flat = frame_diffs.view(B_fd * T_fd, 3, 224, 224)
            diff_proj = self.diff_projector(diffs_flat).view(B_fd, T_fd, -1)  # (B, T-1, 64)
            # Pad to length T (prepend zeros for first frame)
            diff_pad = torch.zeros(B, 1, self.FRAME_DIFF_POOL_DIM, device=device)
            diff_proj = torch.cat([diff_pad, diff_proj], dim=1)  # (B, T, 64)
        else:
            diff_proj = torch.zeros(B, T, self.FRAME_DIFF_POOL_DIM, device=device)

        # Geometry features: use provided geometry, or fall back to zeros
        # geometry_features replaces the old landmark_velocity (which was always zeros).
        if geometry_features is not None:
            geo = geometry_features  # (B, T, 16)
            if geo.shape[-1] != self.GEOMETRY_DIM:
                # Pad or truncate if dim mismatch (e.g., old checkpoints)
                geo = geo[..., :self.GEOMETRY_DIM]
                pad_needed = self.GEOMETRY_DIM - geo.shape[-1]
                if pad_needed > 0:
                    geo = torch.cat([geo, torch.zeros(B, T, pad_needed, device=device)], dim=-1)
        else:
            geo = torch.zeros(B, T, self.GEOMETRY_DIM, device=device)

        # Expand face quality if not per-frame
        if face_quality.dim() == 2:
            face_quality = face_quality.unsqueeze(1).expand(B, T, -1)

        # Expand track continuity to per-frame scalar
        cont = track_continuity.view(B, 1, 1).expand(B, T, 1)

        # Concatenate all per-frame features
        x = torch.cat([
            dino_features,   # (B, T, 384)
            forensic_features,  # (B, T, 256)
            fft_features,    # (B, T, 20)
            diff_proj,       # (B, T, 64)
            geo,             # (B, T, 16) — geometry features (replaces LM_VEL zeros)
            face_quality,    # (B, T, 10)
            cont,            # (B, T, 1)
        ], dim=-1)  # (B, T, 751)

        x = self.input_norm(x)

        # GRU encoding
        gru_out, _ = self.gru(x)  # (B, T, hidden_dim * num_directions)

        # Per-frame suspiciousness scores
        frame_scores = self.frame_score_head(gru_out).squeeze(-1)  # (B, T)
        frame_scores_sig = torch.sigmoid(frame_scores)

        # Video-level aggregation (Section 3.3, Branch 3 of PRD)
        mean_score = frame_scores_sig.mean(dim=1)  # (B,)
        max_score = frame_scores_sig.max(dim=1).values  # (B,)

        # Top-10% window scores
        k = max(1, int(T * 0.1))
        top_k_scores = frame_scores_sig.topk(k, dim=1).values.mean(dim=1)  # (B,)

        # Score dispersion
        score_disp = frame_scores_sig.std(dim=1)  # (B,)

        # Attention-weighted pooling
        attn_pooled, attn_weights = self.attn_pool(gru_out)  # (B, hidden), (B, T)
        attn_score = torch.sigmoid(self.frame_score_head(attn_pooled).squeeze(-1))

        # Temporal quality
        temporal_quality = self.quality_head(gru_out.mean(dim=1))  # (B, 1)

        # Temporal embedding (from attention-pooled GRU output)
        temporal_embedding = self.output_proj(attn_pooled)  # (B, 256)

        return {
            "temporal_embedding": temporal_embedding,          # (B, 256)
            "frame_scores": frame_scores_sig,                  # (B, T)
            "mean_score": mean_score,                          # (B,)
            "max_score": max_score,                            # (B,)
            "top10_score": top_k_scores,                       # (B,)
            "score_dispersion": score_disp,                    # (B,)
            "attn_score": attn_score,                          # (B,)
            "attn_weights": attn_weights,                      # (B, T)
            "temporal_quality": temporal_quality.squeeze(-1),  # (B,)
        }

    def aggregate_video_scores(self, window_scores: list[dict]) -> dict[str, float]:
        """
        Aggregate scores across multiple temporal windows of a video.
        Used during inference when the video is processed window by window.
        """
        if not window_scores:
            return {"video_score": 0.5, "uncertainty": 1.0}

        all_means = [w["mean_score"] for w in window_scores]
        all_maxes = [w["max_score"] for w in window_scores]
        all_top10 = [w["top10_score"] for w in window_scores]

        video_score = float(
            0.4 * max(all_means)
            + 0.3 * max(all_maxes)
            + 0.3 * max(all_top10)
        )
        dispersion = float(torch.tensor(all_means).std())
        return {"video_score": video_score, "score_dispersion": dispersion}
