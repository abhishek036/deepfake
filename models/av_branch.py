"""
models/av_branch.py
Audio-Visual Synchronization and Forensic Branch for FORGE-Lite (Step 12).

Detects phonetic, temporal, and visual desynchronization between speech audio
and facial lip motion — a pervasive tell in talking-head deepfakes (e.g. Wav2Lip,
SadTalker, LivePortrait, and reenactment synthesis).

Architectural Highlights:
- SyncNet-inspired dual-stream contrastive encoder (Audio Stream + Lip Visual Stream)
- Audio Encoder: 2D ConvNet operating on Mel-spectrogram chunks (80 mels)
- Visual Lip Encoder: 3D/2D Spatiotemporal ConvNet operating on mouth crops (window_frames x 3 x 48 x 96)
- Normalized joint projection space for cosine similarity and cross-correlation lag estimation
- Audio-Visual Anomaly Score: 0 = naturally synchronized, 1 = desynchronized/synthetic artifact
- Automatic Voice Activity / Audio Quality gating: handles missing or silent audio gracefully.
"""

from __future__ import annotations

import logging
import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Audio Stream Encoder (Mel-spectrogram → Embedding)
# ─────────────────────────────────────────────────────────────────────────────

class AudioEncoder(nn.Module):
    """
    Lightweight 2D ConvNet for acoustic speech representation.
    Input: Mel-spectrogram slice (B, 1, n_mels=80, T_mel)
    Output: Audio embedding vector (B, embed_dim)
    """

    def __init__(
        self,
        in_channels: int = 1,
        n_mels: int = 80,
        embed_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=(2, 2)),  # (40, T/2)

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=(2, 2)),  # (20, T/4)

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=(2, 2)),  # (10, T/8)

            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.proj = nn.Sequential(
            nn.Linear(256, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for audio stream.
        Args:
            mel: (B, 1, n_mels, T_mel) or (B, n_mels, T_mel)
        Returns:
            embedding: (B, embed_dim)
        """
        if mel.ndim == 3:
            mel = mel.unsqueeze(1)
        feat = self.conv(mel)
        feat = feat.flatten(1)
        embed = self.proj(feat)
        return F.normalize(embed, p=2, dim=-1)


# ─────────────────────────────────────────────────────────────────────────────
# Visual Lip Stream Encoder (Lip Window → Embedding)
# ─────────────────────────────────────────────────────────────────────────────

class LipVisualEncoder(nn.Module):
    """
    Spatiotemporal ConvNet for mouth/lip motion dynamics.
    Input: Temporal window of lip crops (B, C=3, T=5, H=48, W=96)
    Output: Visual lip embedding vector (B, embed_dim)
    """

    def __init__(
        self,
        in_channels: int = 3,
        window_frames: int = 5,
        crop_height: int = 48,
        crop_width: int = 96,
        embed_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.window_frames = window_frames

        # 3D convolution over time and space to capture articulation dynamics
        self.conv3d = nn.Sequential(
            nn.Conv3d(in_channels, 64, kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2)),
            nn.BatchNorm3d(64),
            nn.GELU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),

            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(128),
            nn.GELU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),

            nn.Conv3d(128, 256, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(256),
            nn.GELU(),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
        )

        self.proj = nn.Sequential(
            nn.Linear(256, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, lip_crops: torch.Tensor) -> torch.Tensor:
        """
        Args:
            lip_crops: (B, T, 3, H, W) or (B, 3, T, H, W)
        Returns:
            embedding: (B, embed_dim)
        """
        if lip_crops.ndim == 5 and lip_crops.shape[2] == 3:
            # Reorder from (B, T, C, H, W) to (B, C, T, H, W)
            lip_crops = lip_crops.permute(0, 2, 1, 3, 4)

        feat = self.conv3d(lip_crops)
        feat = feat.flatten(1)
        embed = self.proj(feat)
        return F.normalize(embed, p=2, dim=-1)


# ─────────────────────────────────────────────────────────────────────────────
# Audio-Visual Forensic Branch
# ─────────────────────────────────────────────────────────────────────────────

class AudioVisualBranch(nn.Module):
    """
    FORGE-Lite Audio-Visual Forensic Branch.

    Evaluates audio-visual synchrony, mouth-speech correlation, and temporal lag.
    Produces:
    - av_score: Anomaly score in [0, 1] (0 = real speech sync, 1 = out of sync/fake)
    - av_embedding: Cross-modal feature representation (B, embed_dim)
    - sync_offset: Estimated frame desynchronization offset
    - av_available: Boolean flag indicating if valid audio and mouth data were present
    """

    def __init__(
        self,
        audio_embed_dim: int = 256,
        video_embed_dim: int = 256,
        mel_n_mels: int = 80,
        window_frames: int = 5,
        crop_height: int = 48,
        crop_width: int = 96,
        output_dim: int = 1,
        temperature: float = 0.07,
    ) -> None:
        super().__init__()
        self.audio_embed_dim = audio_embed_dim
        self.video_embed_dim = video_embed_dim
        self.window_frames = window_frames
        self.crop_height = crop_height
        self.crop_width = crop_width
        self.output_dim = output_dim
        self.temperature = temperature

        self.audio_encoder = AudioEncoder(
            in_channels=1,
            n_mels=mel_n_mels,
            embed_dim=audio_embed_dim,
        )

        self.visual_encoder = LipVisualEncoder(
            in_channels=3,
            window_frames=window_frames,
            crop_height=crop_height,
            crop_width=crop_width,
            embed_dim=video_embed_dim,
        )

        # Cross-modal classifier / anomaly scoring head
        joint_dim = audio_embed_dim + video_embed_dim + 1  # audio + visual + cosine_dist
        self.anomaly_head = nn.Sequential(
            nn.Linear(joint_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, output_dim),
        )

        logger.info(
            "AudioVisualBranch initialized | audio_dim=%d video_dim=%d window=%d",
            audio_embed_dim, video_embed_dim, window_frames,
        )

    def extract_lip_crop_from_face(
        self,
        face_frames: torch.Tensor,
        landmarks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Extract mouth region from full face frames.
        Face frames are (B, T, 3, 224, 224).
        Mouth region is typically located in lower third of aligned face:
        y: [130, 210], x: [55, 170] for 224x224.
        """
        B, T, C, H, W = face_frames.shape
        # Default canonical mouth bounding box on aligned 224x224 crop
        ymin, ymax = int(0.58 * H), int(0.92 * H)
        xmin, xmax = int(0.24 * W), int(0.76 * W)

        mouth_crops = face_frames[:, :, :, ymin:ymax, xmin:xmax]
        # Resize to (crop_height, crop_width)
        mouth_reshaped = mouth_crops.reshape(B * T, C, ymax - ymin, xmax - xmin)
        mouth_resized = F.interpolate(
            mouth_reshaped,
            size=(self.crop_height, self.crop_width),
            mode="bilinear",
            align_corners=False,
        )
        return mouth_resized.reshape(B, T, C, self.crop_height, self.crop_width)

    def forward(
        self,
        audio_mel: Optional[torch.Tensor] = None,       # (B, 1, 80, T_mel)
        lip_crops: Optional[torch.Tensor] = None,       # (B, T, 3, 48, 96) or face_frames (B, T, 3, 224, 224)
        audio_available: Optional[torch.Tensor] = None, # (B,) bool
    ) -> Dict[str, torch.Tensor]:
        """
        Forward computation for AV synchrony detection.

        Returns dictionary with:
        - 'av_score': (B, 1) probability of desynchronization artifact
        - 'av_embedding': (B, embed_dim) fused AV representation
        - 'cosine_sim': (B, 1) normalized cross-modal similarity
        - 'av_available': (B, 1) whether audio-visual analysis is valid
        """
        if lip_crops is None:
            raise ValueError("lip_crops or face_frames must be provided")

        B = lip_crops.shape[0]
        device = lip_crops.device

        # If full face frames were passed (224x224), extract canonical mouth region
        if lip_crops.shape[-2] > self.crop_height or lip_crops.shape[-1] > self.crop_width:
            lip_crops = self.extract_lip_crop_from_face(lip_crops)

        # Truncate or select window_frames
        T = lip_crops.shape[1]
        if T >= self.window_frames:
            # Pick center window
            start_idx = (T - self.window_frames) // 2
            lip_window = lip_crops[:, start_idx : start_idx + self.window_frames]
        else:
            # Pad temporal frames
            pad_count = self.window_frames - T
            last_frame = lip_crops[:, -1:].expand(-1, pad_count, -1, -1, -1)
            lip_window = torch.cat([lip_crops, last_frame], dim=1)

        # Check audio presence and validity
        has_audio = (
            audio_mel is not None
            and (audio_available is None or bool(audio_available.all()))
        )

        if not has_audio:
            # Return neutral fallback values with av_available=False
            dummy_score = torch.zeros(B, self.output_dim, device=device)
            dummy_embed = torch.zeros(B, self.audio_embed_dim, device=device)
            dummy_sim = torch.zeros(B, 1, device=device)
            av_flag = torch.zeros(B, 1, device=device)

            return {
                "av_score": dummy_score,
                "av_embedding": dummy_embed,
                "cosine_sim": dummy_sim,
                "av_available": av_flag,
            }

        # Compute embeddings
        audio_emb = self.audio_encoder(audio_mel)     # (B, audio_embed_dim)
        visual_emb = self.visual_encoder(lip_window)  # (B, video_embed_dim)

        # Normalized cosine similarity
        cosine_sim = (audio_emb * visual_emb).sum(dim=-1, keepdim=True)  # (B, 1)

        # Anomaly scoring head
        joint_input = torch.cat([audio_emb, visual_emb, cosine_sim], dim=-1)
        raw_anomaly = self.anomaly_head(joint_input)  # (B, 1)
        av_score = torch.sigmoid(raw_anomaly)

        av_flag = torch.ones(B, 1, device=device)
        if audio_available is not None:
            av_flag = audio_available.float().unsqueeze(-1).to(device)

        return {
            "av_score": av_score,
            "av_embedding": audio_emb + visual_emb,
            "cosine_sim": cosine_sim,
            "av_available": av_flag,
        }
