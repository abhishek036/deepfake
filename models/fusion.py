"""
models/fusion.py
Reliability-Aware Fusion Head for FORGE-Lite.

Replaces XGBoost post-processing with a jointly trained fusion MLP.
Key innovation: quality-conditioned branch gating — learns to reduce
reliance on low-quality or unavailable branches automatically.

PRD Section 3.4:
  branch_weight = softmax(quality_encoder(face_quality, branch_quality, availability))

The model learns these weights from data, not from hand-crafted rules.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

class QualityGate(nn.Module):
    """
    Legacy FORGE-Lite v1.0 Quality Gate.
    """
    def __init__(
        self,
        face_quality_dim: int = 10,
        video_quality_dim: int = 5,
        num_branches: int = 5,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.num_branches = num_branches
        quality_input_dim = face_quality_dim + video_quality_dim + num_branches

        self.gate_net = nn.Sequential(
            nn.Linear(quality_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_branches),
        )

    def forward(
        self,
        face_quality: torch.Tensor,
        video_quality: torch.Tensor,
        availability_mask: torch.Tensor,
    ) -> torch.Tensor:
        gate_input = torch.cat([face_quality, video_quality, availability_mask], dim=-1)
        raw_logits = self.gate_net(gate_input)
        masked_logits = raw_logits.masked_fill(availability_mask == 0, -1e4)
        return F.softmax(masked_logits, dim=-1)


class EvidenceAwareModalityRouter(nn.Module):
    """
    Tier-2 Evidence-Aware Modality Router.
    """
    def __init__(
        self,
        face_quality_dim: int = 10,
        video_quality_dim: int = 5,
        num_branches: int = 5,
        hidden_dim: int = 128,
        ablation_mode: str = "conf_qual",
    ) -> None:
        super().__init__()
        self.num_branches = num_branches
        self.ablation_mode = ablation_mode
        
        # Determine input dimension based on mode
        if ablation_mode == "conf_only":
            quality_input_dim = (num_branches * 2) + num_branches
        else:
            # conf_qual
            quality_input_dim = face_quality_dim + video_quality_dim + (num_branches * 2) + num_branches

        self.router_net = nn.Sequential(
            nn.Linear(quality_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, num_branches),
        )

    def forward(
        self,
        face_quality: torch.Tensor,
        video_quality: torch.Tensor,
        branch_probs: torch.Tensor,       # (B, 5) fake probability per branch
        availability_mask: torch.Tensor,  # (B, 5) float mask
    ) -> torch.Tensor:
        branch_confidence_margin = 2.0 * torch.abs(branch_probs - 0.5)

        if self.ablation_mode == "conf_only":
            router_input = torch.cat([
                branch_probs,
                branch_confidence_margin,
                availability_mask
            ], dim=-1)
        else:
            router_input = torch.cat([
                face_quality,
                video_quality,
                branch_probs,
                branch_confidence_margin,
                availability_mask
            ], dim=-1)

        raw_logits = self.router_net(router_input)
        masked_logits = raw_logits.masked_fill(availability_mask == 0, -1e4)
        return F.softmax(masked_logits, dim=-1)


class ReliabilityAwareFusion(nn.Module):
    """
    FORGE-Lite Tier-2 Modality Router Fusion.
    """

    # Input dimensions
    SEMANTIC_DIM = 384
    FORENSIC_DIM = 256
    FFT_DIM = 20
    TEMPORAL_DIM = 256
    TEMPORAL_SCORES_DIM = 5
    RPPG_DIM = 16
    AV_DIM = 1
    FACE_QUALITY_DIM = 10
    VIDEO_QUALITY_DIM = 5

    NUM_BRANCHES = 5  # semantic, forensic, temporal, rppg, av

    def __init__(
        self,
        hidden_dim: int = 512,
        dropout: float = 0.3,
        mc_dropout_samples: int = 10,
        ablation_mode: str = "conf_qual",
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.mc_dropout_samples = mc_dropout_samples
        self.ablation_mode = ablation_mode

        if self.ablation_mode == "v1":
            # Legacy v1.0 Quality Gate
            self.quality_gate = QualityGate(
                face_quality_dim=self.FACE_QUALITY_DIM,
                video_quality_dim=self.VIDEO_QUALITY_DIM,
                num_branches=self.NUM_BRANCHES,
            )
        else:
            # Tier-2 Modality Router
            self.modality_router = EvidenceAwareModalityRouter(
                face_quality_dim=self.FACE_QUALITY_DIM,
                video_quality_dim=self.VIDEO_QUALITY_DIM,
                num_branches=self.NUM_BRANCHES,
                ablation_mode=self.ablation_mode,
            )

        # Per-branch projection to common dimension
        branch_proj_dim = 128
        self.semantic_proj = nn.Sequential(
            nn.Linear(self.SEMANTIC_DIM, branch_proj_dim),
            nn.LayerNorm(branch_proj_dim),
            nn.GELU(),
        )
        self.forensic_proj = nn.Sequential(
            nn.Linear(self.FORENSIC_DIM + self.FFT_DIM, branch_proj_dim),
            nn.LayerNorm(branch_proj_dim),
            nn.GELU(),
        )
        self.temporal_proj = nn.Sequential(
            nn.Linear(self.TEMPORAL_DIM + self.TEMPORAL_SCORES_DIM, branch_proj_dim),
            nn.LayerNorm(branch_proj_dim),
            nn.GELU(),
        )
        self.rppg_proj = nn.Sequential(
            nn.Linear(self.RPPG_DIM, branch_proj_dim),
            nn.LayerNorm(branch_proj_dim),
            nn.GELU(),
        )
        self.av_proj = nn.Sequential(
            nn.Linear(self.AV_DIM, branch_proj_dim),
            nn.LayerNorm(branch_proj_dim),
            nn.GELU(),
        )
        self.branch_projs = nn.ModuleList([
            self.semantic_proj,
            self.forensic_proj,
            self.temporal_proj,
            self.rppg_proj,
            self.av_proj,
        ])

        # Per-branch fake logit heads
        self.branch_logit_heads = nn.ModuleList([
            nn.Linear(branch_proj_dim, 1) for _ in range(self.NUM_BRANCHES)
        ])

        # Main fusion MLP
        fusion_input_dim = self.NUM_BRANCHES * branch_proj_dim + self.FACE_QUALITY_DIM + self.VIDEO_QUALITY_DIM
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Output heads
        self.fake_logit_head = nn.Linear(hidden_dim // 2, 1)
        self.uncertainty_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Softplus(),  # positive uncertainty
        )

        # MC Dropout for uncertainty estimation
        self.mc_dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        semantic_embedding: torch.Tensor,       # (B, 384)
        forensic_embedding: torch.Tensor,       # (B, 256)
        fft_features: torch.Tensor,             # (B, 20)
        temporal_embedding: torch.Tensor,       # (B, 256)
        temporal_scores: torch.Tensor,          # (B, 5) [mean, max, top10, disp, attn]
        rppg_features: torch.Tensor,            # (B, 16)
        av_score: torch.Tensor,                 # (B, 1)
        face_quality: torch.Tensor,             # (B, 10)
        video_quality: torch.Tensor,            # (B, 5)
        rppg_available: torch.Tensor,           # (B,) bool/float
        av_available: torch.Tensor,             # (B,) bool/float
        temporal_available: Optional[torch.Tensor] = None,  # (B,) bool/float (None -> ones)
        semantic_available: Optional[torch.Tensor] = None,  # (B,) bool/float (None -> ones, for counterfactual attribution)
        forensic_available: Optional[torch.Tensor] = None,  # (B,) bool/float (None -> ones, for counterfactual attribution)
    ) -> dict[str, torch.Tensor]:
        """Forward pass through Evidence-Aware Modality Router Fusion."""
        B = semantic_embedding.shape[0]

        # Build branch availability mask (B, 5)
        if semantic_available is not None:
            semantic_avail = semantic_available.float()
        else:
            semantic_avail = torch.ones(B, device=semantic_embedding.device)

        if forensic_available is not None:
            forensic_avail = forensic_available.float()
        else:
            forensic_avail = torch.ones(B, device=semantic_embedding.device)

        if temporal_available is not None:
            temporal_avail = temporal_available.float()
        else:
            temporal_avail = torch.ones(B, device=semantic_embedding.device)

        availability_mask = torch.stack([
            semantic_avail,
            forensic_avail,
            temporal_avail,
            rppg_available.float(),
            av_available.float(),
        ], dim=1)  # (B, 5)

        # 1. Project each branch to common dimension
        semantic_feat = self.semantic_proj(semantic_embedding) * semantic_avail.unsqueeze(-1)
        forensic_feat = (
            self.forensic_proj(torch.cat([forensic_embedding, fft_features], dim=-1))
            * forensic_avail.unsqueeze(-1)
        )
        temporal_feat = (
            self.temporal_proj(torch.cat([temporal_embedding, temporal_scores], dim=-1))
            * temporal_avail.unsqueeze(-1)
        )
        rppg_feat = self.rppg_proj(rppg_features) * rppg_available.float().unsqueeze(-1)
        av_feat = self.av_proj(av_score) * av_available.float().unsqueeze(-1)

        branch_features = torch.stack([
            semantic_feat, forensic_feat, temporal_feat, rppg_feat, av_feat
        ], dim=1)  # (B, 5, 128)

        # 2. Compute branch evidence logits & probabilities BEFORE routing
        branch_logits = torch.stack([
            self.branch_logit_heads[i](branch_features[:, i, :]).squeeze(-1)
            for i in range(self.NUM_BRANCHES)
        ], dim=1)  # (B, 5)
        
        # Unavailable branches produce neutral 0.5 probability (logit=0)
        branch_logits = branch_logits * availability_mask
        branch_fake_probs = torch.sigmoid(branch_logits)  # (B, 5)
        
        neutral = torch.full_like(branch_fake_probs, 0.5)
        masked_probs = torch.where(
            availability_mask.bool(), branch_fake_probs, neutral
        )
        
        # 3. Dynamic Modality Routing
        if self.ablation_mode == "v1":
            branch_weights = self.quality_gate(
                face_quality, video_quality, availability_mask
            )
        else:
            # Computes context-dependent weights utilizing internal confidence + external quality
            branch_weights = self.modality_router(
                face_quality=face_quality,
                video_quality=video_quality,
                branch_probs=masked_probs,
                availability_mask=availability_mask
            )  # (B, 5)

        # 4. Apply dynamic modality weights
        weighted_features = (branch_features * branch_weights.unsqueeze(-1)).view(B, -1)  # (B, 640)

        # Branch Disagreement
        branch_disagreement = masked_probs.std(dim=1)  # (B,) — calibrated disagreement

        # Evidence coverage (fraction of available branches)
        evidence_coverage = availability_mask.mean(dim=1)  # (B,)


        # Fusion MLP
        fusion_input = torch.cat([weighted_features, face_quality, video_quality], dim=-1)
        fusion_out = self.fusion_mlp(fusion_input)  # (B, hidden//2)

        # Output
        fake_logit = self.fake_logit_head(fusion_out).squeeze(-1)  # (B,)
        uncertainty = self.uncertainty_head(fusion_out).squeeze(-1)  # (B,)

        return {
            "fake_logit": fake_logit,
            "uncertainty": uncertainty,
            "branch_weights": branch_weights,        # (B, 5)
            "branch_disagreement": branch_disagreement,  # (B,)
            "evidence_coverage": evidence_coverage,      # (B,)
            "branch_logits": branch_logits,              # (B, 5)
            "branch_fake_probs": branch_fake_probs,      # (B, 5)
            "availability_mask": availability_mask,      # (B, 5)
            "fusion_features": fusion_out,           # (B, hidden//2) for localization
        }

    @torch.no_grad()
    def predict_with_mc_dropout(
        self,
        n_samples: int = 10,
        **kwargs
    ) -> dict[str, torch.Tensor]:
        """
        Monte Carlo Dropout inference for uncertainty estimation.
        Runs forward pass N times with dropout enabled, returns mean + std.
        """
        self.train()  # enable dropout
        logits = []
        for _ in range(n_samples):
            out = self.forward(**kwargs)
            logits.append(torch.sigmoid(out["fake_logit"]))
        self.eval()

        logits_stack = torch.stack(logits, dim=0)  # (N, B)
        mean_score = logits_stack.mean(dim=0)
        std_score = logits_stack.std(dim=0)

        # Get one non-dropout pass for other outputs
        final_out = self.forward(**kwargs)
        final_out["fake_score_mc"] = mean_score
        final_out["uncertainty_mc"] = std_score
        return final_out
