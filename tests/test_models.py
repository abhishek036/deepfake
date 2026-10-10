"""
tests/test_models.py
Unit and integration tests for all FORGE-Lite neural network models and branches:
- models.srm_filters (SRMFilterBank)
- models.dino_branch (DINOSemanticBranch)
- models.forensic_branch (ForensicResidualBranch)
- models.temporal_branch (TemporalEncoder)
- models.rppg_branch (RPPGBranch)
- models.av_branch (AudioVisualBranch)
- models.fusion (QualityGate, ReliabilityAwareFusion)
- models.localization (LocalizationHead)
- models.calibration (CalibrationLayer)
- models.forge_lite (FORGELite end-to-end)
"""

try:
    import pytest
except ImportError:
    pytest = None
import torch
import torch.nn as nn

from models.av_branch import AudioVisualBranch
from models.calibration import CalibrationLayer
from models.dino_branch import DINOSemanticBranch
from models.forensic_branch import ForensicResidualBranch
from models.forge_lite import ForgeLite
from models.fusion import QualityGate, ReliabilityAwareFusion
from models.localization import LocalizationHead
from models.rppg_branch import RPPGBranch
from models.srm_filters import SRMFilterBank
from models.temporal_branch import TemporalEncoder


# ─────────────────────────────────────────────────────────────────────────────
# SRM Filter Bank Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_srm_filter_bank():
    srm = SRMFilterBank()
    x = torch.randn(2, 1, 64, 64)
    out = srm(x)
    assert out.shape == (2, 30, 64, 64)
    # Weights should not require gradients
    for param in srm.parameters():
        assert not param.requires_grad


# ─────────────────────────────────────────────────────────────────────────────
# Semantic Branch Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_dino_semantic_branch():
    branch = DINOSemanticBranch(adaptation_mode="layernorm", output_dim=384)
    x = torch.randn(2, 3, 224, 224)
    out = branch(x, return_patch_tokens=True)
    assert "cls" in out
    assert out["cls"].shape == (2, 384)
    assert "patch_tokens" in out
    assert out["patch_tokens"].shape == (2, 256, 384)


# ─────────────────────────────────────────────────────────────────────────────
# Forensic Residual Branch Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_forensic_residual_branch():
    branch = ForensicResidualBranch(output_dim=256, fft_output_dim=20)
    x = torch.randn(2, 3, 224, 224)
    out = branch(x)
    assert out["embedding"].shape == (2, 256)
    assert out["fft_features"].shape == (2, 20)
    assert out["feature_map"].ndim == 4
    assert out["feature_map"].shape[0] == 2


# ─────────────────────────────────────────────────────────────────────────────
# Temporal Branch Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_temporal_encoder():
    encoder = TemporalEncoder(hidden_dim=256, num_layers=1)
    B, T = 2, 4
    dino_feat = torch.randn(B, T, 384)
    forensic_feat = torch.randn(B, T, 256)
    fft_feat = torch.randn(B, T, 20)
    frame_diffs = torch.randn(B, T - 1, 3, 224, 224)
    lm_vel = torch.randn(B, T, 10)
    face_q = torch.randn(B, T, 10)
    track_cont = torch.ones(B)

    out = encoder(
        dino_features=dino_feat,
        forensic_features=forensic_feat,
        fft_features=fft_feat,
        frame_diffs=frame_diffs,
        landmark_velocity=lm_vel,
        face_quality=face_q,
        track_continuity=track_cont,
    )
    assert "temporal_embedding" in out
    assert out["temporal_embedding"].shape == (B, 256)
    assert "frame_scores" in out
    assert out["frame_scores"].shape == (B, T)


# ─────────────────────────────────────────────────────────────────────────────
# Audio-Visual Branch Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_audio_visual_branch():
    av = AudioVisualBranch(audio_embed_dim=256, video_embed_dim=256, window_frames=5)

    # Test with valid audio mel and lip crops
    mel = torch.randn(2, 1, 80, 64)
    lip_crops = torch.randn(2, 5, 3, 48, 96)
    out = av(audio_mel=mel, lip_crops=lip_crops)

    assert out["av_score"].shape == (2, 1)
    assert out["av_embedding"].shape == (2, 256)
    assert out["cosine_sim"].shape == (2, 1)
    assert out["av_available"].shape == (2, 1)
    assert bool(out["av_available"].all())

    # Test fallback when audio is missing
    out_no_audio = av(audio_mel=None, lip_crops=lip_crops)
    assert out_no_audio["av_available"].sum().item() == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Reliability-Aware Fusion & Quality Gate Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_quality_gate_and_fusion():
    gate = QualityGate(face_quality_dim=10, video_quality_dim=5, num_branches=5)
    fq = torch.randn(2, 10)
    vq = torch.randn(2, 5)
    avail = torch.tensor([[1.0, 1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 1.0, 0.0, 0.0]])

    weights = gate(fq, vq, avail)
    assert weights.shape == (2, 5)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(2), atol=1e-5)
    assert weights[0, 4].item() < 1e-4

    fusion = ReliabilityAwareFusion(hidden_dim=256)
    fusion_out = fusion(
        semantic_embedding=torch.randn(2, 384),
        forensic_embedding=torch.randn(2, 256),
        fft_features=torch.randn(2, 20),
        temporal_embedding=torch.randn(2, 256),
        temporal_scores=torch.randn(2, 5),
        rppg_features=torch.randn(2, 16),
        av_score=torch.randn(2, 1),
        face_quality=fq,
        video_quality=vq,
        rppg_available=torch.tensor([1.0, 0.0]),
        av_available=torch.tensor([0.0, 0.0]),
    )
    assert "fake_logit" in fusion_out
    assert "uncertainty" in fusion_out
    assert "branch_weights" in fusion_out


# ─────────────────────────────────────────────────────────────────────────────
# Localization Head Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_localization_head():
    head = LocalizationHead(target_size=56)
    patch_tokens = torch.randn(2, 256, 384)
    forensic_map = torch.randn(2, 64, 56, 56)
    pred_mask = head(patch_tokens, forensic_map)
    assert pred_mask.shape == (2, 1, 56, 56)
    assert (pred_mask >= 0.0).all() and (pred_mask <= 1.0).all()


# ─────────────────────────────────────────────────────────────────────────────
# End-to-End FORGE-Lite Model Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_forge_lite_end_to_end():
    model = ForgeLite(
        enable_semantic=True,
        enable_forensic=True,
        enable_temporal=True,
        enable_rppg=False,
        enable_av=False,
        enable_localization=True,
    )
    model.eval()

    B, T = 2, 4
    frames = torch.randn(B, T, 3, 224, 224)
    frame_diffs = torch.randn(B, T - 1, 3, 224, 224)
    face_q = torch.randn(B, 10)
    track_q = torch.randn(B, 5)
    track_cont = torch.ones(B)

    with torch.no_grad():
        out = model(
            frames=frames,
            frame_diffs=frame_diffs,
            face_quality=face_q,
            track_quality=track_q,
            track_continuity=track_cont,
            return_decisions=True,
        )

    assert "fake_score" in out
    assert "fake_logit" in out
    assert "uncertainty" in out
    assert "branch_weights" in out
    assert "pred_mask" in out
    assert out["fake_score"].shape == (B,)
    assert out["pred_mask"].shape == (B, 1, 56, 56)
