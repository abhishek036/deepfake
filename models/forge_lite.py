"""
models/forge_lite.py
FORGE-Lite: Top-level model composing all branches.

The single forward pass:
1. Semantic branch (DINOv2) → CLS + patch tokens
2. Forensic branch (SRM CNN) → embedding + FFT + feature map
3. Temporal encoder (GRU) → temporal embedding + scores
4. rPPG branch (quality-gated) → rPPG features
5. Audio-visual branch (availability-gated) → sync score
6. Reliability-Aware Fusion Head → fake logit + uncertainty
7. Localization Head → 56×56 mask
8. Calibration + Abstention → final decision

Output schema matches PRD Section 4 exactly.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import torch
import torch.nn as nn

from models.dino_branch import DINOSemanticBranch
from models.forensic_branch import ForensicResidualBranch
from models.temporal_branch import TemporalEncoder
from models.geometry_branch import GeometryFeatureExtractor, GEOMETRY_DIM
from models.av_branch import AudioVisualBranch
from models.fusion import ReliabilityAwareFusion
from models.localization import LocalizationHead
from models.calibration import CalibrationLayer, Decision

logger = logging.getLogger(__name__)

MODEL_VERSION = "forge-lite-1.0"
PREPROCESSING_VERSION = "prep-1.0"


class FORGELite(nn.Module):
    """
    FORGE-Lite: Forensic, Robust, Generalizable Evidence Detection.

    Full system combining all branches into a single nn.Module.
    Designed for RTX 4060 8GB — branches processed sequentially to manage VRAM.

    Configuration passed as dicts matching configs/model.yaml structure.
    """

    def __init__(
        self,
        semantic_cfg: Optional[dict] = None,
        forensic_cfg: Optional[dict] = None,
        temporal_cfg: Optional[dict] = None,
        fusion_cfg: Optional[dict] = None,
        localization_cfg: Optional[dict] = None,
        calibration_cfg: Optional[dict] = None,
        geometry_cfg: Optional[dict] = None,
        av_cfg: Optional[dict] = None,
        # Branch enable flags
        enable_semantic: bool = True,
        enable_forensic: bool = True,
        enable_temporal: bool = True,
        enable_rppg: bool = False,      # Added after core model works (Step 12)
        enable_av: bool = True,         # Now enabled — quality-gated by audio availability
        enable_localization: bool = True,
        unweighted_fusion: bool = False,  # For ablation
        ablation_mode: str = "conf_qual", # conf_qual, conf_only, v1
    ) -> None:
        super().__init__()
        self.enable_semantic = enable_semantic
        self.enable_forensic = enable_forensic
        self.enable_temporal = enable_temporal
        self.enable_rppg = enable_rppg
        self.enable_av = enable_av
        self.enable_localization = enable_localization
        self.unweighted_fusion = unweighted_fusion
        self.ablation_mode = ablation_mode

        sc  = semantic_cfg      or {}
        fc  = forensic_cfg      or {}
        tc  = temporal_cfg      or {}
        fsc = fusion_cfg        or {}
        lc  = localization_cfg  or {}
        cc  = calibration_cfg   or {}
        gc  = geometry_cfg      or {}
        avc = av_cfg            or {}

        # ── Branch modules ────────────────────────────────────────────────────
        if enable_semantic:
            self.semantic_branch = DINOSemanticBranch(
                adaptation_mode=sc.get("adaptation_mode", "layernorm"),
                output_dim=sc.get("output_dim", 384),
                dropout=sc.get("dropout", 0.1),
                lora_rank=sc.get("lora_rank", 8),
                lora_alpha=sc.get("lora_alpha", 16),
            )

        if enable_forensic:
            self.forensic_branch = ForensicResidualBranch(
                output_dim=fc.get("output_dim", 256),
                fft_output_dim=fc.get("fft_output_dim", 20),
                dropout=fc.get("dropout", 0.1),
            )

        if enable_temporal:
            self.temporal_encoder = TemporalEncoder(
                hidden_dim=tc.get("hidden_dim", 256),
                num_layers=tc.get("num_layers", 2),
                dropout=tc.get("dropout", 0.2),
                bidirectional=tc.get("bidirectional", False),
            )

        # Geometry branch: always active when temporal is active
        # Projects 16-dim raw geometry features into learnable space
        self.geometry_extractor = GeometryFeatureExtractor(
            input_dim=GEOMETRY_DIM,
            output_dim=gc.get("output_dim", GEOMETRY_DIM),
            hidden_dim=gc.get("hidden_dim", 32),
        )

        # Audio-Visual branch: quality-gated by audio + face track availability
        if enable_av:
            self.av_branch = AudioVisualBranch(
                audio_embed_dim=avc.get("audio_embed_dim", 256),
                video_embed_dim=avc.get("video_embed_dim", 256),
                mel_n_mels=avc.get("mel_n_mels", 80),
                window_frames=avc.get("window_frames", 5),
                crop_height=avc.get("crop_height", 48),
                crop_width=avc.get("crop_width", 96),
            )

        # Fusion head
        self.fusion = ReliabilityAwareFusion(
            hidden_dim=fsc.get("hidden_dim", 512),
            dropout=fsc.get("dropout", 0.3),
            ablation_mode=self.ablation_mode,
        )

        # Localization head
        if enable_localization:
            self.localization_head = LocalizationHead(
                target_size=lc.get("target_size", 56),
            )

        # Calibration + abstention
        self.calibration = CalibrationLayer(
            temperature_init=cc.get("temperature_init", 1.0),
            abstention_config=cc.get("abstention", {}),
        )

        # Zero-embeddings for disabled branches (used in fusion)
        self.register_buffer("_zero_semantic", torch.zeros(384))
        self.register_buffer("_zero_forensic", torch.zeros(256))
        self.register_buffer("_zero_fft", torch.zeros(20))
        self.register_buffer("_zero_temporal", torch.zeros(256))
        self.register_buffer("_zero_temporal_scores", torch.zeros(5))
        self.register_buffer("_zero_rppg", torch.zeros(16))
        self.register_buffer("_zero_av", torch.zeros(1))

        logger.info(
            "FORGELite initialized | semantic=%s forensic=%s temporal=%s geometry=enabled "
            "rPPG=%s av=%s localization=%s",
            enable_semantic, enable_forensic, enable_temporal,
            enable_rppg, enable_av, enable_localization,
        )

    def forward(
        self,
        # Core inputs
        frames: torch.Tensor,               # (B, T, 3, 224, 224)
        frames_aug: Optional[torch.Tensor] = None,           # (B, T, 3, 224, 224) for Codec-Invariant learning
        frame_diffs: Optional[torch.Tensor] = None,          # (B, T-1, 3, 224, 224)
        face_quality: Optional[torch.Tensor] = None,         # (B, 10)
        track_quality: Optional[torch.Tensor] = None,        # (B, 5)
        track_continuity: Optional[torch.Tensor] = None,     # (B,)
        # Masks
        mask: Optional[torch.Tensor] = None,         # (B, 1, 56, 56) GT mask
        mask_available: Optional[torch.Tensor] = None,  # (B,)
        # Optional branch inputs
        rppg_features: Optional[torch.Tensor] = None,   # (B, 16) pre-computed by inference engine
        rppg_available: Optional[torch.Tensor] = None,  # (B,)
        av_score: Optional[torch.Tensor] = None,        # (B, 1) pre-computed by inference engine
        av_available: Optional[torch.Tensor] = None,    # (B,)
        temporal_available: Optional[torch.Tensor] = None,  # (B,) None -> auto-detected from T > 1
        semantic_available: Optional[torch.Tensor] = None,  # (B,) None -> ones (for counterfactual attribution)
        forensic_available: Optional[torch.Tensor] = None,  # (B,) None -> ones (for counterfactual attribution)
        # Geometry features from GeometryFeatureExtractor
        geometry_features: Optional[torch.Tensor] = None,  # (B, T, 16)
        # Audio for AV branch (computed inside forward if enable_av=True)
        audio_mel: Optional[torch.Tensor] = None,       # (B, 1, 80, T_mel)
        # Return calibrated decisions (inference only)
        return_decisions: bool = False,
    ) -> dict:
        """
        Full FORGE-Lite forward pass.

        Returns dict with all outputs needed for loss computation and inference.
        Output schema matches PRD Section 4.
        """
        B, T = frames.shape[:2]
        device = frames.device

        if frame_diffs is None:
            if T > 1:
                frame_diffs = (frames[:, 1:] - frames[:, :-1]).abs()
            else:
                frame_diffs = torch.zeros(B, 1, 3, 224, 224, device=device, dtype=frames.dtype)

        if face_quality is None:
            face_quality = torch.full((B, 10), 0.8, device=device, dtype=frames.dtype)

        if track_quality is None:
            track_quality = torch.full((B, 5), 0.8, device=device, dtype=frames.dtype)

        if track_continuity is None:
            track_continuity = torch.ones(B, device=device, dtype=frames.dtype)

        # ── 0. Geometry features ───────────────────────────────────────────────────────────
        # Project raw geometry features through learnable GeometryFeatureExtractor.
        # Falls back to zeros if not provided (image-mode inference).
        if geometry_features is not None:
            geo_features = self.geometry_extractor(geometry_features)  # (B, T, 16)
        else:
            geo_features = torch.zeros(B, T, GEOMETRY_DIM, device=device)

        # ── 1. Semantic Branch ────────────────────────────────────────────────
        semantic_embedding_aug = None
        if self.enable_semantic:
            sem_requires_grad = any(p.requires_grad for p in self.semantic_branch.parameters())
            with torch.set_grad_enabled(sem_requires_grad and self.training):
                if self.enable_temporal and T > 1:
                    # Process all frames through DINO for temporal sequence
                    frames_flat = frames.view(B * T, 3, 224, 224)
                    with torch.amp.autocast("cuda"):
                        if B * T > 32 and not sem_requires_grad:
                            cls_list, pt_list = [], []
                            for chunk in torch.split(frames_flat, 32):
                                c_out = self.semantic_branch(chunk, return_patch_tokens=True)
                                cls_list.append(c_out["cls"])
                                if c_out.get("patch_tokens") is not None:
                                    pt_list.append(c_out["patch_tokens"])
                            cls_flat = torch.cat(cls_list, dim=0)
                            patch_tokens_flat = torch.cat(pt_list, dim=0) if pt_list else None
                        else:
                            dino_out = self.semantic_branch(frames_flat, return_patch_tokens=True)
                            cls_flat = dino_out["cls"]                          # (B*T, 384)
                            patch_tokens_flat = dino_out.get("patch_tokens")    # (B*T, 256, 384)
                    dino_features = cls_flat.view(B, T, -1)             # (B, T, 384)
                    patch_tokens = patch_tokens_flat[:B] if patch_tokens_flat is not None else None
                    semantic_embedding = dino_features.mean(dim=1)      # (B, 384)
                    
                    if frames_aug is not None:
                        frames_aug_flat = frames_aug.view(B * T, 3, 224, 224)
                        with torch.amp.autocast("cuda"):
                            if B * T > 32 and not sem_requires_grad:
                                cls_list_aug = []
                                for chunk in torch.split(frames_aug_flat, 32):
                                    c_out = self.semantic_branch(chunk, return_patch_tokens=False)
                                    cls_list_aug.append(c_out["cls"])
                                cls_flat_aug = torch.cat(cls_list_aug, dim=0)
                            else:
                                dino_out_aug = self.semantic_branch(frames_aug_flat, return_patch_tokens=False)
                                cls_flat_aug = dino_out_aug["cls"]
                        semantic_embedding_aug = cls_flat_aug.view(B, T, -1).mean(dim=1)
                else:
                    # Spatial only: process representative middle frame (huge speedup)
                    rep_frame = frames[:, T // 2]
                    with torch.amp.autocast("cuda"):
                        dino_out = self.semantic_branch(rep_frame, return_patch_tokens=True)
                    semantic_embedding = dino_out["cls"]                # (B, 384)
                    dino_features = semantic_embedding.unsqueeze(1).expand(B, T, -1)
                    patch_tokens = dino_out.get("patch_tokens")
                    
                    if frames_aug is not None:
                        rep_frame_aug = frames_aug[:, T // 2]
                        with torch.amp.autocast("cuda"):
                            dino_out_aug = self.semantic_branch(rep_frame_aug, return_patch_tokens=False)
                        semantic_embedding_aug = dino_out_aug["cls"]
        else:
            dino_features = self._zero_semantic.expand(B, T, -1)
            semantic_embedding = self._zero_semantic.expand(B, -1)
            patch_tokens = None

        # ── 2. Forensic Branch ────────────────────────────────────────────────
        if self.enable_forensic:
            for_requires_grad = any(p.requires_grad for p in self.forensic_branch.parameters())
            with torch.set_grad_enabled(for_requires_grad and self.training):
                # Per-frame forensic when temporal branch is active
                if self.enable_temporal and T > 1:
                    frames_flat = frames.view(B * T, 3, 224, 224)
                    with torch.amp.autocast("cuda"):
                        if B * T > 32 and not for_requires_grad:
                            emb_list, fft_list, feat_list = [], [], []
                            for chunk in torch.split(frames_flat, 32):
                                c_out = self.forensic_branch(chunk)
                                emb_list.append(c_out["embedding"])
                                fft_list.append(c_out["fft_features"])
                                feat_list.append(c_out["feature_map"])
                            forensic_seq = torch.cat(emb_list, dim=0).view(B, T, -1)
                            fft_seq = torch.cat(fft_list, dim=0).view(B, T, -1)
                            forensic_feature_map = feat_list[0][:B]
                        else:
                            forensic_per_frame = self.forensic_branch(frames_flat)
                            forensic_seq = forensic_per_frame["embedding"].view(B, T, -1)  # (B, T, 256)
                            fft_seq = forensic_per_frame["fft_features"].view(B, T, -1)    # (B, T, 20)
                            forensic_feature_map = forensic_per_frame["feature_map"][:B]
                    # Direct slice instead of redundant forward pass
                    forensic_embedding = forensic_seq[:, T // 2]
                    fft_features = fft_seq[:, T // 2]
                else:
                    # Spatial only: process representative middle frame
                    rep_frame = frames[:, T // 2]  # (B, 3, 224, 224) — middle frame
                    with torch.amp.autocast("cuda"):
                        forensic_out = self.forensic_branch(rep_frame)
                    forensic_embedding = forensic_out["embedding"]  # (B, 256)
                    fft_features = forensic_out["fft_features"]     # (B, 20)
                    forensic_feature_map = forensic_out["feature_map"]  # (B, C, H, W)
                    forensic_seq = forensic_embedding.unsqueeze(1).expand(B, T, -1)
                    fft_seq = fft_features.unsqueeze(1).expand(B, T, -1)
        else:
            forensic_embedding = self._zero_forensic.expand(B, -1)
            fft_features = self._zero_fft.expand(B, -1)
            forensic_feature_map = None
            forensic_seq = self._zero_forensic.expand(B, T, -1)
            fft_seq = self._zero_fft.expand(B, T, -1)

        # ── 3. Temporal Branch ────────────────────────────────────────────────
        if self.enable_temporal and T > 1:
            # Expand face_quality to per-frame
            face_quality_seq = face_quality.unsqueeze(1).expand(B, T, -1)

            temporal_out = self.temporal_encoder(
                dino_features=dino_features,
                forensic_features=forensic_seq,
                fft_features=fft_seq,
                frame_diffs=frame_diffs,
                geometry_features=geo_features,   # (B, T, 16) — real geometry, not zeros
                face_quality=face_quality_seq,
                track_continuity=track_continuity,
            )
            temporal_embedding = temporal_out["temporal_embedding"]  # (B, 256)
            temporal_scores = torch.stack([
                temporal_out["mean_score"],
                temporal_out["max_score"],
                temporal_out["top10_score"],
                temporal_out["score_dispersion"],
                temporal_out["attn_score"],
            ], dim=1)  # (B, 5)
            frame_scores = temporal_out["frame_scores"]  # (B, T)
        else:
            temporal_embedding = self._zero_temporal.expand(B, -1)
            temporal_scores = self._zero_temporal_scores.expand(B, -1)
            frame_scores = torch.zeros(B, T, device=device)

        # ── 4. rPPG Branch (pre-computed features passed in) ─────────────────
        if rppg_features is None:
            rppg_features = self._zero_rppg.expand(B, -1)
        if rppg_available is None:
            rppg_available = torch.zeros(B, dtype=torch.bool, device=device)

        # ── 5. Audio-Visual Branch ───────────────────────────────────────────
        # If av_score is pre-computed by inference engine: use it directly.
        # If audio_mel + frames are provided and enable_av=True: run AVBranch here.
        if av_score is None and self.enable_av and hasattr(self, "av_branch"):
            try:
                av_out = self.av_branch(
                    audio_mel=audio_mel,          # None if no audio → branch returns zeros
                    lip_crops=frames,             # (B, T, 3, 224, 224) — auto-extracts mouth
                    audio_available=av_available,
                )
                av_score = av_out["av_score"]           # (B, 1)
                # av_available is set from branch output
                av_available = av_out["av_available"].squeeze(-1).bool()
            except Exception as e:
                logger.debug("AV branch failed (non-critical): %s", e)
                av_score = self._zero_av.expand(B, -1)
                av_available = torch.zeros(B, dtype=torch.bool, device=device)

        if av_score is None:
            av_score = self._zero_av.expand(B, -1)
        if av_available is None:
            av_available = torch.zeros(B, dtype=torch.bool, device=device)

        # ── 6. Reliability-Aware Fusion ───────────────────────────────────────
        video_quality = track_quality  # (B, 5) — reuse track quality as video quality

        # Auto-determine temporal branch availability: inactive when T <= 1 or enable_temporal=False
        if temporal_available is not None:
            temp_avail = temporal_available
        else:
            temp_avail = (
                torch.ones(B, device=device)
                if (self.enable_temporal and T > 1)
                else torch.zeros(B, device=device)
            )

        fusion_out = self.fusion(
            semantic_embedding=semantic_embedding,
            forensic_embedding=forensic_embedding,
            fft_features=fft_features,
            temporal_embedding=temporal_embedding,
            temporal_scores=temporal_scores,
            rppg_features=rppg_features,
            av_score=av_score,
            face_quality=face_quality,
            video_quality=video_quality,
            rppg_available=rppg_available,
            av_available=av_available,
            temporal_available=temp_avail,
            semantic_available=semantic_available,
            forensic_available=forensic_available,
        )

        fake_logit = fusion_out["fake_logit"]
        uncertainty = fusion_out["uncertainty"]
        branch_weights = fusion_out["branch_weights"]
        branch_disagreement = fusion_out["branch_disagreement"]
        evidence_coverage = fusion_out["evidence_coverage"]

        # ── 7. Localization Head ──────────────────────────────────────────────
        pred_mask = None
        if self.enable_localization and patch_tokens is not None and forensic_feature_map is not None:
            try:
                pred_mask = self.localization_head(patch_tokens, forensic_feature_map)
            except Exception as e:
                logger.debug("Localization head failed: %s", e)

        # ── 8. Calibration + Abstention (inference only) ──────────────────────
        decisions = None
        calibrated_score = torch.sigmoid(fake_logit)
        if return_decisions:
            face_quality_composite = face_quality[:, -1]  # composite is last dim
            cal_out = self.calibration(
                fake_logit=fake_logit,
                uncertainty=uncertainty,
                evidence_coverage=evidence_coverage,
                branch_disagreement=branch_disagreement,
                face_quality_composite=face_quality_composite,
            )
            calibrated_score = cal_out["calibrated_score"]
            decisions = cal_out["decisions"]

        # ── Build output dict ─────────────────────────────────────────────────
        output = {
            # Core predictions
            "fake_logit": fake_logit,                   # (B,) for loss
            "fake_score": calibrated_score,             # (B,) [0,1]
            "uncertainty": uncertainty,                 # (B,)
            "evidence_coverage": evidence_coverage,     # (B,)
            "branch_disagreement": branch_disagreement, # (B,)
            "branch_weights": branch_weights,           # (B, 5)
            "branch_logits": fusion_out.get("branch_logits"),            # (B, 5)
            "branch_fake_probs": fusion_out.get("branch_fake_probs"),    # (B, 5)
            "availability_mask": fusion_out.get("availability_mask"),    # (B, 5)
            # Temporal evidence
            "frame_scores": frame_scores,               # (B, T)
            "temporal_scores": temporal_scores,         # (B, 5)
            # Localization
            "pred_mask": pred_mask,                     # (B, 1, 56, 56) or None
            # Embeddings (for consistency loss)
            "semantic_embedding": semantic_embedding,   # (B, 384)
            "forensic_embedding": forensic_embedding,   # (B, 256)
            "temporal_embedding": temporal_embedding,   # (B, 256)
            # Calibrated decisions (inference only)
            "decisions": decisions,
            # Audio-Visual evidence
            "av_score": av_score,
            "av_available": av_available,
            "model_version": MODEL_VERSION,
            "preprocessing_version": PREPROCESSING_VERSION,
        }
        
        if semantic_embedding_aug is not None:
            output["semantic_embedding_aug"] = semantic_embedding_aug

        return output

    # ── Utility methods ───────────────────────────────────────────────────────

    def count_parameters(self) -> dict[str, int]:
        """Count trainable parameters per branch."""
        counts = {}
        for name, module in [
            ("semantic", getattr(self, "semantic_branch", None)),
            ("forensic", getattr(self, "forensic_branch", None)),
            ("temporal", getattr(self, "temporal_encoder", None)),
            ("fusion", self.fusion),
            ("localization", getattr(self, "localization_head", None)),
            ("calibration", self.calibration),
        ]:
            if module is not None:
                n = sum(p.numel() for p in module.parameters() if p.requires_grad)
                counts[name] = n
        counts["total"] = sum(counts.values())
        return counts

    def get_trainable_parameters(self, phase: str = "fusion"):
        """
        Return optimizer parameter groups for each training phase.
        Prevents accidentally training frozen parameters.
        """
        if phase == "semantic_only":
            if hasattr(self, "semantic_branch"):
                return self.semantic_branch.get_trainable_parameters()
            return []
        elif phase == "forensic_only":
            return list(self.forensic_branch.parameters()) if hasattr(self, "forensic_branch") else []
        elif phase == "temporal_only":
            return list(self.temporal_encoder.parameters()) if hasattr(self, "temporal_encoder") else []
        elif phase == "av_only":
            return list(self.av_branch.parameters()) if hasattr(self, "av_branch") else []
        elif phase == "geometry_only":
            return list(self.geometry_extractor.parameters()) if hasattr(self, "geometry_extractor") else []
        elif phase == "fusion":
            params = list(self.fusion.parameters())
            if hasattr(self, "localization_head"):
                params += list(self.localization_head.parameters())
            if hasattr(self, "geometry_extractor"):
                params += list(self.geometry_extractor.parameters())
            if hasattr(self, "av_branch"):
                params += list(self.av_branch.parameters())
            return params
        elif phase in ("joint_spatial", "spatial"):
            params = []
            if hasattr(self, "semantic_branch"):
                params += self.semantic_branch.get_trainable_parameters()
            if hasattr(self, "forensic_branch"):
                params += list(self.forensic_branch.parameters())
            if hasattr(self, "fusion"):
                params += list(self.fusion.parameters())
            return params
        elif phase == "all":
            return [p for p in self.parameters() if p.requires_grad]
        return []

    @torch.no_grad()
    def measure_vram(self, device: str = "cuda") -> dict[str, float]:
        """Measure peak VRAM usage during inference with a dummy batch."""
        if not torch.cuda.is_available():
            return {"peak_vram_gb": 0.0}
        torch.cuda.reset_peak_memory_stats(device)
        B, T = 1, 16
        dummy_frames = torch.randn(B, T, 3, 224, 224, device=device, dtype=torch.float16)
        dummy_diffs = torch.randn(B, T - 1, 3, 224, 224, device=device, dtype=torch.float16)
        dummy_fq = torch.zeros(B, 10, device=device)
        dummy_tq = torch.zeros(B, 5, device=device)
        dummy_cont = torch.ones(B, device=device)
        t0 = time.time()
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            _ = self.forward(
                frames=dummy_frames, frame_diffs=dummy_diffs,
                face_quality=dummy_fq, track_quality=dummy_tq,
                track_continuity=dummy_cont,
            )
        latency_ms = (time.time() - t0) * 1000
        peak_bytes = torch.cuda.max_memory_allocated(device)
        return {
            "peak_vram_gb": peak_bytes / 1e9,
            "inference_latency_ms": latency_ms,
        }


# Alias for flexible casing
ForgeLite = FORGELite

