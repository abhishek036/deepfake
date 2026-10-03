"""
models/dino_branch.py
DINOv2 ViT-S/14 semantic branch for FORGE-Lite.

Supports 4 adaptation modes (A–D) selected by cross-domain macro-AUROC:
  A: Frozen backbone + linear classifier
  B: Frozen backbone + trainable LayerNorm only (default candidate per PRD)
  C: Frozen backbone + LoRA adapters (requires peft)
  D: Last-block fine-tuning only

Selection rule: choose by cross-domain macro-AUROC, never in-domain alone.

Output:
  - cls_embedding: (B, 384) CLS token embedding for fusion
  - patch_tokens: (B, 256, 384) patch token grid for localization head
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# DINOv2 ViT-S/14: embed_dim=384, patch_size=14
# For 224×224 input: 16×16 = 256 patches
EMBED_DIM = 384
NUM_PATCHES = 256  # 16×16 for 224/14


class DINOSemanticBranch(nn.Module):
    """
    Foundation model semantic branch using DINOv2 ViT-S/14.

    Adaptation modes (PRD Section 3.3, Branch 1):
    - 'frozen_linear': Frozen backbone + linear classifier head only
    - 'layernorm':     Frozen backbone + trainable LayerNorm parameters (recommended)
    - 'lora':          Frozen backbone + low-rank adapters (peft library)
    - 'last_block':    Last transformer block + head unfrozen
    """

    def __init__(
        self,
        adaptation_mode: str = "layernorm",
        output_dim: int = 384,
        dropout: float = 0.1,
        lora_rank: int = 8,
        lora_alpha: int = 16,
    ) -> None:
        super().__init__()
        self.adaptation_mode = adaptation_mode
        self.output_dim = output_dim
        self.embed_dim = EMBED_DIM

        # Load DINOv2 backbone
        self.backbone = self._load_dinov2()

        # Configure adaptation
        self._configure_adaptation(adaptation_mode, lora_rank, lora_alpha)

        # Output projection (optional dim reduction or identity)
        if output_dim != EMBED_DIM:
            self.proj = nn.Linear(EMBED_DIM, output_dim)
        else:
            self.proj = nn.Identity()

        self.dropout = nn.Dropout(p=dropout)
        self.norm = nn.LayerNorm(output_dim)

    def forward(
        self,
        x: torch.Tensor,
        return_patch_tokens: bool = True,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            x: (B, 3, 224, 224) normalized RGB tensor
            return_patch_tokens: whether to return patch tokens for localization

        Returns:
            dict with:
              'cls': (B, output_dim) CLS embedding
              'patch_tokens': (B, 256, output_dim) patch tokens (if requested)
              'patch_mean': (B, output_dim) mean of patch tokens
              'patch_var': (B, output_dim) variance of patch tokens
        """
        dev_type = "cuda" if x.is_cuda else "cpu"
        with torch.amp.autocast(dev_type, enabled=x.is_cuda):
            features = self.backbone.forward_features(x)

        # DINOv2 returns dict with 'x_norm_clstoken' and 'x_norm_patchtokens'
        if isinstance(features, dict):
            cls_token = features.get("x_norm_clstoken", features.get("cls_token"))
            patch_tokens = features.get("x_norm_patchtokens", features.get("patch_tokens"))
        else:
            # Fallback: first token is CLS
            cls_token = features[:, 0]
            patch_tokens = features[:, 1:]

        # Project + normalize CLS
        cls_out = self.norm(self.proj(self.dropout(cls_token)))

        result = {"cls": cls_out}

        if return_patch_tokens and patch_tokens is not None:
            patch_proj = self.proj(patch_tokens)  # (B, 256, output_dim)
            result["patch_tokens"] = patch_proj
            result["patch_mean"] = patch_proj.mean(dim=1)
            result["patch_var"] = patch_proj.var(dim=1)

        return result

    def get_trainable_parameters(self):
        """Return only the parameters that should be updated for this adaptation mode."""
        if self.adaptation_mode == "frozen_linear":
            return list(self.proj.parameters()) + list(self.norm.parameters())
        elif self.adaptation_mode == "layernorm":
            ln_params = [p for n, p in self.backbone.named_parameters() if "norm" in n]
            return ln_params + list(self.proj.parameters()) + list(self.norm.parameters())
        elif self.adaptation_mode == "lora":
            # peft handles trainable params automatically
            return list(self.parameters())
        elif self.adaptation_mode == "last_block":
            last_block_params = []
            blocks = list(self.backbone.blocks) if hasattr(self.backbone, "blocks") else []
            if blocks:
                last_block_params = list(blocks[-1].parameters())
            return last_block_params + list(self.proj.parameters()) + list(self.norm.parameters())
        return list(self.parameters())

    # ── Private helpers ───────────────────────────────────────────────────────

    def _load_dinov2(self) -> nn.Module:
        """Load DINOv2 ViT-S/14 from torch.hub."""
        try:
            model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", verbose=False)
            logger.info("DINOv2 ViT-S/14 loaded from torch.hub")
            return model
        except Exception as e:
            logger.warning("torch.hub DINOv2 load failed (%s), trying HuggingFace...", e)
            return self._load_dinov2_hf()

    def _load_dinov2_hf(self) -> nn.Module:
        """Fallback: load via HuggingFace transformers."""
        try:
            from transformers import AutoModel
            model = AutoModel.from_pretrained(
                "facebook/dinov2-small",
                output_hidden_states=False,
            )
            logger.info("DINOv2 ViT-S/14 loaded from HuggingFace")
            # Wrap to match forward_features interface
            return _HFDinoWrapper(model)
        except Exception as e2:
            logger.error("Both DINOv2 load methods failed. Error: %s", e2)
            logger.info("Falling back to random ViT-S/16 for testing")
            return _RandomViTFallback(embed_dim=EMBED_DIM)

    def _configure_adaptation(
        self, mode: str, lora_rank: int, lora_alpha: int
    ) -> None:
        """Set which parameters are trainable based on adaptation mode."""
        # Start fully frozen
        for param in self.backbone.parameters():
            param.requires_grad = False

        if mode == "frozen_linear":
            pass  # backbone stays frozen

        elif mode == "layernorm":
            # Unfreeze only LayerNorm parameters
            for name, param in self.backbone.named_parameters():
                if "norm" in name.lower() or "ln" in name.lower():
                    param.requires_grad = True

        elif mode == "lora":
            try:
                from peft import get_peft_model, LoraConfig, TaskType
                lora_config = LoraConfig(
                    r=lora_rank,
                    lora_alpha=lora_alpha,
                    target_modules=["qkv", "proj"],
                    lora_dropout=0.1,
                    bias="none",
                )
                self.backbone = get_peft_model(self.backbone, lora_config)
                logger.info("LoRA adapters applied (rank=%d)", lora_rank)
            except ImportError:
                logger.warning("peft not installed. Falling back to layernorm adaptation.")
                self._configure_adaptation("layernorm", lora_rank, lora_alpha)

        elif mode == "last_block":
            # Unfreeze last transformer block
            blocks = list(self.backbone.named_children())
            if blocks:
                last_block_name, last_block = blocks[-1]
                for param in last_block.parameters():
                    param.requires_grad = True
                logger.info("Last block '%s' unfrozen", last_block_name)

        n_trainable = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in self.backbone.parameters())
        logger.info("DINOv2 adaptation='%s': %d / %d params trainable (%.1f%%)",
                    mode, n_trainable, n_total, 100 * n_trainable / max(n_total, 1))


class _HFDinoWrapper(nn.Module):
    """Wraps HuggingFace DINOv2 to match the forward_features interface."""

    def __init__(self, hf_model) -> None:
        super().__init__()
        self.model = hf_model

    def forward_features(self, x: torch.Tensor) -> dict:
        outputs = self.model(x, output_hidden_states=False)
        # HF returns BaseModelOutputWithPooling
        last_hidden = outputs.last_hidden_state  # (B, 1+N, D)
        cls_token = last_hidden[:, 0]            # (B, D)
        patch_tokens = last_hidden[:, 1:]         # (B, N, D)
        return {"x_norm_clstoken": cls_token, "x_norm_patchtokens": patch_tokens}

    def named_parameters(self, *args, **kwargs):
        return self.model.named_parameters(*args, **kwargs)

    def parameters(self, *args, **kwargs):
        return self.model.parameters(*args, **kwargs)


class _RandomViTFallback(nn.Module):
    """Random ViT-like model for testing when DINOv2 is unavailable."""

    def __init__(self, embed_dim: int = 384) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=14, stride=14)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=6, dim_feedforward=1536, dropout=0.1, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

    def forward_features(self, x: torch.Tensor) -> dict:
        patches = self.patch_embed(x)  # (B, D, H/14, W/14)
        B, D, H, W = patches.shape
        patches = patches.flatten(2).transpose(1, 2)  # (B, H*W, D)
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, patches], dim=1)  # (B, 1+N, D)
        out = self.encoder(tokens)
        return {"x_norm_clstoken": out[:, 0], "x_norm_patchtokens": out[:, 1:]}
