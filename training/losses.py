"""
training/losses.py
Composite training loss for FORGE-Lite.

PRD Section 7, Step 10:
  Total loss =
    classification loss
  + consistency loss (stable scores across benign transforms)
  + source-pair metric loss
  + localization loss (when masks exist)
  + temporal consistency loss
  + geometry consistency loss (stable geometry across codec augmentations)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.localization import LocalizationLoss


class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance in deepfake detection.
    
    Combines standard focal weighting with optional pos_weight to handle
    extreme imbalances (e.g. 1:18 real:fake after DF40 integration).
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0,
                 pos_weight: Optional[torch.Tensor] = None) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.register_buffer("pos_weight", pos_weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(
            logits, targets,
            pos_weight=self.pos_weight if self.pos_weight is not None else None,
            reduction="none",
        )
        probs = torch.sigmoid(logits)
        pt = targets * probs + (1 - targets) * (1 - probs)
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        return (focal_weight * bce).mean()


class CodecConsistencyLoss(nn.Module):
    """
    Ensures model produces stable scores across codec variants of the same video.

    For each sample, we expect the embeddings/scores from a clean version
    and a codec-augmented version to be similar (MSE or cosine similarity).

    This teaches: compression ≠ forgery.
    """

    def __init__(self, loss_type: str = "mse", temperature: float = 2.0) -> None:
        super().__init__()
        self.loss_type = loss_type
        self.temperature = temperature

    def forward(
        self,
        embeddings_clean: torch.Tensor,    # (B, D) from clean version
        embeddings_aug: torch.Tensor,      # (B, D) from augmented version
        scores_clean: Optional[torch.Tensor] = None,        # (B,) fake scores from clean
        scores_aug: Optional[torch.Tensor] = None,          # (B,) fake scores from augmented
    ) -> torch.Tensor:
        """
        Penalize large differences in embedding space and score space
        across codec variants.
        """
        if self.loss_type == "mse":
            emb_loss = F.mse_loss(embeddings_clean, embeddings_aug.detach())
        elif self.loss_type == "cosine":
            emb_loss = 1.0 - F.cosine_similarity(embeddings_clean, embeddings_aug.detach()).mean()
        else:
            emb_loss = F.mse_loss(embeddings_clean, embeddings_aug.detach())
            
        if scores_clean is not None and scores_aug is not None:
            score_loss = F.mse_loss(scores_clean, scores_aug.detach())
            return emb_loss + score_loss * 0.5
            
        return emb_loss


class SourcePairMetricLoss(nn.Module):
    """
    Contrastive loss on (real, fake) source pairs.

    Given a real video and its manipulated version (same source identity),
    the embeddings should be distinguishable but the identity representation
    should be similar. This prevents the model from using identity as a shortcut.

    Positive pairs: same label (real-real or fake-fake)
    Negative pairs: different label (real-fake)
    """

    def __init__(self, margin: float = 0.5) -> None:
        super().__init__()
        self.margin = margin

    def forward(
        self,
        embeddings: torch.Tensor,  # (B, D)
        labels: torch.Tensor,      # (B,) 0=real, 1=fake
    ) -> torch.Tensor:
        """Contrastive loss on embedding pairs within the batch."""
        B = embeddings.shape[0]
        if B < 2:
            return torch.tensor(0.0, device=embeddings.device)

        # Normalize embeddings
        emb_norm = F.normalize(embeddings, p=2, dim=-1)
        # Pairwise cosine similarity
        sim_matrix = torch.mm(emb_norm, emb_norm.t())  # (B, B)
        # Labels match matrix
        label_match = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()  # (B, B)

        # Contrastive: similar labels → high sim, different labels → low sim
        positive_loss = (1 - sim_matrix) * label_match
        negative_loss = F.relu(sim_matrix - self.margin) * (1 - label_match)

        # Mask diagonal
        mask = 1 - torch.eye(B, device=embeddings.device)
        loss = ((positive_loss + negative_loss) * mask).sum() / mask.sum().clamp(min=1)
        return loss

class GeometryConsistencyLoss(nn.Module):
    """
    Geometry consistency regularizer.

    Deepfakes often have unstable geometry: landmark positions jitter between
    codec variants because the GAN's spatial blending is not landmark-aware.
    Real faces have stable landmarks regardless of compression level.

    Loss: For real samples, MSE between geometry features of clean and codec-aug versions.
    For fake samples: no penalty (we don’t want to teach the model that fakes have
    stable geometry — some do, and penalizing would suppress a valid signal).

    Usage:
        geometry_consistency_loss = GeometryConsistencyLoss()
        loss = geometry_consistency_loss(
            geo_clean,   # (B, T, 16) geometry from clean video
            geo_aug,     # (B, T, 16) geometry from codec-augmented video
            labels,      # (B,) 0=real, 1=fake
        )
    """

    def __init__(self, loss_type: str = "mse", real_only: bool = True) -> None:
        super().__init__()
        self.loss_type = loss_type
        self.real_only = real_only  # Apply only to real samples

    def forward(
        self,
        geo_clean: torch.Tensor,   # (B, T, 16)
        geo_aug: torch.Tensor,     # (B, T, 16)
        labels: torch.Tensor,      # (B,) 0=real, 1=fake
    ) -> torch.Tensor:
        """
        Compute geometry consistency loss.
        Returns scalar loss (0.0 if no valid samples).
        """
        if geo_clean is None or geo_aug is None:
            return torch.tensor(0.0, device=labels.device)

        if self.real_only:
            real_mask = (labels == 0)  # only real samples
            if not real_mask.any():
                return torch.tensor(0.0, device=labels.device)
            geo_c = geo_clean[real_mask]
            geo_a = geo_aug[real_mask]
        else:
            geo_c, geo_a = geo_clean, geo_aug

        if self.loss_type == "mse":
            return F.mse_loss(geo_c, geo_a)
        else:  # cosine similarity loss
            # Per-frame cosine distance, averaged over B and T
            B, T, D = geo_c.shape
            gc_flat = geo_c.view(B * T, D)
            ga_flat = geo_a.view(B * T, D)
            cos_sim = F.cosine_similarity(gc_flat, ga_flat, dim=-1)  # (B*T,)
            return (1.0 - cos_sim).mean()




class ForgeLoss(nn.Module):
    """
    Composite FORGE-Lite training loss.

    Weights from configs/train.yaml loss_weights section.
    """

    def __init__(
        self,
        classification_weight: float = 1.0,
        consistency_weight: float = 0.3,
        metric_weight: float = 0.2,
        localization_weight: float = 0.5,
        temporal_weight: float = 0.1,
        geometry_weight: float = 0.15,
        branch_weight: float = 1.0,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        pos_weight: Optional[torch.Tensor] = None,  # BCEWithLogitsLoss pos_weight for imbalance
        use_consistency: bool = True,
        use_metric: bool = True,
        use_localization: bool = True,
        use_temporal: bool = True,
        use_geometry: bool = True,
        use_branch_logits: bool = True,
    ) -> None:
        super().__init__()
        self.w_cls = classification_weight
        self.w_cons = consistency_weight
        self.w_metric = metric_weight
        self.w_loc = localization_weight
        self.w_temp = temporal_weight
        self.w_geo = geometry_weight
        self.w_branch = branch_weight

        self.use_consistency = use_consistency
        self.use_metric = use_metric
        self.use_localization = use_localization
        self.use_temporal = use_temporal
        self.use_geometry = use_geometry
        self.use_branch_logits = use_branch_logits

        # pos_weight up-weights the real (positive) class in BCE
        # Critical for 1:18 real:fake imbalance from combined DF40 manifest
        self.cls_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma, pos_weight=pos_weight)
        self.consistency_loss = CodecConsistencyLoss()
        self.metric_loss = SourcePairMetricLoss()
        self.localization_loss = LocalizationLoss()
        self.geometry_loss = GeometryConsistencyLoss(loss_type="mse", real_only=True)


    def forward(
        self,
        # Classification
        fake_logit: torch.Tensor,           # (B,)
        labels: torch.Tensor,               # (B,) 0/1
        # Consistency (optional — only when codec aug applied)
        embeddings_clean: torch.Tensor = None,
        embeddings_aug: torch.Tensor = None,
        scores_clean: torch.Tensor = None,
        scores_aug: torch.Tensor = None,
        # Localization (optional)
        pred_mask: torch.Tensor = None,     # (B, 1, 56, 56)
        gt_mask: torch.Tensor = None,       # (B, 1, 56, 56)
        mask_available: torch.Tensor = None,  # (B,)
        # Temporal frame-level supervision (optional)
        frame_scores: torch.Tensor = None,  # (B, T)
        frame_labels: torch.Tensor = None,  # (B, T) — from video-level label broadcast
        # Embeddings for metric loss
        semantic_embedding: torch.Tensor = None,  # (B, 384)
        # Geometry consistency (optional — only when codec aug applied)
        geometry_clean: torch.Tensor = None,  # (B, T, 16)
        geometry_aug: torch.Tensor = None,    # (B, T, 16)
        # Branch logit calibration (optional)
        branch_logits: torch.Tensor = None,   # (B, 5)
        availability_mask: torch.Tensor = None,  # (B, 5)
    ) -> dict[str, torch.Tensor]:
        """Compute composite loss and return loss dict."""
        losses = {}

        # 1. Classification loss (always)
        cls = self.cls_loss(fake_logit, labels.float())
        losses["classification"] = cls
        total = self.w_cls * cls

        # 2. Codec consistency loss (when augmented version provided)
        if self.use_consistency and embeddings_clean is not None and embeddings_aug is not None:
            sc_clean = torch.sigmoid(scores_clean) if scores_clean is not None else None
            sc_aug = torch.sigmoid(scores_aug) if scores_aug is not None else None
            cons = self.consistency_loss(embeddings_clean, embeddings_aug, sc_clean, sc_aug)
            losses["consistency"] = cons
            total = total + self.w_cons * cons

        # 3. Source-pair metric loss
        if self.use_metric and semantic_embedding is not None:
            metric = self.metric_loss(semantic_embedding, labels)
            losses["metric"] = metric
            total = total + self.w_metric * metric

        # 4. Localization loss (only on valid masks)
        if self.use_localization and pred_mask is not None and gt_mask is not None:
            loc = self.localization_loss(pred_mask, gt_mask, mask_available)
            losses["localization"] = loc
            total = total + self.w_loc * loc

        # 5. Temporal consistency loss
        if self.use_temporal and frame_scores is not None:
            # Broadcast video label to all frames and compute BCE
            if frame_labels is None:
                frame_labels = labels.unsqueeze(-1).expand_as(frame_scores)
            device_type = "cuda" if frame_scores.is_cuda else "cpu"
            with torch.amp.autocast(device_type=device_type, enabled=False):
                temp_loss = F.binary_cross_entropy(
                    frame_scores.float().clamp(1e-6, 1 - 1e-6),
                    frame_labels.float()
                )
            losses["temporal"] = temp_loss
            total = total + self.w_temp * temp_loss

        # 6. Geometry consistency loss (real samples only)
        # Codec compression should not change landmark geometry for real faces.
        # For fakes, landmark jitter under augmentation is actually a signal we want
        # the model to learn — don't penalize instability in fake geometry.
        if self.use_geometry and geometry_clean is not None and geometry_aug is not None:
            geo = self.geometry_loss(geometry_clean, geometry_aug, labels)
            losses["geometry_consistency"] = geo
            total = total + self.w_geo * geo

        # 7. Branch logit heads calibration loss
        if self.use_branch_logits and branch_logits is not None:
            targets = labels.unsqueeze(1).expand_as(branch_logits).float()
            device_type = "cuda" if branch_logits.is_cuda else "cpu"
            with torch.amp.autocast(device_type=device_type, enabled=False):
                bce_per_branch = F.binary_cross_entropy_with_logits(
                    branch_logits.float(), targets, reduction="none"
                )  # (B, 5)
                if availability_mask is not None:
                    mask = availability_mask.float()
                    branch_loss = (bce_per_branch * mask).sum() / mask.sum().clamp(min=1.0)
                else:
                    branch_loss = bce_per_branch.mean()
            losses["branch_logits"] = branch_loss
            total = total + self.w_branch * branch_loss

        losses["total"] = total
        return losses
