"""
models/calibration.py
Temperature scaling calibration and abstention layer for FORGE-Lite.

PRD Section 7, Step 14:
  Combine: calibrated score + prediction uncertainty + input quality + evidence coverage
  → final decision rule (likely_real / likely_fake / insufficient_evidence)

Calibration set must be SEPARATE from test set.
Thresholds tuned on calibration set ONLY — never on test set.

newimp.md P0 corrections:
  - Thresholds must be FROZEN (fixed) outside the final test set.
  - Missing optional branches should disable THAT branch only — not the whole system.
  - Use freeze_threshold() after calibration to lock operating point.
  - Never re-tune threshold after any test-set access.

Metrics: ECE, Brier score, reliability diagram, coverage-risk curve, abstention rate.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class Decision(str, Enum):
    LIKELY_REAL = "likely_real"
    LIKELY_FAKE = "likely_fake"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


@dataclass
class DecisionResult:
    """Full output of the calibration + abstention layer."""
    decision: Decision
    fake_score: float           # raw calibrated fake probability [0, 1]
    calibrated_risk: float      # temperature-scaled probability
    uncertainty: float          # model uncertainty estimate [0, 1]
    evidence_coverage: float    # fraction of branches available [0, 1]
    abstention_reason: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "decision": self.decision.value,
            "fake_score": float(self.fake_score),
            "calibrated_risk": float(self.calibrated_risk),
            "uncertainty": float(self.uncertainty),
            "evidence_coverage": float(self.evidence_coverage),
            "abstention_reason": self.abstention_reason,
        }


class TemperatureScaling(nn.Module):
    """
    Platt Scaling and Temperature Scaling for post-hoc calibration (Guo et al., 2017; Platt, 1999).
    Supports:
    - Pure temperature scaling: z_cal = z / T
    - Affine Platt scaling: z_cal = (z / T) + bias
    """

    def __init__(self, temperature: float = 1.0, bias: float = 0.0) -> None:
        super().__init__()
        import math
        self.log_temperature = nn.Parameter(torch.ones(1) * math.log(max(temperature, 0.01)))
        self.bias = nn.Parameter(torch.ones(1) * float(bias))

    @property
    def temperature(self) -> torch.Tensor:
        return torch.exp(self.log_temperature)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply Platt / temperature scaling to logits."""
        return (logits / self.temperature) + self.bias

    def calibrate(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        fit_bias: bool = True,
        max_iter: int = 200,
        lr: float = 0.05,
    ) -> tuple[float, float]:
        """
        Fit temperature and bias on calibration set.
        Returns (temperature, bias).
        """
        params = [self.log_temperature, self.bias] if fit_bias else [self.log_temperature]
        optimizer = torch.optim.LBFGS(params, lr=lr, max_iter=max_iter)
        criterion = nn.BCEWithLogitsLoss()

        logits = logits.float()
        labels = labels.float()

        def eval_fn():
            optimizer.zero_grad()
            scaled_logits = self.forward(logits)
            loss = criterion(scaled_logits, labels)
            loss.backward()
            return loss

        optimizer.step(eval_fn)
        temp = float(self.temperature.item())
        bias = float(self.bias.item())
        logger.info("Platt/Temperature calibration: T = %.4f, bias = %.4f", temp, bias)
        return temp, bias


class AbstractionLayer:
    """
    Rule-based abstention layer that decides when to return INSUFFICIENT_EVIDENCE.

    Conditions for abstention (any one triggers it):
    - evidence_coverage below minimum threshold
    - uncertainty above maximum threshold
    - branch disagreement above maximum threshold
    - face quality below minimum threshold
    - |score - 0.5| below margin (model is uncertain about the label)
    """

    def __init__(
        self,
        min_evidence_coverage: float = 0.3,
        max_uncertainty: float = 0.35,
        max_branch_disagreement: float = 0.4,
        min_face_quality: float = 0.2,
        score_margin: float = 0.15,
    ) -> None:
        self.min_evidence_coverage = min_evidence_coverage
        self.max_uncertainty = max_uncertainty
        self.max_branch_disagreement = max_branch_disagreement
        self.min_face_quality = min_face_quality
        self.score_margin = score_margin

    def should_abstain(
        self,
        calibrated_score: float,
        uncertainty: float,
        evidence_coverage: float,
        branch_disagreement: float,
        face_quality: float,
    ) -> tuple[bool, Optional[str]]:
        """
        Returns (should_abstain, reason).
        """
        if evidence_coverage < self.min_evidence_coverage:
            return True, f"evidence_coverage={evidence_coverage:.2f} < {self.min_evidence_coverage}"
        if uncertainty > self.max_uncertainty:
            return True, f"uncertainty={uncertainty:.2f} > {self.max_uncertainty}"
        if branch_disagreement > self.max_branch_disagreement:
            return True, f"branch_disagreement={branch_disagreement:.2f} > {self.max_branch_disagreement}"
        if face_quality < self.min_face_quality:
            return True, f"face_quality={face_quality:.2f} < {self.min_face_quality}"
        margin = abs(calibrated_score - 0.5)
        if margin < self.score_margin:
            return True, f"score_margin={margin:.2f} < {self.score_margin}"
        return False, None


class CalibrationLayer(nn.Module):
    """
    Full calibration + abstention layer for FORGE-Lite.

    Post-training: fit temperature on calibration set.
    At inference: apply temperature scaling + abstention rules.
    """

    def __init__(
        self,
        temperature_init: float = 1.0,
        abstention_config: Optional[dict] = None,
    ) -> None:
        super().__init__()
        self.temperature_scaling = TemperatureScaling(temperature_init)

        cfg = abstention_config or {}
        self.abstention_layer = AbstractionLayer(
            min_evidence_coverage=cfg.get("min_evidence_coverage", 0.3),
            max_uncertainty=cfg.get("max_uncertainty", 0.35),
            max_branch_disagreement=cfg.get("max_branch_disagreement", 0.4),
            min_face_quality=cfg.get("min_face_quality", 0.2),
            score_margin=cfg.get("score_margin", 0.15),
        )

    def forward(
        self,
        fake_logit: torch.Tensor,           # (B,)
        uncertainty: torch.Tensor,          # (B,)
        evidence_coverage: torch.Tensor,    # (B,)
        branch_disagreement: torch.Tensor,  # (B,)
        face_quality_composite: torch.Tensor,  # (B,) overall quality score [0,1]
    ) -> dict[str, torch.Tensor]:
        """
        Apply calibration and return decision-ready outputs.

        Returns:
            calibrated_score: (B,) [0, 1] probability of fake
            decision: list of Decision enum values
            abstained: (B,) bool tensor
        """
        calibrated_logit = self.temperature_scaling(fake_logit)
        calibrated_score = torch.sigmoid(calibrated_logit)

        B = calibrated_score.shape[0]
        decisions = []
        abstained = []

        for i in range(B):
            score = float(calibrated_score[i].item())
            unc = float(uncertainty[i].item())
            ev_cov = float(evidence_coverage[i].item())
            br_dis = float(branch_disagreement[i].item())
            fq = float(face_quality_composite[i].item())

            should_abs, reason = self.abstention_layer.should_abstain(
                score, unc, ev_cov, br_dis, fq
            )
            if should_abs:
                decisions.append(Decision.INSUFFICIENT_EVIDENCE)
            elif score >= 0.5:
                decisions.append(Decision.LIKELY_FAKE)
            else:
                decisions.append(Decision.LIKELY_REAL)
            abstained.append(should_abs)

        return {
            "calibrated_score": calibrated_score,
            "decisions": decisions,
            "abstained": torch.tensor(abstained, dtype=torch.bool, device=fake_logit.device),
        }

    def calibrate(
        self,
        val_logits: torch.Tensor,
        val_labels: torch.Tensor,
        fit_bias: bool = True,
    ) -> tuple[float, float]:
        """Fit temperature and bias on validation/calibration set."""
        return self.temperature_scaling.calibrate(val_logits, val_labels, fit_bias=fit_bias)

    def freeze_threshold(self, threshold: float, reason: str = "calibration_set") -> None:
        """
        Lock the operating threshold after calibration-set tuning.

        Per newimp.md P0: thresholds MUST be set outside the final test set.
        Once frozen, this threshold cannot be changed without creating a new
        model version requiring fresh validation.

        Args:
            threshold: the decision threshold to freeze (e.g. 0.5 or tuned value)
            reason: documentation of which dataset/partition was used to select it
        """
        self._frozen_threshold: float = threshold
        self._threshold_frozen: bool = True
        self._threshold_source: str = reason
        logger.info(
            "Threshold FROZEN at %.4f (source: %s). "
            "Do NOT tune further on test set.",
            threshold, reason
        )

    def get_frozen_threshold(self) -> float:
        """Return the frozen threshold. Raises if not yet frozen."""
        if not getattr(self, "_threshold_frozen", False):
            raise RuntimeError(
                "Threshold has not been frozen. Call freeze_threshold(t, reason) "
                "on the calibration partition before test-set evaluation."
            )
        return self._frozen_threshold

    @property
    def is_threshold_frozen(self) -> bool:
        """True if freeze_threshold() has been called."""
        return getattr(self, "_threshold_frozen", False)


# ─────────────────────────────────────────────────────────────────────────────
# Calibration metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_ece(
    probs: np.ndarray, labels: np.ndarray, n_bins: int = 15
) -> float:
    """
    Expected Calibration Error (ECE).
    Measures how well predicted probabilities match empirical frequencies.
    Target: ECE ≤ 0.05 (PRD requirement).
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(probs)

    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        in_bin = (probs >= lo) & (probs < hi)
        if not in_bin.any():
            continue
        bin_acc = labels[in_bin].mean()
        bin_conf = probs[in_bin].mean()
        ece += np.abs(bin_acc - bin_conf) * in_bin.sum() / n

    return float(ece)


def compute_brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    """Brier score: mean squared error between probabilities and binary labels."""
    return float(np.mean((probs - labels) ** 2))


def compute_coverage_risk_curve(
    probs: np.ndarray,
    labels: np.ndarray,
    uncertainties: np.ndarray,
    n_points: int = 20,
) -> dict[str, list[float]]:
    """
    Compute coverage-risk curve for abstention analysis.
    As we abstain more (lower coverage), risk (error rate) on remaining samples decreases.
    """
    thresholds = np.linspace(0, 1, n_points)
    coverages, risks = [], []

    for thresh in thresholds:
        # Keep samples with uncertainty < threshold
        keep = uncertainties < thresh
        if keep.sum() < 5:
            continue
        coverage = float(keep.sum() / len(keep))
        # Error rate on kept samples
        preds = (probs[keep] >= 0.5).astype(float)
        risk = float(np.mean(preds != labels[keep]))
        coverages.append(coverage)
        risks.append(risk)

    return {"coverage": coverages, "risk": risks}
