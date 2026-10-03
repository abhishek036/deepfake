"""FORGE-Lite evaluation suite: video-level, cross-domain, robustness, localization, calibration, and slice analysis."""

from evaluation.evaluate_video import ForgeEvaluator, compute_video_auc, compute_tpr_at_fpr, compute_eer
from evaluation.evaluate_cross_domain import CrossDomainEvaluator
from evaluation.evaluate_robustness import RobustnessEvaluator
from evaluation.evaluate_localization import LocalizationEvaluator, compute_mask_iou_dice
from evaluation.evaluate_calibration import CalibrationEvaluator, compute_detailed_ece
from evaluation.slice_analysis import SliceAnalyzer
from evaluation.bootstrap_ci import compute_all_bootstrap_ci
from evaluation.ablation_runner import AblationRunner

__all__ = [
    "ForgeEvaluator",
    "compute_video_auc",
    "compute_tpr_at_fpr",
    "compute_eer",
    "CrossDomainEvaluator",
    "RobustnessEvaluator",
    "LocalizationEvaluator",
    "compute_mask_iou_dice",
    "CalibrationEvaluator",
    "compute_detailed_ece",
    "SliceAnalyzer",
    "compute_all_bootstrap_ci",
    "AblationRunner",
]
