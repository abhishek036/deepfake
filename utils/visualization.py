"""
utils/visualization.py
Forensic Visualizations and Diagnostic Plotting for FORGE-Lite.

Provides:
- Heatmap overlay: Blends pixel-level manipulation prediction maps onto facial crops
- Frame Anomaly Timeline: Time-series curve of per-frame forgery likelihood
- Physiological rPPG Spectrum Plot: Power spectral density across cardiac band [42-210 BPM]
- Reliability Diagram: Calibration plot comparing confidence bins against observed accuracy
- ROC & DET Curves: Operational trade-off curves with annotated TPR@1%FPR
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Heatmap Overlay Generator
# ─────────────────────────────────────────────────────────────────────────────

def create_heatmap_overlay(
    image: np.ndarray,          # (H, W, 3) in [0, 255] or [0, 1]
    mask: np.ndarray,           # (H, W) in [0, 1]
    alpha: float = 0.5,
    colormap_name: str = "jet",
) -> np.ndarray:
    """
    Generates colorized heatmap overlay blended onto original face crop.
    Returns RGB image as uint8 array (H, W, 3).
    """
    # Normalize image to uint8 [0, 255]
    if image.dtype != np.uint8:
        img_np = np.clip(image * 255.0 if image.max() <= 1.0 else image, 0, 255).astype(np.uint8)
    else:
        img_np = image.copy()

    H, W = img_np.shape[:2]

    # Resize mask if shape differs
    if mask.shape[:2] != (H, W):
        pil_mask = Image.fromarray((mask * 255.0).astype(np.uint8)).resize((W, H), Image.Resampling.BILINEAR)
        norm_mask = np.array(pil_mask).astype(np.float32) / 255.0
    else:
        norm_mask = np.clip(mask, 0.0, 1.0).astype(np.float32)

    # Jet colormap approximation (Blue -> Cyan -> Yellow -> Red)
    r = np.clip(1.5 - np.abs(norm_mask * 4.0 - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(norm_mask * 4.0 - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(norm_mask * 4.0 - 1.0), 0.0, 1.0)
    heatmap = np.stack([r, g, b], axis=-1) * 255.0

    # Alpha blend where mask is active
    weight = (norm_mask[:, :, None] * alpha)
    blended = (1.0 - weight) * img_np.astype(np.float32) + weight * heatmap
    return np.clip(blended, 0, 255).astype(np.uint8)


def create_side_by_side_comparison(
    image: np.ndarray,
    mask: np.ndarray,
    gt_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Creates a composite side-by-side image:
    [Original Face Crop | Predicted Heatmap Overlay | Optional GT Mask]
    """
    overlay = create_heatmap_overlay(image, mask)
    panels = [image.astype(np.uint8), overlay]

    if gt_mask is not None:
        H, W = image.shape[:2]
        if gt_mask.shape[:2] != (H, W):
            gt_pil = Image.fromarray((gt_mask * 255.0).astype(np.uint8)).resize((W, H), Image.Resampling.NEAREST)
            gt_arr = np.array(gt_pil)
        else:
            gt_arr = (gt_mask * 255.0).astype(np.uint8)
        gt_rgb = np.stack([gt_arr, gt_arr, gt_arr], axis=-1)
        panels.append(gt_rgb)

    return np.concatenate(panels, axis=1)


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostic Plots (Matplotlib / Headless Safe)
# ─────────────────────────────────────────────────────────────────────────────

def plot_frame_anomaly_timeline(
    frame_scores: List[float],
    threshold: float = 0.5,
    save_path: Optional[Union[str, Path]] = None,
    title: str = "FORGE-Lite Frame Anomaly Timeline",
) -> Optional[np.ndarray]:
    """
    Plots per-frame fake likelihood across the video duration.
    Returns RGB image array or saves to disk.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 4), dpi=150)
    frames = np.arange(len(frame_scores))

    ax.plot(frames, frame_scores, color="#e74c3c", lw=2, label="Anomaly Probability")
    ax.axhline(threshold, color="#7f8c8d", linestyle="--", lw=1.5, label=f"Decision Threshold ({threshold:.2f})")

    ax.fill_between(frames, threshold, frame_scores, where=(np.array(frame_scores) >= threshold),
                    color="#e74c3c", alpha=0.3, interpolate=True)
    ax.fill_between(frames, 0, frame_scores, where=(np.array(frame_scores) < threshold),
                    color="#2ecc71", alpha=0.15, interpolate=True)

    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Video Frame Index")
    ax.set_ylabel("Synthetic Likelihood")
    ax.set_title(title)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right")

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(save_path), bbox_inches="tight")
        plt.close(fig)
        return None

    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return img


def plot_rppg_spectrum(
    frequencies_hz: np.ndarray,
    psd_values: np.ndarray,
    peak_bpm: float,
    snr_db: float,
    save_path: Optional[Union[str, Path]] = None,
) -> Optional[np.ndarray]:
    """
    Plots rPPG blood volume pulse power spectral density.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 3.5), dpi=150)
    bpm_axis = frequencies_hz * 60.0

    ax.plot(bpm_axis, psd_values, color="#9b59b6", lw=2, label="Power Spectral Density")
    ax.axvline(peak_bpm, color="#e67e22", linestyle="--", lw=1.5, label=f"Detected Peak: {peak_bpm:.1f} BPM (SNR: {snr_db:.1f} dB)")

    ax.set_xlim(42, 210)
    ax.set_xlabel("Heart Rate (Beats Per Minute)")
    ax.set_ylabel("Normalized Power")
    ax.set_title("rPPG Physiological Pulse Frequency Spectrum")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right")

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(save_path), bbox_inches="tight")
        plt.close(fig)
        return None

    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return img
