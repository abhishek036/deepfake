"""
models/rppg_branch.py
Quality-gated rPPG (remote photoplethysmography) branch for FORGE-Lite.

Measures whether a plausible physiological heartbeat signal is present.
rPPG is an AUXILIARY signal — never an authenticity oracle.

PRD Critical Rule:
  "Never output 'no rPPG detected, therefore fake.'"
  "rPPG is available only if all quality gates pass."

Quality gate conditions (all must be met):
  - Track duration ≥ 8 seconds
  - Face area ratio ≥ threshold
  - Track continuity ≥ threshold
  - Motion contamination ≤ threshold

Output: 16-dim rPPG feature vector + availability mask.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# 6 facial ROIs for rPPG extraction
ROI_NAMES = ["forehead", "left_cheek", "right_cheek", "nose_bridge", "chin", "periorbital"]
NUM_ROIS = 6

# BPM valid range: 42–210 BPM
BPM_MIN = 42
BPM_MAX = 210
HZ_LOW = 0.7    # 42/60
HZ_HIGH = 3.5   # 210/60


@dataclass
class RPPGResult:
    """Result of rPPG analysis for one face track."""
    available: bool                 # Did quality gates pass?
    quality_reason: str             # Why unavailable (if applicable)
    snr: float                      # Signal-to-noise ratio in cardiac band
    dominant_freq_hz: float         # Dominant frequency in Hz
    dominant_bpm: float             # Converted to BPM
    bpm_plausible: bool             # Is BPM in 42-210 range?
    band_power: float               # Total power in 0.7-3.5 Hz band
    cross_roi_coherence: float      # Mean coherence between ROI signals
    temporal_stability: float       # Stability of dominant frequency over time
    illumination_variation: float   # Estimated illumination variability
    motion_contamination: float     # Estimated motion artifact contamination
    face_visibility: float          # Overall face visibility score

    def to_feature_vector(self) -> np.ndarray:
        """16-dimensional rPPG feature vector for fusion head."""
        if not self.available:
            # Return zeros with availability=0 for fusion masking
            return np.zeros(16, dtype=np.float32)
        return np.array([
            float(self.snr) / 20.0,                          # normalized SNR
            self.dominant_freq_hz / HZ_HIGH,                  # normalized frequency
            float(self.bpm_plausible),                        # plausibility flag
            self.band_power,                                   # band power
            self.cross_roi_coherence,                         # ROI coherence
            self.temporal_stability,                          # frequency stability
            1.0 - self.illumination_variation,                 # inverse illumination var
            1.0 - self.motion_contamination,                   # inverse motion
            self.face_visibility,                              # face visibility
            float(self.dominant_bpm) / BPM_MAX,               # normalized BPM
            # Additional cross-ROI features (pad to 16)
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ], dtype=np.float32)

    def to_dict(self) -> dict:
        return {
            "rppg_available": self.available,
            "quality_reason": self.quality_reason,
            "snr": float(self.snr),
            "dominant_freq_hz": float(self.dominant_freq_hz),
            "dominant_bpm": float(self.dominant_bpm),
            "bpm_plausible": self.bpm_plausible,
            "band_power": float(self.band_power),
            "cross_roi_coherence": float(self.cross_roi_coherence),
            "temporal_stability": float(self.temporal_stability),
        }


class RPPGBranch:
    """
    Quality-gated rPPG feature extractor.

    This is a signal processing module (not a nn.Module) because MediaPipe
    and scipy are used. The output feature vector is fed to the fusion head
    via a small learnable embedding (RPPGFeatureEncoder).

    Usage:
        branch = RPPGBranch(cfg)
        result = branch.extract(face_crops, fps, track_quality)
        feature_vec = result.to_feature_vector()
    """

    def __init__(
        self,
        min_duration_sec: float = 8.0,
        min_face_area_ratio: float = 0.05,
        min_track_continuity: float = 0.6,
        max_motion_contamination: float = 0.4,
        min_snr: float = 3.0,
        min_coherence: float = 0.4,
        use_mediapipe: bool = True,
    ) -> None:
        self.min_duration_sec = min_duration_sec
        self.min_face_area_ratio = min_face_area_ratio
        self.min_track_continuity = min_track_continuity
        self.max_motion_contamination = max_motion_contamination
        self.min_snr = min_snr
        self.min_coherence = min_coherence
        self.use_mediapipe = use_mediapipe
        self._mp_face_mesh = None

    def check_quality_gate(
        self,
        track_duration_sec: float,
        face_area_ratio: float,
        track_continuity: float,
        motion_contamination: float,
    ) -> tuple[bool, str]:
        """
        Check all quality conditions. Returns (gate_passed, reason).
        ALL conditions must pass for rPPG to be used.
        """
        if track_duration_sec < self.min_duration_sec:
            return False, f"duration={track_duration_sec:.1f}s < {self.min_duration_sec}s"
        if face_area_ratio < self.min_face_area_ratio:
            return False, f"face_area={face_area_ratio:.3f} < {self.min_face_area_ratio}"
        if track_continuity < self.min_track_continuity:
            return False, f"continuity={track_continuity:.2f} < {self.min_track_continuity}"
        if motion_contamination > self.max_motion_contamination:
            return False, f"motion={motion_contamination:.2f} > {self.max_motion_contamination}"
        return True, "OK"

    def extract(
        self,
        face_crops_bgr: list[np.ndarray],   # T frames of 224×224 BGR
        fps: float,
        track_duration_sec: float,
        face_area_ratio: float = 0.1,
        track_continuity: float = 1.0,
        motion_contamination: float = 0.0,
    ) -> RPPGResult:
        """
        Extract rPPG features from a sequence of aligned face crops.

        Returns RPPGResult with availability flag.
        If gate fails, returns RPPGResult(available=False, ...).
        """
        # Check quality gate
        gate_passed, reason = self.check_quality_gate(
            track_duration_sec, face_area_ratio, track_continuity, motion_contamination
        )
        if not gate_passed:
            return RPPGResult(
                available=False, quality_reason=reason,
                snr=0.0, dominant_freq_hz=0.0, dominant_bpm=0.0, bpm_plausible=False,
                band_power=0.0, cross_roi_coherence=0.0, temporal_stability=0.0,
                illumination_variation=0.0, motion_contamination=motion_contamination,
                face_visibility=face_area_ratio,
            )

        # Extract ROI color signals
        roi_signals = self._extract_roi_signals(face_crops_bgr)  # (6, T, 3) BGR
        if roi_signals is None:
            return RPPGResult(
                available=False, quality_reason="roi_extraction_failed",
                snr=0.0, dominant_freq_hz=0.0, dominant_bpm=0.0, bpm_plausible=False,
                band_power=0.0, cross_roi_coherence=0.0, temporal_stability=0.0,
                illumination_variation=0.0, motion_contamination=0.0, face_visibility=0.0,
            )

        # Compute green channel average per ROI (green channel has best rPPG SNR)
        green_signals = roi_signals[:, :, 1]  # (6, T) — green channel
        # Detrend each ROI signal
        detrended = self._detrend(green_signals)

        # Bandpass filter 0.7–3.5 Hz
        filtered = self._bandpass_filter(detrended, fps, HZ_LOW, HZ_HIGH)

        # Compute SNR and dominant frequency for each ROI
        all_snrs = []
        all_freqs = []
        for roi_sig in filtered:
            snr, dom_freq = self._compute_snr_and_freq(roi_sig, fps)
            all_snrs.append(snr)
            all_freqs.append(dom_freq)

        mean_snr = float(np.mean(all_snrs))
        dominant_freq = float(np.median(all_freqs))
        dominant_bpm = dominant_freq * 60.0
        bpm_plausible = BPM_MIN <= dominant_bpm <= BPM_MAX

        # Cross-ROI coherence (correlation between ROI signals)
        coherence = self._cross_roi_coherence(filtered)

        # Band power
        band_power = float(np.mean([
            self._band_power(sig, fps, HZ_LOW, HZ_HIGH) for sig in filtered
        ]))

        # Temporal stability: variance of dominant freq over time windows
        stability = self._temporal_stability(filtered[0], fps)

        # Illumination variation (brightness std across frames)
        illum_var = float(np.std([
            np.mean(frame.astype(np.float32)) / 255.0
            for frame in face_crops_bgr
        ])) * 10.0

        return RPPGResult(
            available=True,
            quality_reason="OK",
            snr=mean_snr,
            dominant_freq_hz=dominant_freq,
            dominant_bpm=dominant_bpm,
            bpm_plausible=bpm_plausible,
            band_power=min(band_power, 1.0),
            cross_roi_coherence=coherence,
            temporal_stability=stability,
            illumination_variation=min(illum_var, 1.0),
            motion_contamination=motion_contamination,
            face_visibility=face_area_ratio,
        )

    # ── Signal processing helpers ─────────────────────────────────────────────

    def _extract_roi_signals(
        self, face_crops: list[np.ndarray]
    ) -> Optional[np.ndarray]:
        """Extract mean RGB color from 6 facial ROIs across frames. Returns (6, T, 3)."""
        T = len(face_crops)
        H = W = 224

        # Fixed ROI definitions (normalized [0,1] coordinates for 224×224 crops)
        # (y_start, y_end, x_start, x_end)
        rois = {
            "forehead":    (0.05, 0.25, 0.25, 0.75),
            "left_cheek":  (0.40, 0.65, 0.05, 0.35),
            "right_cheek": (0.40, 0.65, 0.65, 0.95),
            "nose_bridge": (0.25, 0.55, 0.40, 0.60),
            "chin":        (0.70, 0.90, 0.30, 0.70),
            "periorbital": (0.20, 0.40, 0.15, 0.85),
        }

        signals = np.zeros((6, T, 3), dtype=np.float32)
        for t, frame in enumerate(face_crops):
            if frame is None:
                continue
            for roi_idx, (roi_name, (y0n, y1n, x0n, x1n)) in enumerate(rois.items()):
                y0, y1 = int(y0n * H), int(y1n * H)
                x0, x1 = int(x0n * W), int(x1n * W)
                patch = frame[y0:y1, x0:x1]
                if patch.size > 0:
                    signals[roi_idx, t] = patch.mean(axis=(0, 1))

        return signals

    @staticmethod
    def _detrend(signals: np.ndarray) -> np.ndarray:
        """Remove linear trend from each ROI signal."""
        detrended = np.zeros_like(signals)
        T = signals.shape[1]
        t = np.arange(T, dtype=np.float32)
        for i, sig in enumerate(signals):
            coeffs = np.polyfit(t, sig, 1)
            trend = np.polyval(coeffs, t)
            detrended[i] = sig - trend
        return detrended

    @staticmethod
    def _bandpass_filter(
        signals: np.ndarray, fps: float, low: float, high: float
    ) -> np.ndarray:
        """Butterworth bandpass filter 0.7–3.5 Hz."""
        try:
            from scipy.signal import butter, filtfilt
            nyq = fps / 2.0
            b, a = butter(3, [low / nyq, high / nyq], btype="band")
            return np.array([filtfilt(b, a, sig) for sig in signals])
        except Exception:
            return signals  # fallback: no filter

    @staticmethod
    def _compute_snr_and_freq(signal: np.ndarray, fps: float) -> tuple[float, float]:
        """Compute SNR in cardiac band and dominant frequency."""
        N = len(signal)
        if N < 8:
            return 0.0, 0.0
        fft_mag = np.abs(np.fft.rfft(signal)) ** 2
        freqs = np.fft.rfftfreq(N, d=1.0 / fps)
        cardiac_mask = (freqs >= HZ_LOW) & (freqs <= HZ_HIGH)
        if not cardiac_mask.any():
            return 0.0, 0.0
        total_power = fft_mag.sum() + 1e-12
        band_power = fft_mag[cardiac_mask].sum()
        noise_power = total_power - band_power + 1e-12
        snr = 10 * np.log10(band_power / noise_power + 1e-12)
        dom_freq_idx = fft_mag[cardiac_mask].argmax()
        dom_freq = freqs[cardiac_mask][dom_freq_idx]
        return float(snr), float(dom_freq)

    @staticmethod
    def _cross_roi_coherence(signals: np.ndarray) -> float:
        """Compute mean pairwise correlation across ROI signals."""
        n = len(signals)
        if n < 2:
            return 0.0
        corrs = []
        for i in range(n):
            for j in range(i + 1, n):
                if signals[i].std() > 0 and signals[j].std() > 0:
                    c = float(np.corrcoef(signals[i], signals[j])[0, 1])
                    corrs.append(max(0.0, c))
        return float(np.mean(corrs)) if corrs else 0.0

    @staticmethod
    def _band_power(signal: np.ndarray, fps: float, low: float, high: float) -> float:
        """Fraction of total power in the cardiac frequency band."""
        N = len(signal)
        if N < 4:
            return 0.0
        fft_mag = np.abs(np.fft.rfft(signal)) ** 2
        freqs = np.fft.rfftfreq(N, d=1.0 / fps)
        cardiac = (freqs >= low) & (freqs <= high)
        total = fft_mag.sum() + 1e-12
        return float(fft_mag[cardiac].sum() / total)

    @staticmethod
    def _temporal_stability(signal: np.ndarray, fps: float, window_sec: float = 4.0) -> float:
        """
        Compute temporal stability of the dominant frequency.
        High stability = consistent heartbeat = more likely real.
        """
        N = len(signal)
        window = max(8, int(window_sec * fps))
        if N < window:
            _, dom_freq = RPPGBranch._compute_snr_and_freq(signal, fps)
            return 1.0 if dom_freq > 0 else 0.0

        freqs = []
        for start in range(0, N - window + 1, window // 2):
            seg = signal[start: start + window]
            _, df = RPPGBranch._compute_snr_and_freq(seg, fps)
            if df > 0:
                freqs.append(df)

        if len(freqs) < 2:
            return 0.5
        # Low std of dominant freq across windows → high stability
        stability = 1.0 - min(float(np.std(freqs)) / HZ_HIGH, 1.0)
        return stability


class RPPGFeatureEncoder(nn.Module):
    """
    Small learnable encoder that projects the 16-dim rPPG feature vector
    into a higher-dimensional space for the fusion head.

    Also handles the availability mask — outputs zeros when unavailable.
    """

    def __init__(self, input_dim: int = 16, output_dim: int = 32) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, output_dim),
        )

    def forward(
        self,
        rppg_features: torch.Tensor,    # (B, 16)
        available_mask: torch.Tensor,   # (B,) bool
    ) -> torch.Tensor:
        """
        Returns (B, output_dim) — zeros for unavailable samples.
        """
        encoded = self.encoder(rppg_features)  # (B, output_dim)
        # Zero out unavailable samples
        mask = available_mask.float().unsqueeze(-1)  # (B, 1)
        return encoded * mask
