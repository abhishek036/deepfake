"""
utils/logging_utils.py
Structured Logging and Experiment Tracking Utilities for FORGE-Lite.

Provides:
- Colorized console logging with ISO-8601 timestamps and source line tagging
- Multi-handler configuration (Console + Rotating File Log)
- JSON Lines (JSONL) metrics writer for experiment tracking and metric curves
- Safe scalar converter for PyTorch tensors and NumPy floats/ints
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch


# ─────────────────────────────────────────────────────────────────────────────
# ANSI Color Formatting for Console
# ─────────────────────────────────────────────────────────────────────────────

class ColoredFormatter(logging.Formatter):
    """Console formatter with distinct ANSI colors per log level."""

    COLORS = {
        logging.DEBUG: "\033[36m",     # Cyan
        logging.INFO: "\033[32m",      # Green
        logging.WARNING: "\033[33m",   # Yellow
        logging.ERROR: "\033[31m",     # Red
        logging.CRITICAL: "\033[35;1m",# Bold Magenta
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelno, self.RESET)
        record.levelname = f"{color}{record.levelname:<8}{self.RESET}"
        return super().format(record)


def setup_logger(
    name: str = "forge_lite",
    log_file: Optional[Union[str, Path]] = None,
    level: int = logging.INFO,
    format_str: Optional[str] = None,
) -> logging.Logger:
    """
    Configures and returns a structured logger with console and optional file handlers.
    """
    if format_str is None:
        format_str = "%(asctime)s | %(levelname)s | %(name)s:%(lineno)d | %(message)s"

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    # Avoid duplicate handlers if setup_logger called repeatedly
    if logger.handlers:
        return logger

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(ColoredFormatter(format_str, datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(console_handler)

    # File Handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(str(log_path), encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter(format_str, datefmt="%Y-%m-%d %H:%M:%S"))
        logger.addHandler(file_handler)

    return logger


# ─────────────────────────────────────────────────────────────────────────────
# Safe Metric Serialization & JSONL Metrics Logger
# ─────────────────────────────────────────────────────────────────────────────

def to_json_serializable(val: Any) -> Any:
    """Converts PyTorch tensors, NumPy arrays, and floats to JSON-safe primitives."""
    if isinstance(val, (torch.Tensor,)):
        return val.detach().cpu().item() if val.numel() == 1 else val.detach().cpu().tolist()
    if isinstance(val, (np.ndarray,)):
        return val.item() if val.size == 1 else val.tolist()
    if isinstance(val, (np.floating, float)):
        return float(val) if not np.isnan(val) else None
    if isinstance(val, (np.integer, int)):
        return int(val)
    if isinstance(val, dict):
        return {k: to_json_serializable(v) for k, v in val.items()}
    if isinstance(val, (list, tuple)):
        return [to_json_serializable(v) for v in val]
    return val


class JSONLMetricLogger:
    """
    Appends training and evaluation metrics step-by-step to a JSONL file.

    Usage:
        metrics_logger = JSONLMetricLogger("reports/train_metrics.jsonl")
        metrics_logger.log({"epoch": 1, "loss": 0.42, "lr": 1e-4})
    """

    def __init__(self, file_path: Union[str, Path]) -> None:
        self.file_path = Path(file_path)
        self.file_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, step_metrics: Dict[str, Any], flush: bool = True) -> None:
        """Writes one JSON record with timestamp."""
        payload = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            **to_json_serializable(step_metrics),
        }
        with open(self.file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")
            if flush:
                f.flush()
