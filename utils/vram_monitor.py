"""
utils/vram_monitor.py
Real-time VRAM and Memory Profiler for RTX 4060 8GB Budget.

Enforces the strict 8GB VRAM hardware envelope specified in the FORGE-Lite PRD:
- Context manager `VRAMTracker` tracking step-by-step memory allocation and peak
- Proactive memory headroom checks (triggers warning at >90% of budget)
- Automatic PyTorch CUDA cache clearing and garbage collection
- Function decorator `@monitor_vram` to profile memory consumption per pass
"""

from __future__ import annotations

import gc
import logging
import time
from contextlib import AbstractContextManager
from functools import wraps
from typing import Any, Callable, Dict, Optional

import torch

logger = logging.getLogger(__name__)

# RTX 4060 8GB Hardware Budget (in GiB)
DEFAULT_VRAM_BUDGET_GB: float = 8.0
WARNING_THRESHOLD_RATIO: float = 0.90  # Alert at 7.2 GB


def get_gpu_memory_summary(device: Optional[torch.device] = None) -> Dict[str, float]:
    """
    Returns allocated, reserved, and peak CUDA memory in gigabytes (GiB).
    """
    if not torch.cuda.is_available():
        return {
            "device": "CPU",
            "allocated_gb": 0.0,
            "reserved_gb": 0.0,
            "max_allocated_gb": 0.0,
            "total_budget_gb": 0.0,
        }

    dev = device or torch.cuda.current_device()
    props = torch.cuda.get_device_properties(dev)
    total_gb = props.total_memory / (1024 ** 3)
    allocated_gb = torch.cuda.memory_allocated(dev) / (1024 ** 3)
    reserved_gb = torch.cuda.memory_reserved(dev) / (1024 ** 3)
    max_allocated_gb = torch.cuda.max_memory_allocated(dev) / (1024 ** 3)

    return {
        "device": props.name,
        "allocated_gb": allocated_gb,
        "reserved_gb": reserved_gb,
        "max_allocated_gb": max_allocated_gb,
        "total_budget_gb": total_gb,
    }


def empty_cuda_cache() -> None:
    """
    Explicitly triggers Python garbage collection and clears PyTorch CUDA memory cache.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def check_vram_headroom(
    budget_gb: float = DEFAULT_VRAM_BUDGET_GB,
    threshold: float = WARNING_THRESHOLD_RATIO,
) -> bool:
    """
    Checks whether current VRAM usage is safe. Returns True if headroom is sufficient.
    Logs warning if usage exceeds threshold.
    """
    if not torch.cuda.is_available():
        return True

    mem = get_gpu_memory_summary()
    allocated = mem["allocated_gb"]
    limit = budget_gb * threshold

    if allocated > limit:
        logger.warning(
            "⚠️ VRAM CRITICAL: %.2f GiB / %.2f GiB allocated (threshold: %.2f GiB). Running cache clear.",
            allocated, budget_gb, limit,
        )
        empty_cuda_cache()
        return False
    return True


class VRAMTracker(AbstractContextManager):
    """
    Context manager for measuring and logging GPU memory deltas and peaks.

    Usage:
        with VRAMTracker("DINOv2 Forward Pass"):
            outputs = model(inputs)
    """

    def __init__(self, step_name: str = "Operation", budget_gb: float = DEFAULT_VRAM_BUDGET_GB) -> None:
        self.step_name = step_name
        self.budget_gb = budget_gb
        self.start_time: float = 0.0
        self.start_allocated: float = 0.0

    def __enter__(self) -> "VRAMTracker":
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            self.start_allocated = torch.cuda.memory_allocated() / (1024 ** 3)
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        elapsed = time.perf_counter() - self.start_time
        if torch.cuda.is_available():
            end_allocated = torch.cuda.memory_allocated() / (1024 ** 3)
            peak_allocated = torch.cuda.max_memory_allocated() / (1024 ** 3)
            delta = end_allocated - self.start_allocated

            logger.info(
                "[%s] Completed in %.3fs | Peak VRAM: %.2f GiB | Delta: %+.2f GiB | Total Budget: %.1f GiB",
                self.step_name, elapsed, peak_allocated, delta, self.budget_gb,
            )

            if peak_allocated > self.budget_gb * WARNING_THRESHOLD_RATIO:
                logger.warning(
                    "⚠️ Step [%s] exceeded 90%% VRAM budget (Peak: %.2f GiB / %.1f GiB).",
                    self.step_name, peak_allocated, self.budget_gb,
                )
        else:
            logger.info("[%s] Completed in %.3fs (CPU execution)", self.step_name, elapsed)


def monitor_vram(step_name: Optional[str] = None) -> Callable:
    """
    Decorator to monitor peak VRAM for a function.
    """
    def decorator(func: Callable) -> Callable:
        name = step_name or func.__name__

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with VRAMTracker(name):
                return func(*args, **kwargs)
        return wrapper
    return decorator
