"""FORGE-Lite utilities: VRAM profiler, structured logging, and forensic visualization."""

from utils.vram_monitor import (
    VRAMTracker,
    check_vram_headroom,
    empty_cuda_cache,
    get_gpu_memory_summary,
    monitor_vram,
)
from utils.logging_utils import JSONLMetricLogger, setup_logger, to_json_serializable
from utils.visualization import (
    create_heatmap_overlay,
    create_side_by_side_comparison,
    plot_frame_anomaly_timeline,
    plot_rppg_spectrum,
)

__all__ = [
    "VRAMTracker",
    "check_vram_headroom",
    "empty_cuda_cache",
    "get_gpu_memory_summary",
    "monitor_vram",
    "JSONLMetricLogger",
    "setup_logger",
    "to_json_serializable",
    "create_heatmap_overlay",
    "create_side_by_side_comparison",
    "plot_frame_anomaly_timeline",
    "plot_rppg_spectrum",
]
