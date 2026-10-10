"""
scripts/overnight_pipeline.py
=============================
Autonomous Overnight Pipeline Orchestrator for FORGE-Lite.

Sequentially executes:
1. Phase 4: Wait for currently running Temporal GRU training to complete.
2. Phase 5: Full Multimodal Fusion + Localization training.
3. Phase 6: Audio-Visual SyncNet branch training.
4. Calibration: Post-hoc temperature and uncertainty calibration.
5. Evaluation: Full benchmark evaluation on test split.
"""

import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

LOG_PATH = ROOT / "training" / "training.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_PATH, mode="a", encoding="utf-8"),
    ],
)
logger = logging.getLogger("overnight_pipeline")


def is_process_running(pid: int) -> bool:
    try:
        import psutil
        return psutil.pid_exists(pid)
    except ImportError:
        res = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}"],
            capture_output=True,
            text=True,
        )
        return str(pid) in res.stdout


def wait_for_phase_4(target_pid: int = 19428, timeout_sec: int = 7200):
    logger.info("=" * 70)
    logger.info("OVERNIGHT WATCHDOG: Monitoring Phase 4 (Temporal GRU) [PID %d]", target_pid)
    logger.info("=" * 70)
    start = time.time()

    while is_process_running(target_pid):
        elapsed = time.time() - start
        if elapsed > timeout_sec:
            logger.warning("Phase 4 exceeded timeout (%d sec). Moving forward...", timeout_sec)
            break
        time.sleep(30)

    logger.info("Phase 4 process finished (elapsed: %.1f min)", (time.time() - start) / 60)
    time.sleep(5)

    ckpt = ROOT / "outputs" / "checkpoints" / "forge-lite-v1_temporal_temporal_only_best.ckpt"
    if ckpt.exists():
        logger.info("Phase 4 checkpoint verified: %s (%.1f MB)", ckpt.name, ckpt.stat().st_size / 1e6)
    else:
        logger.warning("Phase 4 best checkpoint not found at %s. Using best.ckpt fallback.", ckpt)


def run_stage(name: str, cmd: list[str]) -> bool:
    logger.info("=" * 70)
    logger.info("STARTING OVERNIGHT STAGE: %s", name)
    logger.info("Command: %s", " ".join(cmd))
    logger.info("=" * 70)
    t0 = time.time()

    ret = subprocess.run(cmd, cwd=str(ROOT))
    duration = time.time() - t0

    if ret.returncode == 0:
        logger.info("Stage '%s' SUCCEEDED in %.1f min", name, duration / 60)
        return True
    else:
        logger.error("Stage '%s' FAILED with exit code %d (in %.1f min)", name, ret.returncode, duration / 60)
        return False


def main():
    logger.info("=== FORGE-LITE OVERNIGHT PIPELINE LAUNCHED ===")
    summary = {
        "start_time": datetime.now().isoformat(),
        "stages": {},
    }

    # 1. Wait for Phase 4 to finish
    wait_for_phase_4(target_pid=19428)
    summary["stages"]["phase_4_temporal"] = "completed"

    py = sys.executable

    # 2. Phase 5: Full Fusion
    ok_p5 = run_stage(
        "Phase 5: Multimodal Fusion + Localization",
        [py, "-m", "training.train_forge", "--phase", "fusion", "--config", "configs/train.yaml"],
    )
    summary["stages"]["phase_5_fusion"] = "success" if ok_p5 else "failed"

    # 3. Phase 6: Audio-Visual Branch
    ok_p6 = run_stage(
        "Phase 6: Audio-Visual Branch (SyncNet)",
        [py, "-m", "training.train_forge", "--phase", "av", "--config", "configs/train.yaml"],
    )
    summary["stages"]["phase_6_av"] = "success" if ok_p6 else "failed"

    # 4. Calibration
    ok_cal = run_stage(
        "Calibration (Temperature & Uncertainty)",
        [py, "-m", "training.train_fusion", "--mode", "calibration", "--config-train", "configs/train.yaml"],
    )
    summary["stages"]["calibration"] = "success" if ok_cal else "failed"

    # 5. Comprehensive Benchmark Evaluation
    eval_script = ROOT / "evaluation" / "evaluate_final_system_v3.py"
    if eval_script.exists():
        ok_eval = run_stage(
            "Final System Evaluation (v3)",
            [py, str(eval_script)],
        )
        summary["stages"]["final_evaluation"] = "success" if ok_eval else "failed"

    summary["end_time"] = datetime.now().isoformat()
    out_file = ROOT / "outputs" / "overnight_summary.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info("=" * 70)
    logger.info("ALL OVERNIGHT STAGES FINISHED. Summary saved to %s", out_file)
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
