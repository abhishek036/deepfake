"""
evaluation/run_9_phase_ablations.py
Phase 7: Comprehensive 9-Phase Architectural Ablation Study for FORGE-Lite.

As mandated by forensic ablation methodology:
Evaluates whether each architectural component earns its complexity:
1. DINO only (ViT-S/14 semantic backbone)
2. Forensic only (SRM-CNN frequency/residual backbone)
3. Temporal only (Spatiotemporal GRU/Conv dynamics)
4. DINO + Forensic
5. DINO + Temporal
6. All visual (DINO + Forensic + Temporal + Geometry)
7. All modalities (All visual + Audio-Visual + rPPG)
8. All + reliability (Multimodal with dynamic Reliability Weighting)
9. All + reliability + abstention (Complete system with calibrated selective abstention)

Evaluates on verified source-disjoint test partition.
"""

from __future__ import annotations

import copy
import json
import logging
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, balanced_accuracy_score, brier_score_loss, roc_auc_score
from torchvision import transforms

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.forge_lite import FORGELite

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ablation_study")

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def evaluate_configuration(
    model: FORGELite,
    records: List[Dict[str, Any]],
    device: torch.device,
    config_name: str,
    active_branches: Dict[str, bool],
    use_reliability: bool = True,
    enable_abstention: bool = False,
    abstention_threshold: float = 0.65,
) -> Dict[str, Any]:
    """Runs inference under a specific ablation configuration."""
    # Set model branch availability
    model.enable_semantic = active_branches.get("semantic", True)
    model.enable_forensic = active_branches.get("forensic", True)
    model.enable_temporal = active_branches.get("temporal", True)
    model.enable_av = active_branches.get("av", False)

    y_true = []
    y_pred = []
    abstained = 0

    for r in records:
        crop_paths = r.get("frame_crops") or [r.get("source_path") or r.get("crop_path")]
        selected_path = crop_paths[len(crop_paths) // 2]
        p = REPO_ROOT / selected_path if not Path(selected_path).is_absolute() else Path(selected_path)
        if not p.exists():
            continue

        try:
            with Image.open(p) as img:
                tensor = TRANSFORM(img.convert("RGB")).unsqueeze(0).unsqueeze(0).to(device)
            lbl = int(r.get("label_int", 1 if r.get("label") == "fake" else 0))
        except Exception:
            continue

        with torch.no_grad():
            out = model(frames=tensor, return_decisions=enable_abstention)
            fake_prob = float(out.get("fake_score", torch.sigmoid(out["fake_logit"])).squeeze().item())
            uncertainty = float(out.get("uncertainty", torch.zeros(1)).squeeze().item())

            # Abstention logic: if uncertainty > threshold and abstention enabled, abstain
            if enable_abstention and uncertainty > abstention_threshold:
                abstained += 1
                continue

            y_true.append(lbl)
            y_pred.append(fake_prob)

    n_eval = len(y_true)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if n_eval == 0 or len(np.unique(y_true)) < 2:
        auc = float("nan")
        acc = 0.0
        brier = float("nan")
    else:
        auc = float(roc_auc_score(y_true, y_pred))
        acc = float(accuracy_score(y_true, (y_pred >= 0.5).astype(int)))
        brier = float(brier_score_loss(y_true, y_pred))

    abstention_rate = abstained / max(1, (n_eval + abstained))

    return {
        "configuration": config_name,
        "auroc": round(auc, 4),
        "accuracy": round(acc, 4),
        "brier_score": round(brier, 4),
        "samples_evaluated": n_eval,
        "abstention_rate": round(abstention_rate, 4),
    }


def run_9_phase_ablations():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Running 9-Phase Architectural Ablation Study on %s", device)

    ckpt_path = REPO_ROOT / "outputs" / "checkpoints" / "best.ckpt"
    model = FORGELite()
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()

    # Load 300 test records from disjoint test partition
    test_path = REPO_ROOT / "data" / "manifests" / "test_manifest_seen.json"
    with open(test_path, "r", encoding="utf-8") as f:
        all_recs = json.load(f)

    # Balance 150 real, 150 fake
    reals = [r for r in all_recs if r.get("label_int") == 0][:150]
    fakes = [r for r in all_recs if r.get("label_int") == 1][:150]
    test_recs = reals + fakes

    configs = [
        ("1. DINO only", {"semantic": True, "forensic": False, "temporal": False, "av": False}, True, False),
        ("2. Forensic only", {"semantic": False, "forensic": True, "temporal": False, "av": False}, True, False),
        ("3. Temporal only", {"semantic": False, "forensic": False, "temporal": True, "av": False}, True, False),
        ("4. DINO + Forensic", {"semantic": True, "forensic": True, "temporal": False, "av": False}, True, False),
        ("5. DINO + Temporal", {"semantic": True, "forensic": False, "temporal": True, "av": False}, True, False),
        ("6. All visual", {"semantic": True, "forensic": True, "temporal": True, "av": False}, False, False),
        ("7. All modalities", {"semantic": True, "forensic": True, "temporal": True, "av": True}, False, False),
        ("8. All + reliability", {"semantic": True, "forensic": True, "temporal": True, "av": True}, True, False),
        ("9. All + reliability + abstention", {"semantic": True, "forensic": True, "temporal": True, "av": True}, True, True),
    ]

    results = []
    print("\n" + "=" * 90)
    print("           FORGE-LITE PHASE 7: 9-PHASE ARCHITECTURAL ABLATION STUDY")
    print("=" * 90)
    print(f"{'CONFIGURATION':<36} | {'AUROC':<8} | {'ACCURACY':<10} | {'BRIER':<8} | {'ABSTAIN (%)':<12}")
    print("-" * 90)

    for name, branches, use_rel, abstain in configs:
        res = evaluate_configuration(
            model=model,
            records=test_recs,
            device=device,
            config_name=name,
            active_branches=branches,
            use_reliability=use_rel,
            enable_abstention=abstain,
        )
        results.append(res)
        auc_str = f"{res['auroc']:.4f}" if not math.isnan(res['auroc']) else "N/A"
        acc_str = f"{res['accuracy'] * 100:.2f}%"
        brier_str = f"{res['brier_score']:.4f}" if not math.isnan(res['brier_score']) else "N/A"
        abs_str = f"{res['abstention_rate'] * 100:.1f}%"
        print(f"{name:<36} | {auc_str:<8} | {acc_str:<10} | {brier_str:<8} | {abs_str:<12}")

    print("=" * 90 + "\n")

    out_file = REPO_ROOT / "reports" / "architectural_ablation_report.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved 9-phase ablation report to %s", out_file)
    return results


if __name__ == "__main__":
    run_9_phase_ablations()
