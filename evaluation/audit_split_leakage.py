"""
evaluation/audit_split_leakage.py
Phase 1: Split & Data Leakage Audit for FORGE-Lite.

1. Audits legacy split files and manifests for source identity leakage.
2. Quantifies exact cross-partition contamination (train vs val vs test).
3. Builds verified, strictly source-disjoint partitions:
   - data/manifests/train_manifest_disjoint.json (Model training)
   - data/manifests/calibration_manifest.json    (Post-hoc calibration & threshold tuning)
   - data/manifests/test_manifest_seen.json       (Untouched test evaluation for seen domains)
   - data/manifests/test_manifest_unseen.json     (Untouched test evaluation for unseen domains)
4. Programmatically proves zero identity overlap across partitions.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("split_audit")


def extract_identities(record: Dict[str, Any]) -> Set[str]:
    """
    Extracts all participating source identities from a video record.
    For FaceForensics++ (e.g. 000_003.mp4): both actor 000 and actor 003.
    For Celeb-DF (e.g. id0_id1_0000.mp4): both id0 and id1.
    For single images: record_id or image filename stem.
    """
    src_path = record.get("source_path") or record.get("video_id") or ""
    stem = Path(src_path).stem
    identities = set()

    dataset = record.get("dataset", "").lower()
    if "faceforensics" in dataset or "ff" in dataset:
        parts = stem.split("_")
        for p in parts:
            if p.isdigit():
                identities.add(f"ff_{p}")
    elif "celeb" in dataset:
        parts = stem.split("_")
        for p in parts:
            if p.startswith("id") and len(p) > 2:
                identities.add(f"cdf_{p}")
    
    if not identities:
        # Fallback to source_id or root stem
        sid = record.get("source_id") or stem.split("_")[0]
        identities.add(f"src_{sid}")

    return identities


def audit_legacy_splits(combined_manifest: Path = REPO_ROOT / "data" / "manifests" / "combined.jsonl") -> Dict[str, Any]:
    """Inspects legacy combined.jsonl for source identity leakage."""
    logger.info("Auditing legacy splits in: %s", combined_manifest)
    if not combined_manifest.exists():
        logger.warning("Legacy manifest not found: %s", combined_manifest)
        return {}

    split_records = defaultdict(list)
    split_identities = defaultdict(set)
    split_videos = defaultdict(set)

    with open(combined_manifest, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            s_name = r.get("split", "unknown")
            split_records[s_name].append(r)
            split_videos[s_name].add(r.get("source_path", ""))
            idents = extract_identities(r)
            split_identities[s_name].update(idents)

    leakage_matrix = {}
    splits = ["train", "val", "test"]
    for i in range(len(splits)):
        for j in range(i + 1, len(splits)):
            s1, s2 = splits[i], splits[j]
            overlap_ids = split_identities[s1] & split_identities[s2]
            overlap_vids = split_videos[s1] & split_videos[s2]
            leakage_matrix[f"{s1}_vs_{s2}"] = {
                "shared_identities": len(overlap_ids),
                "shared_videos": len(overlap_vids),
                "examples": list(overlap_ids)[:5],
            }
            logger.warning(
                "Legacy Leakage [%s vs %s]: %d shared identities (Shared videos: %d). Examples: %s",
                s1, s2, len(overlap_ids), len(overlap_vids), list(overlap_ids)[:5],
            )

    return {
        "split_counts": {s: len(recs) for s, recs in split_records.items()},
        "split_identities": {s: len(ids) for s, ids in split_identities.items()},
        "leakage": leakage_matrix,
    }


def build_source_disjoint_partitions(
    combined_path: Path = REPO_ROOT / "data" / "manifests" / "combined.jsonl",
    final_test_path: Path = REPO_ROOT / "data" / "manifests" / "final_dataset_test.jsonl",
    talkinghead_path: Path = REPO_ROOT / "data" / "manifests" / "talkingheadbench.jsonl",
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Groups all videos by connected source identities (graph components)
    and allocates them deterministically into source-disjoint partitions:
    - train: ~65%
    - calibration: ~15%
    - test_seen: ~20%
    - test_unseen: 100% held-out unseen generative models
    """
    logger.info("Building strictly source-disjoint partitions...")

    # 1. Gather all video sources from combined.jsonl with existing crops
    by_video = defaultdict(list)
    with open(combined_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            by_video[r["source_path"]].append(r)

    valid_videos = []
    for v_path, f_list in by_video.items():
        # verify crops exist
        valid = [f for f in f_list if (REPO_ROOT / f.get("crop_path", "")).exists()]
        if len(valid) >= 2:
            valid.sort(key=lambda x: x.get("frame_idx", 0))
            first = valid[0]
            lbl = int(first.get("label", 0))
            idents = extract_identities(first)
            valid_videos.append({
                "video_id": v_path,
                "source_path": v_path,
                "dataset": first.get("dataset", "combined"),
                "label_int": lbl,
                "label": "fake" if lbl == 1 else "real",
                "manipulation_family": first.get("manipulation", "unknown"),
                "generator_family": first.get("manipulation", "unknown"),
                "identities": sorted(list(idents)),
                "frame_crops": [f.get("crop_path") for f in valid],
                "track_length": len(valid),
                "track_continuity": 1.0,
                "sufficient_for_analysis": True,
            })

    logger.info("Total verified multi-frame videos: %d", len(valid_videos))

    # 2. Build connected component clusters of identities (handling pairs like A_B where A and B co-occur)
    parent = {}

    def find(x):
        if parent.setdefault(x, x) != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    for v in valid_videos:
        id_list = v["identities"]
        if len(id_list) > 1:
            for i in range(len(id_list) - 1):
                union(id_list[i], id_list[i + 1])
        elif len(id_list) == 1:
            find(id_list[0])

    # Map each video to its root identity group
    video_by_cluster = defaultdict(list)
    for v in valid_videos:
        root_id = find(v["identities"][0])
        video_by_cluster[root_id].append(v)

    cluster_keys = sorted(list(video_by_cluster.keys()))
    logger.info("Total independent identity clusters: %d", len(cluster_keys))

    # 3. Deterministic split allocation based on hash of root cluster ID
    train_records = []
    calib_records = []
    test_seen_records = []

    train_clusters = set()
    calib_clusters = set()
    test_clusters = set()

    for root_id in cluster_keys:
        h = int(hashlib.md5(f"{seed}:{root_id}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
        v_list = video_by_cluster[root_id]

        if h < 0.65:
            train_records.extend(v_list)
            train_clusters.add(root_id)
            for r in v_list:
                r["split"] = "train"
        elif h < 0.80:
            calib_records.extend(v_list)
            calib_clusters.add(root_id)
            for r in v_list:
                r["split"] = "calibration"
        else:
            test_seen_records.extend(v_list)
            test_clusters.add(root_id)
            for r in v_list:
                r["split"] = "test_seen"

    # 4. Gather Unseen Test Set
    test_unseen_records = []
    if final_test_path.exists():
        with open(final_test_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                cp = item.get("crop_path") or item.get("source_path")
                if cp and Path(cp).exists():
                    lbl = int(item.get("label", 0))
                    test_unseen_records.append({
                        "video_id": item.get("record_id", str(cp)),
                        "source_path": str(cp),
                        "dataset": "final_dataset_test",
                        "label_int": lbl,
                        "label": "fake" if lbl == 1 else "real",
                        "manipulation_family": item.get("manipulation", "Generative"),
                        "generator_family": item.get("manipulation", "Generative"),
                        "difficulty": item.get("difficulty", "Medium"),
                        "identities": [item.get("record_id", str(cp))],
                        "frame_crops": [str(cp)],
                        "track_length": 1,
                        "split": "test_unseen",
                        "track_continuity": 1.0,
                        "sufficient_for_analysis": True,
                    })

    if talkinghead_path.exists():
        with open(talkinghead_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                cp = item.get("crop_path") or item.get("source_path")
                if cp and Path(cp).exists():
                    lbl = int(item.get("label", 1))
                    test_unseen_records.append({
                        "video_id": item.get("record_id", str(cp)),
                        "source_path": str(cp),
                        "dataset": "talkingheadbench",
                        "label_int": lbl,
                        "label": "fake" if lbl == 1 else "real",
                        "manipulation_family": "TalkingHeadSynthesis",
                        "generator_family": "TalkingHeadSynthesis",
                        "identities": [item.get("record_id", str(cp))],
                        "frame_crops": [str(cp)],
                        "track_length": 1,
                        "split": "test_unseen",
                        "audio_path": item.get("audio_path"),
                        "mel_path": item.get("mel_path"),
                        "audio_available": bool(item.get("audio_path") or item.get("mel_path")),
                        "track_continuity": 1.0,
                        "sufficient_for_analysis": True,
                    })

    # 5. Formal Source-Disjoint Proof Verification
    train_ids = set().union(*[r["identities"] for r in train_records])
    calib_ids = set().union(*[r["identities"] for r in calib_records])
    test_seen_ids = set().union(*[r["identities"] for r in test_seen_records])
    test_unseen_ids = set().union(*[r["identities"] for r in test_unseen_records])

    leak_train_calib = train_ids & calib_ids
    leak_train_seen = train_ids & test_seen_ids
    leak_calib_seen = calib_ids & test_seen_ids
    leak_train_unseen = train_ids & test_unseen_ids

    assert len(leak_train_calib) == 0, f"Leakage between Train and Calib: {leak_train_calib}"
    assert len(leak_train_seen) == 0, f"Leakage between Train and Seen Test: {leak_train_seen}"
    assert len(leak_calib_seen) == 0, f"Leakage between Calib and Seen Test: {leak_calib_seen}"
    assert len(leak_train_unseen) == 0, f"Leakage between Train and Unseen Test: {leak_train_unseen}"

    logger.info("=" * 70)
    logger.info("SOURCE-DISJOINT MATHEMATICAL PROOF PASSED:")
    logger.info("  Train vs Calibration Identity Overlap:      0 (Disjoint)")
    logger.info("  Train vs Seen Test Identity Overlap:        0 (Disjoint)")
    logger.info("  Calibration vs Seen Test Identity Overlap:  0 (Disjoint)")
    logger.info("  Train vs Unseen Test Identity Overlap:      0 (Disjoint)")
    logger.info("=" * 70)

    # 6. Save Clean Partitions
    out_dir = REPO_ROOT / "data" / "manifests"
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "train_manifest_disjoint.json": train_records,
        "calibration_manifest.json": calib_records,
        "test_manifest_seen.json": test_seen_records,
        "test_manifest_unseen.json": test_unseen_records,
    }

    for fname, data in paths.items():
        p = out_dir / fname
        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        real_c = sum(1 for r in data if r["label_int"] == 0)
        fake_c = sum(1 for r in data if r["label_int"] == 1)
        logger.info("Saved %-32s: %5d records (%d Real, %d Fake)", fname, len(data), real_c, fake_c)

    audit_summary = {
        "verified_source_disjoint": True,
        "train_samples": len(train_records),
        "calibration_samples": len(calib_records),
        "test_seen_samples": len(test_seen_records),
        "test_unseen_samples": len(test_unseen_records),
        "overlaps": {
            "train_vs_calibration": len(leak_train_calib),
            "train_vs_test_seen": len(leak_train_seen),
            "calibration_vs_test_seen": len(leak_calib_seen),
            "train_vs_test_unseen": len(leak_train_unseen),
        },
    }

    report_out = REPO_ROOT / "reports" / "split_leakage_audit_report.json"
    report_out.parent.mkdir(parents=True, exist_ok=True)
    with open(report_out, "w", encoding="utf-8") as f:
        json.dump(audit_summary, f, indent=2)
    logger.info("Saved split audit report to: %s", report_out)

    return audit_summary


def main():
    audit_legacy = audit_legacy_splits()
    audit_new = build_source_disjoint_partitions()
    print("\n" + "=" * 80)
    print("                      FORGE-LITE SPLIT & DATA LEAKAGE AUDIT")
    print("=" * 80)
    print("LEGACY MANIFEST AUDIT:")
    if audit_legacy.get("leakage"):
        for k, v in audit_legacy["leakage"].items():
            print(f"  {k:<20}: {v['shared_identities']} shared identities (Examples: {v['examples']})")
    print("-" * 80)
    print("NEW SOURCE-DISJOINT PARTITIONS (AUDITED & LOCKED):")
    print(f"  Train Manifest:       data/manifests/train_manifest_disjoint.json ({audit_new['train_samples']} videos)")
    print(f"  Calibration Manifest: data/manifests/calibration_manifest.json    ({audit_new['calibration_samples']} videos)")
    print(f"  Seen Test Manifest:   data/manifests/test_manifest_seen.json       ({audit_new['test_seen_samples']} videos)")
    print(f"  Unseen Test Manifest: data/manifests/test_manifest_unseen.json     ({audit_new['test_unseen_samples']} samples)")
    print("-" * 80)
    print("MATHEMATICAL PROOF:")
    for k, v in audit_new["overlaps"].items():
        print(f"  {k:<26}: {v} Overlapping Identities (STRICTLY ZERO)")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
