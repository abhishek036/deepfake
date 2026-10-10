"""
scripts/make_celeb_manifests.py
===============================
Extracts clean Celeb-DF native manifests for real and fake crops
in the schema expected by evaluation scripts.

Schema:
  reals: {"label": 0, "crop_path": ..., "source_path": ..., "video_id": "id0_0000", "frame_idx": ..., "split": ...}
  fakes: {"label": 1, "source_path": ..., "method": "Celeb-synthesis", "forgery_type": "face_swap", "video_id": "id0_id16_0000", "frame_idx": ..., "split": ...}

Usage:
  python scripts/make_celeb_manifests.py --manifest data/manifests/celeb_df.jsonl --out-reals data/celeb_native_reals.jsonl --out-fakes data/celeb_native_fakes.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path


def sha256_file(p: Path | str) -> str | None:
    p = Path(p)
    if not p.exists():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while c := f.read(8 << 20):
            h.update(c)
    return h.hexdigest()


def norm_path(p: str) -> str:
    return str(Path(str(p).replace("\\", "/")))


def main():
    parser = argparse.ArgumentParser(description="Extract Celeb-DF native manifests")
    parser.add_argument("--manifest", default="data/manifests/celeb_df.jsonl")
    parser.add_argument("--out-reals", default="data/celeb_native_reals.jsonl")
    parser.add_argument("--out-fakes", default="data/celeb_native_fakes.jsonl")
    parser.add_argument("--max-pairs-per-video", type=int, default=10, help="Max frames per video to balance dataset")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    print(f"Reading Celeb-DF manifest: {manifest_path}...")
    reals_by_key = {}
    fakes_by_key = []

    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            crop = r.get("crop_path")
            if not crop or not Path(crop).exists():
                continue

            gid = str(r.get("source_group_id", ""))
            fidx = int(r.get("frame_idx", 0))
            split = str(r.get("split", "train"))

            if r.get("label") == 0:
                reals_by_key[(gid, fidx)] = {
                    "label": 0,
                    "crop_path": norm_path(crop),
                    "source_path": norm_path(r.get("source_path", "")),
                    "video_id": gid,
                    "frame_idx": fidx,
                    "split": split,
                }
            else:
                parts = gid.split("_")
                target_video = f"{parts[0]}_{parts[2]}" if len(parts) == 3 else gid
                fakes_by_key.append({
                    "label": 1,
                    "source_path": norm_path(crop),
                    "method": "Celeb-synthesis",
                    "forgery_type": "face_swap",
                    "video_id": gid,
                    "target_video": target_video,
                    "frame_idx": fidx,
                    "split": split,
                    "video_path": norm_path(r.get("source_path", "")),
                })

    print(f"Total valid real frames: {len(reals_by_key)}")
    print(f"Total valid fake frames: {len(fakes_by_key)}")

    # Extract paired frames
    matched_fakes = []
    matched_reals_dict = {}
    video_pair_counts = Counter()

    for fk in fakes_by_key:
        key = (fk["target_video"], fk["frame_idx"])
        if key in reals_by_key:
            if video_pair_counts[fk["target_video"]] < args.max_pairs_per_video:
                video_pair_counts[fk["target_video"]] += 1
                matched_fakes.append(fk)
                matched_reals_dict[key] = reals_by_key[key]

    matched_reals = list(matched_reals_dict.values())
    print(f"Extracted matched pairs: {len(matched_fakes)} fakes, {len(matched_reals)} unique reals")
    print(f"Unique target videos: {len(video_pair_counts)}")

    out_reals_path = Path(args.out_reals)
    out_fakes_path = Path(args.out_fakes)

    out_reals_path.parent.mkdir(parents=True, exist_ok=True)
    out_fakes_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_reals_path, "w", encoding="utf-8") as f:
        for r in matched_reals:
            f.write(json.dumps(r) + "\n")

    with open(out_fakes_path, "w", encoding="utf-8") as f:
        for r in matched_fakes:
            f.write(json.dumps(r) + "\n")

    print(f"Wrote {len(matched_reals)} records to {out_reals_path}")
    print(f"Wrote {len(matched_fakes)} records to {out_fakes_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
