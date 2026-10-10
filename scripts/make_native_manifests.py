"""
make_native_manifests.py  --  put in scripts/   (stdlib only)

Splits data/manifests/faceforensics.jsonl into two manifests in the schema used by region_probe / linear_probe_beforeafter /
paired_diagnostic:
  fakes: {label:1, source_path:<crop jpg>, method, forgery_type, video_id:'000_003', frame_idx, split, video_path}
  reals: {label:0, crop_path:<crop jpg>, source_path:<original video>, video_id:'000', frame_idx, split}
Only the five FF++ methods with ids like 000_003 are kept (DeepFakeDetection has a different naming scheme and is dropped).

One line:
  python scripts/make_native_manifests.py --manifest data/manifests/faceforensics.jsonl --out-fakes data/ff_native_fakes.jsonl --out-reals data/ff_native_reals.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

METHODS = {"Deepfakes", "FaceSwap", "FaceShifter", "Face2Face", "NeuralTextures"}
PAIR = re.compile(r"\d{3}_\d{3}")
VID = re.compile(r"\d{3}")


def norm(p):
    return str(Path(str(p).replace("\\", "/")))


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while c := f.read(8 << 20):
            h.update(c)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out-fakes", required=True)
    ap.add_argument("--out-reals", required=True)
    ap.add_argument("--check-n", type=int, default=400, help="sample size for the crop-exists check")
    a = ap.parse_args()

    fakes, reals, skipped = [], [], Counter()
    with open(a.manifest, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            o = json.loads(line)
            crop = o.get("crop_path")
            if not crop:
                skipped["no_crop_path"] += 1
                continue
            if o.get("label") == 1:
                m, gid = o.get("manipulation"), str(o.get("source_group_id", ""))
                if m in METHODS and PAIR.fullmatch(gid):
                    fakes.append({"label": 1, "source_path": norm(crop), "method": m, "forgery_type": o.get("forgery_type"),
                                  "video_id": gid, "frame_idx": o.get("frame_idx"), "split": o.get("split"),
                                  "video_path": norm(o.get("source_path", ""))})
                else:
                    skipped[f"fake_dropped:{m}"] += 1
            elif o.get("label") == 0:
                stem = Path(norm(o.get("source_path", ""))).stem
                if VID.fullmatch(stem):
                    reals.append({"label": 0, "split": o.get("split"), "crop_path": norm(crop),
                                  "source_path": norm(o.get("source_path", "")), "video_id": stem,
                                  "frame_idx": o.get("frame_idx")})
                else:
                    skipped["real_dropped_non_ffpp_name"] += 1

    for path, rows in ((a.out_fakes, fakes), (a.out_reals, reals)):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    print(f"fakes kept {len(fakes)}: {dict(Counter(r['method'] for r in fakes))}")
    print(f"  forgery_type: {dict(Counter(r['forgery_type'] for r in fakes))}   split: {dict(Counter(r['split'] for r in fakes))}")
    print(f"reals kept {len(reals)}: {len({r['video_id'] for r in reals})} videos   split: {dict(Counter(r['split'] for r in reals))}")
    print("skipped:", dict(skipped))

    rng = random.Random(0)
    for name, rows, key in (("fake", fakes, "source_path"), ("real", reals, "crop_path")):
        s = rng.sample(rows, min(a.check_n, len(rows)))
        ok = sum(Path(r[key]).exists() for r in s)
        print(f"crop files exist ({name}, sample of {len(s)}): {ok}/{len(s)}")

    real_idx = {(r["video_id"], r["frame_idx"]) for r in reals}
    for part in (0, 1):
        hit = sum((r["video_id"].split("_")[part], r["frame_idx"]) in real_idx for r in fakes)
        print(f"fakes whose real frame exists if parent = id[{part}]: {hit}/{len(fakes)} ({hit / max(1, len(fakes)):.1%})")
    print("(coverage is only a hint; paired_diagnostic.py decides the parent id by measuring background differences)")
    print("hashes:", {Path(p).name: sha256_file(p)[:12] for p in (a.manifest, a.out_fakes, a.out_reals)})
    return 0


if __name__ == "__main__":
    sys.exit(main())
