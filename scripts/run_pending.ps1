# run_pending.ps1  --  put in scripts\ ; run from the repo root (C:\deepfake):
#   powershell -ExecutionPolicy Bypass -File scripts\run_pending.ps1
# Runs every pending experiment one after another (a failing step does not stop the rest), keeps each step's console
# output in reports\handoff\log_<step>.txt, then builds reports\handoff\HANDOFF.md (the one file to upload).
# Expect roughly 30-60 minutes on the RTX 4060 (most of it is DINOv2 embedding).

$ErrorActionPreference = "Continue"
New-Item -ItemType Directory -Force reports\handoff | Out-Null

$native = "--confounded-real-manifest data/ff_native_reals.jsonl --clean-real-manifest data/ff_native_reals.jsonl --fake-manifest data/ff_native_fakes.jsonl --families composited --parent-part 0"
$df40   = "--confounded-real-manifest data/manifests/faceforensics.jsonl --clean-real-manifest data/ffpp_matched_reals_fsfr.jsonl --fake-manifest data/df40_fsfr_subset.jsonl --families composited"

$steps = [ordered]@{
  "paired_stats"       = "-m evaluation.paired_stats --fakes data/ff_native_fakes.jsonl --reals data/ff_native_reals.jsonl --out reports/paired_stats.json"
  "region_native"      = "-m evaluation.region_probe --real-manifest data/ff_native_reals.jsonl --real-split any --fake-manifest data/ff_native_fakes.jsonl --families composited reenactment --neg-control-families composited reenactment --n 600 --out reports/region_probe_native.json"
  "lp_native"          = "-m evaluation.linear_probe_beforeafter $native --cache reports/emb_cache_native_none.pkl --out reports/lp_native.json"
  "lp_native_face_out" = "-m evaluation.linear_probe_beforeafter $native --mask face_out --cache reports/emb_cache_native_face_out.pkl --out reports/lp_native_face_out.json"
  "lp_native_face_only"= "-m evaluation.linear_probe_beforeafter $native --mask face_only --cache reports/emb_cache_native_face_only.pkl --out reports/lp_native_face_only.json"
  "lp_size_control"    = "-m evaluation.linear_probe_beforeafter $df40 --fakes-per-scene 3 --cache reports/emb_cache_size.pkl --out reports/lp_size_control.json"
}

foreach ($name in $steps.Keys) {
  Write-Host "=== $name ==="
  $argList = $steps[$name] -split ' '
  & python @argList 2>&1 | Tee-Object -FilePath "reports\handoff\log_$name.txt"
}

Write-Host "=== collecting project state ==="
& python scripts/collect_project_state.py 2>&1 | Tee-Object -FilePath "reports\handoff\log_collect.txt"
Write-Host "Done. Upload reports\handoff\HANDOFF.md"
