"""
collect_project_state.py  --  put in scripts/ and run from the repo root:  python scripts/collect_project_state.py

READ-ONLY. Gathers, in one pass, everything needed to judge the project's condition and writes
    reports/handoff/HANDOFF.md   (upload this one file)      reports/handoff/project_state.json
Sections: environment, git, repo size, configs, manifests (schema + label/split counts), datasets on disk, checkpoints,
ledgers (stale-hash check), training logs, claims audit of the paper draft, release hygiene (paths, secrets, large files),
plus the tails of reports/handoff/log_*.txt if run_pending.ps1 produced them.
User names in paths and secret-looking values are masked. Use --no-torch to skip opening checkpoints.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as md
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path.cwd()
T0 = time.time()
USER = re.compile(r"([A-Za-z]:[\\/]Users[\\/])[^\\/\s\"']+", re.I)
SECRET = re.compile(r"\b(api[_-]?key|token|secret|passw(?:or)?d|authorization)(\s*[:=]\s*)['\"]?[^\s'\",]{6,}", re.I)
SKIP = {".git", ".venv", "venv", "env", "__pycache__", "node_modules", ".idea", ".vscode", ".mypy_cache", ".pytest_cache"}
RISKY = ["held-out", "held out", "unseen", "zero-shot", "genuine", "proves", "proven", "sota", "state-of-the-art", "100%", "guarantee",
         "cryptograph", "tamper", "world's best", "universal", "first "]
PKGS = ["torch", "torchvision", "transformers", "numpy", "scikit-learn", "scipy", "opencv-python", "opencv-python-headless", "insightface",
        "onnxruntime", "onnxruntime-gpu", "pillow", "timm", "PyYAML", "pandas"]


def san(s):
    return SECRET.sub(lambda m: f"{m.group(1)}{m.group(2)}<masked>", USER.sub(r"\1<user>", str(s)))


def left(a):
    return a.max_seconds - (time.time() - T0)


def sh(cmd, timeout=40):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=ROOT).stdout.strip()
    except Exception as e:
        return f"<error {type(e).__name__}>"


def sha(p, cap=2 << 30):
    p = Path(p)
    if not p.exists() or p.stat().st_size > cap:
        return None
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while c := f.read(8 << 20):
            h.update(c)
    return h.hexdigest()


def walk(base, depth):
    base = Path(base)
    for dp, dn, fn in os.walk(base):
        dn[:] = [d for d in dn if d not in SKIP]
        if len(Path(dp).relative_to(base).parts) >= depth:
            dn[:] = []
        yield Path(dp), fn


def human(n):
    for u in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or u == "TB":
            return f"{n:.1f} {u}"
        n /= 1024


def sec_env(a):
    j = {"python": sys.version.split()[0], "platform": platform.platform(), "packages": {}}
    for p in PKGS:
        try:
            j["packages"][p] = md.version(p)
        except md.PackageNotFoundError:
            pass
    j["nvidia_smi"] = sh(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"])
    du = shutil.disk_usage(ROOT)
    j["disk_free_gb"] = round(du.free / 1e9, 1)
    j["disk_total_gb"] = round(du.total / 1e9, 1)
    return j


def sec_git(a):
    j = {"head": sh(["git", "rev-parse", "HEAD"]), "branch": sh(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
         "dirty_files": len([x for x in sh(["git", "status", "--porcelain"]).splitlines() if x.strip()]),
         "log": sh(["git", "log", "--oneline", "-n", "30"]).splitlines(),
         "remote": re.sub(r"//[^@/]+@", "//<cred>@", sh(["git", "remote", "-v"]))}
    big = []
    for f in sh(["git", "ls-files"], 60).splitlines():
        try:
            s = (ROOT / f).stat().st_size
            if s > 20 << 20:
                big.append((f, human(s)))
        except OSError:
            pass
    j["tracked_files_over_20MB"] = big[:20]
    j["has_gitignore"] = (ROOT / ".gitignore").exists()
    return j


def sec_repo(a):
    rows = {}
    for d in sorted(p for p in ROOT.iterdir() if p.is_dir() and p.name not in SKIP):
        n = py = loc = 0
        for dp, fn in walk(d, 4):
            for f in fn:
                n += 1
                if f.endswith(".py"):
                    py += 1
                    try:
                        loc += sum(1 for _ in open(dp / f, "rb"))
                    except OSError:
                        pass
            if n > 300000 or left(a) < 0:
                break
        rows[d.name] = {"files": n, "py_files": py, "py_loc": loc}
    top = [p.name for p in ROOT.iterdir() if p.is_file()]
    return {"dirs": rows, "top_level_files": sorted(top)[:60],
            "scripts": {d: sorted(x.name for x in (ROOT / d).glob("*.py"))[:60] for d in ("evaluation", "scripts", "training", "preprocessing", "models") if (ROOT / d).exists()}}


def sec_configs(a):
    out = {}
    d = ROOT / "configs"
    for f in sorted(list(d.glob("*.y*ml")) + list(d.glob("*.json")))[:12] if d.exists() else []:
        try:
            out[f.name] = [san(x.rstrip()) for x in f.read_text(encoding="utf-8", errors="replace").splitlines()[: a.config_lines]]
        except OSError:
            pass
    return out


def sec_manifests(a):
    out = {}
    files = sorted(set((ROOT / "data" / "manifests").glob("*.json*")) | set((ROOT / "data").glob("*.jsonl"))) if (ROOT / "data").exists() else []
    for f in files:
        if left(a) < 20:
            out[f.name] = "skipped (time budget)"
            continue
        info = {"size": human(f.stat().st_size), "sha256_12": (sha(f, 1 << 30) or "")[:12]}
        if f.suffix == ".jsonl":
            keys, cnt, n, first = Counter(), {k: Counter() for k in ("label", "split", "dataset", "forgery_type", "manipulation", "method", "generator_family")}, 0, None
            with open(f, "r", encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    try:
                        o = json.loads(line)
                    except ValueError:
                        continue
                    n += 1
                    first = first or o
                    if n <= 200:
                        keys.update(o.keys())
                    for k in cnt:
                        if k in o:
                            cnt[k][str(o[k])] += 1
                    if n >= a.max_lines or left(a) < 10:
                        info["truncated_at"] = n
                        break
            info.update({"records": n, "fields": sorted(keys), "counts": {k: dict(v.most_common(25)) for k, v in cnt.items() if v},
                         "first_record": {k: san(v) for k, v in (first or {}).items()}})
        out[f.name] = info
    return out


def sec_datasets(a):
    out = {}
    for root in ("datasets", "data/raw", "data/crops"):
        r = ROOT / root
        if not r.exists():
            continue
        for d in sorted(p for p in r.iterdir() if p.is_dir())[:40]:
            if left(a) < 15:
                out[f"{root}/{d.name}"] = "skipped (time budget)"
                continue
            n, size, ext, t1 = 0, 0, Counter(), time.time()
            for dp, fn in walk(d, 6):
                for f in fn:
                    n += 1
                    ext[Path(f).suffix.lower()] += 1
                    if n % 50 == 0 or n < 50:
                        pass
                    try:
                        size += (dp / f).stat().st_size
                    except OSError:
                        pass
                if n > 400000 or time.time() - t1 > a.dir_seconds:
                    ext["<truncated>"] += 1
                    break
            out[f"{root}/{d.name}"] = {"files": n, "size": human(size), "top_ext": dict(ext.most_common(4))}
    return out


def sec_ckpt(a):
    out = {}
    cands = []
    for pat in ("*.ckpt", "*.pt", "*.pth"):
        cands += list((ROOT / "outputs").rglob(pat)) if (ROOT / "outputs").exists() else []
    for f in sorted(cands, key=lambda p: -p.stat().st_mtime)[:25]:
        info = {"size": human(f.stat().st_size), "modified": time.strftime("%Y-%m-%d %H:%M", time.localtime(f.stat().st_mtime)),
                "sha256_12": (sha(f) or "")[:12] if left(a) > 30 else None}
        if f.name == "best.ckpt" and not a.no_torch:
            try:
                import torch
                blob = torch.load(f, map_location="cpu", weights_only=False)
                info["top_level_keys"] = list(blob.keys())[:30] if isinstance(blob, dict) else type(blob).__name__
                sd = blob.get("model_state_dict", blob) if isinstance(blob, dict) else {}
                tens = [v for v in sd.values() if hasattr(v, "numel")]
                info["tensors"], info["params"] = len(tens), int(sum(v.numel() for v in tens))
                for k in ("epoch", "global_step", "metrics", "val_metrics", "best_metric", "config", "args", "phase"):
                    if isinstance(blob, dict) and k in blob:
                        info[k] = san(json.dumps(blob[k], default=str)[:700])
            except Exception as e:
                info["load_error"] = f"{type(e).__name__}: {str(e)[:120]}"
        out[str(f.relative_to(ROOT)).replace("\\", "/")] = info
    return out


def sec_ledgers(a):
    cur = {}
    for d in ("evaluation", "scripts", "training", "preprocessing"):
        for f in (ROOT / d).glob("*.py") if (ROOT / d).exists() else []:
            cur[sha(f)] = f"{d}/{f.name}"
    out = {}
    for f in sorted((ROOT / "reports").glob("*.json"))[:60] if (ROOT / "reports").exists() else []:
        try:
            j = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            out[f.name] = "unreadable"
            continue
        info = {"modified": time.strftime("%Y-%m-%d %H:%M", time.localtime(f.stat().st_mtime)), "size": human(f.stat().st_size),
                "keys": list(j.keys())[:14] if isinstance(j, dict) else type(j).__name__}
        h = (j.get("script_sha256") or j.get("reproducibility", {}).get("audit_script_sha256")) if isinstance(j, dict) else None
        if h:
            info["script_hash_check"] = f"matches current {cur[h]}" if h in cur else "STALE: no current script has this hash"
        out[f.name] = info
    return out


def sec_logs(a):
    out = {}
    for d in ("outputs/logs", "logs", "runs", "lightning_logs", "wandb", "outputs/tensorboard"):
        p = ROOT / d
        if p.exists():
            files = [x for x in p.rglob("*") if x.is_file()]
            out[d] = {"files": len(files), "latest": time.strftime("%Y-%m-%d %H:%M", time.localtime(max((x.stat().st_mtime for x in files), default=0)))}
    return out


def sec_claims(a):
    out = {}
    cands = [p for p in list(ROOT.glob("*.md")) + list((ROOT / "paper").glob("*.*") if (ROOT / "paper").exists() else []) +
             list((ROOT / "docs").glob("*.md") if (ROOT / "docs").exists() else []) +
             list((ROOT / "drafts").glob("*.md") if (ROOT / "drafts").exists() else []) if re.search(r"paper|compendium|draft|manuscript", p.name, re.I)]
    for p in cands[:6]:
        hits = []
        for i, line in enumerate(p.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            low = line.lower()
            for t in RISKY:
                if t in low:
                    hits.append((i, t, san(line.strip()[:110])))
        out[p.name] = {"risky_term_hits": len(hits), "first_hits": hits[: a.claim_hits]}
    return out


def sec_hygiene(a):
    tracked = sh(["git", "ls-files"], 60).splitlines() or [str(p.relative_to(ROOT)) for p in ROOT.rglob("*.py")][:1500]
    abs_paths, secrets, scanned = Counter(), Counter(), 0
    for f in tracked[:3000]:
        p = ROOT / f
        if p.suffix.lower() not in {".py", ".md", ".yaml", ".yml", ".json", ".txt", ".ps1", ".cfg", ".toml"}:
            continue
        try:
            if p.stat().st_size > 1 << 20:
                continue
            t = p.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        scanned += 1
        n = len(re.findall(r"[A-Za-z]:[\\/](?:Users|deepfake)[\\/]", t))
        if n:
            abs_paths[f] = n
        if SECRET.search(t):
            secrets[f] += 1
    return {"text_files_scanned": scanned, "files_with_absolute_windows_paths": dict(abs_paths.most_common(15)),
            "files_with_secret_like_strings": dict(secrets.most_common(10)),
            "license": any((ROOT / n).exists() for n in ("LICENSE", "LICENSE.md", "LICENSE.txt")),
            "readme": any((ROOT / n).exists() for n in ("README.md", "readme.md")),
            "requirements": [n for n in ("requirements.txt", "pyproject.toml", "environment.yml") if (ROOT / n).exists()],
            "env_files": [str(p.relative_to(ROOT)) for p in ROOT.glob(".env*")], "tests_dir": (ROOT / "tests").exists()}


def sec_run_logs(a):
    out = {}
    for f in sorted((ROOT / "reports" / "handoff").glob("log_*.txt")) if (ROOT / "reports" / "handoff").exists() else []:
        out[f.name] = [san(x.rstrip()) for x in f.read_text(encoding="utf-8", errors="replace").splitlines()[-a.log_tail:]]
    return out


def md_render(state):
    L = ["# FORGE-Lite / audit project: HANDOFF", f"generated {time.strftime('%Y-%m-%d %H:%M')} from {san(ROOT)}", ""]
    for name, val in state.items():
        L += [f"## {name}", "```json", san(json.dumps(val, indent=1, default=str)), "```", ""]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="reports/handoff")
    ap.add_argument("--max-seconds", type=int, default=300)
    ap.add_argument("--dir-seconds", type=int, default=25)
    ap.add_argument("--max-lines", type=int, default=1_000_000)
    ap.add_argument("--config-lines", type=int, default=150)
    ap.add_argument("--claim-hits", type=int, default=40)
    ap.add_argument("--log-tail", type=int, default=70)
    ap.add_argument("--no-torch", action="store_true")
    a = ap.parse_args()
    state = {}
    for name, fn in (("environment", sec_env), ("git", sec_git), ("repo_size", sec_repo), ("configs", sec_configs), ("manifests", sec_manifests),
                     ("datasets_on_disk", sec_datasets), ("checkpoints", sec_ckpt), ("ledgers", sec_ledgers), ("training_logs", sec_logs),
                     ("claims_audit", sec_claims), ("release_hygiene", sec_hygiene), ("run_logs_tail", sec_run_logs)):
        t = time.time()
        try:
            state[name] = fn(a)
        except Exception as e:
            state[name] = f"<section failed: {type(e).__name__}: {str(e)[:150]}>"
        print(f"{name:18s} {time.time() - t:6.1f}s")
    out = ROOT / a.out_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "project_state.json").write_text(san(json.dumps(state, indent=1, default=str)), encoding="utf-8")
    (out / "HANDOFF.md").write_text(md_render(state), encoding="utf-8")
    print(f"\nwrote {out / 'HANDOFF.md'}  ({(out / 'HANDOFF.md').stat().st_size / 1024:.0f} KB): upload this one file")
    return 0


if __name__ == "__main__":
    sys.exit(main())
