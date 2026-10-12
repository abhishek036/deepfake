"""
scripts/export_clean_repo.py
Creates a lightweight, clean git repository export for submission or remote publishing.
Excludes large video files, caches, crops, and historical bloated git pack files.

Usage:
    python scripts/export_clean_repo.py --out-dir ../deepfake_clean
Self-test:
    python scripts/export_clean_repo.py --self-test
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# File patterns and directories to explicitly include
INCLUDE_DIRS = [
    "configs",
    "drafts",
    "evaluation",
    "models",
    "preprocessing",
    "scripts",
    "training",
    "utils",
    "tests",
]

INCLUDE_ROOT_FILES = [
    "ANALYSIS_LOG.md",
    "MASTER_PLAN.md",
    "PREREG.md",
    "PROJECT_OVERVIEW.md",
    "README.md",
    "RESEARCH_PAPER_COMPENDIUM.md",
    "requirements.txt",
    ".gitignore",
]

EXCLUDE_EXTENSIONS = {
    ".mp4", ".avi", ".mov", ".mkv",
    ".pkl", ".pth", ".pt", ".ckpt", ".bin",
    ".pyc", ".pyo",
}

EXCLUDE_DIRS = {
    ".git", "__pycache__", ".pytest_cache", ".cache", "env", "venv",
    "crops", "raw", "scratch",
}


def copy_clean_tree(src_root: Path, dst_root: Path) -> int:
    dst_root.mkdir(parents=True, exist_ok=True)
    n_copied = 0

    # Root files
    for fname in INCLUDE_ROOT_FILES:
        sp = src_root / fname
        if sp.exists() and sp.is_file():
            dp = dst_root / fname
            dp.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(sp, dp)
            n_copied += 1

    # Include dirs
    for dname in INCLUDE_DIRS:
        sp = src_root / dname
        if not sp.exists():
            continue
        for root, dirs, files in os.walk(sp):
            dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
            rel_dir = Path(root).relative_to(src_root)
            dst_dir = dst_root / rel_dir
            dst_dir.mkdir(parents=True, exist_ok=True)
            for f in files:
                ext = Path(f).suffix.lower()
                if ext in EXCLUDE_EXTENSIONS:
                    continue
                shutil.copy2(Path(root) / f, dst_dir / f)
                n_copied += 1

    # Reports: include JSON ledgers only, exclude .pkl caches
    rep_src = src_root / "reports"
    if rep_src.exists():
        rep_dst = dst_root / "reports"
        rep_dst.mkdir(parents=True, exist_ok=True)
        for f in rep_src.glob("*.json"):
            shutil.copy2(f, rep_dst / f.name)
            n_copied += 1

    # Data: manifests and split metadata only (no raw videos, no crops)
    data_src = src_root / "data"
    if data_src.exists():
        data_dst = dst_root / "data"
        data_dst.mkdir(parents=True, exist_ok=True)
        for f in data_src.glob("*.jsonl"):
            # skip manifests larger than 25MB if any
            if f.stat().st_size < 25 * 1024 * 1024:
                shutil.copy2(f, data_dst / f.name)
                n_copied += 1
        splits_src = data_src / "splits"
        if splits_src.exists():
            splits_dst = data_dst / "splits"
            splits_dst.mkdir(parents=True, exist_ok=True)
            for f in splits_src.glob("*"):
                if f.is_file() and f.stat().st_size < 10 * 1024 * 1024:
                    shutil.copy2(f, splits_dst / f.name)
                    n_copied += 1

    return n_copied


def init_git(dst_root: Path) -> bool:
    try:
        subprocess.run(["git", "init"], cwd=str(dst_root), check=True, capture_output=True)
        subprocess.run(["git", "add", "."], cwd=str(dst_root), check=True, capture_output=True)
        subprocess.run(
            [
                "git",
                "-c", "user.name=Abhi",
                "-c", "user.email=abhi@mits.ac.in",
                "commit",
                "-m", "feat: initial clean reproducible benchmark and audit release"
            ],
            cwd=str(dst_root), check=True, capture_output=True
        )
        return True
    except Exception as e:
        print(f"Warning: git init/commit failed: {e}")
        return False


def self_test() -> int:
    tmp_src = Path(tempfile.mkdtemp())
    tmp_dst = Path(tempfile.mkdtemp())
    try:
        (tmp_src / "README.md").write_text("# Test Repo\n", encoding="utf-8")
        (tmp_src / "scripts").mkdir()
        (tmp_src / "scripts" / "test.py").write_text("print('hello')\n", encoding="utf-8")
        (tmp_src / "reports").mkdir()
        (tmp_src / "reports" / "audit.json").write_text('{"status": "ok"}\n', encoding="utf-8")
        (tmp_src / "reports" / "cache.pkl").write_text("binary blob\n", encoding="utf-8")
        n = copy_clean_tree(tmp_src, tmp_dst)
        ok = (
            n == 3
            and (tmp_dst / "README.md").exists()
            and (tmp_dst / "scripts" / "test.py").exists()
            and (tmp_dst / "reports" / "audit.json").exists()
            and not (tmp_dst / "reports" / "cache.pkl").exists()
        )
        git_ok = init_git(tmp_dst)
        print("export_clean_repo self-test", "PASSED" if (ok and git_ok) else "FAILED")
        return 0 if (ok and git_ok) else 1
    finally:
        shutil.rmtree(tmp_src, ignore_errors=True)
        shutil.rmtree(tmp_dst, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser(description="Export a clean, lightweight reproducible repo")
    ap.add_argument("--out-dir", default=None, help="Destination directory for clean repo")
    ap.add_argument("--src-dir", default=".", help="Source repo directory")
    ap.add_argument("--no-git-init", action="store_true", help="Do not initialize a git repo in out-dir")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        return self_test()

    if not args.out_dir:
        sys.exit("Error: --out-dir is required (or use --self-test)")

    src = Path(args.src_dir).resolve()
    dst = Path(args.out_dir).resolve()

    print(f"Exporting clean repository from {src} -> {dst}...")
    n = copy_clean_tree(src, dst)
    print(f"Copied {n} clean project files.")

    if not args.no_git_init:
        print("Initializing clean git repository...")
        if init_git(dst):
            print("Successfully initialized git repository with initial commit.")

    print(f"\nDone! Clean repository ready at: {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
