import os
from pathlib import Path
from collections import Counter

REPO_ROOT = Path(__file__).resolve().parent.parent
base = REPO_ROOT / "data" / "raw" / "DF40"

# Sample DF40_train/fomm
fomm = base / "DF40_train" / "fomm"
print("=== DF40_train/fomm/ first 20 entries ===")
all_fomm = sorted(fomm.rglob("*"))
for f in all_fomm[:20]:
    rel = str(f).replace(str(base), "")
    kind = "DIR " if f.is_dir() else "FILE"
    print(f"  [{kind}] {rel}")

print()
print("=== fomm top-level subdirectories (first 5) ===")
subdirs = [x for x in sorted(fomm.iterdir()) if x.is_dir()]
for d in subdirs[:5]:
    children = list(d.iterdir())
    print(f"  {d.name}/  ({len(children)} children, first: {children[0].name if children else 'empty'})")

print()
print("=== Extension counts in fomm/ ===")
exts = Counter(f.suffix.lower() for f in fomm.rglob("*") if f.is_file())
for ext, n in sorted(exts.items(), key=lambda x: -x[1]):
    label = ext if ext else "(no ext)"
    print(f"  {label}: {n}")

print()
print("=== inswap/ top-level (first 10) ===")
inswap = base / "inswap"
for item in sorted(inswap.iterdir())[:10]:
    kind = "DIR " if item.is_dir() else "FILE"
    print(f"  [{kind}] {item.name}")
