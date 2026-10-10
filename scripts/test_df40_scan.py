from preprocessing.preprocess_all import _scan_df40, _df40_method_family
from pathlib import Path
from collections import Counter

root = Path("data/raw/DF40")
items = _scan_df40(root, max_videos=None)

labels   = Counter("real" if i[1]==0 else "fake" for i in items)
methods  = Counter(i[2] for i in items if i[1]==1)
families = Counter(_df40_method_family(i[2]) for i in items if i[1]==1)

print(f"Total items : {len(items)}")
print(f"  Real      : {labels['real']}")
print(f"  Fake      : {labels['fake']}")
print()
print("Methods found:")
for m, n in sorted(methods.items()):
    fam = _df40_method_family(m)
    print(f"  {m:<22} {n:>8} videos   [{fam}]")
print()
print("By family:")
for fam, n in sorted(families.items()):
    print(f"  {fam:<28} {n:>9} videos")
print()
unknown = [m for m in methods if _df40_method_family(m) == "unknown"]
if unknown:
    print(f"WARNING - unknown family methods: {unknown}")
else:
    print("All methods mapped to a family - OK")
