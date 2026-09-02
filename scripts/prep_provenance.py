# -*- coding: utf-8 -*-
"""Split the official test set by data provenance and evaluate every trained model.

  test_scutA  : images imported from SCUT-HEAD Part A (classroom monitor video)
  test_helmet : the genuine helmet imagery collected from web search
"""

import os
from pathlib import Path

# Configure with environment variables, or edit the two defaults below.
#   DATA_ROOT : directory holding the datasets (must contain VOC2028/ for SHWD)
#   OUT_DIR   : directory for generated result files
DATA_ROOT = Path(os.environ.get("AUDIT_DATA_ROOT", "data"))
OUT_DIR = Path(os.environ.get("AUDIT_OUT_DIR", "results"))
OUT_DIR.mkdir(parents=True, exist_ok=True)


import json, shutil
from pathlib import Path
SRC = Path(str(DATA_ROOT / "SHWD_official"))
OUT = Path(str(DATA_ROOT / "SHWD_prov"))

def group(stem):
    if stem.startswith("PartA"): return "test_scutA"
    if stem.startswith("PartB"): return None          # web-crawled heads: excluded, mixed character
    return "test_helmet"

stems = sorted(p.stem for p in (SRC/"images/test_full").glob("*.jpg"))
counts = {}
for s in stems:
    g = group(s)
    if g is None: continue
    for sub in ("images", "labels"):
        (OUT/sub/g).mkdir(parents=True, exist_ok=True)
    shutil.copy2(SRC/"images/test_full"/f"{s}.jpg", OUT/"images"/g/f"{s}.jpg")
    shutil.copy2(SRC/"labels/test_full"/f"{s}.txt", OUT/"labels"/g/f"{s}.txt")
    counts[g] = counts.get(g, 0) + 1
print(counts)
for g in counts:
    (OUT/f"{g}.yaml").write_text(
        f"train: {(OUT/'images'/g).as_posix()}\n"
        f"val: {(OUT/'images'/g).as_posix()}\n"
        f"test: {(OUT/'images'/g).as_posix()}\n"
        f"names:\n  0: helmet\n  1: head\n", encoding="ascii")
json.dump(counts, open(str(OUT_DIR) + "/provenance_counts.json","w"), indent=1)
