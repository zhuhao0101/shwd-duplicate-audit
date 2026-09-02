# -*- coding: utf-8 -*-
"""Build training sets for the causal leakage experiment.

  A  = full official trainval                       (already built: SHWD_official/images/train)
  B  = trainval with every image that near-duplicates a TEST image removed
  C  = random subset of trainval, same size as B    (controls for training-set size)

Test sets are unchanged, so the only variable between A/B/C is which training
images are present.
"""

import os
from pathlib import Path

# Configure with environment variables, or edit the two defaults below.
#   DATA_ROOT : directory holding the datasets (must contain VOC2028/ for SHWD)
#   OUT_DIR   : directory for generated result files
DATA_ROOT = Path(os.environ.get("AUDIT_DATA_ROOT", "data"))
OUT_DIR = Path(os.environ.get("AUDIT_OUT_DIR", "results"))
OUT_DIR.mkdir(parents=True, exist_ok=True)


import json, random, shutil
from pathlib import Path
import numpy as np

SRC = Path(str(DATA_ROOT / "SHWD_official"))
OUT = Path(str(DATA_ROOT / "SHWD_causal"))
CACHE = Path(str(OUT_DIR) + "/hashes_shwd.npz")
HAMMING_T, CORR_MIN, SEED = 6, 0.90, 0

z = np.load(CACHE, allow_pickle=True)
ph, th, stems = z["ph"], z["th"], [str(s) for s in z["stems"]]
idx = {s: i for i, s in enumerate(stems)}

train_names = sorted(p.stem for p in (SRC / "images/train").glob("*.jpg"))
test_names = sorted(p.stem for p in (SRC / "images/test_full").glob("*.jpg"))
tr = np.array([idx[s] for s in train_names])
te = np.array([idx[s] for s in test_names])

d = np.empty((len(tr), len(te)), np.uint8)
for i in range(0, len(tr), 256):
    d[i:i+256] = np.bitwise_count(np.bitwise_xor(ph[tr[i:i+256], None], ph[te][None, :])).astype(np.uint8)

contaminating = []
for i, name in enumerate(train_names):
    js = np.where(d[i] <= HAMMING_T)[0]
    if any(float(th[tr[i]] @ th[te[j]]) >= CORR_MIN for j in js):
        contaminating.append(name)
contam = set(contaminating)
B = [n for n in train_names if n not in contam]
random.seed(SEED)
C = sorted(random.sample(train_names, len(B)))

print(f"trainval A = {len(train_names)}")
print(f"contaminating (near-dup of some test image) = {len(contaminating)}")
print(f"B (deduplicated)  = {len(B)}")
print(f"C (random, size-matched) = {len(C)}  | of which contaminating = {len(set(C) & contam)}")

for tag, names in (("trainB", B), ("trainC", C)):
    for sub in ("images", "labels"):
        (OUT / sub / tag).mkdir(parents=True, exist_ok=True)
    for n in names:
        shutil.copy2(SRC / "images/train" / f"{n}.jpg", OUT / "images" / tag / f"{n}.jpg")
        shutil.copy2(SRC / "labels/train" / f"{n}.txt", OUT / "labels" / tag / f"{n}.txt")
    print(f"  built {tag}: {len(names)}")

# test sets: reuse SRC via symlink-free copy of the yaml pointing at SRC paths
for tag in ("trainB", "trainC"):
    for variant in ("test_leaked", "test_clean", "test_full"):
        (OUT / f"{tag}_{variant}.yaml").write_text(
            f"path: {SRC.as_posix()}\n"
            f"train: {(OUT/'images'/tag).as_posix()}\n"
            f"val: images/{variant}\ntest: images/{variant}\n"
            f"names:\n  0: helmet\n  1: head\n", encoding="ascii")

json.dump({"A": len(train_names), "contaminating": len(contaminating),
           "B": len(B), "C": len(C),
           "C_contaminating_retained": len(set(C) & contam),
           "contaminating_names": contaminating},
          open(str(OUT_DIR) + "/causal_split.json", "w"), indent=1)
print("done ->", OUT)
