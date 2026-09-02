# -*- coding: utf-8 -*-
"""Quantify train->test near-duplicate leakage under SHWD's official split.

Caches hashes to hashes_shwd.npz so downstream analyses are cheap.
"""

import os
from pathlib import Path

# Configure with environment variables, or edit the two defaults below.
#   DATA_ROOT : directory holding the datasets (must contain VOC2028/ for SHWD)
#   OUT_DIR   : directory for generated result files
DATA_ROOT = Path(os.environ.get("AUDIT_DATA_ROOT", "data"))
OUT_DIR = Path(os.environ.get("AUDIT_OUT_DIR", "results"))
OUT_DIR.mkdir(parents=True, exist_ok=True)


import json
from pathlib import Path
import numpy as np, imagehash
from PIL import Image

ROOT = Path(str(DATA_ROOT / "SHWD/VOC2028"))
IMGD, SETS = ROOT / "JPEGImages", ROOT / "ImageSets/Main"
CACHE = Path(str(OUT_DIR) + "/hashes_shwd.npz")
HAMMING_T, CORR_MIN = 6, 0.90


def sig(p):
    im = Image.open(p); im.draft("L", (256, 256)); im = im.convert("L")
    b = imagehash.phash(im, hash_size=8).hash.flatten()
    v = int("".join("1" if x else "0" for x in b), 2)
    t = np.asarray(im.resize((32, 32), Image.BILINEAR), np.float32).ravel()
    t -= t.mean(); n = np.linalg.norm(t)
    return v, (t / n if n > 1e-6 else t)


stems = sorted(p.stem for p in IMGD.glob("*.jpg"))
idx = {s: i for i, s in enumerate(stems)}

if CACHE.exists():
    z = np.load(CACHE, allow_pickle=True)
    ph, th = z["ph"], z["th"]
    print("loaded cached hashes")
else:
    ph, th = [], []
    for i, s in enumerate(stems):
        v, t = sig(IMGD / f"{s}.jpg"); ph.append(v); th.append(t)
        if (i + 1) % 2000 == 0: print(f"  hashed {i+1}/{len(stems)}", flush=True)
    ph = np.array(ph, np.uint64); th = np.vstack(th)
    np.savez_compressed(CACHE, ph=ph, th=th, stems=np.array(stems))
    print("cached hashes ->", CACHE)


def load(name):
    f = SETS / f"{name}.txt"
    return [l.split()[0] for l in f.read_text().split("\n") if l.strip()]


splits = {n: load(n) for n in ("train", "val", "test", "trainval")}
print({k: len(v) for k, v in splits.items()})

tr = np.array([idx[s] for s in splits["trainval"] if s in idx])
te = np.array([idx[s] for s in splits["test"] if s in idx])
print(f"trainval={len(tr)}  test={len(te)}")

# distance test x trainval
d = np.empty((len(te), len(tr)), np.uint8)
for i in range(0, len(te), 256):
    d[i:i+256] = np.bitwise_count(
        np.bitwise_xor(ph[te[i:i+256], None], ph[tr][None, :])).astype(np.uint8)

leaked, pairs = [], 0
for i in range(len(te)):
    js = np.where(d[i] <= HAMMING_T)[0]
    hit = [j for j in js if float(th[te[i]] @ th[tr[j]]) >= CORR_MIN]
    if hit:
        leaked.append(i); pairs += len(hit)

res = {
    "test_images": int(len(te)),
    "trainval_images": int(len(tr)),
    "test_images_with_near_dup_in_trainval": len(leaked),
    "leakage_pct_of_test": round(100 * len(leaked) / len(te), 2),
    "total_leaking_pairs": pairs,
    "params": {"hamming": HAMMING_T, "corr_min": CORR_MIN},
}
print(json.dumps(res, indent=2))

clean = [stems[te[i]] for i in range(len(te)) if i not in set(leaked)]
Path(str(OUT_DIR) + "/test_clean.txt").write_text("\n".join(clean), encoding="ascii")
Path(str(OUT_DIR) + "/leakage.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
print(f"clean test list: {len(clean)} images -> test_clean.txt")
