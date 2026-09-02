# -*- coding: utf-8 -*-
"""Build a YOLO dataset from SHWD's OFFICIAL split, plus three test variants:
   test_full (1517) / test_leaked (dupes of training data) / test_clean (rest).
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
import xml.etree.ElementTree as ET
import numpy as np

ROOT = Path(str(DATA_ROOT / "SHWD/VOC2028"))
OUT = Path(str(DATA_ROOT / "SHWD_official"))
CACHE = Path(str(OUT_DIR) + "/hashes_shwd.npz")
CLASSES = {"hat": 0, "person": 1}
HAMMING_T, CORR_MIN = 6, 0.90

z = np.load(CACHE, allow_pickle=True)
ph, th, stems = z["ph"], z["th"], [str(s) for s in z["stems"]]
idx = {s: i for i, s in enumerate(stems)}
S = ROOT / "ImageSets/Main"
ld = lambda n: [l.split()[0] for l in (S / f"{n}.txt").read_text().split("\n") if l.strip()]
trainval, test = ld("trainval"), ld("test")

tr = np.array([idx[s] for s in trainval if s in idx])
te_names = [s for s in test if s in idx]
te = np.array([idx[s] for s in te_names])

d = np.empty((len(te), len(tr)), np.uint8)
for i in range(0, len(te), 256):
    d[i:i+256] = np.bitwise_count(np.bitwise_xor(ph[te[i:i+256], None], ph[tr][None, :])).astype(np.uint8)

leaked, clean = [], []
for i, name in enumerate(te_names):
    js = np.where(d[i] <= HAMMING_T)[0]
    (leaked if any(float(th[te[i]] @ th[tr[j]]) >= CORR_MIN for j in js) else clean).append(name)
print(f"trainval={len(trainval)}  test_full={len(te_names)}  leaked={len(leaked)}  clean={len(clean)}")


def convert(stem, split):
    x = ET.parse(ROOT / "Annotations" / f"{stem}.xml").getroot()
    sz = x.find("size"); W, H = int(sz.find("width").text), int(sz.find("height").text)
    if W <= 0 or H <= 0: return False
    lines = []
    for o in x.iter("object"):
        c = o.find("name").text.strip()
        if c not in CLASSES: continue
        b = o.find("bndbox")
        x1, y1, x2, y2 = (float(b.find(k).text) for k in ("xmin", "ymin", "xmax", "ymax"))
        cx, cy, w, h = (x1+x2)/2/W, (y1+y2)/2/H, (x2-x1)/W, (y2-y1)/H
        if w <= 0 or h <= 0: continue
        lines.append(f"{CLASSES[c]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    src = ROOT / "JPEGImages" / f"{stem}.jpg"
    if not src.exists(): return False
    (OUT/"images"/split).mkdir(parents=True, exist_ok=True)
    (OUT/"labels"/split).mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, OUT/"images"/split/f"{stem}.jpg")
    (OUT/"labels"/split/f"{stem}.txt").write_text("\n".join(lines), encoding="ascii")
    return True


for split, names in (("train", trainval), ("test_full", te_names),
                     ("test_leaked", leaked), ("test_clean", clean)):
    n = sum(convert(s, split) for s in names)
    print(f"  {split}: {n} images")

for variant in ("test_full", "test_leaked", "test_clean"):
    (OUT / f"shwd_{variant}.yaml").write_text(
        f"path: {OUT.as_posix()}\ntrain: images/train\nval: images/{variant}\n"
        f"test: images/{variant}\nnames:\n  0: helmet\n  1: head\n", encoding="ascii")

json.dump({"trainval": len(trainval), "test_full": len(te_names),
           "leaked": len(leaked), "clean": len(clean),
           "leaked_names": leaked}, open(str(OUT_DIR) + "/official_split.json", "w"), indent=1)
print("done ->", OUT)
