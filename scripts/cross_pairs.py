# -*- coding: utf-8 -*-
"""List every SHWD <-> external near-duplicate pair and prepare it for manual verification.

External collection: keremberke/hard-hat-detection, test split (2,001 images), the same
split used in the cross-dataset check of the manuscript. Same two-stage procedure and
thresholds as the main audit (Hamming <= 6, NCC >= 0.90).

Outputs (audit/):
  hashes_hhw_test.npz     cached signatures of the external split
  cross_pairs.json        every confirmed pair with distances, SHWD split and source prefix
  cross_pairs.csv         rating sheet (decision column empty)
  montages_cross/         one contact sheet per 6 pairs, side by side, labelled
"""

import os
from pathlib import Path

# Configure with environment variables, or edit the two defaults below.
#   DATA_ROOT : directory holding the datasets (must contain SHWD/VOC2028 for SHWD)
#   OUT_DIR   : directory for generated result files
DATA_ROOT = Path(os.environ.get("AUDIT_DATA_ROOT", "data"))
OUT_DIR = Path(os.environ.get("AUDIT_OUT_DIR", "results"))
OUT_DIR.mkdir(parents=True, exist_ok=True)


import json, csv, io, sys
from pathlib import Path
import numpy as np, imagehash
from PIL import Image, ImageDraw
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

AUD = OUT_DIR
SHWD_IMG = Path(str(DATA_ROOT / "SHWD/VOC2028/JPEGImages"))
EXT = Path(str(DATA_ROOT / "HHW_kb/test"))
OFF = Path(str(DATA_ROOT / "SHWD_official"))
HAMMING_T, CORR_MIN = 6, 0.90


def sig(p):
    im = Image.open(p); im.draft("L", (256, 256)); im = im.convert("L")
    b = imagehash.phash(im, hash_size=8).hash.flatten()
    v = int("".join("1" if x else "0" for x in b), 2)
    t = np.asarray(im.resize((32, 32), Image.BILINEAR), np.float32).ravel()
    t -= t.mean(); n = np.linalg.norm(t)
    return v, (t / n if n > 1e-6 else t)


z = np.load(AUD / "hashes_shwd.npz", allow_pickle=True)
ph_a, th_a, stems_a = z["ph"], z["th"], [str(s) for s in z["stems"]]

cache = AUD / "hashes_hhw_test.npz"
ext_files = sorted(p for p in EXT.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
if cache.exists():
    zz = np.load(cache, allow_pickle=True); ph_b, th_b, names_b = zz["ph"], zz["th"], [str(s) for s in zz["names"]]
else:
    ph_b, th_b, names_b = [], [], []
    for p in ext_files:
        v, t = sig(p); ph_b.append(v); th_b.append(t); names_b.append(p.name)
    ph_b = np.array(ph_b, np.uint64); th_b = np.vstack(th_b)
    np.savez_compressed(cache, ph=ph_b, th=th_b, names=np.array(names_b))
print(f"SHWD {len(stems_a)}  external {len(names_b)}")

d = np.empty((len(ph_a), len(ph_b)), np.uint8)
for i in range(0, len(ph_a), 512):
    d[i:i + 512] = np.bitwise_count(np.bitwise_xor(ph_a[i:i + 512, None], ph_b[None, :])).astype(np.uint8)
ii, jj = np.where(d <= HAMMING_T)
ncc = np.array([float(th_a[a] @ th_b[b]) for a, b in zip(ii, jj)])
keep = ncc >= CORR_MIN
print(f"stage-1 candidates {len(ii)}  confirmed {int(keep.sum())}")

test_names = {p.stem for p in (OFF / "images/test_full").glob("*.jpg")}
def source(stem):
    if stem.startswith("PartA"): return "SCUT-HEAD Part A"
    if stem.startswith("PartB"): return "SCUT-HEAD Part B"
    return "helmet imagery"

pairs = []
for k, (a, b, c) in enumerate(zip(ii[keep], jj[keep], ncc[keep]), 1):
    pairs.append({"pair_id": f"X{k:03d}", "shwd": stems_a[a], "external": names_b[b],
                  "hamming": int(d[a, b]), "ncc": round(float(c), 4),
                  "shwd_split": "test" if stems_a[a] in test_names else "trainval",
                  "shwd_source": source(stems_a[a])})
pairs.sort(key=lambda r: (r["hamming"], -r["ncc"]))
for k, r in enumerate(pairs, 1): r["pair_id"] = f"X{k:03d}"
json.dump({"external_collection": "keremberke/hard-hat-detection, test split",
           "external_images": len(names_b), "shwd_images": len(stems_a),
           "params": {"hamming": HAMMING_T, "corr_min": CORR_MIN},
           "stage1_candidates": int(len(ii)), "confirmed_pairs": len(pairs),
           "shwd_images_matched": len({r["shwd"] for r in pairs}),
           "external_images_matched": len({r["external"] for r in pairs}),
           "pairs": pairs}, open(AUD / "cross_pairs.json", "w"), indent=1)
with open(AUD / "cross_pairs.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["pair_id", "shwd", "external", "hamming", "ncc", "shwd_split", "shwd_source", "decision", "note"])
    for r in pairs: w.writerow([r["pair_id"], r["shwd"], r["external"], r["hamming"], r["ncc"], r["shwd_split"], r["shwd_source"], "", ""])

# contact sheets
md = AUD / "montages_cross"; md.mkdir(exist_ok=True)
S, PAD = 240, 8
per = 6
for s in range(0, len(pairs), per):
    chunk = pairs[s:s + per]
    sheet = Image.new("RGB", (2 * S + 3 * PAD, len(chunk) * (S + 26) + PAD), "white")
    dr = ImageDraw.Draw(sheet)
    for r_i, r in enumerate(chunk):
        y = PAD + r_i * (S + 26)
        for c_i, p in enumerate((SHWD_IMG / f"{r['shwd']}.jpg", EXT / r["external"])):
            im = Image.open(p).convert("RGB"); im.thumbnail((S, S))
            sheet.paste(im, (PAD + c_i * (S + PAD), y + 18))
        dr.text((PAD, y + 2), f"{r['pair_id']}   H={r['hamming']}  NCC={r['ncc']:.3f}   left: SHWD {r['shwd']} ({r['shwd_split']}, {r['shwd_source']})   right: external", fill="black")
    sheet.save(md / f"cross_{s // per + 1:02d}.jpg", quality=90)
print("sheets ->", md)
print(json.dumps({k: sum(1 for r in pairs if r["shwd_split"] == k) for k in ("trainval", "test")}))
print(json.dumps({k: sum(1 for r in pairs if r["shwd_source"] == k) for k in ("helmet imagery", "SCUT-HEAD Part A", "SCUT-HEAD Part B")}))
