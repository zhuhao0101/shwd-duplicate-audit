# -*- coding: utf-8 -*-
"""Cluster and verify intra-dataset near-duplicates; export montages for manual review."""

import os
from pathlib import Path

# Configure with environment variables, or edit the two defaults below.
#   DATA_ROOT : directory holding the datasets (must contain VOC2028/ for SHWD)
#   OUT_DIR   : directory for generated result files
DATA_ROOT = Path(os.environ.get("AUDIT_DATA_ROOT", "data"))
OUT_DIR = Path(os.environ.get("AUDIT_OUT_DIR", "results"))
OUT_DIR.mkdir(parents=True, exist_ok=True)


import argparse, json, random
from pathlib import Path
import numpy as np
from PIL import Image
import imagehash

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
HAMMING_T, CORR_MIN, THUMB = 6, 0.90, 32


def sig(p):
    try:
        im = Image.open(p); im.draft("L", (256, 256)); im = im.convert("L")
        b = imagehash.phash(im, hash_size=8).hash.flatten()
        v = int("".join("1" if x else "0" for x in b), 2)
        t = np.asarray(im.resize((THUMB, THUMB), Image.BILINEAR), np.float32).ravel()
        t -= t.mean(); n = np.linalg.norm(t)
        return v, (t / n if n > 1e-6 else t), float(np.asarray(im).std())
    except Exception:
        return None


ap = argparse.ArgumentParser()
ap.add_argument("--dir", required=True); ap.add_argument("--name", default="DS")
ap.add_argument("--out", default="intra.json"); ap.add_argument("--montage-dir", default="montages")
a = ap.parse_args()

paths = sorted(p for p in Path(a.dir).rglob("*") if p.suffix.lower() in IMG_EXT)
print(f"{a.name}: {len(paths)} images", flush=True)
ph, th, sd, keep = [], [], [], []
for i, p in enumerate(paths):
    s = sig(p)
    if s: ph.append(s[0]); th.append(s[1]); sd.append(s[2]); keep.append(p)
    if (i + 1) % 2000 == 0: print(f"  hashed {i+1}", flush=True)
ph = np.array(ph, np.uint64); th = np.vstack(th); sd = np.array(sd)

# pairwise
d = np.empty((len(ph), len(ph)), np.uint8)
for i in range(0, len(ph), 512):
    d[i:i+512] = np.bitwise_count(np.bitwise_xor(ph[i:i+512, None], ph[None, :])).astype(np.uint8)
np.fill_diagonal(d, 99)

pairs = []
for i, j in zip(*np.where(d <= HAMMING_T)):
    if i < j:
        c = float(th[i] @ th[j])
        if c >= CORR_MIN:
            pairs.append((int(i), int(j), int(d[i, j]), round(c, 4)))
print("confirmed pairs:", len(pairs), flush=True)

# connected components
parent = list(range(len(keep)))
def find(x):
    while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
    return x
for i, j, _, _ in pairs:
    ri, rj = find(i), find(j)
    if ri != rj: parent[ri] = rj
comp = {}
for i in range(len(keep)):
    comp.setdefault(find(i), []).append(i)
clusters = sorted([c for c in comp.values() if len(c) > 1], key=len, reverse=True)

sizes = [len(c) for c in clusters]
res = {
    "name": a.name, "images": len(keep), "confirmed_pairs": len(pairs),
    "clusters": len(clusters),
    "images_in_clusters": int(sum(sizes)),
    "pct_images_in_clusters": round(100 * sum(sizes) / len(keep), 2),
    "largest_clusters": sizes[:15],
    "low_texture_images(std<20)": int((sd < 20).sum()),
    "params": {"hamming": HAMMING_T, "corr_min": CORR_MIN},
}
print(json.dumps(res, indent=2))

# montages for manual verification: 6 random clusters, up to 5 images each
md = Path(a.montage_dir); md.mkdir(parents=True, exist_ok=True)
random.seed(0)
for k, cl in enumerate(random.sample(clusters, min(6, len(clusters)))):
    sel = cl[:5]
    ims = [Image.open(keep[i]).convert("RGB").resize((200, 200)) for i in sel]
    sheet = Image.new("RGB", (200 * len(ims), 200), "white")
    for n, im in enumerate(ims): sheet.paste(im, (200 * n, 0))
    sheet.save(md / f"{a.name}_cluster{k}_size{len(cl)}.jpg", quality=88)
    res.setdefault("montage_members", {})[f"cluster{k}"] = [keep[i].name for i in sel]
print("montages ->", md)
Path(a.out).write_text(json.dumps(res, indent=2), encoding="utf-8")
