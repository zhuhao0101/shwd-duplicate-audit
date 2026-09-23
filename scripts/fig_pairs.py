# -*- coding: utf-8 -*-
"""Figure: representative accepted, rejected and near-miss pairs with their matcher values.

Pairs are taken from the blind validation sample (validation_key.csv) so that every example
shown is also part of the reported manual rating. Every annotated head/helmet box is
pixelated before display, using the dataset's own bounding-box annotations.
"""

import os
from pathlib import Path

# Configure with environment variables, or edit the two defaults below.
#   DATA_ROOT : directory holding the datasets (must contain SHWD/VOC2028 for SHWD)
#   OUT_DIR   : directory for generated result files
DATA_ROOT = Path(os.environ.get("AUDIT_DATA_ROOT", "data"))
OUT_DIR = Path(os.environ.get("AUDIT_OUT_DIR", "results"))
OUT_DIR.mkdir(parents=True, exist_ok=True)


import csv, os, io, sys
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

AUD = r"results"
IMG = str(DATA_ROOT / "SHWD/VOC2028/JPEGImages")
ANN = str(DATA_ROOT / "SHWD/VOC2028/Annotations")
OUT = r"figures"
INK, MUT = "#1A1A1A", "#5A5A5A"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "svg.fonttype": "none"})

rows = list(csv.DictReader(open(os.path.join(AUD, "validation_key.csv"))))
for r in rows:
    r["hamming"] = int(r["hamming"]); r["ncc"] = float(r["ncc"])

def pick(pred):
    for r in rows:
        if pred(r): return r
    raise LookupError

sel = [
    ("accepted", pick(lambda r: r["stratum"] == "accepted-random" and r["hamming"] == 0)),
    ("accepted", pick(lambda r: r["stratum"] == "accepted-random" and r["hamming"] == 4)),
    ("accepted", pick(lambda r: r["stratum"] == "accepted-boundary" and r["ncc"] < 0.91)),
    ("rejected", pick(lambda r: r["stratum"] == "rejected-boundary" and r["ncc"] >= 0.88)),
    ("rejected", pick(lambda r: r["stratum"] == "rejected-random" and r["ncc"] < 0.80)),
    ("near-miss", pick(lambda r: r["stratum"] == "nearmiss" and r["ncc"] >= 0.95)),
]


def anonymised(stem, block=14):
    im = Image.open(os.path.join(IMG, f"{stem}.jpg")).convert("RGB")
    xml = os.path.join(ANN, f"{stem}.xml")
    if os.path.exists(xml):
        for obj in ET.parse(xml).getroot().iter("object"):
            bb = obj.find("bndbox")
            x1, y1, x2, y2 = (int(float(bb.find(k).text)) for k in ("xmin", "ymin", "xmax", "ymax"))
            x1, y1 = max(0, x1), max(0, y1); x2, y2 = min(im.width, x2), min(im.height, y2)
            if x2 - x1 < 2 or y2 - y1 < 2: continue
            reg = im.crop((x1, y1, x2, y2))
            small = reg.resize((max(1, (x2 - x1) // block), max(1, (y2 - y1) // block)), Image.BILINEAR)
            im.paste(small.resize((x2 - x1, y2 - y1), Image.NEAREST), (x1, y1))
    return im


fig, axes = plt.subplots(2, 6, figsize=(7.2, 2.25), gridspec_kw={"wspace": 0.04, "hspace": 0.55, "left": 0.01, "right": 0.99, "top": 0.90, "bottom": 0.09})
for k, (kind, r) in enumerate(sel):
    row, col = divmod(k, 3)
    for j, stem in enumerate((r["image_a"], r["image_b"])):
        ax = axes[row, col * 2 + j]
        ax.imshow(anonymised(stem)); ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set_edgecolor("#B0B0B0"); s.set_linewidth(0.6)
    axl = axes[row, col * 2]
    label = f"({chr(97 + k)}) {kind}   H = {r['hamming']}, NCC = {r['ncc']:.3f}"
    axl.set_title(label, fontsize=6.9, loc="left", pad=3, color=INK, x=0.0)
fig.text(0.5, 0.01, "Pairs (a)–(c): accepted by both stages;  (d)–(e): Stage-1 candidates rejected by Stage 2;  (f): Hamming 8, would pass Stage 2 alone.  Annotated head regions are pixelated for display.",
         ha="center", fontsize=6.4, color=MUT)
for ext in ("svg", "pdf"):
    fig.savefig(os.path.join(OUT, f"fig4_pairs.{ext}"), bbox_inches="tight", pad_inches=0.05)
fig.savefig(os.path.join(OUT, "fig4_pairs.png"), dpi=600, bbox_inches="tight", pad_inches=0.05, facecolor="white")
print("saved fig4_pairs;", [(kind, r["pair_id"], r["image_a"], r["image_b"]) for kind, r in sel])
