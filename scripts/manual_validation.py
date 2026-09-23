# -*- coding: utf-8 -*-
"""Draw the blind manual-validation sample of the two-stage matcher.

Five strata (seed 0):
  accepted-random     100 confirmed pairs drawn uniformly            -> precision, overall
  accepted-boundary   100 confirmed pairs with Hamming 5-6, NCC < 0.93 -> precision at the weak end
  rejected-boundary    60 Stage-1 candidates rejected with 0.85 <= NCC < 0.90 -> what Stage 2 removes
  rejected-random      40 rejected candidates drawn uniformly
  nearmiss             50 pairs at Hamming 7-8 with NCC >= 0.90        -> what Stage 1 misses

The 350 pairs are shuffled into one sequence and shown WITHOUT their stratum, distance or
correlation (blind rating). Raters fill the decision column of rating_sheet.csv:
  ND  near-duplicate: same scene, same viewpoint, same people in essentially the same
      positions (frames seconds apart, or a re-encoded / rescaled copy)
  SC  same camera or scene, but the people or their positions differ substantially
  NO  unrelated content
The stratum and the matcher values are kept in validation_key.csv and joined afterwards
by summarize_validation.py.
"""

import os
from pathlib import Path

# Configure with environment variables, or edit the two defaults below.
#   DATA_ROOT : directory holding the datasets (must contain SHWD/VOC2028 for SHWD)
#   OUT_DIR   : directory for generated result files
DATA_ROOT = Path(os.environ.get("AUDIT_DATA_ROOT", "data"))
OUT_DIR = Path(os.environ.get("AUDIT_OUT_DIR", "results"))
OUT_DIR.mkdir(parents=True, exist_ok=True)


import json, csv, random, io, sys
from pathlib import Path
from PIL import Image, ImageDraw
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

AUD = OUT_DIR
IMG = Path(str(DATA_ROOT / "SHWD/VOC2028/JPEGImages"))
OUTD = AUD / "validation_sheets"; OUTD.mkdir(exist_ok=True)
rng = random.Random(0)

conf = json.load(open(AUD / "pairs_confirmed.json"))["pairs"]
rej = json.load(open(AUD / "pairs_rejected.json"))["pairs"]
nm = json.load(open(AUD / "pairs_nearmiss.json"))["pairs"]
print(f"confirmed {len(conf)}  rejected {len(rej)}  near-miss {len(nm)}")

strata = {
    "accepted-random": rng.sample(conf, 100),
    "accepted-boundary": rng.sample([p for p in conf if p[2] >= 5 and p[3] < 0.93], 100),
    "rejected-boundary": rng.sample([p for p in rej if 0.85 <= p[3] < 0.90], 60),
    "rejected-random": rng.sample(rej, 40),
    "nearmiss": rng.sample(nm, 50),
}
for k, v in strata.items():
    print(f"  {k:18s} {len(v)}")

items = [(k, p) for k, v in strata.items() for p in v]
rng.shuffle(items)
key_rows, sheet_rows = [], []
for n, (k, (a, b, h, c)) in enumerate(items, 1):
    pid = f"V{n:03d}"
    key_rows.append([pid, k, a, b, h, c])
    sheet_rows.append([pid, a, b, "", ""])
with open(AUD / "validation_key.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["pair_id", "stratum", "image_a", "image_b", "hamming", "ncc"]); w.writerows(key_rows)
with open(AUD / "rating_sheet.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["pair_id", "image_a", "image_b", "decision", "note"]); w.writerows(sheet_rows)

S, PAD, PER = 300, 10, 5
for s in range(0, len(items), PER):
    chunk = items[s:s + PER]
    sheet = Image.new("RGB", (2 * S + 3 * PAD, len(chunk) * (S + 24) + PAD), "white")
    dr = ImageDraw.Draw(sheet)
    for r_i, (k, (a, b, h, c)) in enumerate(chunk):
        y = PAD + r_i * (S + 24)
        for c_i, stem in enumerate((a, b)):
            im = Image.open(IMG / f"{stem}.jpg").convert("RGB"); im.thumbnail((S, S))
            sheet.paste(im, (PAD + c_i * (S + PAD), y + 16))
        dr.text((PAD, y + 1), f"V{s + r_i + 1:03d}", fill="black")
    sheet.save(OUTD / f"sheet_{s // PER + 1:02d}.jpg", quality=88)
print(f"{len(items)} pairs -> {OUTD} ({(len(items) + PER - 1) // PER} sheets); rating_sheet.csv; validation_key.csv")
