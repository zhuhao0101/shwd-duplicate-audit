# -*- coding: utf-8 -*-
"""Join the blind ratings with the validation key and summarise them for the manuscript.

Inputs : rating_sheet.csv (decision filled in by the raters: ND / SC / NO),
         validation_key.csv, cross_pairs.csv (decision column for the 42 external pairs)
Outputs: validation_summary.json, cross_validation_summary.json
The script refuses to run on an incomplete sheet.
"""

import os
from pathlib import Path

# Configure with environment variables, or edit the two defaults below.
#   DATA_ROOT : directory holding the datasets (must contain SHWD/VOC2028 for SHWD)
#   OUT_DIR   : directory for generated result files
DATA_ROOT = Path(os.environ.get("AUDIT_DATA_ROOT", "data"))
OUT_DIR = Path(os.environ.get("AUDIT_OUT_DIR", "results"))
OUT_DIR.mkdir(parents=True, exist_ok=True)


import csv, json, io, sys, collections
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
AUD = OUT_DIR
CODES = ("ND", "SC", "NO")

key = {r["pair_id"]: r for r in csv.DictReader(open(AUD / "validation_key.csv"))}
rated = list(csv.DictReader(open(AUD / "rating_sheet.csv")))
missing = [r["pair_id"] for r in rated if r["decision"].strip().upper() not in CODES]
if missing:
    sys.exit(f"{len(missing)} pairs without a valid decision (first: {missing[:5]}); nothing written")

by = collections.defaultdict(lambda: collections.Counter())
for r in rated:
    by[key[r["pair_id"]]["stratum"]][r["decision"].strip().upper()] += 1
out = {"n": len(rated), "codes": {"ND": "near-duplicate", "SC": "same camera, different content", "NO": "unrelated"},
       "strata": {}}
for s, c in by.items():
    n = sum(c.values())
    out["strata"][s] = {"n": n, **{k: c[k] for k in CODES}, **{f"pct_{k}": round(100 * c[k] / n, 1) for k in CODES}}
acc = collections.Counter()
for s in ("accepted-random", "accepted-boundary"):
    acc.update({k: by[s][k] for k in CODES})
n = sum(acc.values())
out["accepted_all"] = {"n": n, **{k: acc[k] for k in CODES}, **{f"pct_{k}": round(100 * acc[k] / n, 1) for k in CODES}}
json.dump(out, open(AUD / "validation_summary.json", "w"), indent=1)
print(json.dumps(out, indent=1))

xr = list(csv.DictReader(open(AUD / "cross_pairs.csv")))
xm = [r["pair_id"] for r in xr if r["decision"].strip().upper() not in CODES]
if xm:
    print(f"cross-dataset sheet incomplete ({len(xm)} pairs); cross_validation_summary.json not written")
else:
    c = collections.Counter(r["decision"].strip().upper() for r in xr)
    xo = {"n": len(xr), **{k: c[k] for k in CODES}}
    json.dump(xo, open(AUD / "cross_validation_summary.json", "w"), indent=1)
    print("cross-dataset:", xo)
