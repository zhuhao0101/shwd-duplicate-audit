# -*- coding: utf-8 -*-
"""Evaluate every trained configuration/seed on the provenance-split test subsets."""

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
import numpy as np
from ultralytics import YOLO

RUNS = Path(str(DATA_ROOT / "leak_runs"))
PROV = Path(str(DATA_ROOT / "SHWD_prov"))
SEED0 = {"A": "official", "B": "dedup", "C": "control"}
SEEDS = [0, 1, 2]
SUBSETS = ("test_scutA", "test_helmet")

def name(cfg, s): return SEED0[cfg] if s == 0 else f"{cfg}_s{s}"

def main():
    raw = {}
    for cfg in ("A", "B", "C"):
        for s in SEEDS:
            w = RUNS / name(cfg, s) / "weights/best.pt"
            if not w.exists(): continue
            for v in SUBSETS:
                r = YOLO(str(w)).val(data=str(PROV/f"{v}.yaml"), split="test", imgsz=640,
                                     batch=1, verbose=False, plots=False, workers=0)
                per = {int(c): float(a) for c, a in zip(r.box.ap_class_index, r.box.ap50)}
                rec = {"seed": s, "mAP50": round(r.box.map50*100, 2)}
                if 0 in per: rec["helmet_AP50"] = round(per[0]*100, 2)
                if 1 in per: rec["head_AP50"] = round(per[1]*100, 2)
                raw.setdefault(cfg, {}).setdefault(v, []).append(rec)
                print(cfg, s, v, raw[cfg][v][-1], flush=True)
    agg = {c: {v: {k: {"mean": round(float(np.mean([x[k] for x in lst])), 2),
                       "std": round(float(np.std([x[k] for x in lst], ddof=1)), 2)}
                   for k in ("mAP50", "helmet_AP50", "head_AP50")
                   if all(k in x for x in lst)}
               for v, lst in d.items()} for c, d in raw.items()}
    json.dump({"raw": raw, "agg": agg}, open(str(OUT_DIR) + "/provenance_result.json", "w"), indent=2)
    print("\n=== mAP@0.5 by provenance (mean ± std, 3 seeds) ===")
    for v in SUBSETS:
        print(f"\n[{v}]")
        for c in "ABC":
            a = agg[c][v]["mAP50"]; print(f"  {c}: {a['mean']:6.2f} ± {a['std']:.2f}")
        print(f"  A-B = {agg['A'][v]['mAP50']['mean']-agg['B'][v]['mAP50']['mean']:+.2f}   "
              f"C-B = {agg['C'][v]['mAP50']['mean']-agg['B'][v]['mAP50']['mean']:+.2f}")

if __name__ == "__main__":
    main()
