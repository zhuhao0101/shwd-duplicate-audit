# -*- coding: utf-8 -*-
"""Re-evaluate every configuration using last.pt (final epoch, no model selection).

The original runs used val = test_full, so best.pt was selected on the test data.
last.pt is the final-epoch weight and involves no test-set-based selection.
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
import numpy as np
from ultralytics import YOLO

RUNS = Path(str(DATA_ROOT / "leak_runs"))
OFF = Path(str(DATA_ROOT / "SHWD_official"))
PROV = Path(str(DATA_ROOT / "SHWD_prov"))
SEED0 = {"A": "official", "B": "dedup", "C": "control"}
SEEDS = [0, 1, 2]
MAIN = ("test_leaked", "test_clean", "test_full")
PRV = ("test_scutA", "test_helmet")

def rn(cfg, s): return SEED0[cfg] if s == 0 else f"{cfg}_s{s}"

def yaml_main(v):
    p = OFF / f"_lp_{v}.yaml"
    p.write_text(f"train: {(OFF/'images/train').as_posix()}\n"
                 f"val: {(OFF/'images'/v).as_posix()}\n"
                 f"test: {(OFF/'images'/v).as_posix()}\n"
                 f"names:\n  0: helmet\n  1: head\n", encoding="ascii")
    return str(p)

def agg(raw):
    return {c: {v: {k: {"mean": round(float(np.mean([x[k] for x in lst])), 2),
                        "std": round(float(np.std([x[k] for x in lst], ddof=1)), 2)}
                    for k in ("mAP50", "mAP50_95", "helmet_AP50", "head_AP50")
                    if all(k in x for x in lst)}
                for v, lst in d.items()} for c, d in raw.items()}

def run(subsets, yaml_fn, out):
    raw = {}
    for cfg in ("A", "B", "C"):
        for s in SEEDS:
            w = RUNS / rn(cfg, s) / "weights/last.pt"
            if not w.exists():
                print("MISSING", w); continue
            for v in subsets:
                r = YOLO(str(w)).val(data=yaml_fn(v), split="test", imgsz=640,
                                     batch=1, verbose=False, plots=False, workers=0)
                per = {int(c): float(a) for c, a in zip(r.box.ap_class_index, r.box.ap50)}
                rec = {"seed": s, "mAP50": round(r.box.map50*100, 2),
                       "mAP50_95": round(r.box.map*100, 2)}
                if 0 in per: rec["helmet_AP50"] = round(per[0]*100, 2)
                if 1 in per: rec["head_AP50"] = round(per[1]*100, 2)
                raw.setdefault(cfg, {}).setdefault(v, []).append(rec)
                print(cfg, s, v, rec, flush=True)
    a = agg(raw)
    json.dump({"raw": raw, "agg": a, "weights": "last.pt"}, open(out, "w"), indent=2)
    return a

def main():
    print("########## MAIN (official split subsets) ##########", flush=True)
    m = run(MAIN, yaml_main, str(OUT_DIR) + "/seeds_result_lastpt.json")
    print("\n########## PROVENANCE ##########", flush=True)
    p = run(PRV, lambda v: str(PROV/f"{v}.yaml"), str(OUT_DIR) + "/provenance_result_lastpt.json")

    print("\n=== last.pt : mAP@0.5 mean ± std ===")
    for v in MAIN + PRV:
        src = m if v in MAIN else p
        print(f"\n[{v}]")
        for c in "ABC":
            a = src[c][v]["mAP50"]; print(f"  {c}: {a['mean']:6.2f} ± {a['std']:.2f}")
        print(f"  A-B = {src['A'][v]['mAP50']['mean']-src['B'][v]['mAP50']['mean']:+.2f}   "
              f"C-B = {src['C'][v]['mAP50']['mean']-src['B'][v]['mAP50']['mean']:+.2f}")
    print("\n=== mAP@0.5:0.95 effect on contaminated subset (checks the disputed sentence) ===")
    for c in "AC":
        print(f"  {c}-B = {m[c]['test_leaked']['mAP50_95']['mean']-m['B']['test_leaked']['mAP50_95']['mean']:+.2f}")

if __name__ == "__main__":
    main()
