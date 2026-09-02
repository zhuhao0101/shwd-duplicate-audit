# -*- coding: utf-8 -*-
"""Repeat the causal leakage experiment over 3 seeds and aggregate mean +/- std.

Configs (training data differs; test sets are identical):
  A = full official trainval (6064, contains 1534 near-duplicates of test)
  B = deduplicated trainval  (4530, contains 0)
  C = random size-matched    (4530, retains 1132)
Resumable: any run whose best.pt already exists is skipped.
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
CAU = Path(str(DATA_ROOT / "SHWD_causal"))
SEEDS = [0, 1, 2]
TRAIN_DIR = {"A": OFF/"images/train", "B": CAU/"images/trainB", "C": CAU/"images/trainC"}
SEED0_NAME = {"A": "official", "B": "dedup", "C": "control"}
TESTS = ("test_leaked", "test_clean", "test_full")


def yaml_for(cfg, variant):
    p = CAU / f"_{cfg}_{variant}.yaml"
    p.write_text(f"train: {TRAIN_DIR[cfg].as_posix()}\n"
                 f"val: {(OFF/'images'/variant).as_posix()}\n"
                 f"test: {(OFF/'images'/variant).as_posix()}\n"
                 f"names:\n  0: helmet\n  1: head\n", encoding="ascii")
    return str(p)


def run_name(cfg, seed):
    return SEED0_NAME[cfg] if seed == 0 else f"{cfg}_s{seed}"


def main():
    for seed in SEEDS:
        for cfg in ("A", "B", "C"):
            w = RUNS / run_name(cfg, seed) / "weights/best.pt"
            if w.exists():
                print(f"skip {cfg} seed{seed} (exists)", flush=True); continue
            print(f"=== training {cfg} seed {seed} ===", flush=True)
            YOLO("yolov8n.pt").train(
                data=yaml_for(cfg, "test_full"), epochs=40, imgsz=640, batch=16,
                seed=seed, project=str(RUNS), name=run_name(cfg, seed), exist_ok=True,
                pretrained=True, deterministic=False, patience=50, workers=4)
            print(f"TRAINED {cfg} seed{seed}", flush=True)

    raw = {}
    for cfg in ("A", "B", "C"):
        for seed in SEEDS:
            w = RUNS / run_name(cfg, seed) / "weights/best.pt"
            if not w.exists(): continue
            for v in TESTS:
                r = YOLO(str(w)).val(data=yaml_for(cfg, v), split="test", imgsz=640,
                                     batch=1, verbose=False, plots=False, workers=0)
                raw.setdefault(cfg, {}).setdefault(v, []).append(
                    {"seed": seed, "mAP50": round(r.box.map50*100,2),
                     "mAP50_95": round(r.box.map*100,2),
                     "helmet_AP50": round(float(r.box.ap50[0])*100,2),
                     "head_AP50": round(float(r.box.ap50[1])*100,2)})
                print(cfg, v, raw[cfg][v][-1], flush=True)

    agg = {}
    for cfg, d in raw.items():
        agg[cfg] = {}
        for v, lst in d.items():
            agg[cfg][v] = {k: {"mean": round(float(np.mean([x[k] for x in lst])),2),
                               "std": round(float(np.std([x[k] for x in lst], ddof=1)) if len(lst)>1 else 0.0, 2),
                               "n": len(lst)}
                           for k in ("mAP50","mAP50_95","helmet_AP50","head_AP50")}
    json.dump({"raw": raw, "agg": agg}, open(str(OUT_DIR) + "/seeds_result.json","w"), indent=2)

    print("\n=== MEAN +/- STD (n seeds) ===")
    for v in TESTS:
        print(f"\n[{v}]")
        for cfg in ("A","B","C"):
            a = agg[cfg][v]
            print(f"  {cfg}: mAP50={a['mAP50']['mean']:.2f}+/-{a['mAP50']['std']:.2f}  "
                  f"mAP50:95={a['mAP50_95']['mean']:.2f}+/-{a['mAP50_95']['std']:.2f}  (n={a['mAP50']['n']})")
    print("\n=== EFFECT SIZES (mAP50) ===")
    for v in TESTS:
        ab = agg["A"][v]["mAP50"]["mean"] - agg["B"][v]["mAP50"]["mean"]
        cb = agg["C"][v]["mAP50"]["mean"] - agg["B"][v]["mAP50"]["mean"]
        print(f"  {v:12s}  A-B={ab:+6.2f}   C-B={cb:+6.2f}")

if __name__ == "__main__":
    main()
