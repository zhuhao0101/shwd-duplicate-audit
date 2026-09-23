# -*- coding: utf-8 -*-
"""Train and evaluate the cluster-disjoint experiment (D vs L, three seeds).

Protocol identical to the main A/B/C campaign (YOLOv8n, 40 epochs, 640 px,
batch 16, pretrained COCO weights). No evaluation data is seen during training:
the validation entry of the data file points at a 32-image stub of training
images and every reported number comes from the final-epoch weights (last.pt).

Usage: python run_disjoint.py --variant multi|giant
"""

import os
from pathlib import Path

# Configure with environment variables, or edit the two defaults below.
#   DATA_ROOT : directory holding the datasets (must contain SHWD/VOC2028 for SHWD)
#   OUT_DIR   : directory for generated result files
DATA_ROOT = Path(os.environ.get("AUDIT_DATA_ROOT", "data"))
OUT_DIR = Path(os.environ.get("AUDIT_OUT_DIR", "results"))
OUT_DIR.mkdir(parents=True, exist_ok=True)


import argparse, json, io, sys
from pathlib import Path
import numpy as np
from ultralytics import YOLO
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)

ap = argparse.ArgumentParser(); ap.add_argument("--variant", default="multi"); a = ap.parse_args()
VAR = a.variant
DATA = Path(str(DATA_ROOT / "SHWD_disjoint") + ("" if VAR == "multi" else "_giant"))
OFF = Path(str(DATA_ROOT / "SHWD_official"))
PROV = Path(str(DATA_ROOT / "SHWD_prov"))
RUNS = Path(str(DATA_ROOT / "disjoint_runs"))
OUT = OUT_DIR / f"disjoint_result_{VAR}.json"
SEEDS = [0, 1, 2]
CFGS = {"D": "trainD", "L": "trainL"}
NAMES = "names:\n  0: helmet\n  1: head\n"


def _complete(run_dir, epochs=40):
    """A run counts as trained only if its results.csv shows every epoch."""
    f = run_dir / "results.csv"
    try:
        return sum(1 for ln in open(f)) - 1 >= epochs
    except FileNotFoundError:
        return False


def yaml_train(cfg):
    p = DATA / f"_{cfg}_train.yaml"
    p.write_text(f"train: {(DATA/'images'/CFGS[cfg]).as_posix()}\n"
                 f"val: {(DATA/'images/valstub').as_posix()}\n" + NAMES, encoding="ascii")
    return str(p)


EVAL = {"testH": DATA / "images/testH",
        "test_clean": OFF / "images/test_clean",
        "test_helmet": PROV / "images/test_helmet",
        "test_scutA": PROV / "images/test_scutA"}


def yaml_eval(name):
    p = DATA / f"_eval_{name}.yaml"
    p.write_text(f"train: {EVAL[name].as_posix()}\nval: {EVAL[name].as_posix()}\n"
                 f"test: {EVAL[name].as_posix()}\n" + NAMES, encoding="ascii")
    return str(p)


def main():
    for seed in SEEDS:
        for cfg in CFGS:
            name = f"{VAR}_{cfg}_s{seed}"
            w = RUNS / name / "weights/last.pt"
            if _complete(RUNS / name):
                print(f"skip {name}"); continue
            if w.exists():
                print(f"=== resuming {name} ===")
                YOLO(str(w)).train(resume=True)
            else:
                print(f"=== training {name} ===")
                YOLO("yolov8n.pt").train(data=yaml_train(cfg), epochs=40, imgsz=640, batch=16, seed=seed,
                                         project=str(RUNS), name=name, exist_ok=True, pretrained=True,
                                         deterministic=False, patience=50, workers=4, val=False, plots=False)
            print(f"TRAINED {name}")

    raw = {}
    for cfg in CFGS:
        for seed in SEEDS:
            w = RUNS / f"{VAR}_{cfg}_s{seed}" / "weights/last.pt"
            if not w.exists():
                print("MISSING", w); continue
            for ev in EVAL:
                r = YOLO(str(w)).val(data=yaml_eval(ev), split="test", imgsz=640, batch=1,
                                     verbose=False, plots=False, workers=0)
                per = {int(c): float(x) for c, x in zip(r.box.ap_class_index, r.box.ap50)}
                rec = {"seed": seed, "mAP50": round(r.box.map50 * 100, 2), "mAP50_95": round(r.box.map * 100, 2)}
                if 0 in per: rec["helmet_AP50"] = round(per[0] * 100, 2)
                if 1 in per: rec["head_AP50"] = round(per[1] * 100, 2)
                raw.setdefault(cfg, {}).setdefault(ev, []).append(rec)
                print(cfg, seed, ev, rec)

    agg = {c: {v: {k: {"mean": round(float(np.mean([x[k] for x in lst])), 2),
                       "std": round(float(np.std([x[k] for x in lst], ddof=1)), 2) if len(lst) > 1 else 0.0,
                       "n": len(lst)}
                   for k in ("mAP50", "mAP50_95", "helmet_AP50", "head_AP50") if all(k in x for x in lst)}
               for v, lst in d.items()} for c, d in raw.items()}
    split = json.load(open(OUT_DIR / f"disjoint_split_{VAR}.json"))
    json.dump({"variant": VAR, "weights": "last.pt", "raw": raw, "agg": agg,
               "split": {k: split[k] for k in ("H_clusters", "testH", "leak", "trainD", "trainD_partA", "trainL", "trainL_partA")}},
              open(OUT, "w"), indent=2)
    print("\n=== L - D (mAP@0.5) ===")
    for ev in EVAL:
        if ev in agg.get("D", {}) and ev in agg.get("L", {}):
            print(f"  {ev:12s} D={agg['D'][ev]['mAP50']['mean']:.2f}  L={agg['L'][ev]['mAP50']['mean']:.2f}  "
                  f"L-D={agg['L'][ev]['mAP50']['mean']-agg['D'][ev]['mAP50']['mean']:+.2f}")
    print("saved", OUT)


if __name__ == "__main__":
    main()
