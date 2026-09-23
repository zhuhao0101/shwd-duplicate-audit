# -*- coding: utf-8 -*-
"""Repeat the A (official trainval) vs B (deduplicated) contrast with other detectors.

  --model yolov8s   : same family, 3x the parameters, three seeds
  --model rtdetr-l  : detection transformer, single seed (memory-bound on an 8 GB GPU)

Same 40-epoch, 640 px protocol; validation entry points at a stub of training
images; final-epoch weights are evaluated on the official test subsets and on the
provenance subsets.
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
from ultralytics import YOLO, RTDETR
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="yolov8s")
ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--imgsz", type=int, default=640)
ap.add_argument("--epochs", type=int, default=40)
a = ap.parse_args()
MODEL = a.model
OFF = Path(str(DATA_ROOT / "SHWD_official"))
CAU = Path(str(DATA_ROOT / "SHWD_causal"))
PROV = Path(str(DATA_ROOT / "SHWD_prov"))
DJ = Path(str(DATA_ROOT / "SHWD_disjoint"))          # reuse its 32-image validation stub
RUNS = Path(str(DATA_ROOT / "arch_runs"))
OUT = OUT_DIR / f"arch_result_{MODEL}.json"
TRAIN = {"A": OFF / "images/train", "B": CAU / "images/trainB"}
NAMES = "names:\n  0: helmet\n  1: head\n"
EVAL = {"test_leaked": OFF / "images/test_leaked", "test_clean": OFF / "images/test_clean",
        "test_full": OFF / "images/test_full", "test_scutA": PROV / "images/test_scutA",
        "test_helmet": PROV / "images/test_helmet"}


def _complete(run_dir, epochs=40):
    """A run counts as trained only if its results.csv shows every epoch."""
    f = run_dir / "results.csv"
    try:
        return sum(1 for ln in open(f)) - 1 >= epochs
    except FileNotFoundError:
        return False


def yaml_train(cfg):
    p = RUNS / f"_{MODEL}_{cfg}_train.yaml"; p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(f"train: {TRAIN[cfg].as_posix()}\nval: {(DJ/'images/valstub').as_posix()}\n" + NAMES, encoding="ascii")
    return str(p)


def yaml_eval(name):
    p = RUNS / f"_eval_{name}.yaml"
    p.write_text(f"train: {EVAL[name].as_posix()}\nval: {EVAL[name].as_posix()}\ntest: {EVAL[name].as_posix()}\n" + NAMES,
                 encoding="ascii")
    return str(p)


def load(weights):
    return RTDETR(weights) if MODEL.startswith("rtdetr") else YOLO(weights)


def main():
    for seed in a.seeds:
        for cfg in TRAIN:
            name = f"{MODEL}_{cfg}_s{seed}"
            w = RUNS / name / "weights/last.pt"
            if _complete(RUNS / name, a.epochs):
                print(f"skip {name}"); continue
            if w.exists():
                print(f"=== resuming {name} ===")
                load(str(w)).train(resume=True)
            else:
                print(f"=== training {name} ===")
                load(f"{MODEL}.pt").train(data=yaml_train(cfg), epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, seed=seed,
                                          project=str(RUNS), name=name, exist_ok=True, pretrained=True,
                                          deterministic=False, patience=50, workers=4, val=False, plots=False)
            print(f"TRAINED {name}")

    raw = {}
    for cfg in TRAIN:
        for seed in a.seeds:
            w = RUNS / f"{MODEL}_{cfg}_s{seed}" / "weights/last.pt"
            if not w.exists():
                print("MISSING", w); continue
            for ev in EVAL:
                r = load(str(w)).val(data=yaml_eval(ev), split="test", imgsz=a.imgsz, batch=1,
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
    json.dump({"model": MODEL, "weights": "last.pt", "seeds": a.seeds, "batch": a.batch,
               "imgsz": a.imgsz, "epochs": a.epochs, "raw": raw, "agg": agg},
              open(OUT, "w"), indent=2)
    print(f"\n=== {MODEL}: A - B (mAP@0.5) ===")
    for ev in EVAL:
        if ev in agg.get("A", {}) and ev in agg.get("B", {}):
            print(f"  {ev:12s} A={agg['A'][ev]['mAP50']['mean']:.2f}  B={agg['B'][ev]['mAP50']['mean']:.2f}  "
                  f"A-B={agg['A'][ev]['mAP50']['mean']-agg['B'][ev]['mAP50']['mean']:+.2f}")
    print("saved", OUT)


if __name__ == "__main__":
    main()
