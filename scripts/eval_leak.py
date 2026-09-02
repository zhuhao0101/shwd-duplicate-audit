# -*- coding: utf-8 -*-
"""Evaluate one checkpoint on full / leaked / clean SHWD test subsets."""

import os
from pathlib import Path

# Configure with environment variables, or edit the two defaults below.
#   DATA_ROOT : directory holding the datasets (must contain VOC2028/ for SHWD)
#   OUT_DIR   : directory for generated result files
DATA_ROOT = Path(os.environ.get("AUDIT_DATA_ROOT", "data"))
OUT_DIR = Path(os.environ.get("AUDIT_OUT_DIR", "results"))
OUT_DIR.mkdir(parents=True, exist_ok=True)


import json
from ultralytics import YOLO

W = str(DATA_ROOT / "leak_runs/official/weights/best.pt")

def main():
    res = {}
    for v in ("test_full", "test_leaked", "test_clean"):
        m = YOLO(W)
        r = m.val(data=rf"" + str(DATA_ROOT / "SHWD_official/shwd_{v}.yaml"), split="test",
                  imgsz=640, batch=1, verbose=False, plots=False, workers=0)
        res[v] = {"mAP50": round(r.box.map50*100, 2), "mAP50_95": round(r.box.map*100, 2),
                  "P": round(r.box.mp*100, 2), "R": round(r.box.mr*100, 2),
                  "helmet_AP50": round(float(r.box.ap50[0])*100, 2),
                  "head_AP50": round(float(r.box.ap50[1])*100, 2)}
        print(v, res[v], flush=True)
    res["inflation_mAP50"] = round(res["test_full"]["mAP50"] - res["test_clean"]["mAP50"], 2)
    res["leaked_minus_clean_mAP50"] = round(res["test_leaked"]["mAP50"] - res["test_clean"]["mAP50"], 2)
    res["inflation_mAP50_95"] = round(res["test_full"]["mAP50_95"] - res["test_clean"]["mAP50_95"], 2)
    res["leaked_minus_clean_mAP50_95"] = round(res["test_leaked"]["mAP50_95"] - res["test_clean"]["mAP50_95"], 2)
    json.dump(res, open(str(OUT_DIR) + "/leak_effect.json", "w"), indent=2)
    print("\n=== HEADLINE ===")
    print("full - clean   (mAP@0.5)     :", res["inflation_mAP50"])
    print("leaked - clean (mAP@0.5)     :", res["leaked_minus_clean_mAP50"])
    print("full - clean   (mAP@0.5:0.95):", res["inflation_mAP50_95"])
    print("leaked - clean (mAP@0.5:0.95):", res["leaked_minus_clean_mAP50_95"])

if __name__ == "__main__":
    main()
