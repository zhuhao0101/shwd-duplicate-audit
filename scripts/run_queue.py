# -*- coding: utf-8 -*-
"""Run the revision experiments one after another on the single GPU.

Order reflects priority for the response to reviewers:
  1. cluster-disjoint experiment, multi-session hold-out (D vs L, 3 seeds)
  2. YOLOv8s, A vs B, 3 seeds
  3. cluster-disjoint experiment, largest-cluster hold-out (robustness)
  4. RT-DETR-l, A vs B, 1 seed (batch 8, falling back to 4 on memory errors)
Every phase is resumable and writes its own result file when complete.
"""

import os
from pathlib import Path

# Configure with environment variables, or edit the two defaults below.
#   DATA_ROOT : directory holding the datasets (must contain SHWD/VOC2028 for SHWD)
#   OUT_DIR   : directory for generated result files
DATA_ROOT = Path(os.environ.get("AUDIT_DATA_ROOT", "data"))
OUT_DIR = Path(os.environ.get("AUDIT_OUT_DIR", "results"))
OUT_DIR.mkdir(parents=True, exist_ok=True)


import subprocess, sys, time, io
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
HERE = Path(__file__).parent
LOG = HERE / "queue_log.txt"

PHASES = [
    ("disjoint-multi", ["run_disjoint.py", "--variant", "multi"]),
    ("yolov8s", ["run_arch.py", "--model", "yolov8s"]),
    ("disjoint-giant", ["run_disjoint.py", "--variant", "giant"]),
    ("rtdetr-l", ["run_arch.py", "--model", "rtdetr-l", "--seeds", "0", "--batch", "8"]),
]


def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run(name, args):
    log(f"START {name}")
    t0 = time.time()
    with open(HERE / f"log_{name}.txt", "a", encoding="utf-8") as out:
        rc = subprocess.call([sys.executable] + args, cwd=str(HERE), stdout=out, stderr=subprocess.STDOUT)
    log(f"END {name} rc={rc} elapsed={(time.time()-t0)/3600:.2f} h")
    return rc


if __name__ == "__main__":
    for name, args in PHASES:
        rc = run(name, args)
        if rc != 0 and name == "rtdetr-l":
            log("rtdetr-l failed at batch 8; retrying with batch 4")
            run("rtdetr-l-b4", ["run_arch.py", "--model", "rtdetr-l", "--seeds", "0", "--batch", "4"])
    log("QUEUE COMPLETE")
