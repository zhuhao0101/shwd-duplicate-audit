# -*- coding: utf-8 -*-
"""Compare content statistics of the leaked vs clean test subsets."""

import os
from pathlib import Path

# Configure with environment variables, or edit the two defaults below.
#   DATA_ROOT : directory holding the datasets (must contain VOC2028/ for SHWD)
#   OUT_DIR   : directory for generated result files
DATA_ROOT = Path(os.environ.get("AUDIT_DATA_ROOT", "data"))
OUT_DIR = Path(os.environ.get("AUDIT_OUT_DIR", "results"))
OUT_DIR.mkdir(parents=True, exist_ok=True)


import numpy as np
from pathlib import Path
R = Path(str(DATA_ROOT / "SHWD_official/labels"))
for split in ("test_leaked", "test_clean"):
    objs, areas, cls = [], [], []
    for f in (R/split).glob("*.txt"):
        ls = [l.split() for l in f.read_text().split("\n") if l.strip()]
        objs.append(len(ls))
        for p in ls:
            cls.append(int(p[0])); areas.append(float(p[3])*float(p[4]))
    a = np.array(areas); o = np.array(objs); c = np.array(cls)
    print(f"{split:12s} imgs={len(o):5d}  objs/img={o.mean():6.2f}  "
          f"median_box_area={np.median(a)*100:6.3f}%  "
          f"tiny(<0.1% area)={100*(a<0.001).mean():5.1f}%  "
          f"helmet={100*(c==0).mean():5.1f}%  head={100*(c==1).mean():5.1f}%")
