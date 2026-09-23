# -*- coding: utf-8 -*-
"""Cluster-disjoint leakage experiment: build training/test sets.

Question: does the A-B gap come from near-duplicate frames of the test images
being present in training, or merely from more same-domain (classroom) exposure?

Design (domain composition held fixed):
  pool   = official trainval (6064 images)
  H      = a set of near-duplicate clusters drawn from the pool, every member a
           SCUT-HEAD Part A classroom frame; each cluster is split at random into
           a TEST half and a LEAK half
  testH  = the TEST halves                            (evaluation set)
  trainD = pool minus every image of H                (cluster-disjoint: no image
           in trainD shares a cluster with any testH image)
  trainL = trainD with |LEAK| of its Part A images replaced by the LEAK halves
           (same size and same number of Part A images as trainD; the only
           difference is that trainL contains near-duplicate frames of testH)

Stage 1/2 thresholds and the hash cache are those of the main audit.
"""

import os
from pathlib import Path

# Configure with environment variables, or edit the two defaults below.
#   DATA_ROOT : directory holding the datasets (must contain SHWD/VOC2028 for SHWD)
#   OUT_DIR   : directory for generated result files
DATA_ROOT = Path(os.environ.get("AUDIT_DATA_ROOT", "data"))
OUT_DIR = Path(os.environ.get("AUDIT_OUT_DIR", "results"))
OUT_DIR.mkdir(parents=True, exist_ok=True)


import json, random, shutil, sys, io
from pathlib import Path
import numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

SRC = Path(str(DATA_ROOT / "SHWD_official"))
OUT = Path(str(DATA_ROOT / "SHWD_disjoint"))
AUD = OUT_DIR
CACHE = AUD / "hashes_shwd.npz"
HAMMING_T, CORR_MIN = 6, 0.90
SEED = 0
MIN_CLUSTER = 8         # pool members; smaller clusters cannot be split meaningfully
PLAN_ONLY = "--plan" in sys.argv
# variant "multi": hold out ~450 pool images drawn from the Part A clusters other than the
#   largest one (that cluster alone, 641 pool images, is a single fixed camera and stays in
#   training for both arms; the held-out frames therefore come from several sessions)
# variant "giant": hold out the largest cluster only, as a robustness check
VARIANT = "giant" if "--giant" in sys.argv else "multi"
TARGET_H = 450 if VARIANT == "multi" else 10**9
if VARIANT == "giant":
    OUT = Path(str(DATA_ROOT / "SHWD_disjoint_giant"))

z = np.load(CACHE, allow_pickle=True)
ph, th, stems = z["ph"], z["th"], [str(s) for s in z["stems"]]
idx = {s: i for i, s in enumerate(stems)}
N = len(stems)

# ---- all confirmed pairs over the whole dataset (Stage 1 Hamming, Stage 2 NCC)
d = np.empty((N, N), np.uint8)
for i in range(0, N, 512):
    d[i:i + 512] = np.bitwise_count(np.bitwise_xor(ph[i:i + 512, None], ph[None, :])).astype(np.uint8)
np.fill_diagonal(d, 99)
ii, jj = np.where(np.triu(d <= HAMMING_T, 1))
ncc = np.array([float(th[a] @ th[b]) for a, b in zip(ii, jj)], np.float64)
keep = ncc >= CORR_MIN
pairs = [(int(a), int(b), int(d[a, b]), round(float(c), 4)) for a, b, c in zip(ii[keep], jj[keep], ncc[keep])]
print(f"stage-1 candidates {len(ii)}  stage-2 confirmed {len(pairs)}")

parent = list(range(N))
def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]; x = parent[x]
    return x
for a, b, _, _ in pairs:
    ra, rb = find(a), find(b)
    if ra != rb: parent[ra] = rb
comp = {}
for i in range(N):
    comp.setdefault(find(i), []).append(i)
clusters = sorted([c for c in comp.values() if len(c) > 1], key=len, reverse=True)
print(f"clusters {len(clusters)}  images in clusters {sum(map(len, clusters))}")

# full cluster membership and pair list are audit artefacts in their own right
json.dump({"params": {"hamming": HAMMING_T, "corr_min": CORR_MIN},
           "clusters": [[stems[i] for i in c] for c in clusters]},
          open(AUD / "clusters_full.json", "w"), indent=0)
json.dump({"params": {"hamming": HAMMING_T, "corr_min": CORR_MIN},
           "pairs": [[stems[a], stems[b], h, c] for a, b, h, c in pairs]},
          open(AUD / "pairs_confirmed.json", "w"), indent=0)
rej = ~keep
json.dump({"note": "stage-1 candidates (Hamming <= 6) rejected by stage 2 (NCC < 0.90)",
           "pairs": [[stems[a], stems[b], int(d[a, b]), round(float(c), 4)] for a, b, c in zip(ii[rej], jj[rej], ncc[rej])]},
          open(AUD / "pairs_rejected.json", "w"), indent=0)
mi, mj = np.where(np.triu((d >= 7) & (d <= 8), 1))
mncc = np.array([float(th[a] @ th[b]) for a, b in zip(mi, mj)], np.float64)
nm = mncc >= CORR_MIN
json.dump({"note": "pairs just outside stage 1 (Hamming 7-8) that would pass stage 2 (NCC >= 0.90)",
           "n_candidates_h7_8": int(len(mi)),
           "pairs": [[stems[a], stems[b], int(d[a, b]), round(float(c), 4)] for a, b, c in zip(mi[nm], mj[nm], mncc[nm])]},
          open(AUD / "pairs_nearmiss.json", "w"), indent=0)
print(f"rejected {int(rej.sum())}  near-miss (H7-8, NCC>=0.90) {int(nm.sum())} of {len(mi)}")

# ---- pool = official trainval
pool = sorted(p.stem for p in (SRC / "images/train").glob("*.jpg"))
pool_set = set(pool)
is_partA = lambda s: s.startswith("PartA")
partA_pool = [s for s in pool if is_partA(s)]
print(f"pool {len(pool)}  Part A in pool {len(partA_pool)}")

rows = []
for k, c in enumerate(clusters):
    names = [stems[i] for i in c]
    inpool = [s for s in names if s in pool_set]
    rows.append({"cluster": k, "size": len(c), "in_pool": len(inpool),
                 "partA_frac_pool": (sum(map(is_partA, inpool)) / len(inpool)) if inpool else 0.0,
                 "members_pool": inpool})
print("\ntop clusters (size / in pool / Part A share of pool part):")
for r in rows[:25]:
    print(f"  c{r['cluster']:<4d} {r['size']:5d} {r['in_pool']:5d}  {r['partA_frac_pool']:.2f}")
eligible = [r for r in rows if r["in_pool"] >= MIN_CLUSTER and r["partA_frac_pool"] == 1.0]
largest = max(eligible, key=lambda r: r["in_pool"])
eligible = [largest] if VARIANT == "giant" else [r for r in eligible if r is not largest]
print(f"\neligible clusters (pool part >= {MIN_CLUSTER}, all Part A): {len(eligible)}, "
      f"images {sum(r['in_pool'] for r in eligible)}")
if PLAN_ONLY:
    sys.exit(0)

# ---- choose H: random eligible clusters until TARGET_H reached
rng = random.Random(SEED)
order = eligible[:]; rng.shuffle(order)
H, tot = [], 0
for r in order:
    if tot >= TARGET_H: break
    H.append(r); tot += r["in_pool"]
H_names = [s for r in H for s in r["members_pool"]]
testH, leakH = [], []
for r in H:
    m = r["members_pool"][:]; rng.shuffle(m)
    half = len(m) // 2
    testH += m[:half]; leakH += m[half:]
H_set = set(H_names)
trainD = [s for s in pool if s not in H_set]
D_partA = [s for s in trainD if is_partA(s)]
swap_out = set(rng.sample(D_partA, len(leakH)))
trainL = sorted([s for s in trainD if s not in swap_out] + leakH)

print(f"\nH: {len(H)} clusters, {len(H_names)} images -> testH {len(testH)}, leak {len(leakH)}")
print(f"trainD {len(trainD)} (Part A {len(D_partA)})")
print(f"trainL {len(trainL)} (Part A {sum(map(is_partA, trainL))}; of which leak frames {len(leakH)})")

# sanity: no trainD image shares a cluster with a testH image
cl_of = {stems[i]: k for k, c in enumerate(clusters) for i in c}
tH = {cl_of[s] for s in testH}
assert not any(cl_of.get(s) in tH for s in trainD), "trainD is not cluster-disjoint"
assert len(trainD) == len(trainL)
assert sum(map(is_partA, trainL)) == len(D_partA)

# ---- materialise
for tag, names, src in (("trainD", trainD, "train"), ("trainL", trainL, "train"), ("testH", testH, "train")):
    for sub in ("images", "labels"):
        (OUT / sub / tag).mkdir(parents=True, exist_ok=True)
    for n in names:
        shutil.copy2(SRC / "images" / src / f"{n}.jpg", OUT / "images" / tag / f"{n}.jpg")
        shutil.copy2(SRC / "labels" / src / f"{n}.txt", OUT / "labels" / tag / f"{n}.txt")
    print(f"  built {tag}: {len(names)}")
# a tiny validation stub of training images so that no evaluation data is touched during training
stub = [s for s in trainD if s in set(trainL)][:32]
for sub in ("images", "labels"):
    (OUT / sub / "valstub").mkdir(parents=True, exist_ok=True)
for n in stub:
    shutil.copy2(SRC / "images/train" / f"{n}.jpg", OUT / "images/valstub" / f"{n}.jpg")
    shutil.copy2(SRC / "labels/train" / f"{n}.txt", OUT / "labels/valstub" / f"{n}.txt")

json.dump({"variant": VARIANT, "seed": SEED, "target_H": TARGET_H, "min_cluster": MIN_CLUSTER,
           "H_clusters": [{"cluster": r["cluster"], "size": r["size"], "in_pool": r["in_pool"]} for r in H],
           "testH": len(testH), "leak": len(leakH),
           "trainD": len(trainD), "trainD_partA": len(D_partA),
           "trainL": len(trainL), "trainL_partA": sum(map(is_partA, trainL)),
           "pool": len(pool), "pool_partA": len(partA_pool),
           "testH_names": sorted(testH), "leak_names": sorted(leakH), "swapped_out": sorted(swap_out)},
          open(AUD / f"disjoint_split_{VARIANT}.json", "w"), indent=1)
print("done ->", OUT)
