# -*- coding: utf-8 -*-
"""Duplicate / near-duplicate audit between two image collections.

Two-stage matching, as required for a defensible audit:
  Stage 1 (recall): perceptual hash (pHash, 64-bit) with Hamming distance <= T.
                    Tolerant to rescaling, re-compression, minor crops.
  Stage 2 (precision): confirm every stage-1 candidate with a second,
                    independent signal -- normalised grayscale correlation on a
                    32x32 thumbnail -- and keep only pairs above CORR_MIN.

Reports: exact duplicates (identical bytes), near-duplicates, per-collection
intra-set duplicates, and a random sample of matched pairs for manual review.

Usage:
  python audit_overlap.py --a " + str(DATA_ROOT / "SHWD/VOC2028/JPEGImages") + " --a-name SHWD \
                          --b " + str(DATA_ROOT / "HHW/data") + " --b-name HardHatWorkers
"""

import os
from pathlib import Path

# Configure with environment variables, or edit the two defaults below.
#   DATA_ROOT : directory holding the datasets (must contain VOC2028/ for SHWD)
#   OUT_DIR   : directory for generated result files
DATA_ROOT = Path(os.environ.get("AUDIT_DATA_ROOT", "data"))
OUT_DIR = Path(os.environ.get("AUDIT_OUT_DIR", "results"))
OUT_DIR.mkdir(parents=True, exist_ok=True)


import argparse, hashlib, json, random, sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
import imagehash

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
HAMMING_T = 6        # stage-1 threshold on 64-bit pHash
CORR_MIN = 0.90      # stage-2 confirmation threshold
THUMB = 32


def list_images(root):
    root = Path(root)
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in IMG_EXT)


def signature(path):
    """Return (md5, phash_uint64, thumb_vector) or None on failure."""
    try:
        raw = path.read_bytes()
        md5 = hashlib.md5(raw).hexdigest()
        im = Image.open(path)
        im.draft("L", (256, 256))          # fast approximate decode for JPEG
        im = im.convert("L")
        ph = imagehash.phash(im, hash_size=8)
        bits = ph.hash.flatten()
        ph_int = int("".join("1" if b else "0" for b in bits), 2)
        t = np.asarray(im.resize((THUMB, THUMB), Image.BILINEAR), dtype=np.float32).ravel()
        t -= t.mean()
        n = np.linalg.norm(t)
        t = t / n if n > 1e-6 else t
        return md5, ph_int, t
    except Exception:
        return None


def build(paths, label):
    md5s, phs, thumbs, kept = [], [], [], []
    for i, p in enumerate(paths):
        s = signature(p)
        if s is None:
            continue
        md5s.append(s[0]); phs.append(s[1]); thumbs.append(s[2]); kept.append(p)
        if (i + 1) % 1000 == 0:
            print(f"  [{label}] hashed {i+1}/{len(paths)}", flush=True)
    return md5s, np.array(phs, dtype=np.uint64), np.vstack(thumbs), kept


def hamming_matrix(a, b, chunk=512):
    """Pairwise Hamming distance between two uint64 arrays, chunked."""
    out = np.empty((len(a), len(b)), dtype=np.uint8)
    for i in range(0, len(a), chunk):
        x = a[i:i + chunk, None]
        out[i:i + chunk] = np.bitwise_count(np.bitwise_xor(x, b[None, :])).astype(np.uint8)
    return out


def intra_duplicates(phs, thumbs, paths):
    d = hamming_matrix(phs, phs)
    np.fill_diagonal(d, 99)
    pairs = []
    for i, j in zip(*np.where(d <= HAMMING_T)):
        if i < j and float(thumbs[i] @ thumbs[j]) >= CORR_MIN:
            pairs.append((int(i), int(j)))
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True); ap.add_argument("--a-name", default="A")
    ap.add_argument("--b", required=True); ap.add_argument("--b-name", default="B")
    ap.add_argument("--out", default="audit_result.json")
    ap.add_argument("--sample", type=int, default=12)
    args = ap.parse_args()

    pa, pb = list_images(args.a), list_images(args.b)
    print(f"{args.a_name}: {len(pa)} images\n{args.b_name}: {len(pb)} images", flush=True)
    if not pa or not pb:
        sys.exit("one collection is empty")

    print("hashing...", flush=True)
    md5a, pha, tha, pa = build(pa, args.a_name)
    md5b, phb, thb, pb = build(pb, args.b_name)

    # ---- exact byte-identical
    setb = defaultdict(list)
    for j, m in enumerate(md5b):
        setb[m].append(j)
    exact = [(i, setb[m][0]) for i, m in enumerate(md5a) if m in setb]

    # ---- near-duplicate: stage 1 then stage 2
    print("matching...", flush=True)
    d = hamming_matrix(pha, phb)
    cand = np.argwhere(d <= HAMMING_T)
    near = []
    for i, j in cand:
        c = float(tha[i] @ thb[j])
        if c >= CORR_MIN:
            near.append((int(i), int(j), int(d[i, j]), round(c, 4)))

    a_matched = {i for i, _, _, _ in near}
    b_matched = {j for _, j, _, _ in near}

    intra_a = intra_duplicates(pha, tha, pa)
    intra_b = intra_duplicates(phb, thb, pb)

    res = {
        "a_name": args.a_name, "b_name": args.b_name,
        "a_count": len(pa), "b_count": len(pb),
        "exact_identical_pairs": len(exact),
        "near_duplicate_pairs": len(near),
        "a_images_with_match": len(a_matched),
        "b_images_with_match": len(b_matched),
        "a_overlap_pct": round(100 * len(a_matched) / len(pa), 2),
        "b_overlap_pct": round(100 * len(b_matched) / len(pb), 2),
        "intra_a_duplicate_pairs": len(intra_a),
        "intra_b_duplicate_pairs": len(intra_b),
        "params": {"hamming_threshold": HAMMING_T, "corr_min": CORR_MIN},
    }

    random.seed(0)
    res["sample_pairs"] = [
        {"a": str(pa[i]), "b": str(pb[j]), "hamming": h, "corr": c}
        for i, j, h, c in random.sample(near, min(args.sample, len(near)))
    ]

    Path(args.out).write_text(json.dumps(res, indent=2), encoding="utf-8")
    print("\n" + "=" * 60)
    for k in ("a_count", "b_count", "exact_identical_pairs", "near_duplicate_pairs",
              "a_overlap_pct", "b_overlap_pct", "intra_a_duplicate_pairs",
              "intra_b_duplicate_pairs"):
        print(f"  {k:28s} {res[k]}")
    print("=" * 60)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
