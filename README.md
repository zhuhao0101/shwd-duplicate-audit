# Near-duplicate audit of the SHWD benchmark

Code and result artifacts for a near-duplicate audit of the Safety Helmet Wearing
Dataset (SHWD) and for a controlled experiment measuring the effect of the
train/test contamination it reveals.

## Headline measurements

| Quantity | Value |
|---|---|
| Images belonging to a near-duplicate cluster | 2,288 / 7,581 (30.2%) |
| Near-duplicate clusters | 193 |
| Largest cluster (one fixed camera) | 802 images |
| Official test images with a near-duplicate in trainval | 421 / 1,517 (27.8%) |
| Stability across matching thresholds | 22.4% – 29.3% |
| mAP@0.5 inflation, contaminated test subset | ~4.4 pp |
| mAP@0.5 inflation, full official test set | ~1.5–2.0 pp |
| Training images removable with no clean-subset loss | 1,534 (25.3%) |

## Layout

```
scripts/   audit and experiment code (CPU only, except the training runs)
results/   measured outputs in JSON
splits/    test_clean.txt - the cleaned 1,096-image evaluation split (image stems)
```

## Requirements

```
pip install -r requirements.txt
```

## Data

Obtain SHWD (VOC2028 layout) from its official distribution and place it so that
`$AUDIT_DATA_ROOT/VOC2028/{Annotations,JPEGImages,ImageSets}` exists. No images are
redistributed here.

## Configuration

Every script reads two environment variables, with the defaults shown:

```
AUDIT_DATA_ROOT   dataset directory        (default: ./data)
AUDIT_OUT_DIR     output directory         (default: ./results)
```

## Reproducing

Audit only (no GPU, a few minutes):

```
python scripts/leakage.py          # hashes, train->test leakage, cleaned split
python scripts/intra_analysis.py   # whole-dataset clustering
python scripts/audit_overlap.py    # cross-collection check (optional)
```

Controlled experiment (GPU; nine training runs of roughly 50 min each):

```
python scripts/prep_official.py
python scripts/prep_causal.py
python scripts/run_seeds.py
python scripts/eval_lastpt.py      # evaluation from final-epoch weights
```

Evaluation uses final-epoch weights throughout; no checkpoint or hyperparameter
selection is performed against any test subset.

## Cleaned evaluation split

`splits/test_clean.txt` lists the 1,096 official-test image stems that have no
confirmed near-duplicate inside the official trainval split (pHash Hamming distance
<= 6, confirmed by 32x32 thumbnail normalised cross-correlation >= 0.90). Reporting
results on this split alongside the official one is recommended.

## License

MIT for code and result files. Image lists reference an externally distributed
dataset; no images are included.
