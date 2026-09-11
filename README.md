# Trustworthy, Resource-Efficient Vision Transformers for Cervical Cytology Screening

Research code for the flagship project under the program *"Trustworthy and
Resource-Efficient AI for Early Cancer Screening in Rural and Medically
Underserved Clinical Settings."* See `01_accessible_cancer_screening.md` for
the original proposal and the published proposal artifact (linked in project
notes) for the full literature-grounded revision — gap analysis, architecture
design, research questions/hypotheses, experimental design, and the
national-importance evidence review.

## Status

Currently executing the Month 1–2 timeline item: dataset acquisition and
pipeline scaffolding for the primary application (cervical cytology). See
`docs/dataset_provenance.md` for what's been verified so far and what's
still in progress.

## Layout

```
src/cervical_screening/
  data/
    crop_sipakmed.py   # derive native-resolution single-cell crops from raw SIPaKMeD
    make_splits.py      # stratified train/val/calibration/test manifest
    dataset.py           # torch Dataset reading the manifest
    transforms.py         # ImageNet-normalized train/eval transforms
  models/
    registry.py           # RQ1 baseline registry (CNN / full ViT / efficient ViT), all timm tags verified against the pinned timm version
  eval/
    metrics.py             # sensitivity/specificity/F1/AUROC/AUPRC/balanced accuracy — not accuracy-only, per proposal S14
    efficiency.py           # measured params/CPU latency/RSS delta — not asserted from architecture name
  train.py                  # RQ1 baseline training + evaluation entrypoint
```

## Setup

```
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
```

Requires a working `~/.kaggle/kaggle.json` for dataset acquisition.

## Running the RQ1 baseline benchmark (once data is in place)

```
python -m cervical_screening.data.crop_sipakmed \
  --raw-root data/raw/sipakmed_full --out-root data/processed/sipakmed_crops

python -m cervical_screening.data.make_splits \
  --crops-root data/processed/sipakmed_crops --out-csv data/processed/sipakmed_splits.csv

python -m cervical_screening.train \
  --model mobilevit_s --manifest data/processed/sipakmed_splits.csv --out-dir runs/mobilevit_s
```

Repeat the last command once per entry in `MODEL_REGISTRY`
(`src/cervical_screening/models/registry.py`) to reproduce the full
efficiency/accuracy baseline table.
