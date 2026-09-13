"""Seed-variance setup: identical data prep to the original RQ1/RQ4 Herlev+
Mendeley LBC sweep (docs/rq1_results.md), used to measure seed-to-seed
variance on the two closest calls flagged in paper1.tex's Threats to
Validity section (MobileViTv2 0.5418 vs MobileNetV2 0.5391 on Mendeley LBC,
a 0.0027 gap). No APCEM data needed here.

Run once per session via `colab exec -f setup.py`; every later script in
this directory assumes rp, manifest_csv, device already in the kernel's
persistent namespace.
"""

import sys
sys.path.insert(0, "/content")
import run_pipeline as rp

print("downloading datasets...", flush=True)
rp.download_all()

crops_root = rp.DATA_ROOT / "processed/sipakmed_crops"
counts = rp.crop_all(rp.DATA_ROOT / "raw/sipakmed_full", crops_root)
print("crop counts:", counts, "total:", sum(counts.values()), flush=True)
assert sum(counts.values()) == 4049

manifest_csv = rp.DATA_ROOT / "processed/sipakmed_splits.csv"
rp.build_manifest(crops_root, manifest_csv)

device = rp.select_device()
import torch
print("device:", device, "-", torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU", flush=True)

print("SETUP_DONE", flush=True)
