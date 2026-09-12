"""Month 11 setup: download + crop SIPaKMeD, build the training manifest,
select device, train EfficientViT-M0 (RQ1's winning backbone -- this is the
flagship model the resource-efficiency and robustness story is about, unlike
RQ3/RQ5 which needed a pure-ViT backbone for explainability reasons).

Only SIPaKMeD is needed here (no Herlev/Mendeley): the degradation suite and
OOD sanity check both evaluate against the in-distribution test split plus
synthetic corruptions/images, not cross-dataset shift.

Run this once per session via `colab exec -f setup.py`; every later step in
this directory assumes the globals it sets (rp, cs, manifest_csv, device,
model) are already in the kernel's persistent namespace.
"""

import sys
sys.path.insert(0, "/content")
import run_pipeline as rp

import cervical_screening as cs  # noqa: F401

print("downloading SIPaKMeD...", flush=True)
rp.kaggle_download("marinaeplissiti/sipakmed", rp.DATA_ROOT / "raw/sipakmed_full")

crops_root = rp.DATA_ROOT / "processed/sipakmed_crops"
counts = rp.crop_all(rp.DATA_ROOT / "raw/sipakmed_full", crops_root)
print("crop counts:", counts, "total:", sum(counts.values()), flush=True)
assert sum(counts.values()) == 4049

manifest_csv = rp.DATA_ROOT / "processed/sipakmed_splits.csv"
rp.build_manifest(crops_root, manifest_csv)

device = rp.select_device()
import torch
print("device:", device, "-", torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU", flush=True)

from cervical_screening.uncertainty.train_variants import train_baseline

MODEL_NAME = "efficientvit_m0"
print(f"training {MODEL_NAME}...", flush=True)
result = train_baseline(MODEL_NAME, manifest_csv, device, epochs=30, batch_size=32, patience=7, seed=13)
model = result["model"]
print("best_val_loss:", result["best_val_loss"], flush=True)

print("SETUP_DONE", flush=True)
