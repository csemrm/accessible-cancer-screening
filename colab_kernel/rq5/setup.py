"""RQ5 setup: download SIPaKMeD + Herlev + Mendeley LBC, crop SIPaKMeD,
build the 5-class training manifest and the binary shift manifests, select
device, train DeiT-Tiny (same backbone RQ3 used, for the same reason: it's a
standard ViT that relevancy propagation is actually defined for -- see
cervical_screening.explainability's docstring).

Run this once per session via `colab exec -f setup.py`; every later step in
this directory assumes the globals it sets (rp, cs, manifest_csv, device,
herlev_rows, mendeley_rows, model, fitted_temperature) are already in the
kernel's persistent namespace.
"""

import sys
sys.path.insert(0, "/content")
import run_pipeline as rp  # the existing RQ1 helper module, already uploaded

import cervical_screening as cs  # noqa: F401 -- side effect: registers the package
from cervical_screening.data.harmonize_labels import HERLEV_TO_BINARY, MENDELEY_LBC_TO_BINARY, scan_binary_manifest

print("downloading datasets...", flush=True)
rp.download_all()

crops_root = rp.DATA_ROOT / "processed/sipakmed_crops"
counts = rp.crop_all(rp.DATA_ROOT / "raw/sipakmed_full", crops_root)
print("crop counts:", counts, "total:", sum(counts.values()), flush=True)
assert sum(counts.values()) == 4049

manifest_csv = rp.DATA_ROOT / "processed/sipakmed_splits.csv"
rp.build_manifest(crops_root, manifest_csv)

herlev_rows = scan_binary_manifest(rp.DATA_ROOT / "raw/herlev/Herlev Dataset", HERLEV_TO_BINARY)
mendeley_rows = scan_binary_manifest(rp.DATA_ROOT / "raw/mendeley_lbc", MENDELEY_LBC_TO_BINARY)
print("herlev:", len(herlev_rows), "mendeley_lbc:", len(mendeley_rows), flush=True)
assert len(herlev_rows) == 917

device = rp.select_device()
import torch
print("device:", device, "-", torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU", flush=True)

from cervical_screening.uncertainty.train_variants import train_baseline, predict_baseline
from cervical_screening.uncertainty import calibration

MODEL_NAME = "deit_tiny_patch16_224"
print(f"training {MODEL_NAME}...", flush=True)
result = train_baseline(MODEL_NAME, manifest_csv, device, epochs=30, batch_size=32, patience=7, seed=13)
model = result["model"]
print("best_val_loss:", result["best_val_loss"], flush=True)

print("SETUP_DONE", flush=True)
