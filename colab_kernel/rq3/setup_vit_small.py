"""RQ3 second-backbone setup: identical to setup.py, but trains ViT-Small
instead of DeiT-Tiny, to test whether relevancy propagation's clean win
(Section RQ3 results) generalizes beyond one specific pure-ViT backbone.
ViT-Small is structurally identical to DeiT-Tiny for this purpose (timm
VisionTransformer, global_pool="token", num_prefix_tokens=1, 12 blocks --
verified directly, not assumed), so the explainability code
(attention_rollout/relevancy_propagation/gradcam, including the
CLS-pooling Grad-CAM fix) needs no changes, only a different MODEL_NAME.

Run this once per session via `colab exec -f setup_vit_small.py`; every
later step assumes the globals it sets (rp, cs, manifest_csv,
nucleus_manifest_csv, crops_root, device, model) are already in the
kernel's persistent namespace.
"""

import sys
from pathlib import Path

sys.path.insert(0, "/content")
import run_pipeline as rp

import cervical_screening as cs  # noqa: F401
from cervical_screening.data.crop_sipakmed import CLASSES, crop_class

print("downloading datasets...", flush=True)
rp.download_all()

crops_root = rp.DATA_ROOT / "processed/sipakmed_crops"
nucleus_manifest_csv = rp.DATA_ROOT / "processed/nucleus_manifest.csv"

import csv

raw_root = rp.DATA_ROOT / "raw/sipakmed_full"
nucleus_rows: list[dict] = []
counts = {}
for class_name in CLASSES:
    n = crop_class(raw_root, crops_root, class_name, margin=4.0, nucleus_rows=nucleus_rows)
    counts[class_name] = n
print("crop counts:", counts, "total:", sum(counts.values()), flush=True)
assert sum(counts.values()) == 4049

nucleus_manifest_csv.parent.mkdir(parents=True, exist_ok=True)
with open(nucleus_manifest_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["crop_path", "crop_width", "crop_height", "nucleus_polygons"])
    writer.writeheader()
    writer.writerows(nucleus_rows)
print(f"nucleus manifest: {len(nucleus_rows)} rows -> {nucleus_manifest_csv}", flush=True)

manifest_csv = rp.DATA_ROOT / "processed/sipakmed_splits.csv"
rp.build_manifest(crops_root, manifest_csv)

device = rp.select_device()
import torch
print("device:", device, "-", torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU", flush=True)

from cervical_screening.uncertainty.train_variants import train_baseline

MODEL_NAME = "vit_small_patch16_224"
print(f"training {MODEL_NAME}...", flush=True)
result = train_baseline(MODEL_NAME, manifest_csv, device, epochs=30, batch_size=32, patience=7, seed=13)
model = result["model"]
print("best_val_loss:", result["best_val_loss"], flush=True)

print("SETUP_DONE", flush=True)
