"""RQ2 setup: download datasets, crop SIPaKMeD, build the SIPaKMeD training
manifest and the Herlev/Mendeley LBC binary shift manifests, select device,
import the real cervical_screening.uncertainty package (uploaded as a zip
and unzipped to /content/cervical_screening -- see docs/colab_cli_pipeline.md
for why this replaced re-inlining the same logic in every Colab script).

Run this once per session via `colab exec -f setup.py`; every later step in
this directory assumes the globals it sets (rp, cs, manifest_csv, device,
herlev_rows, mendeley_rows, ALL_RESULTS) are already in the kernel's
persistent namespace.
"""

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, "/content")
import run_pipeline as rp  # the existing RQ1 helper module, already uploaded

import cervical_screening as cs  # noqa: F401 -- imported for its side effect of registering the package
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

ALL_RESULTS = {}
print("SETUP_DONE", flush=True)
