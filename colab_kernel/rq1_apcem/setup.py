"""RQ1 3rd-shift-dataset setup: download + crop SIPaKMeD (same as the
original RQ1/RQ4 pipeline), unzip the pre-uploaded APCEM archive (no
Kaggle mirror exists for it, unlike SIPaKMeD/Herlev/Mendeley LBC -- it must
be uploaded directly via `colab upload` before running this script), and
build both the SIPaKMeD training manifest and the APCEM binary shift
manifest.

Run once per session via `colab exec -f setup.py`, after uploading
apcem_clean.zip to /content/apcem_clean.zip. Every later step assumes the
globals it sets (rp, cs, manifest_csv, apcem_rows, device) are already in
the kernel's persistent namespace.
"""

import subprocess
import sys
sys.path.insert(0, "/content")
import run_pipeline as rp

import cervical_screening as cs  # noqa: F401
from cervical_screening.data.harmonize_labels import APCEM_TO_BINARY, scan_binary_manifest

print("downloading SIPaKMeD...", flush=True)
rp.kaggle_download("marinaeplissiti/sipakmed", rp.DATA_ROOT / "raw/sipakmed_full")

crops_root = rp.DATA_ROOT / "processed/sipakmed_crops"
counts = rp.crop_all(rp.DATA_ROOT / "raw/sipakmed_full", crops_root)
print("crop counts:", counts, "total:", sum(counts.values()), flush=True)
assert sum(counts.values()) == 4049

manifest_csv = rp.DATA_ROOT / "processed/sipakmed_splits.csv"
rp.build_manifest(crops_root, manifest_csv)

print("unzipping APCEM...", flush=True)
apcem_root = rp.DATA_ROOT / "raw/apcem"
apcem_root.mkdir(parents=True, exist_ok=True)
subprocess.run(["unzip", "-oq", "/content/apcem_clean.zip", "-d", str(apcem_root)], check=True)

apcem_rows = scan_binary_manifest(apcem_root / "Enhanced Cervical Cytology Image dataset", APCEM_TO_BINARY)
print("apcem:", len(apcem_rows), flush=True)
assert len(apcem_rows) == 4253

device = rp.select_device()
import torch
print("device:", device, "-", torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU", flush=True)

print("SETUP_DONE", flush=True)
