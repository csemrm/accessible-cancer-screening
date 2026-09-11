"""Retrain the best RQ1 model (DeiT-Tiny, F1=0.9607 at 5.5M params, from the
10/12-model benchmark already completed across several preempted sessions)
and immediately run the cross-dataset shift eval (S16/RQ4) against it in the
same session -- combined into one script/one exec call to minimize the
number of round-trips exposed to Colab free-tier GPU preemption, which
killed five sessions in a row at random points (15-65 min lifetimes) while
running colab_kernel/run_pipeline.py's per-model loop.
"""

from __future__ import annotations

import json
import sys

sys.path.insert(0, "/content")
import run_pipeline as rp  # noqa: E402

BEST_MODEL = "deit_tiny_patch16_224"


def main() -> None:
    print("downloading datasets...", flush=True)
    rp.download_all()

    crops_root = rp.DATA_ROOT / "processed/sipakmed_crops"
    counts = rp.crop_all(rp.DATA_ROOT / "raw/sipakmed_full", crops_root)
    print("crop counts:", counts, "total:", sum(counts.values()), flush=True)
    assert sum(counts.values()) == 4049

    manifest_csv = rp.DATA_ROOT / "processed/sipakmed_splits.csv"
    rp.build_manifest(crops_root, manifest_csv)

    device = rp.select_device()
    print("device:", device, flush=True)

    print(f"=== training {BEST_MODEL} ===", flush=True)
    result = rp.train_one(BEST_MODEL, manifest_csv, device, epochs=30, batch_size=32,
                           lr=3e-4, patience=7, seed=13)
    print("TRAIN_RESULT_START")
    print(json.dumps(result))
    print("TRAIN_RESULT_END")

    print("=== cross-dataset shift eval ===", flush=True)
    shift = rp.run_cross_dataset_eval(BEST_MODEL)
    print("SHIFT_RESULT_START")
    print(json.dumps(shift))
    print("SHIFT_RESULT_END")


if __name__ == "__main__":
    main()
