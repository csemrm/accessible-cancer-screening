"""Retrain one RQ1 registry model and immediately run the cross-dataset
shift eval (S16/RQ4) against it in the same session -- combined into one
script/one exec call to minimize the number of round-trips exposed to
Colab free-tier GPU preemption, which killed five sessions in a row at
random points (15-65 min lifetimes) while running
colab_kernel/run_pipeline.py's per-model loop.

Usage: colab-cli runs this file's code inside an already-running Jupyter
kernel (not as a fresh `python script.py` subprocess), so sys.argv reflects
the *kernel's* own launch args (e.g. `-f <connection_file>.json`), not
anything passed on the `colab exec -f ...` command line -- there is no way
to pass a real CLI argument through that path. To target a model other than
the default, set a global TARGET_MODEL before running this file, e.g. via a
preceding `colab exec -s <session>` call with stdin:
    echo "TARGET_MODEL = 'mobilevit_s'" | colab exec -s <session> --timeout 10

Defaults to efficientvit_m0 -- the overall best RQ1 result (96.54% balanced
accuracy at 2.16M params) -- so the domain-shift finding first observed on
DeiT-Tiny (95.0% in-distribution -> 52.3% on Mendeley LBC, see
docs/rq1_results.md) can be checked against the top performer rather than
resting on one model.
"""

from __future__ import annotations

import json
import sys

sys.path.insert(0, "/content")
import run_pipeline as rp  # noqa: E402

DEFAULT_MODEL = "efficientvit_m0"


def main() -> None:
    model_name = globals().get("TARGET_MODEL", DEFAULT_MODEL)

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

    print(f"=== training {model_name} ===", flush=True)
    result = rp.train_one(model_name, manifest_csv, device, epochs=30, batch_size=32,
                           lr=3e-4, patience=7, seed=13)
    print("TRAIN_RESULT_START")
    print(json.dumps(result))
    print("TRAIN_RESULT_END")

    print("=== cross-dataset shift eval ===", flush=True)
    shift = rp.run_cross_dataset_eval(model_name)
    print("SHIFT_RESULT_START")
    print(json.dumps(shift))
    print("SHIFT_RESULT_END")


if __name__ == "__main__":
    main()
