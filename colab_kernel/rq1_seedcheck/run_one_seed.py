"""Trains one model at one seed and evaluates on Herlev + Mendeley LBC.
Set TARGET_MODEL and SEED via a preceding inline `colab exec`, e.g.:
    echo "TARGET_MODEL = 'mobilenetv2_100'
SEED = 17" | colab exec -s <session> --timeout 10

Assumes setup.py already ran in this session (rp, manifest_csv, device).
"""

import json

TARGET_MODEL = globals().get("TARGET_MODEL", "mobilenetv2_100")
SEED = globals().get("SEED", 13)
print(f"=== training {TARGET_MODEL} seed={SEED} ===", flush=True)

result = rp.train_one(TARGET_MODEL, manifest_csv, device, epochs=30, batch_size=32,
                       lr=3e-4, patience=7, seed=SEED)
shift = rp.run_cross_dataset_eval(TARGET_MODEL)

report = {
    "model": TARGET_MODEL,
    "seed": SEED,
    "in_distribution": result["test_metrics"],
    "shift_results": shift["shift_results"],
}

print("RESULT_JSON_START")
print(json.dumps(report, indent=2))
print("RESULT_JSON_END")
