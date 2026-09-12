"""Trains one model (TARGET_MODEL, set via a preceding inline `colab exec`
since sys.argv reflects the Jupyter kernel's own launch args, not this
script's) and evaluates it zero-shot on APCEM, the 3rd cross-dataset shift
target. Reuses run_pipeline.train_one() unchanged (it already saves a
checkpoint to RUNS_ROOT/<model_name>/model.pt), so a session death after
this script completes does not lose the trained weights -- only a death
mid-training does.

Assumes setup.py already ran in this session (rp, manifest_csv,
apcem_rows, device in the persistent namespace).
"""

import json

import torch
from torch.utils.data import DataLoader

TARGET_MODEL = globals().get("TARGET_MODEL", "efficientvit_m0")
print(f"=== training {TARGET_MODEL} ===", flush=True)

result = rp.train_one(TARGET_MODEL, manifest_csv, device, epochs=30, batch_size=32,
                       lr=3e-4, patience=7, seed=13)
print("in-distribution test_metrics:", result["test_metrics"], flush=True)

input_size = rp.input_size_for(TARGET_MODEL)
model = rp.build_model(TARGET_MODEL, pretrained=False)
model.load_state_dict(torch.load(rp.RUNS_ROOT / TARGET_MODEL / "model.pt", map_location="cpu"))
model = model.to(device)

eval_tf = rp.build_transform(input_size, train=False)
apcem_loader = DataLoader(rp.BinaryDataset(apcem_rows, eval_tf), batch_size=32, shuffle=False, num_workers=2)
y_prob, y_true = rp.predict_binary(model, apcem_loader, device)
apcem_metrics = rp.compute_metrics(y_true, y_prob)
print("apcem shift metrics:", apcem_metrics, flush=True)

report = {
    "model": TARGET_MODEL,
    "in_distribution": result["test_metrics"],
    "efficiency": result["efficiency"],
    "apcem_shift": apcem_metrics,
    "n_apcem": len(apcem_rows),
}

print("RESULT_JSON_START")
print(json.dumps(report, indent=2))
print("RESULT_JSON_END")
