"""Seed-configurable variant of run_one_model.py, for measuring seed-to-seed
variance on the two closest APCEM calls flagged in paper1.tex's Threats to
Validity (EfficientViT-B0 0.4481 vs DeiT-Tiny 0.4469 -- a 0.0012 gap).

Set TARGET_MODEL and SEED via a preceding inline `colab exec`, e.g.:
    echo "TARGET_MODEL = 'deit_tiny_patch16_224'
SEED = 17" | colab exec -s <session> --timeout 10

Assumes setup.py already ran in this session (rp, manifest_csv, apcem_rows,
device in the persistent namespace) -- same assumption as run_one_model.py.
Each run overwrites the previous seed's checkpoint on the remote VM; only
the printed JSON (captured client-side before the next seed starts) needs
to survive.
"""

import json

import torch
from torch.utils.data import DataLoader

TARGET_MODEL = globals().get("TARGET_MODEL", "efficientvit_m0")
SEED = globals().get("SEED", 13)
print(f"=== training {TARGET_MODEL} seed={SEED} ===", flush=True)

result = rp.train_one(TARGET_MODEL, manifest_csv, device, epochs=30, batch_size=32,
                       lr=3e-4, patience=7, seed=SEED)
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
    "seed": SEED,
    "in_distribution": result["test_metrics"],
    "efficiency": result["efficiency"],
    "apcem_shift": apcem_metrics,
    "n_apcem": len(apcem_rows),
}

print("RESULT_JSON_START")
print(json.dumps(report, indent=2))
print("RESULT_JSON_END")
