"""Full-registry seed-variance sweep: trains ONE model at ONE seed, then
evaluates the same trained checkpoint zero-shot on all three external shift
datasets (Herlev, Mendeley LBC, APCEM) -- one training run instead of three,
since the point is measuring seed-to-seed variance, not re-deriving the
already-established shift results.

Set TARGET_MODEL and SEED via a preceding inline `colab exec`, e.g.:
    echo "TARGET_MODEL = 'resnet50'
SEED = 17" | colab exec -s <session> --timeout 10

Assumes colab_kernel/rq1_apcem/setup.py already ran in this session (rp,
manifest_csv, device, apcem_rows all in the persistent namespace -- its
rp.download_all() call also fetches Herlev + Mendeley LBC, so
rp.run_cross_dataset_eval() works without any extra setup).
"""

import json

import torch
from torch.utils.data import DataLoader

TARGET_MODEL = globals().get("TARGET_MODEL", "efficientvit_m0")
SEED = globals().get("SEED", 13)
print(f"=== training {TARGET_MODEL} seed={SEED} (full 3-way shift eval) ===", flush=True)

result = rp.train_one(TARGET_MODEL, manifest_csv, device, epochs=30, batch_size=32,
                       lr=3e-4, patience=7, seed=SEED)
print("in-distribution test_metrics:", result["test_metrics"], flush=True)

# Herlev + Mendeley LBC (reloads the just-saved checkpoint internally)
shift = rp.run_cross_dataset_eval(TARGET_MODEL)
print("herlev/mendeley shift_results:", shift["shift_results"], flush=True)

# APCEM (reuse the same checkpoint)
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
    "herlev": shift["shift_results"]["herlev"],
    "mendeley_lbc": shift["shift_results"]["mendeley_lbc"],
    "apcem": apcem_metrics,
}

print("RESULT_JSON_START")
print(json.dumps(report, indent=2))
print("RESULT_JSON_END")
