"""RQ2 method 4/4 (evaluation, checkpointed variant): rebuild all 5 ensemble
members from uploaded state_dict checkpoints (see run_ensemble_member_ckpt.py)
instead of assuming they trained in this same kernel session, then evaluate.

Requires setup.py and common_eval.py already run in this session, and all 5
checkpoint files already uploaded to /content/ (ENSEMBLE_SEEDS names them).
"""

import json

import torch

from cervical_screening.models.registry import build_model
from cervical_screening.uncertainty.train_variants import predict_ensemble

ENSEMBLE_SEEDS = [13, 17, 29, 41, 53]
MODEL = "efficientvit_m0"

ensemble_models = []
for seed in ENSEMBLE_SEEDS:
    m = build_model(MODEL).to(device)
    state = torch.load(f"/content/ensemble_member_seed{seed}.pt", map_location=device)
    m.load_state_dict(state)
    m.eval()
    ensemble_models.append(m)
print(f"loaded {len(ensemble_models)} ensemble members from checkpoints", flush=True)

assert len(ensemble_models) == 5


def predict_fn(loader):
    probs, variance, labels = predict_ensemble(ensemble_models, loader, device)
    confidence = -variance
    return probs, confidence, labels


ensemble_report = evaluate_method("deep_ensemble", predict_fn, EVAL_LOADERS)
ensemble_report["n_members"] = len(ensemble_models)
ensemble_report["seeds"] = ENSEMBLE_SEEDS
ALL_RESULTS["deep_ensemble"] = ensemble_report

print("RESULT_JSON_START")
print(json.dumps(ensemble_report))
print("RESULT_JSON_END")
