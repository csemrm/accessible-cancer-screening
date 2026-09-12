"""RQ2 method 4/4 (evaluation): run once after all 5 run_ensemble_member.py
calls have appended to `ensemble_models`."""

import json

from cervical_screening.uncertainty.train_variants import predict_ensemble

assert len(ensemble_models) == 5, f"expected 5 ensemble members, got {len(ensemble_models)}"


def predict_fn(loader):
    probs, variance, labels = predict_ensemble(ensemble_models, loader, device)
    confidence = -variance
    return probs, confidence, labels


ensemble_report = evaluate_method("deep_ensemble", predict_fn, EVAL_LOADERS)
ensemble_report["n_members"] = len(ensemble_models)
ALL_RESULTS["deep_ensemble"] = ensemble_report

print("RESULT_JSON_START")
print(json.dumps(ensemble_report))
print("RESULT_JSON_END")

print("ALL_RESULTS_JSON_START")
print(json.dumps(ALL_RESULTS))
print("ALL_RESULTS_JSON_END")
