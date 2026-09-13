"""Seed-variance check for the baseline (temp-scaled softmax) method, to
give it the same seed-to-seed rigor already applied to MC Dropout.

Set SEED via a preceding inline `colab exec`, e.g.:
    echo "SEED = 17" | colab exec -s <session> --timeout 10

Assumes rq2/setup.py and rq2/common_eval.py already ran this session.
"""

import json

from cervical_screening.uncertainty import calibration
from cervical_screening.uncertainty.train_variants import predict_baseline, train_baseline

MODEL = "efficientvit_m0"
SEED = globals().get("SEED", 13)

result = train_baseline(MODEL, manifest_csv, device, epochs=30, batch_size=32, patience=7, seed=SEED)
baseline_model = result["model"]

_, cal_labels, cal_logits = predict_baseline(baseline_model, EVAL_LOADERS["calibration"], device)
fitted_temperature = calibration.fit_temperature(cal_logits, cal_labels)


def predict_fn(loader):
    probs, labels, _logits = predict_baseline(baseline_model, loader, device, temperature=fitted_temperature)
    confidence = probs.max(axis=1)
    return probs, confidence, labels


baseline_report = evaluate_method(f"baseline_seed{SEED}", predict_fn, EVAL_LOADERS)

report = {
    "seed": SEED,
    "mendeley_lbc_rc_auc": baseline_report["shift"]["mendeley_lbc"]["selective"]["risk_coverage_auc"],
    "herlev_rc_auc": baseline_report["shift"]["herlev"]["selective"]["risk_coverage_auc"],
    "full_report": baseline_report,
}

print("RESULT_JSON_START")
print(json.dumps(report))
print("RESULT_JSON_END")
