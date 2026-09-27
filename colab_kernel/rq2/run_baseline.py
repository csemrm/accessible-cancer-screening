"""RQ2 method 1/4: plain softmax + post-hoc temperature scaling."""

import json

from cervical_screening.uncertainty import calibration
from cervical_screening.uncertainty.train_variants import predict_baseline, train_baseline

MODEL = globals().get("MODEL", "efficientvit_m0")

result = train_baseline(MODEL, manifest_csv, device, epochs=30, batch_size=32, patience=7, seed=13)
baseline_model = result["model"]

# fit temperature on the calibration split, frozen before any test/shift evaluation
_, cal_labels, cal_logits = predict_baseline(baseline_model, EVAL_LOADERS["calibration"], device)
fitted_temperature = calibration.fit_temperature(cal_logits, cal_labels)
print("fitted temperature:", fitted_temperature, flush=True)


def predict_fn(loader):
    probs, labels, _logits = predict_baseline(baseline_model, loader, device, temperature=fitted_temperature)
    confidence = probs.max(axis=1)
    return probs, confidence, labels


baseline_report = evaluate_method("baseline_temp_scaled", predict_fn, EVAL_LOADERS)
baseline_report["fitted_temperature"] = fitted_temperature
ALL_RESULTS["baseline_temp_scaled"] = baseline_report

print("RESULT_JSON_START")
print(json.dumps(baseline_report))
print("RESULT_JSON_END")
