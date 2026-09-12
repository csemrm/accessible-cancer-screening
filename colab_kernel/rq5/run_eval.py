"""RQ5: does composing calibration + conformal prediction + explainability +
selective prediction degrade anything, and does combining two native
uncertainty signals (temp-scaled confidence, conformal set size) into one
abstention rule beat either alone (H5)?

Assumes setup.py already ran in this session (manifest_csv, device, model,
herlev_rows, mendeley_rows in the persistent namespace) and that
colab_kernel/rq2/common_eval.py has already been exec'd (EVAL_LOADERS,
evaluate_method -- reused unchanged, since it only needs those same globals).
"""

import json

import numpy as np
import torch

from cervical_screening.uncertainty import calibration
from cervical_screening.uncertainty.conformal import (
    nonconformity_scores, stable_conformal_threshold, prediction_sets, average_set_size,
)
from cervical_screening.uncertainty.selective import selective_report
from cervical_screening.uncertainty.train_variants import predict_baseline
from cervical_screening.eval.cross_dataset import group_to_binary_probs
from cervical_screening.integration.combined_selective import rank_normalize, combined_confidence
from cervical_screening.integration.pipeline_latency import measure_bare_forward_latency, measure_explained_latency

ALPHA = 0.1

# --- fit temperature + conformal threshold on the calibration split ---
_, cal_labels, cal_logits = predict_baseline(model, EVAL_LOADERS["calibration"], device)
fitted_temperature = calibration.fit_temperature(cal_logits, cal_labels)
cal_probs = calibration.apply_temperature(cal_logits, fitted_temperature)
print("fitted temperature:", fitted_temperature, flush=True)

conformal_scores = nonconformity_scores(cal_probs, cal_labels)
conformal_threshold = stable_conformal_threshold(conformal_scores, alpha=ALPHA)
print("conformal threshold:", conformal_threshold, flush=True)

# --- structural check: calibration must not change the argmax (H5's
# "no degradation" claim, verified rather than assumed) ---
raw_probs = calibration.softmax(cal_logits)
assert (raw_probs.argmax(axis=1) == cal_probs.argmax(axis=1)).all(), \
    "temperature scaling changed at least one argmax -- H5's calibration-is-a-no-op assumption is false here"
print("VERIFIED: temperature scaling changes no argmax on the calibration split", flush=True)


def _signals_report(probs5, labels_or_binary_correct, is_binary_task: bool, binary_probs=None):
    """Builds conf-only / set-only / combined selective_report()s for one split.
    probs5: (n, 5) temp-scaled softmax over the 5 SIPaKMeD classes (conformal
    sets are always computed at this native 5-class level, since that's what
    the calibration split's threshold was fit against, regardless of whether
    the eval task itself is 5-class or the harmonized binary task)."""
    pred_sets = prediction_sets(probs5, conformal_threshold)
    set_size = pred_sets.sum(axis=1).astype(float)

    if is_binary_task:
        conf = binary_probs.max(axis=1)
        correct = labels_or_binary_correct
    else:
        conf = probs5.max(axis=1)
        correct = labels_or_binary_correct

    combined = combined_confidence(conf, set_size)

    return {
        "avg_conformal_set_size": average_set_size(pred_sets),
        "confidence_only": selective_report(conf, correct.astype(float)),
        "set_size_only": selective_report(rank_normalize(-set_size), correct.astype(float)),
        "combined": selective_report(combined, correct.astype(float)),
    }


# --- in-distribution test set ---
test_probs, test_labels, test_logits = predict_baseline(model, EVAL_LOADERS["test"], device,
                                                          temperature=fitted_temperature)
test_correct = (test_probs.argmax(axis=1) == test_labels)
in_dist_report = _signals_report(test_probs, test_correct, is_binary_task=False)

# --- shift sets (Herlev, Mendeley LBC), binary-harmonized ---
shift_reports = {}
for shift_name in ("herlev", "mendeley_lbc"):
    shift_probs5, shift_binary_labels, _logits = predict_baseline(
        model, EVAL_LOADERS[shift_name], device, temperature=fitted_temperature)
    binary_probs = group_to_binary_probs(shift_probs5)
    binary_correct = (binary_probs.argmax(axis=1) == shift_binary_labels)
    shift_reports[shift_name] = _signals_report(shift_probs5, binary_correct, is_binary_task=True,
                                                  binary_probs=binary_probs)

# --- latency: cost of returning a prediction with an explanation, on CPU ---
cpu_model = model.to("cpu")
bare_latency = measure_bare_forward_latency(cpu_model, input_size=224, n_warmup=5, n_iters=30)
explained_latency = measure_explained_latency(cpu_model, input_size=224, grid_size=14, n_warmup=5, n_iters=30)
model.to(device)  # move back for anything downstream in this session
print("bare forward median ms:", bare_latency["median_latency_ms"], flush=True)
print("forward+explanation median ms:", explained_latency["median_latency_ms"], flush=True)

report = {
    "fitted_temperature": fitted_temperature,
    "conformal_threshold": conformal_threshold,
    "argmax_unchanged_by_calibration": True,
    "in_distribution": in_dist_report,
    "shift": shift_reports,
    "latency_cpu_ms": {
        "bare_forward_median": bare_latency["median_latency_ms"],
        "forward_plus_explanation_median": explained_latency["median_latency_ms"],
    },
}

print("RESULT_JSON_START")
print(json.dumps(report, indent=2))
print("RESULT_JSON_END")
