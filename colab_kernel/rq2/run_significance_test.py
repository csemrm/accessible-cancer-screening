"""RQ2 follow-up: is MC Dropout's apparent Mendeley LBC advantage over the
deep ensemble (0.285 vs 0.473 risk-coverage AUC) real, or within bootstrap
noise? Paired bootstrap significance test on the same per-sample
predictions.

Assumes setup.py, common_eval.py, run_mc_dropout.py (mc_dropout_model in
namespace), and the 5 ensemble members (ensemble_models, either trained
in-session or loaded from checkpoints via run_ensemble_eval_from_ckpts.py)
have all already run in this session.
"""

import json

from cervical_screening.eval.cross_dataset import group_to_binary_probs
from cervical_screening.uncertainty.significance import paired_bootstrap_rc_auc_test
from cervical_screening.uncertainty.train_variants import predict_mc_dropout, predict_ensemble

# --- MC Dropout on Mendeley LBC ---
mc_probs, mc_entropy, mc_labels = predict_mc_dropout(mc_dropout_model, EVAL_LOADERS["mendeley_lbc"],
                                                       device, n_passes=20)
mc_binary_probs = group_to_binary_probs(mc_probs)
mc_correct = (mc_binary_probs.argmax(axis=1) == mc_labels).astype(float)
mc_confidence = -mc_entropy

# --- Deep ensemble on Mendeley LBC ---
ens_probs, ens_variance, ens_labels = predict_ensemble(ensemble_models, EVAL_LOADERS["mendeley_lbc"], device)
ens_binary_probs = group_to_binary_probs(ens_probs)
ens_correct = (ens_binary_probs.argmax(axis=1) == ens_labels).astype(float)
ens_confidence = -ens_variance

assert (mc_labels == ens_labels).all(), "MC Dropout and ensemble evaluated on different sample order"

result = paired_bootstrap_rc_auc_test(
    confidence_a=mc_confidence, correct_a=mc_correct,
    confidence_b=ens_confidence, correct_b=ens_correct,
    n_boot=2000, seed=13,
)
result["method_a"] = "mc_dropout"
result["method_b"] = "deep_ensemble"
result["condition"] = "mendeley_lbc"

print("RESULT_JSON_START")
print(json.dumps(result, indent=2))
print("RESULT_JSON_END")
