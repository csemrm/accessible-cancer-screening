"""Seed-variance check: how much does MC Dropout's Mendeley LBC risk-coverage
AUC swing across independent training runs of the identical recipe? Motivated
by the original run (0.285, paper2.tex Table 1) vs. the significance-test
retrain (0.503, both nominally seed=13) landing far apart despite identical
code. This runs N more retrains at different seeds to characterize the swing
with more than 2 data points, rather than treat it as a single anecdote.

Set SEED via a preceding inline `colab exec`, e.g.:
    echo "SEED = 17" | colab exec -s <session> --timeout 10

Assumes setup.py and common_eval.py already ran in this session (manifest_csv,
device, EVAL_LOADERS in the persistent namespace) -- same assumption as
run_mc_dropout.py.
"""

import json

from cervical_screening.uncertainty.train_variants import predict_mc_dropout, train_mc_dropout

MODEL = "efficientvit_m0"
N_PASSES = 20
SEED = globals().get("SEED", 13)

result = train_mc_dropout(MODEL, manifest_csv, device, dropout_rate=0.3, epochs=30, batch_size=32,
                           patience=7, seed=SEED)
mc_model = result["model"]


def predict_fn(loader):
    mean_probs, entropy, labels = predict_mc_dropout(mc_model, loader, device, n_passes=N_PASSES)
    confidence = -entropy
    return mean_probs, confidence, labels

mendeley_report = evaluate_method(f"mc_dropout_seed{SEED}", predict_fn, {"test": EVAL_LOADERS["test"],
                                                                          "calibration": EVAL_LOADERS["calibration"],
                                                                          "herlev": EVAL_LOADERS["herlev"],
                                                                          "mendeley_lbc": EVAL_LOADERS["mendeley_lbc"]})

report = {
    "seed": SEED,
    "mendeley_lbc_rc_auc": mendeley_report["shift"]["mendeley_lbc"]["selective"]["risk_coverage_auc"],
    "mendeley_lbc_full_coverage_acc": mendeley_report["shift"]["mendeley_lbc"]["selective"]["full_coverage_accuracy"],
    "full_report": mendeley_report,
}

print("RESULT_JSON_START")
print(json.dumps(report))
print("RESULT_JSON_END")
