"""Seed-variance check for evidential deep learning, to give it the same
seed-to-seed rigor already applied to MC Dropout.

Set SEED via a preceding inline `colab exec`, e.g.:
    echo "SEED = 17" | colab exec -s <session> --timeout 10

Assumes rq2/setup.py and rq2/common_eval.py already ran this session.
"""

import json

from cervical_screening.uncertainty.train_variants import predict_evidential, train_evidential

MODEL = "efficientvit_m0"
NUM_CLASSES = 5
SEED = globals().get("SEED", 13)

result = train_evidential(MODEL, manifest_csv, device, num_classes=NUM_CLASSES, annealing_epochs=10,
                           epochs=30, batch_size=32, patience=7, seed=SEED)
evidential_model = result["model"]


def predict_fn(loader):
    probs, vac, labels = predict_evidential(evidential_model, loader, device, num_classes=NUM_CLASSES)
    confidence = 1.0 - vac
    return probs, confidence, labels


evidential_report = evaluate_method(f"evidential_seed{SEED}", predict_fn, EVAL_LOADERS)

report = {
    "seed": SEED,
    "mendeley_lbc_rc_auc": evidential_report["shift"]["mendeley_lbc"]["selective"]["risk_coverage_auc"],
    "herlev_rc_auc": evidential_report["shift"]["herlev"]["selective"]["risk_coverage_auc"],
    "full_report": evidential_report,
}

print("RESULT_JSON_START")
print(json.dumps(report))
print("RESULT_JSON_END")
