"""RQ2 method 3/4: evidential deep learning (Sensoy et al., NeurIPS 2018) --
the single-forward-pass method this whole RQ exists to test."""

import json

from cervical_screening.uncertainty.train_variants import predict_evidential, train_evidential

MODEL = "efficientvit_m0"
NUM_CLASSES = 5

result = train_evidential(MODEL, manifest_csv, device, num_classes=NUM_CLASSES, annealing_epochs=10,
                           epochs=30, batch_size=32, patience=7, seed=13)
evidential_model = result["model"]


def predict_fn(loader):
    probs, vac, labels = predict_evidential(evidential_model, loader, device, num_classes=NUM_CLASSES)
    confidence = 1.0 - vac
    return probs, confidence, labels


evidential_report = evaluate_method("evidential", predict_fn, EVAL_LOADERS)
ALL_RESULTS["evidential"] = evidential_report

print("RESULT_JSON_START")
print(json.dumps(evidential_report))
print("RESULT_JSON_END")
