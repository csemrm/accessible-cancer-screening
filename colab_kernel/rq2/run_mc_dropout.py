"""RQ2 method 2/4: MC Dropout (Gal & Ghahramani, ICML 2016)."""

import json

from cervical_screening.uncertainty.train_variants import predict_mc_dropout, train_mc_dropout

MODEL = globals().get("MODEL", "efficientvit_m0")
N_PASSES = 20

result = train_mc_dropout(MODEL, manifest_csv, device, dropout_rate=0.3, epochs=30, batch_size=32,
                           patience=7, seed=13)
mc_dropout_model = result["model"]


def predict_fn(loader):
    mean_probs, entropy, labels = predict_mc_dropout(mc_dropout_model, loader, device, n_passes=N_PASSES)
    confidence = -entropy
    return mean_probs, confidence, labels


mc_dropout_report = evaluate_method("mc_dropout", predict_fn, EVAL_LOADERS)
mc_dropout_report["n_passes"] = N_PASSES
mc_dropout_report["dropout_rate"] = result["dropout_rate"]
ALL_RESULTS["mc_dropout"] = mc_dropout_report

print("RESULT_JSON_START")
print(json.dumps(mc_dropout_report))
print("RESULT_JSON_END")
