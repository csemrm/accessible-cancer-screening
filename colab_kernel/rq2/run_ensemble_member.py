"""RQ2 method 4/4 (member training): deep ensemble (Lakshminarayanan et al.,
NeurIPS 2017). One member per exec call -- set ENSEMBLE_SEED via a preceding
inline `colab exec` before running this file each time, e.g.:
    echo "ENSEMBLE_SEED = 17" | colab exec -s <session> --timeout 10
Appends the trained model to the persistent `ensemble_models` list; run 5
times with 5 different seeds, then run run_ensemble_eval.py once.
"""

from cervical_screening.uncertainty.train_variants import train_baseline

MODEL = "efficientvit_m0"
seed = globals().get("ENSEMBLE_SEED", 13)

if "ensemble_models" not in globals():
    ensemble_models = []

print(f"=== training ensemble member, seed={seed} ===", flush=True)
result = train_baseline(MODEL, manifest_csv, device, epochs=30, batch_size=32, patience=7, seed=seed)
ensemble_models.append(result["model"])
print(f"ENSEMBLE_MEMBER_DONE seed={seed} total_members={len(ensemble_models)}", flush=True)
