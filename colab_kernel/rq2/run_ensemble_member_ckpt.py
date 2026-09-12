"""RQ2 method 4/4 (member training, checkpointed variant): deep ensemble
(Lakshminarayanan et al., NeurIPS 2017), one member per exec call.

Unlike run_ensemble_member.py, this version does NOT rely on all 5 members
plus the final eval sharing one uninterrupted kernel session -- three straight
Colab session deaths mid-RQ2 made that assumption too fragile. Instead each
member's trained weights are saved to /content/ and then pulled to the local
machine with `colab download` right after this script returns, so progress
survives even if the session dies before the next member starts.

Set ENSEMBLE_SEED via a preceding inline `colab exec` before running this,
e.g.:
    echo "ENSEMBLE_SEED = 17" | colab exec -s <session> --timeout 10
"""

from cervical_screening.uncertainty.train_variants import train_baseline

MODEL = "efficientvit_m0"
seed = globals().get("ENSEMBLE_SEED", 13)

print(f"=== training ensemble member, seed={seed} ===", flush=True)
result = train_baseline(MODEL, manifest_csv, device, epochs=30, batch_size=32, patience=7, seed=seed)
member_model = result["model"]

ckpt_path = f"/content/ensemble_member_seed{seed}.pt"
import torch
torch.save(member_model.state_dict(), ckpt_path)
print(f"ENSEMBLE_MEMBER_DONE seed={seed} ckpt={ckpt_path}", flush=True)
