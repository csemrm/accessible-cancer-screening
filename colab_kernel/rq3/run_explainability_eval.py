"""RQ3: compare relevancy propagation (primary method) against attention
rollout and Grad-CAM (baselines) on deletion/insertion faithfulness and
nucleus-overlap localization (proposal S17, H3).

Assumes setup.py already ran in this session (manifest_csv, nucleus_manifest_csv,
crops_root, device, model all in the persistent namespace).
"""

import csv
import json
import random

import torch

from cervical_screening.data.dataset import CLASSES, SipakmedManifestDataset
from cervical_screening.data.transforms import build_transform
from cervical_screening.explainability.attention_rollout import attention_rollout
from cervical_screening.explainability.relevancy_propagation import relevancy_propagation
from cervical_screening.explainability.gradcam import vit_gradcam
from cervical_screening.explainability.faithfulness import deletion_auc, insertion_auc
from cervical_screening.explainability.nucleus_overlap import rasterize_nucleus_mask, concentration_ratio

INPUT_SIZE = 224
GRID_SIZE = INPUT_SIZE // 16
NUM_PREFIX_TOKENS = getattr(model, "num_prefix_tokens", 1)
N_PER_CLASS = 10
SEED = 13

# --- load nucleus polygons, keyed by crop path relative to crops_root ---
nucleus_lookup = {}
with open(nucleus_manifest_csv, newline="") as f:
    for row in csv.DictReader(f):
        polygons = json.loads(row["nucleus_polygons"])
        if polygons:
            nucleus_lookup[row["crop_path"]] = {
                "width": int(row["crop_width"]),
                "height": int(row["crop_height"]),
                "polygons": polygons,
            }
print(f"nucleus polygons available for {len(nucleus_lookup)} crops", flush=True)

# --- stratified sample of the test split ---
eval_tf = build_transform(INPUT_SIZE, train=False)
test_ds = SipakmedManifestDataset(manifest_csv, "test", transform=None)  # no transform: need raw PIL for nucleus join
by_class: dict[int, list[int]] = {i: [] for i in range(len(CLASSES))}
for idx, (_path, label) in enumerate(test_ds.samples):
    by_class[label].append(idx)

rng = random.Random(SEED)
sampled_indices = []
for label, indices in by_class.items():
    rng.shuffle(indices)
    sampled_indices.extend(indices[:N_PER_CLASS])
print(f"sampled {len(sampled_indices)} test images across {len(CLASSES)} classes", flush=True)

METHODS = ["relevancy_propagation", "attention_rollout", "gradcam"]
per_method = {m: {"deletion_auc": [], "insertion_auc": [], "concentration_ratio": []} for m in METHODS}
n_nucleus_matched = 0

model.eval()
for i, idx in enumerate(sampled_indices):
    path, label = test_ds.samples[idx]
    from PIL import Image
    pil_image = Image.open(path).convert("RGB")
    image = eval_tf(pil_image).unsqueeze(0).to(device)

    with torch.no_grad():
        pred_class = int(model(image).argmax(dim=-1).item())

    heatmaps = {
        "attention_rollout": attention_rollout(model, image, grid_size=GRID_SIZE,
                                                num_prefix_tokens=NUM_PREFIX_TOKENS)[0],
        "relevancy_propagation": relevancy_propagation(model, image, target_class=pred_class,
                                                        grid_size=GRID_SIZE,
                                                        num_prefix_tokens=NUM_PREFIX_TOKENS)[0],
        "gradcam": vit_gradcam(model, image, target_class=pred_class, grid_size=GRID_SIZE,
                                num_prefix_tokens=NUM_PREFIX_TOKENS)[0],
    }

    # crop_path relative to crops_root, matching how nucleus_manifest.csv stores it
    rel_path = str(path.relative_to(crops_root))
    nucleus_entry = nucleus_lookup.get(rel_path)
    if nucleus_entry is not None:
        n_nucleus_matched += 1
        mask = rasterize_nucleus_mask(nucleus_entry["polygons"], nucleus_entry["width"],
                                       nucleus_entry["height"], INPUT_SIZE).to(device)

    for method_name, heatmap in heatmaps.items():
        per_method[method_name]["deletion_auc"].append(
            deletion_auc(model, image, heatmap, target_class=pred_class, n_steps=20))
        per_method[method_name]["insertion_auc"].append(
            insertion_auc(model, image, heatmap, target_class=pred_class, n_steps=20))
        if nucleus_entry is not None:
            per_method[method_name]["concentration_ratio"].append(concentration_ratio(heatmap, mask))

    if (i + 1) % 10 == 0:
        print(f"  processed {i + 1}/{len(sampled_indices)}", flush=True)

print(f"nucleus polygon matched for {n_nucleus_matched}/{len(sampled_indices)} sampled images", flush=True)


def _mean(xs):
    return float(sum(xs) / len(xs)) if xs else None


report = {
    "n_samples": len(sampled_indices),
    "n_nucleus_matched": n_nucleus_matched,
    "methods": {
        m: {
            "deletion_auc_mean": _mean(v["deletion_auc"]),
            "insertion_auc_mean": _mean(v["insertion_auc"]),
            "concentration_ratio_mean": _mean(v["concentration_ratio"]),
            "n_concentration_samples": len(v["concentration_ratio"]),
        }
        for m, v in per_method.items()
    },
}

ALL_RESULTS = globals().get("ALL_RESULTS", {})
ALL_RESULTS["rq3_explainability"] = report
globals()["ALL_RESULTS"] = ALL_RESULTS

print("RESULT_JSON_START")
print(json.dumps(report, indent=2))
print("RESULT_JSON_END")
