"""Month 11: synthetic degradation robustness suite + OOD sanity check
(proposal S16), on EfficientViT-M0, RQ1's winning backbone.

Assumes setup.py already ran in this session (manifest_csv, device, model
in the persistent namespace).
"""

import csv
import json

import numpy as np
import torch
from PIL import Image

from cervical_screening.data.transforms import build_transform
from cervical_screening.eval.metrics import compute_metrics
from cervical_screening.robustness.degradation import (
    jpeg_compress, gaussian_blur, brightness_contrast,
    JPEG_QUALITIES, BLUR_RADII, BRIGHTNESS_CONTRAST_FACTORS,
)
from cervical_screening.robustness.ood_sanity import build_ood_batch

INPUT_SIZE = 224
BATCH_SIZE = 32
eval_tf = build_transform(INPUT_SIZE, train=False)

test_rows = []
with open(manifest_csv, newline="") as f:
    for row in csv.DictReader(f):
        if row["split"] == "test":
            test_rows.append((row["path"], int(row["class_idx"])))
print(f"test set: {len(test_rows)} images", flush=True)


@torch.no_grad()
def predict_with_corruption(corrupt_fn) -> dict:
    """corrupt_fn(PIL.Image) -> PIL.Image, or None for no corruption."""
    all_probs, all_labels = [], []
    batch_imgs, batch_labels = [], []

    def flush():
        if not batch_imgs:
            return
        x = torch.stack(batch_imgs).to(device)
        probs = torch.softmax(model(x), dim=-1).cpu().numpy()
        all_probs.append(probs)
        all_labels.extend(batch_labels)
        batch_imgs.clear()
        batch_labels.clear()

    for path, label in test_rows:
        img = Image.open(path).convert("RGB")
        if corrupt_fn is not None:
            img = corrupt_fn(img)
        batch_imgs.append(eval_tf(img))
        batch_labels.append(label)
        if len(batch_imgs) == BATCH_SIZE:
            flush()
    flush()

    probs = np.concatenate(all_probs)
    labels = np.array(all_labels)
    return compute_metrics(labels, probs)


model.eval()

degradation_report = {"clean": predict_with_corruption(None)}
print("clean balanced_accuracy:", degradation_report["clean"]["balanced_accuracy"], flush=True)

degradation_report["jpeg_compression"] = {}
for q in JPEG_QUALITIES:
    m = predict_with_corruption(lambda img, q=q: jpeg_compress(img, q))
    degradation_report["jpeg_compression"][str(q)] = m
    print(f"  jpeg quality={q}: balanced_accuracy={m['balanced_accuracy']:.4f}", flush=True)

degradation_report["gaussian_blur"] = {}
for r in BLUR_RADII:
    m = predict_with_corruption(lambda img, r=r: gaussian_blur(img, r))
    degradation_report["gaussian_blur"][str(r)] = m
    print(f"  blur radius={r}: balanced_accuracy={m['balanced_accuracy']:.4f}", flush=True)

degradation_report["brightness_contrast"] = {}
for f in BRIGHTNESS_CONTRAST_FACTORS:
    m = predict_with_corruption(lambda img, f=f: brightness_contrast(img, f, f))
    degradation_report["brightness_contrast"][str(f)] = m
    print(f"  brightness/contrast factor={f}: balanced_accuracy={m['balanced_accuracy']:.4f}", flush=True)

# --- OOD sanity check: is max-softmax confidence lower on obviously-not-cytology images? ---
ood_images = build_ood_batch(size=INPUT_SIZE, n_noise=20)
with torch.no_grad():
    x_ood = torch.stack([eval_tf(img) for img in ood_images]).to(device)
    ood_confidence = torch.softmax(model(x_ood), dim=-1).max(dim=-1).values.cpu().numpy()

with torch.no_grad():
    in_dist_sample = test_rows[:len(ood_images) * 3]
    x_in = torch.stack([eval_tf(Image.open(p).convert("RGB")) for p, _ in in_dist_sample]).to(device)
    in_dist_confidence = torch.softmax(model(x_in), dim=-1).max(dim=-1).values.cpu().numpy()

ood_report = {
    "n_ood_images": len(ood_images),
    "ood_confidence_mean": float(ood_confidence.mean()),
    "ood_confidence_median": float(np.median(ood_confidence)),
    "n_in_distribution_sample": len(in_dist_sample),
    "in_distribution_confidence_mean": float(in_dist_confidence.mean()),
    "in_distribution_confidence_median": float(np.median(in_dist_confidence)),
}
print("OOD mean confidence:", ood_report["ood_confidence_mean"],
      "| in-dist mean confidence:", ood_report["in_distribution_confidence_mean"], flush=True)

report = {"degradation": degradation_report, "ood_sanity": ood_report}

print("RESULT_JSON_START")
print(json.dumps(report, indent=2))
print("RESULT_JSON_END")
