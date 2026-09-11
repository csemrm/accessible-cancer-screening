"""Zero-shot cross-dataset domain-shift evaluation (proposal S16/RQ4).

Loads a checkpoint trained on SIPaKMeD's native 5-class task (train.py) and
evaluates it on Herlev or Mendeley LBC by grouping its 5-class softmax
output into the shared binary (normal/abnormal) decision from
harmonize_labels.py — no retraining, no calibration on the shift set. The
model's SIPaKMeD-side binary performance (computed the same way, on
SIPaKMeD's own held-out test split) is the in-distribution reference this
zero-shot number should be compared against, which is what quantifies
degradation under shift rather than just reporting an isolated accuracy.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from cervical_screening.data.dataset import CLASSES
from cervical_screening.data.harmonize_labels import ABNORMAL, NORMAL, SIPAKMED_TO_BINARY
from cervical_screening.data.transforms import build_transform
from cervical_screening.device import select_device
from cervical_screening.eval.metrics import compute_metrics
from cervical_screening.models.registry import build_model, input_size_for

# Fixed positions in the 5-class softmax output that get summed into each
# binary bucket. Order matters: this must track CLASSES in dataset.py.
_ABNORMAL_IDX = [i for i, c in enumerate(CLASSES) if SIPAKMED_TO_BINARY[c] == ABNORMAL]
_NORMAL_IDX = [i for i, c in enumerate(CLASSES) if SIPAKMED_TO_BINARY[c] == NORMAL]


def group_to_binary_probs(probs_5class: np.ndarray) -> np.ndarray:
    """(N, 5) softmax -> (N, 2) [P(normal), P(abnormal)]."""
    p_normal = probs_5class[:, _NORMAL_IDX].sum(axis=1)
    p_abnormal = probs_5class[:, _ABNORMAL_IDX].sum(axis=1)
    return np.stack([p_normal, p_abnormal], axis=1)


class BinaryManifestDataset(Dataset):
    """Reads a make_shift_manifest.py CSV (path, binary_label, split, source_dataset)."""

    LABEL_TO_IDX = {NORMAL: 0, ABNORMAL: 1}

    def __init__(self, manifest_csv: Path, transform):
        self.transform = transform
        self.samples = []
        with open(manifest_csv, newline="") as f:
            for row in csv.DictReader(f):
                self.samples.append((Path(row["path"]), self.LABEL_TO_IDX[row["binary_label"]]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        return self.transform(image), label


@torch.no_grad()
def predict_binary_probs(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_probs, all_labels = [], []
    for images, labels in loader:
        logits = model(images.to(device))
        probs_5 = torch.softmax(logits, dim=1).cpu().numpy()
        all_probs.append(group_to_binary_probs(probs_5))
        all_labels.append(labels.numpy())
    return np.concatenate(all_probs), np.concatenate(all_labels)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="registry name, e.g. mobilevit_s")
    parser.add_argument("--checkpoint", type=Path, required=True, help="model.pt from train.py")
    parser.add_argument("--shift-manifest", type=Path, required=True,
                         help="CSV from make_shift_manifest.py")
    parser.add_argument("--source-dataset-name", default="", help="label for the results row")
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    device = select_device()
    input_size = input_size_for(args.model)

    model = build_model(args.model, pretrained=False)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model = model.to(device)

    dataset = BinaryManifestDataset(args.shift_manifest, build_transform(input_size, train=False))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    y_prob, y_true = predict_binary_probs(model, loader, device)
    metrics = compute_metrics(y_true, y_prob)

    result = {
        "model": args.model,
        "checkpoint": str(args.checkpoint),
        "shift_dataset": args.source_dataset_name or args.shift_manifest.stem,
        "n_samples": len(dataset),
        "binary_metrics": metrics,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
