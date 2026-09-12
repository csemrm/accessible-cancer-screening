"""Shared evaluation harness for all four RQ2 uncertainty methods. Run once
after setup.py; defines DataLoaders and evaluate_method(), which every
per-method script (train_baseline.py, train_mc_dropout.py, etc.) calls with
its own predict_fn.

Each predict_fn must have the signature
    predict_fn(loader) -> (probs_5class, confidence, labels)
where confidence is a "higher = more trustworthy" scalar per sample --
max-softmax-probability for the plain baseline, negative predictive entropy
for MC Dropout, 1-vacuity for evidential, negative ensemble variance for the
ensemble. Using each method's own native uncertainty signal here (not a
common one imposed after the fact) is the actual point of RQ2/H2: whether
evidential's cheap single-pass signal is as good as the others' expensive
multi-pass ones, not whether they can all be forced into the same shape.
"""

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from cervical_screening.data.dataset import SipakmedManifestDataset
from cervical_screening.data.transforms import build_transform
from cervical_screening.data.harmonize_labels import ABNORMAL, NORMAL
from cervical_screening.eval.cross_dataset import group_to_binary_probs
from cervical_screening.uncertainty import calibration, conformal, selective


class BinaryRowsDataset(Dataset):
    LABEL_TO_IDX = {NORMAL: 0, ABNORMAL: 1}

    def __init__(self, rows, transform):
        self.rows = rows
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        path, label = self.rows[idx]
        return self.transform(Image.open(path).convert("RGB")), self.LABEL_TO_IDX[label]


INPUT_SIZE = 224  # efficientvit_m0's native size; kept as a module constant since every RQ2 run uses one fixed backbone


def build_eval_loaders(input_size: int = INPUT_SIZE, batch_size: int = 32):
    eval_tf = build_transform(input_size, train=False)
    test_ds = SipakmedManifestDataset(manifest_csv, "test", eval_tf)
    cal_ds = SipakmedManifestDataset(manifest_csv, "calibration", eval_tf)
    herlev_ds = BinaryRowsDataset(herlev_rows, eval_tf)
    mendeley_ds = BinaryRowsDataset(mendeley_rows, eval_tf)
    return {
        "test": DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2),
        "calibration": DataLoader(cal_ds, batch_size=batch_size, shuffle=False, num_workers=2),
        "herlev": DataLoader(herlev_ds, batch_size=batch_size, shuffle=False, num_workers=2),
        "mendeley_lbc": DataLoader(mendeley_ds, batch_size=batch_size, shuffle=False, num_workers=2),
    }


def evaluate_method(method_name: str, predict_fn, loaders: dict) -> dict:
    test_probs, test_conf, test_labels = predict_fn(loaders["test"])
    test_correct = (test_probs.argmax(axis=1) == test_labels)

    cal_probs, cal_conf, cal_labels = predict_fn(loaders["calibration"])

    result = {
        "method": method_name,
        "in_distribution": {
            "calibration": calibration.calibration_report(test_probs, test_labels),
            "selective": selective.selective_report(test_conf, test_correct.astype(float)),
            "conformal": conformal.conformal_report(cal_probs, cal_labels, test_probs, test_labels),
        },
        "shift": {},
    }

    for shift_name in ("herlev", "mendeley_lbc"):
        shift_probs5, shift_conf, shift_binary_labels = predict_fn(loaders[shift_name])
        binary_probs = group_to_binary_probs(shift_probs5)
        binary_correct = (binary_probs.argmax(axis=1) == shift_binary_labels)
        result["shift"][shift_name] = {
            "calibration": calibration.calibration_report(binary_probs, shift_binary_labels),
            "selective": selective.selective_report(shift_conf, binary_correct.astype(float)),
            "n_samples": len(shift_binary_labels),
        }

    return result


EVAL_LOADERS = build_eval_loaders()
print("EVAL_LOADERS ready:", {k: len(v.dataset) for k, v in EVAL_LOADERS.items()}, flush=True)
