"""Diagnostic metrics for the RQ1 benchmark (proposal S14).

Deliberately not accuracy-only: balanced accuracy and macro sensitivity/
specificity are reported because SIPaKMeD's five classes are imbalanced
(787-831 images per class here, but far more skewed in the Bethesda-mapped
clinical setting this stands in for), and per-class recall is what a
screening task actually cares about.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)


def specificity_per_class(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    total = cm.sum()
    spec = np.zeros(num_classes)
    for c in range(num_classes):
        tp = cm[c, c]
        fn = cm[c, :].sum() - tp
        fp = cm[:, c].sum() - tp
        tn = total - tp - fn - fp
        spec[c] = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    return spec


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    """y_true: (N,) int labels. y_prob: (N, C) predicted class probabilities."""
    num_classes = y_prob.shape[1]
    y_pred = y_prob.argmax(axis=1)

    metrics = {
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall_sensitivity": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_specificity": float(np.nanmean(specificity_per_class(y_true, y_pred, num_classes))),
    }

    # One-vs-rest AUROC/AUPRC — only defined when every class is present in y_true.
    try:
        y_onehot = np.eye(num_classes)[y_true]
        metrics["macro_auroc"] = roc_auc_score(y_onehot, y_prob, average="macro", multi_class="ovr")
        metrics["macro_auprc"] = average_precision_score(y_onehot, y_prob, average="macro")
    except ValueError:
        metrics["macro_auroc"] = float("nan")
        metrics["macro_auprc"] = float("nan")

    return metrics
