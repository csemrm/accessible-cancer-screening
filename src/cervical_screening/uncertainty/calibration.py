"""Calibration metrics and temperature scaling (proposal S18/RQ2).

Pure numpy, framework-agnostic: every function takes plain numpy arrays of
logits/probabilities and integer labels, so the same code evaluates a
softmax baseline, an MC-Dropout mean, an ensemble average, or an evidential
head's expected probability without caring which produced it.
"""

from __future__ import annotations

import numpy as np


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(z)
    return exp / exp.sum(axis=1, keepdims=True)


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """ECE: bins predictions by max-class confidence, weights each bin's
    |accuracy - mean confidence| by its share of samples."""
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correct = (predictions == labels).astype(np.float64)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(labels)
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        in_bin = (confidences > lo) & (confidences <= hi) if lo > 0 else (confidences >= lo) & (confidences <= hi)
        if not in_bin.any():
            continue
        bin_acc = correct[in_bin].mean()
        bin_conf = confidences[in_bin].mean()
        ece += (in_bin.sum() / n) * abs(bin_acc - bin_conf)
    return float(ece)


def brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    """Multiclass Brier score: mean squared error between predicted
    probabilities and the one-hot true label, summed over classes."""
    num_classes = probs.shape[1]
    onehot = np.eye(num_classes)[labels]
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def negative_log_likelihood(probs: np.ndarray, labels: np.ndarray, eps: float = 1e-12) -> float:
    true_class_probs = probs[np.arange(len(labels)), labels]
    return float(-np.mean(np.log(np.clip(true_class_probs, eps, 1.0))))


def fit_temperature(logits: np.ndarray, labels: np.ndarray,
                     t_min: float = 0.1, t_max: float = 5.0, n_steps: int = 200) -> float:
    """Grid search for the temperature T minimizing NLL of softmax(logits/T)
    on a held-out calibration set. Grid search over this single parameter is
    exact enough and avoids pulling in a gradient-based optimizer for what
    is a one-dimensional, well-behaved (typically unimodal) objective."""
    candidates = np.linspace(t_min, t_max, n_steps)
    best_t, best_nll = 1.0, float("inf")
    for t in candidates:
        probs = softmax(logits / t)
        nll = negative_log_likelihood(probs, labels)
        if nll < best_nll:
            best_nll, best_t = nll, t
    return float(best_t)


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    return softmax(logits / temperature)


def calibration_report(probs: np.ndarray, labels: np.ndarray) -> dict:
    return {
        "ece": expected_calibration_error(probs, labels),
        "brier": brier_score(probs, labels),
        "nll": negative_log_likelihood(probs, labels),
    }
