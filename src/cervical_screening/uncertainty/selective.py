"""Selective prediction / risk-coverage evaluation (proposal S8/S18, RQ2/RQ4).

A confidence score (any scalar where higher = more trustworthy — max
softmax probability, 1-vacuity for an evidential head, negative predictive
entropy for MC Dropout/ensembles) ranks predictions from most to least
trustworthy. At each coverage level (the fraction of samples the system
would answer rather than abstain on, if predictions were kept in confidence
order), risk is the error rate among the kept predictions. This is the
actual trustworthiness signal proposal S16 needs: whether abstention rises
where accuracy would otherwise fail silently, not just an isolated accuracy
number.
"""

from __future__ import annotations

import numpy as np


def risk_coverage_curve(confidence: np.ndarray, correct: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Returns (coverage, risk) arrays, coverage ascending from 1/n to 1.0.
    correct: boolean/0-1 array, 1 if the prediction was right."""
    order = np.argsort(-confidence)  # most confident first
    correct_sorted = correct[order].astype(np.float64)
    n = len(correct)
    cum_correct = np.cumsum(correct_sorted)
    coverage = np.arange(1, n + 1) / n
    risk = 1.0 - (cum_correct / np.arange(1, n + 1))
    return coverage, risk


def risk_coverage_auc(confidence: np.ndarray, correct: np.ndarray) -> float:
    coverage, risk = risk_coverage_curve(confidence, correct)
    # manual trapezoidal integration: np.trapz was removed in NumPy 2.0
    # (renamed np.trapezoid), so avoid depending on either name existing.
    return float(np.sum((risk[1:] + risk[:-1]) / 2.0 * np.diff(coverage)))


def selective_accuracy_at_coverage(confidence: np.ndarray, correct: np.ndarray, target_coverage: float) -> float:
    """Accuracy among the target_coverage fraction of most-confident predictions."""
    n = len(correct)
    k = max(1, int(round(target_coverage * n)))
    order = np.argsort(-confidence)
    kept = correct[order[:k]]
    return float(kept.mean())


def abstention_rate(confidence: np.ndarray, threshold: float) -> float:
    """Fraction of samples that would be abstained on at a fixed confidence threshold."""
    return float((confidence < threshold).mean())


def select_threshold_for_target_risk(confidence: np.ndarray, correct: np.ndarray,
                                       target_risk: float) -> float:
    """The confidence threshold that would have been picked on THIS data to
    keep selective risk at or below target_risk, for use as a frozen
    operating point evaluated on separate held-out data (proposal S10: the
    threshold must be fit on a validation split and frozen before evaluation
    on test/shift data, never re-tuned there)."""
    coverage, risk = risk_coverage_curve(confidence, correct)
    order = np.argsort(-confidence)
    confidence_sorted = confidence[order]
    feasible = risk <= target_risk
    if not feasible.any():
        # no coverage level meets the target risk; abstain on everything
        return float(np.inf)
    # highest coverage (lowest threshold) among feasible points
    best_idx = np.where(feasible)[0].max()
    return float(confidence_sorted[best_idx])


def selective_report(confidence: np.ndarray, correct: np.ndarray) -> dict:
    return {
        "risk_coverage_auc": risk_coverage_auc(confidence, correct),
        "selective_accuracy_at_0.80": selective_accuracy_at_coverage(confidence, correct, 0.80),
        "selective_accuracy_at_0.90": selective_accuracy_at_coverage(confidence, correct, 0.90),
        "full_coverage_accuracy": float(correct.mean()),
    }
