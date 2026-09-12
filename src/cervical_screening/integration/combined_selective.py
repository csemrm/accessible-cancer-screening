"""Combines temperature-scaled confidence with conformal prediction-set size
into one abstention signal, and lets selective.selective_report() score it
against each signal alone -- the one genuinely non-trivial composition
question in RQ5/H5 (see this package's docstring)."""

from __future__ import annotations

import numpy as np


def rank_normalize(x: np.ndarray) -> np.ndarray:
    """Ranks in [0, 1], higher input value -> higher rank."""
    if len(x) <= 1:
        return np.zeros_like(x, dtype=float)
    order = np.argsort(x)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(x))
    return ranks / (len(x) - 1)


def combined_confidence(prob_confidence: np.ndarray, conformal_set_sizes: np.ndarray) -> np.ndarray:
    """prob_confidence: higher = more confident (e.g. temp-scaled max-softmax).
    conformal_set_sizes: LOWER = more confident (smaller prediction set), so its
    rank is inverted before averaging. Returns a combined score, higher = more
    confident, on the same [0, 1] rank scale as either input alone -- avoids the
    unit mismatch of directly summing a probability and a set-size count."""
    conf_rank = rank_normalize(prob_confidence)
    set_rank = rank_normalize(-conformal_set_sizes)
    return 0.5 * conf_rank + 0.5 * set_rank
