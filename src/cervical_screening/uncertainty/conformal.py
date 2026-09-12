"""Small-sample-corrected conformal prediction (proposal S10/S18/RQ2).

Standard split conformal prediction gives a distribution-free coverage
guarantee from a single calibration split, but at the ~1,000-4,000-image
scale of the public cytology datasets this project uses, a single random
calibration split gives an unstable threshold — a different split can give
a meaningfully different set size at the same nominal coverage. The
correction here is a resampling-averaged threshold: repeatedly split-
conformal-calibrate over bootstrap resamples of the calibration set and
average the resulting threshold, rather than trusting one split. This is
not the literature-standard jackknife+/CV+ (which recomputes residuals from
models refit without each held-out point) — refitting the model dozens of
times is not affordable at this stage — but it directly addresses the
instability a single small calibration split introduces, and that
trade-off is stated explicitly rather than implied away.
"""

from __future__ import annotations

import numpy as np


def nonconformity_scores(probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """1 - P(true class), the standard softmax nonconformity score."""
    return 1.0 - probs[np.arange(len(labels)), labels]


def split_conformal_threshold(scores: np.ndarray, alpha: float) -> float:
    """The finite-sample-corrected (1-alpha) quantile of calibration
    nonconformity scores (Vovk et al.): ceil((n+1)(1-alpha))/n, clipped to 1.0
    when n is too small to support the target coverage at all."""
    n = len(scores)
    q_level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(scores, q_level, method="higher"))


def stable_conformal_threshold(scores: np.ndarray, alpha: float,
                                 n_resamples: int = 200, resample_frac: float = 0.8,
                                 seed: int = 13) -> float:
    """Small-sample correction: average the split-conformal threshold over
    bootstrap resamples of the calibration scores, rather than trusting a
    single small split's quantile estimate."""
    rng = np.random.default_rng(seed)
    n = len(scores)
    resample_n = max(10, int(round(resample_frac * n)))
    thresholds = []
    for _ in range(n_resamples):
        idx = rng.choice(n, size=resample_n, replace=True)
        thresholds.append(split_conformal_threshold(scores[idx], alpha))
    return float(np.mean(thresholds))


def prediction_sets(probs: np.ndarray, threshold: float) -> np.ndarray:
    """Boolean (n_samples, n_classes) matrix: True where a class is included
    in that sample's conformal prediction set."""
    return probs >= (1.0 - threshold)


def realized_coverage(pred_sets: np.ndarray, labels: np.ndarray) -> float:
    included = pred_sets[np.arange(len(labels)), labels]
    return float(included.mean())


def average_set_size(pred_sets: np.ndarray) -> float:
    return float(pred_sets.sum(axis=1).mean())


def conformal_report(cal_probs: np.ndarray, cal_labels: np.ndarray,
                      eval_probs: np.ndarray, eval_labels: np.ndarray,
                      alpha: float = 0.1) -> dict:
    """Fits the threshold on (cal_probs, cal_labels) and reports realized
    coverage + average set size on a SEPARATE (eval_probs, eval_labels) —
    fitting and evaluating on the same split would overstate coverage."""
    scores = nonconformity_scores(cal_probs, cal_labels)
    threshold_naive = split_conformal_threshold(scores, alpha)
    threshold_stable = stable_conformal_threshold(scores, alpha)

    sets_naive = prediction_sets(eval_probs, threshold_naive)
    sets_stable = prediction_sets(eval_probs, threshold_stable)

    return {
        "target_coverage": 1 - alpha,
        "naive": {
            "threshold": threshold_naive,
            "realized_coverage": realized_coverage(sets_naive, eval_labels),
            "avg_set_size": average_set_size(sets_naive),
        },
        "stable": {
            "threshold": threshold_stable,
            "realized_coverage": realized_coverage(sets_stable, eval_labels),
            "avg_set_size": average_set_size(sets_stable),
        },
    }
