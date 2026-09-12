"""Paired bootstrap significance test between two uncertainty methods'
risk-coverage AUC on the same samples (RQ2 follow-up: is MC Dropout's
Mendeley LBC advantage over the deep ensemble real, or within noise?).

Nonparametric rather than assuming a normal sampling distribution for
risk-coverage AUC, and paired (same resampled indices for both methods)
since both methods are evaluated on the identical held-out samples --
consistent with this project's existing bootstrap-resampling precedent in
conformal.stable_conformal_threshold, which uses the same resampling
philosophy for the same small-sample-size reason.
"""

from __future__ import annotations

import numpy as np

from cervical_screening.uncertainty.selective import risk_coverage_auc


def paired_bootstrap_rc_auc_test(confidence_a: np.ndarray, correct_a: np.ndarray,
                                   confidence_b: np.ndarray, correct_b: np.ndarray,
                                   n_boot: int = 2000, seed: int = 13) -> dict:
    """Tests whether method A's risk-coverage AUC is significantly LOWER
    (better) than method B's, on the same n samples. Returns the observed
    difference (B - A; positive means A is better), a 95% bootstrap CI on
    that difference, and a one-sided p-value for "A is better than B."
    """
    assert len(confidence_a) == len(correct_a) == len(confidence_b) == len(correct_b)
    n = len(confidence_a)
    rng = np.random.default_rng(seed)

    observed_a = risk_coverage_auc(confidence_a, correct_a)
    observed_b = risk_coverage_auc(confidence_b, correct_b)
    observed_diff = observed_b - observed_a

    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        auc_a = risk_coverage_auc(confidence_a[idx], correct_a[idx])
        auc_b = risk_coverage_auc(confidence_b[idx], correct_b[idx])
        diffs[i] = auc_b - auc_a

    ci_low, ci_high = np.percentile(diffs, [2.5, 97.5])
    # one-sided p-value: fraction of bootstrap replicates where A is NOT better than B
    p_a_better = float((diffs <= 0).mean())

    return {
        "n_samples": n,
        "n_bootstrap": n_boot,
        "method_a_rc_auc": observed_a,
        "method_b_rc_auc": observed_b,
        "observed_diff_b_minus_a": observed_diff,
        "bootstrap_ci_95": [float(ci_low), float(ci_high)],
        "p_value_a_better_one_sided": p_a_better,
    }
