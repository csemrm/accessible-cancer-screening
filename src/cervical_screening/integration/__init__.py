"""RQ5 system-integration package (proposal S15/S18, H5).

RQ2's uncertainty methods and RQ3's explainability methods were each
evaluated in isolation (uncertainty method comparisons hold the backbone
fixed; explainability comparisons run post-hoc on a trained model without
touching its weights). Because every property in this project's pipeline
attaches to a fixed trained backbone rather than being jointly optimized,
most of H5's literal claim -- that adding calibration/explainability/
selective-prediction doesn't degrade each other's target metrics -- holds
close to trivially by construction: temperature scaling and conformal
calibration are both monotonic transforms of a frozen model's probabilities
and cannot change its argmax or accuracy, and explainability is a read-only
post-hoc analysis. The one place composition is genuinely non-trivial is
selective prediction: does COMBINING two different native uncertainty
signals (temperature-scaled confidence and conformal prediction-set size)
into one abstention rule actually help, or is one signal just noise on top
of the other? `combined_selective.py` is that empirical test.
"""
