# RQ2 results: does uncertainty quantification catch EfficientViT-M0's known shift failure?

## Setup

RQ1/RQ4 established that EfficientViT-M0 (2.16M params, best in-distribution SIPaKMeD
accuracy of the 12 models tested) collapses to near/below-chance balanced accuracy on
the Mendeley LBC cross-dataset shift, while degrading more gracefully on Herlev. RQ2
asks a narrower, practically important question: **does the model's own uncertainty
signal correctly flag this failure**, or does it stay falsely confident? A screening
tool that fails silently is far more dangerous than one that fails and says so.

Four uncertainty methods were trained on the identical SIPaKMeD split (same manifest,
same 60/15/10/15 train/val/calibration/test partition) with EfficientViT-M0 held fixed
as the backbone, isolating the uncertainty method rather than architecture:

1. **Baseline** -- plain softmax + post-hoc temperature scaling (Guo et al., 2017)
2. **MC Dropout** -- Gal & Ghahramani, ICML 2016 (dropout rate 0.3, 20 stochastic passes)
3. **Evidential deep learning** -- Sensoy et al., NeurIPS 2018 (Dirichlet evidence, vacuity as the uncertainty signal)
4. **Deep ensemble** -- Lakshminarayanan et al., NeurIPS 2017 (5 independently-seeded members, prediction variance as the uncertainty signal)

Each method's own native confidence/uncertainty signal is used for selective prediction
(max-softmax for baseline, `-entropy` for MC Dropout, `1-vacuity` for evidential,
`-variance` for the ensemble) -- the point of RQ2 is whether each method's *cheapest
native signal* works, not whether all four can be forced through a common post-hoc
calibration step.

All four were evaluated on the same three splits: in-distribution SIPaKMeD test set,
Herlev (moderate shift), and Mendeley LBC (severe shift, the dataset where this
backbone is known to fail).

## Results

| Method | In-dist ECE | In-dist full-cov acc | Herlev risk-cov AUC | Herlev full-cov acc | Mendeley risk-cov AUC | Mendeley full-cov acc | Mendeley sel-acc@90% cov |
|---|---|---|---|---|---|---|---|
| Baseline (temp-scaled) | 0.012 | 0.967 | 0.193 | 0.694 | **0.515** | 0.530 | 0.522 |
| MC Dropout | 0.013 | 0.972 | 0.203 | 0.658 | **0.285** | 0.579 | 0.583 |
| Evidential | 0.159 | 0.967 | 0.167 | 0.677 | **0.476** | 0.543 | 0.539 |
| Deep ensemble | 0.012 | 0.974 | **0.147** | **0.771** | 0.473 | 0.537 | 0.544 |

(Risk-coverage AUC: lower is better -- it is the integral of error rate over the
coverage curve, so a well-behaved uncertainty signal keeps risk near zero until
coverage is forced high. Full JSON results, including calibration/conformal detail,
are in `runs/colab_rq2/*.json`.)

## Findings

**In-distribution, all four methods are close.** Baseline, MC Dropout, and deep
ensemble all calibrate well (ECE ~0.012-0.013) with full-coverage accuracy in the
0.967-0.974 range, consistent with RQ1's finding that this backbone is genuinely
strong in-distribution. Evidential is the outlier: its raw belief masses are
markedly less calibrated (ECE 0.159) even though its point accuracy matches the
others (0.967) -- evidential's probabilities need their own post-hoc correction,
which this comparison deliberately did not apply, to make its calibration numbers
comparable to the others'.

**On Herlev (moderate shift), the deep ensemble is the clear winner.** Its
risk-coverage AUC (0.147) and full-coverage accuracy (0.771) are both the best of
the four methods by a real margin. Member disagreement is doing genuine work here.

**On Mendeley LBC -- the shift that matters most, because it's where the backbone
catastrophically fails -- the answer is much less encouraging.** MC Dropout's
entropy is the only signal that shows a substantial, unambiguous improvement over
the baseline: risk-coverage AUC drops from 0.515 (baseline) to 0.285, and
full-coverage accuracy rises from 0.530 to 0.579. Evidential (0.476) and the deep
ensemble (0.473) are both only marginally better than the plain temperature-scaled
baseline (0.515) -- essentially within noise of each other, despite the ensemble's
clear win on Herlev. **Member disagreement, which correctly flags the milder
Herlev shift, does not reliably track the specific failure mode behind the
Mendeley collapse.** This is a genuinely surprising result relative to the prior
that ensembles are the most robust uncertainty method under shift, and it is
reported as-is rather than smoothed over.

**None of the four methods reach a reliable operating point on Mendeley.** Even MC
Dropout's best result -- selective accuracy of 0.583 at 90% coverage on a *binary*
(normal/abnormal) task -- is barely better than chance after abstaining on the
least-confident 10% of samples. For deployment purposes, none of these methods
alone would justify an "abstain and refer" policy on this severe a domain shift;
the honest conclusion is that a single fixed backbone with a single uncertainty
method cannot be assumed reliable off known training distribution, no matter which
uncertainty method sits on top of it.

## Status against the RQ2 hypotheses (proposal S18)

Full hypothesis language and the RQ1/RQ4 half of this table are in
`docs/rq1_results.md`. RQ2-relevant hypotheses:

- **H2** (evidential's cheap single-pass signal matches expensive multi-pass methods): **not supported**. Evidential's Mendeley risk-coverage AUC (0.476) is close to the ensemble's (0.473) but both are clearly worse than MC Dropout's (0.285) -- evidential does not "punch above its weight" here relative to MC Dropout specifically, though it does roughly match the ensemble at a fraction of the training cost (1 model vs. 5).
- **H3** (member disagreement generalizes as a shift-robustness signal across shift severities): **partially refuted**. True on Herlev, false on Mendeley -- the first direct evidence in this project that a method's shift-detection quality is itself shift-magnitude-dependent, not a fixed property of the method.

## Reproducibility notes

- All four methods share one manifest, one calibration split, one backbone init recipe (see `src/cervical_screening/uncertainty/train_variants.py`); only the loss/inference path differs.
- MC Dropout required a code-level fix before any GPU time was spent: `efficientvit_m0.r224_in1k`'s only `nn.Dropout` module ships at `p=0.0` by default, which would make MC Dropout completely degenerate. `set_dropout_rate()` (`src/cervical_screening/uncertainty/mc_dropout.py`) forces a real rate before training starts, since MC Dropout's Bayesian-approximation justification requires dropout to be active during training, not just patched in at inference.
- Ensemble members are trained and checkpointed independently (`colab_kernel/rq2/run_ensemble_member_ckpt.py` saves each member's state_dict to disk immediately after training) rather than assuming all 5 trainings plus the final eval survive one uninterrupted Colab session -- repeated mid-run session deaths during this project's execution made that assumption unsafe.
