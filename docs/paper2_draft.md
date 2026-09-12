# Paper 2 (draft): No Single Uncertainty-Quantification Method Reliably Flags Domain-Shift Failure in a Cervical Cytology Classifier

**Status: internal draft, journal-article format.** Target venue: a
UQ-focused medical-imaging journal, per the proposal's publication plan
(Section 25). Covers RQ2 (uncertainty/calibration head-to-head) and the
calibration-under-shift question it raises. Companion papers: Paper 1
(RQ1+RQ4, the efficiency benchmark and shift-collapse finding this paper's
backbone is chosen to interrogate) and Paper 3 (RQ3+RQ5+Month 11,
explainability and system integration), drafted separately.

**All numbers in this draft are pulled directly from `docs/rq2_results.md`
and the JSON result files it cites (`runs/colab_rq2/*.json`). Nothing here
is extrapolated or projected beyond what these completed experiments
measured.**

---

## Abstract

A vision transformer that classifies cervical cytology images with 96%+
in-distribution accuracy but collapses to near- or below-chance accuracy
under real cross-institution domain shift — as our companion paper
(Paper 1) shows EfficientViT-M0 does on the Mendeley LBC dataset — is only
as dangerous as its confidence is falsely high when it fails. We compare
four uncertainty-quantification methods on that exact backbone, using each
method's own cheapest native confidence signal (temperature-scaled
max-softmax, MC Dropout entropy, evidential vacuity, deep-ensemble
variance), evaluated in distribution and on both a moderate (Herlev) and
severe (Mendeley LBC) external shift. In distribution, three of the four
methods calibrate well (expected calibration error 0.012-0.013); evidential
deep learning's raw belief masses are markedly less calibrated (0.159) at
matched point accuracy. Under moderate shift, a five-member deep ensemble
is unambiguously the best method by every metric measured. Under severe
shift — the condition that matters most, since it is where the backbone is
known to fail — the ranking inverts: MC Dropout's entropy is the only
signal that substantially improves shift-failure detection over the
uncalibrated baseline (risk-coverage AUC 0.515 to 0.285), while the deep
ensemble's advantage does not transfer (0.473, barely better than
baseline) despite being the clear winner on the milder shift. No method
reaches a reliable abstention operating point on the severe shift
condition: even MC Dropout's best selective accuracy at 90% coverage
(0.583, on a binary task) is barely better than chance after abstaining on
the least-confident decile. We conclude that uncertainty-method ranking is
itself shift-magnitude-dependent, not a fixed property of the method, and
that no single fixed uncertainty-quantification choice can be assumed safe
across the range of shift severity a deployed screening tool will
encounter.

## 1. Introduction

A model that fails silently — misclassifying confidently under conditions
its training data did not cover — is more dangerous in a screening context
than one that fails and flags the failure. Paper 1 established that this is
not a hypothetical concern for this project's own backbone:
EfficientViT-M0, the best in-distribution model across a 12-model registry,
is simultaneously the worst-performing model under cross-dataset domain
shift, dropping below the chance floor on the more severe of two external
cytology datasets tested. This paper asks the natural next question: does
any standard uncertainty-quantification method's own native confidence
signal correctly detect that specific failure, using only the cheapest
version of each method (a single native signal, not a method-specific
post-hoc correction layered on top) so the comparison reflects what a
deployed system would actually have access to without extra engineering?

**Contributions:**
1. A four-method uncertainty-quantification comparison (temperature-scaled
   softmax, MC Dropout, evidential deep learning, deep ensembles) on a
   backbone with a documented, severe, real (not synthetic) distribution-
   shift failure mode, rather than a synthetic corruption benchmark.
2. Evidence that uncertainty-method ranking itself depends on shift
   severity: the method that wins under moderate shift (deep ensemble)
   is not the method that wins under severe shift (MC Dropout), and the
   gap between them under severe shift is large (0.473 vs. 0.285
   risk-coverage AUC) while their gap under moderate shift favors the
   ensemble even more strongly (0.147 vs. 0.203).
3. A direct, quantified answer to whether any of the four methods reaches
   a *reliable* abstention policy under severe shift: none does, and we
   report the closest result (MC Dropout, 0.583 selective accuracy at 90%
   coverage) as still practically insufficient for an "abstain and refer"
   deployment policy, rather than rounding it up to a success.
4. A methods-level finding independent of the uncertainty comparison
   itself: MC Dropout is completely degenerate on this exact backbone
   unless a code-level fix is applied before training, not just at
   inference (Section 3.3) — a reproducibility hazard for anyone applying
   MC Dropout to `efficientvit_m0.r224_in1k` or similarly-configured timm
   models without checking.

## 2. Related Work

Temperature scaling (Guo et al., ICML 2017) remains the standard post-hoc
softmax calibration baseline. MC Dropout (Gal & Ghahramani, ICML 2016)
approximates Bayesian inference via dropout retained at test time; deep
ensembles (Lakshminarayanan et al., NeurIPS 2017) instead use disagreement
across independently-trained models; evidential deep learning (Sensoy et
al., NeurIPS 2018) reframes classification as Dirichlet evidence
accumulation, producing a single-pass uncertainty signal (vacuity) without
the multi-pass or multi-model cost the other two methods incur. The
proposal's own gap analysis (Section 5) identifies a conformal-prediction-
and-Monte-Carlo-inference paper specifically on cervical cancer screening
uncertainty (2024) as adjacent work; this paper differs by comparing four
methods head-to-head on a documented real (not assumed) shift-failure
case, rather than proposing one method in isolation.

## 3. Methods

### 3.1 Backbone

EfficientViT-M0, held fixed across all four methods to isolate the
uncertainty-quantification method rather than architecture — the same
backbone Paper 1 shows to be the best in-distribution and worst
under-shift model in its 12-model registry, making it the model for which
this question is most practically urgent.

### 3.2 Four methods, each method's own cheapest native signal

1. **Baseline**: cross-entropy training, post-hoc temperature scaling
   fit on a held-out calibration split; confidence = temperature-scaled
   max-softmax probability.
2. **MC Dropout**: dropout rate 0.3, 20 stochastic forward passes at
   inference; confidence = negative predictive entropy of the mean
   prediction across passes.
3. **Evidential deep learning**: Dirichlet evidence output via a softplus
   activation and the standard evidential loss (Type II MLE plus an
   annealed KL-divergence-to-uniform penalty); confidence = 1 - vacuity.
4. **Deep ensemble**: 5 independently-seeded members, each trained with
   the same recipe as the baseline; confidence = negative variance of the
   ensemble's mean predicted probability across members.

All four share one manifest, one calibration split, and one backbone
initialization recipe; only the loss function and inference procedure
differ (`src/cervical_screening/uncertainty/train_variants.py`).

### 3.3 A methods-level fix required before any GPU time was spent

`efficientvit_m0.r224_in1k`'s only `nn.Dropout` module ships at `p=0.0` by
default — verified directly via model inspection, not assumed — which
would make MC Dropout completely degenerate (zero stochasticity across
passes). A dropout rate is forced to a real, non-zero value before
training begins, not just patched in at inference time, since MC Dropout's
Bayesian-approximation justification specifically requires dropout to be
active during training as well as at test time. We flag this because the
failure mode is silent (MC Dropout would simply produce zero-variance,
falsely confident "uncertainty" estimates) rather than an error, and is
specific to this exact timm configuration rather than a general property
of MC Dropout.

### 3.4 Evaluation

All four methods evaluated identically on three splits: the SIPaKMeD
in-distribution test set, Herlev (moderate shift), and Mendeley LBC (severe
shift, harmonized to a shared binary normal/abnormal label as in Paper 1).
Metrics: expected calibration error (ECE), full-coverage accuracy,
risk-coverage AUC (integral of error rate over the coverage curve — lower
is better, since a well-behaved confidence signal keeps risk near zero
until coverage is forced high), and selective accuracy at fixed coverage
levels.

## 4. Results

| Method | In-dist ECE | In-dist full-cov acc | Herlev risk-cov AUC | Herlev full-cov acc | Mendeley risk-cov AUC | Mendeley full-cov acc | Mendeley sel-acc@90% cov |
|---|---|---|---|---|---|---|---|
| Baseline (temp-scaled) | 0.012 | 0.967 | 0.193 | 0.694 | 0.515 | 0.530 | 0.522 |
| MC Dropout | 0.013 | 0.972 | 0.203 | 0.658 | **0.285** | 0.579 | **0.583** |
| Evidential | 0.159 | 0.967 | 0.167 | 0.677 | 0.476 | 0.543 | 0.539 |
| Deep ensemble | 0.012 | 0.974 | **0.147** | **0.771** | 0.473 | 0.537 | 0.544 |

**In distribution, three methods are close; evidential is the outlier.**
Baseline, MC Dropout, and the deep ensemble all calibrate well (ECE
0.012-0.013) with full-coverage accuracy in the 0.967-0.974 range.
Evidential deep learning's raw belief masses are markedly less calibrated
(ECE 0.159) at matched point accuracy (0.967) — its probabilities would
need their own post-hoc correction, deliberately not applied here, to make
this comparison's calibration numbers fair to it.

**Under moderate shift (Herlev), the deep ensemble wins clearly.** Its
risk-coverage AUC (0.147) and full-coverage accuracy (0.771) both beat the
other three methods by a real margin — member disagreement is doing
genuine, useful work at this shift severity.

**Under severe shift (Mendeley LBC) — the condition that matters most,
since it is where this exact backbone is known to fail — the ranking
inverts.** MC Dropout's entropy is the only signal showing a substantial,
unambiguous improvement over the uncalibrated baseline (risk-coverage AUC
0.515 to 0.285; full-coverage accuracy 0.530 to 0.579). Evidential (0.476)
and the deep ensemble (0.473) are both only marginally better than the
plain baseline — essentially within noise of each other — despite the
ensemble's clear win on the milder shift one row above. Member
disagreement, which correctly flags moderate shift, does not reliably
track the specific failure mode behind the severe collapse.

**No method reaches a reliable abstention operating point under severe
shift.** Even MC Dropout's best result — 0.583 selective accuracy at 90%
coverage on a binary task — is barely better than chance after abstaining
on the least-confident decile. We report this plainly rather than treating
"best of four" as equivalent to "good enough": for deployment purposes,
none of these four methods alone would currently justify an "abstain and
refer" policy on a shift this severe.

## 5. Discussion

The central methodological finding is that **uncertainty-method ranking is
itself a function of shift severity, not a fixed property of the method**.
A reader who evaluated these four methods only in distribution, or only
under the milder of the two shifts tested, would conclude the deep
ensemble is the best choice and MC Dropout is mediocre — the opposite of
what the severe-shift evaluation shows. This has a direct practical
consequence: a system designer who validates an uncertainty method under
one shift condition and deploys it expecting the same ranking to hold
under a more severe, unseen shift is making an assumption this data does
not support. It also sharpens Paper 1's finding rather than merely
extending it: Paper 1 showed no *architectural* property predicts shift
robustness; this paper shows no *uncertainty-quantification* property
predicts shift-detection quality either, at least not one held fixed across
shift severities. Together, they argue that a trustworthy deployment
cannot rely on a single fixed choice at either level (backbone or
uncertainty method) validated once and assumed stable.

## 6. Limitations

- Single backbone (EfficientViT-M0); whether this specific inversion
  (ensemble best under moderate shift, MC Dropout best under severe shift)
  holds for other architectures is untested — Paper 3's explainability/
  integration work uses a different backbone (DeiT-Tiny) for unrelated
  reasons and does not re-run this comparison there.
- Two shift conditions (Herlev, Mendeley LBC); "moderate" and "severe" are
  descriptive labels based on the accuracy drop observed, not independently
  validated shift-magnitude measurements.
- Evidential deep learning's calibration numbers are reported without the
  post-hoc correction its raw outputs would need for a fully fair
  comparison to the other three methods' already-calibrated numbers; this
  is stated as a limitation of the comparison as run, not hidden.
- Deep ensemble results depend on the specific 5 seeds used; seed-to-seed
  ensemble variance was not separately measured.
- No trained-weight checkpoints persist beyond the ensemble members'
  checkpointed state (saved specifically to survive Colab session
  interruptions during training — see Section 3 reproducibility notes in
  `docs/rq2_results.md`); reproducing any row of this table from scratch
  means retraining.

## 7. Conclusion

Comparing four uncertainty-quantification methods' own cheapest native
confidence signals on a backbone with a documented real domain-shift
failure shows that method ranking inverts between moderate and severe
shift: deep ensembles win under moderate shift, MC Dropout wins under
severe shift, and neither method's advantage transfers to the other
condition. No method reaches a reliable abstention operating point under
the severe shift condition that matters most for this project's own
backbone. The practical conclusion is that a single fixed uncertainty-
quantification method, chosen and validated once, cannot be assumed to
remain the right choice as shift severity varies — this itself needs to be
tested, not assumed, for any deployed system relying on model confidence to
decide when to abstain.

## Data and code availability

Training/evaluation code and raw per-method JSON results
(`runs/colab_rq2/*.json`) are in the project repository.

## References

- Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. "On Calibration of Modern Neural Networks." ICML 2017.
- Gal, Y., & Ghahramani, Z. "Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning." ICML 2016.
- Lakshminarayanan, B., Pritzel, A., & Blundell, C. "Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles." NeurIPS 2017.
- Sensoy, M., Kaplan, L., & Kandemir, M. "Evidential Deep Learning to Quantify Classification Uncertainty." NeurIPS 2018.
- Herlev dataset and Mendeley LBC Pap smear dataset: **citations require verification** — not independently confirmed by this project beyond the Kaggle mirror pages used for data acquisition.
