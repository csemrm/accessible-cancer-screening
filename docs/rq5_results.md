# RQ5 results: does composing the four properties degrade anything, and does combining uncertainty signals help?

## Setup

RQ5/H5 asks whether efficiency, calibration, explainability, and selective
prediction can be integrated without one degrading another. Backbone:
DeiT-Tiny (same as RQ3, and for the same reason -- relevancy propagation
needs a standard multi-head self-attention transformer, which EfficientViT-M0
is not), retrained fresh (best val loss 0.0918, 24 epochs, early-stopped).

**Two of H5's four sub-claims hold by construction, not by discovery, given
this project's design.** Every property here attaches to a fixed trained
backbone rather than being jointly optimized:

- **Calibration cannot degrade accuracy.** Temperature scaling is a monotonic
  transform of a frozen model's logits, so it cannot change which class has
  the highest probability. Verified directly rather than assumed: temperature
  scaling changed the predicted class for **zero** calibration-split samples.
- **Explainability cannot degrade the model.** Relevancy propagation is a
  read-only post-hoc analysis (a second forward+backward pass that never
  updates weights), so by construction it cannot change accuracy or
  calibration.

The one place composition is genuinely non-trivial -- and the actual
empirical question this section tests -- is **selective prediction**: does
combining two different native uncertainty signals (temperature-scaled
confidence, and conformal prediction-set size) into one abstention rule
help, hurt, or do nothing, relative to using either signal alone? And
separately: **what does adding an explanation actually cost**, in CPU
latency, for a deployment context that assumes no GPU?

## Results: combining confidence and conformal set size for selective prediction

Risk-coverage AUC (lower is better) for each signal, on the same
temperature-scaled + conformally-calibrated DeiT-Tiny, at three operating
conditions:

| Condition | Confidence only | Conformal set size only | Combined (rank-averaged) |
|---|---|---|---|
| In-distribution (SIPaKMeD test) | **0.0050** | 0.0834 | 0.0173 |
| Herlev (moderate shift) | **0.1190** | 0.3281 | 0.1793 |
| Mendeley LBC (severe shift) | 0.3244 | 0.4395 | **0.2855** |

## Findings

**Combining the two signals is not a uniform win -- it helps exactly where
confidence alone was already struggling, and hurts everywhere else.** In
distribution and on Herlev, temperature-scaled confidence alone is already
an excellent signal (risk-coverage AUC 0.005 and 0.119); adding the coarser,
much-noisier conformal set-size signal (0.083 and 0.328 alone -- clearly
worse than confidence alone in both cases) dilutes it, and the combined
score is worse than confidence alone (though still much better than set
size alone). But on Mendeley LBC -- the severe shift where confidence alone
degrades the most (0.324) -- the combined signal is actually the *best* of
the three (0.285, beating confidence alone). This is a real, useful, and
non-obvious result: **naive rank-averaged composition of two uncertainty
signals is condition-dependent, not universally helpful or harmful**, which
is the same shape of finding RQ2 produced for individual uncertainty
methods (no single choice wins everywhere) -- now shown to extend to
*combinations* of signals, not just to the choice between methods.

**The cost of an explanation, measured, not assumed.** On CPU (the
no-GPU-assumed deployment case proposal S19 cares about), a bare forward
pass has a median latency of 49.6ms; a prediction that also returns a
relevancy-propagation explanation for its own predicted class costs 164.1ms
-- **about 3.3x** the bare-prediction cost. This is a real, reportable
number for the Month 11 resource-efficiency benchmarking to build on, not a
qualitative "explanations are expensive" claim.

## Status against the RQ5 hypothesis (proposal S18)

- **H5** (the four properties compose without degrading each other): **partially supported, more precisely than the literal hypothesis anticipated.** Calibration and explainability provably cannot degrade accuracy given this project's post-hoc architecture (verified, not assumed). Selective-prediction composition is genuinely conditional: naive combination of two uncertainty signals helps under severe shift and hurts under mild/no shift, rather than uniformly helping or hurting. A production system would need a rule for WHEN to combine signals, not a fixed always-combine or never-combine policy.

## Implementation notes

- `src/cervical_screening/integration/combined_selective.py` rank-averages
  temperature-scaled confidence with (inverted) conformal prediction-set size
  into one [0,1] abstention score, avoiding the unit mismatch of directly
  summing a probability and a small integer count.
- Conformal sets are always computed at the native 5-class SIPaKMeD level
  (matching what the calibration split's threshold was fit against), even
  when scoring the harmonized binary shift task -- consistent with RQ2's
  precedent of evaluating each method's own native signal against the
  harmonized correctness label, rather than re-deriving a binary-specific
  conformal procedure.
- `src/cervical_screening/integration/pipeline_latency.py` measures the CPU
  cost of a bare forward pass vs. a forward+backward+rollout explanation;
  temperature scaling and conformal set construction are both O(1) operations
  on a logit/probability vector already in hand and are not separately
  profiled (their cost is unmeasurably small next to a transformer forward
  pass).
- `colab_kernel/rq5/` holds the two-script pipeline (`setup.py` trains
  DeiT-Tiny and builds the SIPaKMeD + Herlev/Mendeley manifests;
  `run_eval.py` fits calibration+conformal, runs the three-signal
  comparison, and measures latency), reusing `colab_kernel/rq2/common_eval.py`
  unchanged for the evaluation loaders.
