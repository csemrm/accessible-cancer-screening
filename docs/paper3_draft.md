# Paper 3 (draft): Explainability Fidelity and Deployment Cost in a Trustworthy Cervical Cytology Screening Pipeline

**Status: internal draft, Month 12.** Synthesizes RQ3 (explainability),
RQ5 (system integration), and the Month 11 resource-efficiency/robustness
benchmark, per the publication plan in the project proposal (Section 25).
Companion papers: Paper 1 (RQ1 efficiency/accuracy benchmark + RQ4
cross-dataset shift) and Paper 2 (RQ2 uncertainty/calibration comparison)
cover the results this paper builds on but are drafted separately. Target
venue: MICCAI main track or IEEE TMI, per the proposal's publication plan.

**All numbers in this draft are pulled directly from `docs/rq3_results.md`,
`docs/rq5_results.md`, and `docs/month11_results.md`, and from the JSON
results files those documents cite (`runs/colab_rq3/`, `runs/colab_rq5/`,
`runs/efficiency/`, `runs/month11_robustness/`). Nothing here is
extrapolated or projected beyond what those completed experiments measured.**

---

## Abstract

Vision-transformer explainability methods for medical image classification
are typically evaluated against one baseline at a time, on whichever
backbone a given paper happens to train, without asking whether the choice
of explainability method matters more or less than the choice of
uncertainty-quantification method for the same underlying trustworthiness
goal. We evaluate three explainability methods (relevancy propagation,
attention rollout, and Grad-CAM adapted for vision transformers) on a
cervical cytology classifier, using both behavioral faithfulness (deletion/
insertion) and a real, expert-annotated localization ground truth (nucleus
boundaries from the SIPaKMeD dataset) rather than a derived proxy. Relevancy
propagation wins cleanly on all three metrics. We then ask a system-level
question our own prior uncertainty-quantification comparison (a companion
paper) could not answer alone: does composing two independently-useful
uncertainty signals — temperature-scaled confidence and conformal
prediction-set size — into one abstention rule help, hurt, or do nothing?
The answer is conditional: composition helps exactly where the better
individual signal was already struggling (under severe distribution shift)
and hurts where it was already excellent (in-distribution and under mild
shift). Finally, we measure what trustworthiness costs in practice:
resource-efficiency claims for lightweight vision transformers, evaluated
by parameter count and FLOPs alone, do not reliably predict real CPU
inference latency, and returning an explanation alongside a prediction
costs roughly 3.3x a bare forward pass on CPU. Taken together, these
results argue that explainability method choice is more reliably
consequential than uncertainty-signal composition strategy, and that both
claims require direct measurement rather than architectural or
metric-count inference.

## 1. Introduction

Cervical cancer screening in specialist-scarce settings is a plausible
target for lightweight, on-device AI assistance, but a screening tool that
fails *silently* — misclassifying confidently under conditions its training
data didn't cover — is more dangerous than one that fails and says so. Prior
work in this project (Paper 1) established that a lightweight hybrid vision
transformer (EfficientViT-M0) matches or exceeds much larger CNN and
full-size ViT baselines on in-distribution cervical cytology accuracy, and
that this same backbone can collapse to near-chance accuracy under
cross-dataset distribution shift with no static architectural property
(parameter count, FLOPs, or architecture-family lineage) predicting which
models fail safely. Paper 2 asked whether any of four uncertainty-
quantification methods' native confidence signals correctly flag that
collapse, and found a genuinely mixed answer: MC Dropout's entropy signal
is the only one that substantially improves shift-failure detection on the
severe shift condition, while a deep ensemble — the best method
in-distribution and under mild shift — does not transfer that advantage to
the severe case.

This paper asks the analogous question for **explainability** — does the
choice of explainability method matter as much, and in the same
condition-dependent way? — and then asks a genuinely different question
neither prior paper could answer: given that no single component (backbone,
uncertainty method) is reliably best across all conditions, does
**composing** two different native uncertainty signals into one system
help, or does composition just average away whichever signal was better?
Finally, since a lightweight-architecture claim is only as good as its
measured deployment cost, we report the actual CPU latency and memory cost
of the full pipeline, not an estimate from parameter count.

**Contributions:**
1. A three-way explainability comparison (relevancy propagation vs.
   attention rollout vs. Grad-CAM) on deletion/insertion faithfulness *and*
   real expert-annotated nucleus-localization ground truth, where relevancy
   propagation wins unambiguously on all three metrics — a cleaner result
   than either companion paper's uncertainty-method comparisons produced.
2. The first system-level test (to our knowledge, within this project's own
   scope) of whether combining two uncertainty signals for selective
   prediction helps: a conditional answer (helps under severe shift, hurts
   in-distribution and under mild shift) that extends the "no single best
   method" finding from individual methods to method *combinations*.
3. A measured, not estimated, resource-efficiency and deployment-cost
   analysis: parameter count and FLOPs do not reliably predict CPU latency
   across a 12-model registry, and an explanation costs ~3.3x a bare
   forward pass.

## 2. Related Work

Relevancy propagation for transformers (Chefer, Gur & Wolf, CVPR 2021)
extends attention rollout (Abnar & Zuidema, 2020) by weighting each
transformer block's attention matrix with its gradient with respect to the
target class before the same rollout aggregation, making the resulting
relevance map class-discriminative rather than a fixed structural property
of attention. Grad-CAM (Selvaraju et al., 2017) was originally designed for
CNN feature maps and has been adapted to transformers by targeting a block's
token-wise activations instead of a convolutional feature map; we identify
and document (Section 3.2) a specific failure mode of this adaptation for
CLS-token-pooled ViTs that does not appear to be widely flagged in prior
adaptations. MC Dropout (Gal & Ghahramani, ICML 2016), deep ensembles
(Lakshminarayanan et al., NeurIPS 2017), and evidential deep learning
(Sensoy et al., NeurIPS 2018) are the three uncertainty-quantification
methods our companion paper (Paper 2) compares against temperature-scaled
softmax; conformal prediction's finite-sample coverage guarantee (Vovk et
al.) is the calibration method this paper's selective-prediction
composition experiment builds on. Deletion/insertion faithfulness (Petsiuk
et al., RISE, BMVC 2018) is the behavioral-faithfulness metric used
throughout. Directly adjacent to this paper's own scope: Mohiuddin, Rahman
et al., "Accelerated and Accurate Cervical Cancer Diagnosis Using a Novel
Stacking Ensemble Method with Explainable AI" (International Conference on
Quantum Photonics, Artificial Intelligence, 2025) combines a stacking
ensemble with explainable AI for cervical cancer diagnosis; this paper
differs in evaluating explainability methods against a real expert-
annotated localization ground truth (nucleus polygons) and behavioral
faithfulness metrics rather than an ensemble-accuracy framing, and in
explicitly comparing three explainability methods head-to-head rather than
applying one. Rahman et al., "MaizeFormerX: a lightweight vision
transformer with cross-scale attention for explainable maize leaf disease
diagnosis" (Scientific Reports, 2026) applies a parallel lightweight-ViT-
plus-explainability design philosophy to a different domain (plant disease
diagnosis), motivating this paper's own choice to hold efficiency and
explainability as joint, not competing, design goals. Two further papers
from the same author sit in adjacent clinical-AI territory without
overlapping this paper's specific claims: Gony, Shuvra, Poudel, Dhakal,
Rahman et al., "Dual-View Semi-Supervised Retinal Disease Recognition via
Chromatic Structural Decomposition and Cross-View Contrastive Learning"
(International Conference on Power, Electronics, Communications, 2026) is
a medical-image disease-recognition system in a different clinical domain
(retinal imaging) using a semi-supervised, non-transformer-explainability
approach; Ahmed, Ahmed, Mow, Taha, Barua, Rahman et al., "An Explainable
Multimodal AI Framework with Reinforcement Learning for Post-Surgical
Clinical Decision Support" (medRxiv, 2026) applies explainable AI to a
clinical decision-support setting downstream of diagnosis (post-surgical
care) rather than a screening/diagnosis task itself. Both establish this
author's continuing research program in trustworthy, explainable clinical
AI, of which this paper's cervical cytology system is one instance. The
project's own literature-gap analysis (proposal Section 5)
identifies specific adjacent 2024-2026 work combining subsets of these
methods for cervical or oral cancer imaging; none combine an
efficiency-budgeted architecture, single-pass uncertainty, small-sample-
correct conformal calibration, transformer-native explainability, and
cross-dataset robustness evaluation together, which remains this project's
positioning (not re-litigated here; see the proposal for the full gap
discussion, which is marked for re-verification with a fresh literature
search near submission given how fast this intersection moves).

## 3. Methods

### 3.1 Backbone

Paper 1's winning backbone, EfficientViT-M0, uses a hybrid convolutional +
cascaded-group-attention design. Relevancy propagation and attention
rollout are both defined for standard multi-head self-attention transformer
blocks; EfficientViT-M0 has no established propagation rule for either
method. Rather than force an ad hoc extension onto an architecture these
methods were not designed for, this paper uses **DeiT-Tiny** — the
proposal's "efficient ViT" baseline that is a standard `timm`
`VisionTransformer`, achieving 0.9606 balanced accuracy on the same
SIPaKMeD in-distribution task in Paper 1 (vs. EfficientViT-M0's 0.9654) —
retrained fresh for this paper (best validation loss 0.0918-0.0920 across
the two training runs used in Sections 4.2-4.3, both early-stopped around
epoch 24-29). This substitution is a limitation, not a hidden assumption:
Section 6 states plainly that whether these findings transfer to
EfficientViT-M0 or another hybrid architecture is untested.

### 3.2 Explainability methods

All three methods produce a relevance map over the model's 14x14 patch grid
(DeiT-Tiny, 224px input, 16px patches), upsampled to image resolution.

- **Attention rollout**: raw softmax attention, averaged over heads,
  chained across all 12 blocks via the standard rollout recurrence (adding
  the identity matrix per block to account for the residual connection,
  row-normalizing, then chain-multiplying). Class-agnostic.
- **Relevancy propagation**: identical rollout recurrence, but each block's
  contribution is `ReLU(grad(attention) ⊙ attention)`, averaged over heads,
  where the gradient is with respect to the model's own predicted class's
  logit. Class-discriminative.
- **Grad-CAM (ViT-adapted)**: gradient-weighted channel importance applied
  to a transformer block's token activations rather than a CNN's
  convolutional feature map. **Implementation note**: the target layer must
  be the *second-to-last* transformer block, not the last. DeiT-Tiny pools
  only the CLS token for classification (`global_pool="token"`), so no
  layer downstream of the final block mixes information back into patch-
  token positions — verified directly by inspecting gradients rather than
  assumed: every patch token's gradient at the last block's output was
  exactly zero, while the second-to-last block's patch-token gradients
  vary normally (the final block's attention still mixes them into CLS).
  This produces a completely degenerate (all-zero) heatmap if
  misconfigured, silently rather than with an error, and is the standard
  adaptation used for CLS-pooled ViTs in the Grad-CAM-for-transformers
  literature — we simply verified it rather than assumed it, via a
  CPU-only unit test before spending any GPU time.

### 3.3 Faithfulness and localization metrics

**Deletion/insertion AUC** (Petsiuk et al.): patches are ranked by a
method's own relevance score and progressively replaced with (deletion) or
revealed from (insertion) a heavily Gaussian-blurred version of the image,
tracking the model's predicted-class probability at each step; the area
under the resulting probability-vs-fraction curve is reported (lower is
better for deletion, higher is better for insertion). Blurring rather than
zeroing or mean-filling the removed region avoids a trivially
out-of-distribution deletion baseline.

**Nucleus concentration ratio**: SIPaKMeD ships real, expert-marked nucleus
boundary polygons per cell (used originally only for crop derivation); we
persist these in crop-local pixel coordinates and compute the fraction of a
heatmap's total relevance mass falling inside the nucleus region, divided
by the nucleus's share of image area. A ratio of 1.0 is exactly chance
level; higher values mean a heatmap concentrates relevance in the nucleus
more than chance would predict — clinically the correct place to
concentrate, since nuclear morphology (enlargement, hyperchromasia,
irregular contour) is the primary diagnostic feature under the Bethesda
system. This is a genuine expert annotation, not a derived proxy, which we
consider a methodological improvement over evaluating against a segmentation
model's own output.

### 3.4 Calibration, conformal prediction, and the combined selective-prediction signal

Temperature scaling (Guo et al.) is fit on a held-out calibration split
disjoint from both training and test data. Conformal prediction sets are
built via a resampling-averaged split-conformal threshold (bootstrap-
averaging the finite-sample-corrected quantile across the calibration set,
addressing the instability a single small calibration split introduces at
this project's ~1,000-4,000-image dataset scale — not the literature-
standard jackknife+/CV+, which would require refitting the model per
held-out point, not affordable at this project's compute budget, and stated
as such). The **combined selective-prediction signal** rank-normalizes
temperature-scaled max-softmax confidence and (inverted) conformal
prediction-set size independently, then averages their ranks — avoiding the
unit mismatch of directly summing a probability and a small integer count.

### 3.5 Efficiency and latency measurement

Parameters and FLOPs (via `torch.utils.flop_counter.FlopCounterMode`) are
architecture properties independent of trained weight values, so measured
without any training. CPU latency and memory (isolated per model in its own
subprocess — see Section 4.4's methodological note) are measured with
`pretrained=False` weights for the same reason. The "cost of an explanation"
is measured as a full forward+backward+rollout pass for relevancy
propagation's own predicted-class explanation, against a bare forward pass,
both on CPU (the no-GPU-assumed deployment case this project's efficiency
claim targets).

## 4. Experiments

### 4.1 Data

SIPaKMeD (Plissiti et al., ICIP 2018): 4,049 single-cell images, 5
cytomorphology classes, split 60/15/10/15 into train/val/calibration/test.
Cross-dataset shift results (Herlev, Mendeley LBC — citations for both
require verification; not independently confirmed by this project beyond
the Kaggle mirror pages used to acquire them) are reported in Paper 1/2 and
not repeated here, since RQ3/RQ5's questions are answered on the
in-distribution split and (for RQ5's selective-prediction test) the same
shift sets Paper 2 used, for direct comparability of the composition result
against Paper 2's individual-method numbers.

### 4.2 Explainability results (RQ3)

| Method | Deletion AUC (lower better) | Insertion AUC (higher better) | Nucleus concentration (higher better, 1.0 = chance) |
|---|---|---|---|
| **Relevancy propagation** | **0.734** | **0.970** | **2.714** |
| Grad-CAM (ViT-adapted) | 0.818 | 0.877 | 1.472 |
| Attention rollout | 0.825 | 0.936 | 1.177 |

(50 SIPaKMeD test images, 10 per class, stratified, fixed seed; all 50
matched to a real nucleus polygon.)

Relevancy propagation wins every comparison, and by a wide margin on
nucleus concentration specifically (2.71x chance vs. attention rollout's
1.18x — barely above chance). We attribute this to a mechanistic
difference rather than measurement noise: attention rollout is
class-agnostic, an inherent disadvantage on an inherently
class-discriminative faithfulness test, while relevancy propagation is both
class-discriminative *and* integrates the gradient-weighted signal across
all twelve transformer blocks, rather than one block's channel-gradient
signal (Grad-CAM) or attention's unweighted structural pattern (rollout).
Unlike Paper 2's uncertainty-method comparison, no method's ranking flips
across the three metrics here — a materially cleaner result.

### 4.3 System-integration results (RQ5)

Two of H5's four proposed sub-claims (proposal Section 8) hold by
construction given this project's post-hoc-attachment architecture, and
were verified rather than assumed: temperature scaling changed **zero**
argmax predictions on the calibration split (it is a monotonic transform of
frozen logits), and explainability computation is a read-only analysis that
cannot touch model weights. The genuinely open question is whether
composing two uncertainty signals for selective prediction helps:

| Condition | Confidence only | Conformal set size only | Combined (rank-averaged) |
|---|---|---|---|
| In-distribution (SIPaKMeD test) | **0.0050** | 0.0834 | 0.0173 |
| Herlev (moderate shift) | **0.1190** | 0.3281 | 0.1793 |
| Mendeley LBC (severe shift) | 0.3244 | 0.4395 | **0.2855** |

(Risk-coverage AUC, lower is better.)

Composition is not a uniform win. In distribution and under moderate shift,
confidence alone is already the best individual signal, and combining it
with the noisier discrete conformal-set-size signal (clearly worse than
confidence alone in both conditions) dilutes it — the combined score lands
between the two individual signals but worse than the better one alone.
Under severe shift, however — the condition where confidence alone
degrades the most — the combined signal is the *best* of the three,
beating confidence alone. This mirrors the shape of Paper 2's finding for
individual uncertainty methods (no single choice wins everywhere), now
extended to combinations of signals: a deployed system would need a rule
for *when* to combine signals (e.g., conditioned on a shift-detection
signal itself), not a fixed always-combine or never-combine policy.

### 4.4 Resource efficiency and deployment robustness (Month 11)

**Efficiency is not predictable from architecture name or FLOP count
alone.** Across all 12 registry models (params/FLOPs measured architecture-
only, no training needed), EfficientNet-Lite0 (3.38M params, 0.77G FLOPs)
has more than 4x ResNet-50's CPU latency (137.7ms vs. 32.0ms) despite
ResNet-50 having roughly 10x the FLOPs — depthwise-separable convolutions
do not map efficiently onto generic CPU BLAS kernels the way dense
convolutions do. Even EfficientViT-M0, the overall winner on
params+FLOPs+accuracy, is not the fastest model in wall-clock CPU terms:
DeiT-Tiny (15.4ms) and LeViT-128 (17.5ms) are both faster despite 10-14x
more FLOPs. A methodological note: an initial implementation measured
params/FLOPs/latency for all 12 models in one process before measuring
memory, and every model's memory delta after the first (larger) model
silently reported 0.0 — `ru_maxrss` is a monotonically non-decreasing
high-water mark for the whole process, not a live delta, so a later
model's forward pass never exceeds whatever peak an earlier, larger model
already established. Fixed by isolating each model's memory measurement in
its own subprocess; we flag this because it is a general footgun for
anyone building a similar multi-model efficiency harness, not specific to
this codebase.

**Deployment robustness**: EfficientViT-M0, evaluated under synthetic JPEG
compression, Gaussian blur, and brightness/contrast shift (5 severities
each), degrades gracefully (no threshold-effect cliff) under all three, but
Gaussian blur is by far the most damaging — costing 32.8 points of balanced
accuracy at its most severe tested level, roughly double JPEG's worst case
(15.5 points) or brightness/contrast's (18.1 points). This tracks a
plausible clinical mechanism: Bethesda-system cytology diagnosis depends on
fine nuclear texture that blur destroys directly, while JPEG at moderate
quality and brightness/contrast shifts leave more of that structure intact.
A synthetic OOD sanity check (25 images: random noise, solid colors,
checkerboards) shows mean confidence dropping from 0.985 (real
in-distribution images) to 0.743 (OOD) — real and meaningful, but far above
the 0.20 chance level for this 5-class task. We report this as a partial
result: confidence alone is a genuine but weak OOD signal on its own.

**The cost of trustworthiness, measured**: on CPU, a bare DeiT-Tiny forward
pass has a median latency of 49.6ms; a prediction that also returns a
relevancy-propagation explanation for its own predicted class costs
164.1ms — about 3.3x the bare-prediction cost.

## 5. Discussion

Two threads run through this paper and its companions. First,
**explainability method choice produced the cleanest, most unambiguous
result in the project so far** — relevancy propagation wins every metric,
with no condition-dependent flip — while **uncertainty method choice and
uncertainty-signal composition both produced conditional, no-single-winner
results**. This asymmetry is itself informative: it suggests that for this
task and backbone family, picking the right explainability method is a
more reliably beneficial engineering decision than picking (or combining)
the right uncertainty-quantification method, where the honest answer is "it
depends on the operating condition." Second, **every efficiency and
robustness claim in this paper required direct measurement to be trusted**:
FLOP count did not predict CPU latency; parameter count did not predict
memory footprint without fixing a real measurement bug first; and OOD
confidence dropped only partially, not to the clean near-chance level a
reader might assume from an "OOD detection" framing. None of these findings
would have been visible from architectural reasoning or from citing prior
literature's general claims about efficient architectures or OOD detection
without this project's own direct measurement.

## 6. Limitations

- **Backbone mismatch with Paper 1's winner.** RQ3/RQ5 and the Month 11
  degradation/OOD check together use two different backbones (DeiT-Tiny for
  RQ3/RQ5, EfficientViT-M0 for the Month 11 degradation/OOD/latency-of-
  explanation numbers are *not* run on the same model as the efficiency
  table's EfficientViT-M0 latency row — the explanation-cost number in
  Section 4.4 is specifically for DeiT-Tiny, not EfficientViT-M0, since
  relevancy propagation is not defined for the latter). Whether relevancy
  propagation's clean win, or the selective-prediction composition result,
  hold on a hybrid conv+attention backbone is untested.
- **Single-institution, small-scale data.** SIPaKMeD is ~4,000 images from
  one source; generalization beyond it is not established by this paper.
- **CPU latency measured on one shared consumer laptop**, not dedicated or
  isolated benchmark hardware — a ~30% run-to-run swing was observed for
  the same model in repeated measurements, though relative ordering across
  the 12 models was stable and reproduced. True edge/mobile-class device
  measurement (an actual phone or single-board computer) was not attempted.
- **OOD sanity check uses a small, synthetic sample** (25 images) rather
  than a natural out-of-domain image dataset; the qualitative finding
  (confidence drops but not to chance) should be treated as suggestive, not
  a precise estimate.
- **No clinical or expert-review validation.** Explanation-localization
  fidelity against nucleus polygons is not equivalent to a cytotechnologist
  or pathologist rating whether an explanation is clinically meaningful;
  this project's proposal explicitly scopes an expert-review pilot as
  separate future work, not yet run.

## 7. Conclusion

Relevancy propagation is a clearly better explainability method than
attention rollout or Grad-CAM for this cervical cytology classifier, on
both behavioral faithfulness and real expert-annotated localization
ground truth — the most unambiguous positive result this project has
produced. Composing two uncertainty signals for selective prediction, by
contrast, helps only under the specific condition where the better
individual signal was already failing, extending this project's broader
finding (no single fixed choice is reliably best across all conditions)
from individual methods to combinations of methods. Resource-efficiency and
robustness claims for lightweight vision transformers require direct
measurement in both directions: FLOP-efficient does not mean
latency-efficient, and "confidence drops on OOD input" does not mean
"confidence reaches a safe abstention level."

## Data and code availability

All training/evaluation code, JSON result files, and this draft are in the
project repository (private during development; see the project's
publication and NIW-adjacent national-importance sections for the intended
public-release plan, subject to dataset license verification — proposal
Section 11 flags this as not yet independently confirmed).

## References

Citations verified elsewhere in this project's own code/docs (not
independently re-verified for this draft; a full reference list with exact
venue/page details should be compiled from primary sources before
submission):

- Chefer, H., Gur, S., & Wolf, L. "Transformer Interpretability Beyond Attention Visualization." CVPR 2021.
- Abnar, S., & Zuidema, W. "Quantifying Attention Flow in Transformers." 2020.
- Selvaraju, R. R., et al. "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization." 2017.
- Gal, Y., & Ghahramani, Z. "Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning." ICML 2016.
- Lakshminarayanan, B., Pritzel, A., & Blundell, C. "Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles." NeurIPS 2017.
- Sensoy, M., Kaplan, L., & Kandemir, M. "Evidential Deep Learning to Quantify Classification Uncertainty." NeurIPS 2018.
- Petsiuk, V., Das, A., & Saenko, K. "RISE: Randomized Input Sampling for Explanation of Black-box Models." BMVC 2018.
- Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. "On Calibration of Modern Neural Networks." ICML 2017.
- Plissiti, M. E., et al. "SIPAKMED: A New Dataset for Feature and Image Based Classification of Normal and Pathological Cervical Cells in Pap Smear Images." ICIP 2018.
- Mohiuddin, A. B., Rahman, M. M., et al. "Accelerated and Accurate Cervical Cancer Diagnosis Using a Novel Stacking Ensemble Method with Explainable AI." International Conference on Quantum Photonics, Artificial Intelligence, 2025.
- Rahman, M. M., et al. "MaizeFormerX: a lightweight vision transformer with cross-scale attention for explainable maize leaf disease diagnosis." Scientific Reports, 2026.
- Gony, M. N., Shuvra, S. M. K., Poudel, R., Dhakal, K., Rahman, M. M., et al. "Dual-View Semi-Supervised Retinal Disease Recognition via Chromatic Structural Decomposition and Cross-View Contrastive Learning." International Conference on Power, Electronics, Communications, 2026.
- Ahmed, M., Ahmed, F., Mow, S. M., Taha, P. A., Barua, S., Rahman, M. M., et al. "An Explainable Multimodal AI Framework with Reinforcement Learning for Post-Surgical Clinical Decision Support." medRxiv, 2026.
- Herlev dataset and Mendeley LBC Pap smear dataset: **citations require verification** — not independently confirmed by this project beyond the Kaggle mirror pages used for data acquisition; do not cite a specific author/venue for either without first locating and confirming the primary source.
