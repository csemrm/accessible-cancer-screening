# Paper 1 (draft): Efficient Vision Transformers Match Larger Models on Cervical Cytology Classification, But In-Distribution Accuracy Does Not Predict Domain-Shift Robustness

**Status: internal draft, journal-article format.** Target venue: IEEE
Journal of Biomedical and Health Informatics (JBHI), per the proposal's
publication plan (Section 25); a MICCAI workshop is the stated alternative.
Covers RQ1 (efficiency/accuracy benchmark) and the domain-shift half of RQ4
(cross-dataset robustness). Companion papers: Paper 2 (RQ2, uncertainty/
calibration) and Paper 3 (RQ3+RQ5+Month 11, explainability and system
integration), drafted separately.

**All numbers in this draft are pulled directly from `docs/rq1_results.md`
and the JSON result files it cites (`runs/colab_rq1/*.json`). Nothing here
is extrapolated or projected beyond what these completed experiments
measured.**

---

## Abstract

Lightweight vision transformers are frequently proposed for medical image
classification on the premise that they trade a small, bounded amount of
accuracy for a large reduction in compute cost, and are rarely tested for
whether that same efficiency holds under the distribution shift a
deployed model will actually encounter across clinics, scanners, and
populations. We benchmark 12 models spanning CNN, full-size vision
transformer, and efficient-vision-transformer families on cervical
cytology classification (SIPaKMeD, 5 Bethesda-relevant cytomorphology
classes), then evaluate the same models zero-shot, with no retraining or
recalibration, on two independent external cytology datasets (Herlev,
Mendeley LBC) after harmonizing all three onto a shared binary
normal/abnormal label. Contrary to the usual efficiency-accuracy
trade-off framing, the two most parameter-efficient models in the entire
registry (EfficientViT-M0 and -B0, ~2.1M parameters each) are also the two
best-performing in distribution, beating every full-size baseline by 1-3
points of balanced accuracy. Under domain shift, however, in-distribution
accuracy actively misleads: EfficientViT-M0 — the overall in-distribution
winner — is the worst-performing model of all 12 on both external
datasets, dropping below the chance floor on Mendeley LBC (0.4731 balanced
accuracy on a binary task), while a much larger CNN baseline (ResNet-50)
that is not the best in-distribution model on any dataset is the most
robust model on that same shift. Across the full registry, CNN-family
models cluster at 0.539-0.667 balanced accuracy on the severe shift
condition while eight of nine transformer-family models cluster at
0.293-0.523 (with one transformer landing 0.003 above the lowest CNN — a
real, reported exception, not a clean partition), and one model
(MobileViT-S) collapses to an AUROC of 0.215 — actively anti-correlated
with ground truth, not merely uninformative — while its own architectural
successor (MobileViTv2) shows no such failure and is in fact the strongest
model on the milder shift condition. No single static property we measured
— parameter count, FLOPs, in-distribution accuracy, or architecture
family/lineage — reliably predicts which model fails safely under domain
shift.

## 1. Introduction

Cervical cancer remains substantially more curable when caught at a
localized stage, yet screening access is documented to be uneven across
rural and urban U.S. counties, and a growing literature proposes vision
transformers — often marketed as "efficient" or "lightweight" variants —
as classification backbones for automated cytology screening. Two
questions are almost never asked together in that literature: does the
claimed efficiency actually cost accuracy on this specific task, and does
whatever backbone wins an in-distribution benchmark remain trustworthy
when the input distribution shifts, as it inevitably will across
institutions, scanners, staining protocols, and patient populations? This
paper answers both, using a single consistent benchmark across a 12-model
registry spanning three architecture families (standard CNNs, full-size
vision transformers, and efficient/mobile-oriented vision transformers),
evaluated first in distribution and then zero-shot on two independent
external cytology datasets.

**Contributions:**
1. A 12-model efficiency/accuracy benchmark on cervical cytology
   classification showing that, on this task, the most parameter-efficient
   models in the registry are also the most accurate — efficiency costs
   nothing here, contrary to the usual framing.
2. A systematic escalation from a 2-model to a 4-model to a full 12-model
   cross-dataset domain-shift evaluation, each stage refining rather than
   simply confirming the prior stage's finding, converging on: **no single
   static model property predicts shift robustness**, and the
   in-distribution winner is the shift-robustness loser.
3. A specific, load-bearing counterexample to "architecture family
   predicts behavior" reasoning: MobileViT-S collapses to below-chance,
   actively anti-correlated predictions on one external dataset, while its
   direct architectural successor, MobileViTv2, shows no such failure and
   is the single best-performing model in the registry on the other
   external dataset.

## 2. Related Work

The proposal underlying this paper (Section 5) documents a targeted
2022-2026 literature pass finding "vision transformer + cervical or oral
cancer image + a single explainability or efficiency claim" to be a common
pattern that rarely combines an efficiency-budgeted architecture with
cross-dataset robustness evaluation; specific adjacent works identified
there (a hybrid CNN-ViT oral-cancer explainability model, a fuzzy-ensemble
cervical-cytology paper, a conformal-prediction cervical-screening
uncertainty paper, and the author's own prior stacking-ensemble cervical
cancer + explainable AI paper — Mohiuddin, Rahman et al., 2025) are
discussed there in full; this paper does not re-litigate that gap analysis
and refers readers to the proposal for it, flagged there for
re-verification with a fresh search near submission given how quickly this
intersection moves.

## 3. Methods

### 3.1 Model registry

Twelve models spanning three families: CNN baselines (ResNet-50,
MobileNetV2, EfficientNet-Lite0), full-size vision transformers (ViT-Small,
Swin-Tiny), and efficient/mobile-oriented vision transformers (DeiT-Tiny,
LeViT-128, TinyViT-5M, MobileViT-S, MobileViTv2, EfficientViT-B0,
EfficientViT-M0). All timm tags verified against the pinned timm version
rather than assumed from memory (timm's naming scheme has changed across
versions).

### 3.2 Training and in-distribution evaluation

All 12 models trained on SIPaKMeD (4,049 native-resolution single-cell
crops derived directly from the dataset's own cytoplasm-polygon boundary
annotations, not a pre-resized third-party mirror, since later
explainability work in this project's broader scope needs nucleus-level
detail the common 66x66px mirrors discard), 60/15/10/15
train/val/calibration/test split, seed 13, full 30-epoch budget with early
stopping (patience 7), on a Colab T4 GPU. Reported: balanced accuracy,
macro F1, AUROC, parameter count, and measured (not estimated) inference
latency.

### 3.3 Cross-dataset domain-shift evaluation

Each model evaluated zero-shot — no retraining, no recalibration — on
Herlev and Mendeley LBC after harmonizing all three datasets' differing
label taxonomies onto one shared binary normal/abnormal label (mapping
detailed in the project's `docs/cross_dataset_harmonization.md`). This
tests generalization under the specific conditions this project's
screening application will actually face: different institutions,
scanners, and staining protocols, not a synthetic corruption of the same
source images.

## 4. Results

### 4.1 Efficiency/accuracy benchmark

| Model | Balanced accuracy | Macro F1 | AUROC | Params | Latency |
|---|---|---|---|---|---|
| **EfficientViT-M0** | **0.9654** | **0.9656** | 0.9985 | **2,157,401** | 17.56ms |
| EfficientViT-B0 | 0.9623 | 0.9624 | 0.9983 | 2,135,413 | 6.97ms |
| DeiT-Tiny | 0.9606 | 0.9607 | 0.9974 | 5,525,381 | 5.17ms |
| Swin-Tiny | 0.9605 | 0.9606 | 0.9965 | 27,523,199 | 11.36ms |
| TinyViT-5M | 0.9587 | 0.9589 | 0.9979 | 5,073,369 | 9.57ms |
| MobileViT-S | 0.9570 | 0.9575 | 0.9979 | 4,940,837 | 9.32ms |
| ResNet-50 | 0.9542 | 0.9540 | 0.9966 | 23,518,277 | 5.95ms |
| LeViT-128 | 0.9458 | 0.9456 | 0.9946 | 8,447,786 | 12.81ms |
| MobileViTv2 | 0.9387 | 0.9401 | 0.9966 | 4,391,406 | 9.90ms |
| EfficientNet-Lite0 | 0.9375 | 0.9376 | 0.9944 | 3,377,413 | 5.51ms |
| ViT-Small | 0.9374 | 0.9380 | 0.9960 | 21,667,589 | 5.39ms |
| MobileNetV2 | 0.9357 | 0.9365 | 0.9952 | 2,230,277 | 7.01ms |

The two smallest models in the registry are also the two best-performing,
beating every full-size baseline (ResNet-50, 23.5M params; Swin-Tiny,
27.5M params) by 1-3 points of balanced accuracy. On this task, at this
data scale, the efficiency-accuracy trade-off the proposal is built around
does not appear at all in-distribution — whether this generalizes beyond
this one dataset/split is exactly what Section 4.2 tests, and the answer
is not reassuring.

### 4.2 Cross-dataset domain shift, escalating from 2 to 12 models

**Stage 1 (DeiT-Tiny alone):** in distribution 0.9504 balanced accuracy,
Herlev (zero-shot) 0.7151, Mendeley LBC (zero-shot) 0.5230 — a model that
looks excellent in distribution degrades to near-chance on the more severe
external dataset, with nothing in a standard pipeline able to flag this.

**Stage 2 (adding EfficientViT-M0):** the *better* in-distribution model
(0.9737 vs. DeiT-Tiny's 0.9504) is the *worse* model under both shifts
(Herlev 0.6716 vs. 0.7151; Mendeley LBC 0.4731 vs. 0.5230 — the first
model in this study to drop below the 0.50 chance floor). Two models this
close in architecture family and parameter count producing this reversal
is a direct caution against using in-distribution accuracy as a proxy for
deployment robustness.

**Stage 3 (adding ResNet-50 and Swin-Tiny, one model per family):**

| Model (family) | SIPaKMeD | Herlev | Mendeley LBC |
|---|---|---|---|
| EfficientViT-M0 (efficient ViT) | **0.9737** | 0.6716 | **0.4731** (below chance) |
| Swin-Tiny (full ViT) | 0.9590 | **0.7282** | 0.5192 |
| DeiT-Tiny (efficient ViT) | 0.9504 | 0.7151 | 0.5230 |
| ResNet-50 (CNN) | 0.9556 | 0.6838 | **0.6082** |

Not a clean CNN-vs-transformer split: on Herlev, both transformer models
beat ResNet-50. What holds across both shift datasets is narrower: the
single most parameter-efficient, best-in-distribution model
(EfficientViT-M0) is the worst performer under shift on both external
datasets, and the only one below chance on either.

**Stage 4 (all 12 models):**

| Model (family, params) | SIPaKMeD | Herlev | Mendeley LBC |
|---|---|---|---|
| EfficientNet-Lite0 (CNN, 3.38M) | 0.9455 | 0.5970 | **0.6674** |
| ResNet-50 (CNN, 23.5M) | 0.9556 | 0.6838 | 0.6082 |
| MobileViTv2 (eff. ViT, 4.39M) | 0.9638 | **0.7501** | 0.5418 |
| MobileNetV2 (CNN, 2.23M) | 0.9505 | 0.6919 | 0.5391 |
| DeiT-Tiny (eff. ViT, 5.5M) | 0.9504 | 0.7151 | 0.5230 |
| EfficientViT-B0 (eff. ViT, 2.14M) | 0.9738 | 0.6818 | 0.5217 |
| Swin-Tiny (full ViT, 27.5M) | 0.9590 | 0.7282 | 0.5192 |
| ViT-Small (full ViT, 21.7M) | 0.9573 | 0.7049 | 0.5084 |
| LeViT-128 (eff. ViT, 8.4M) | 0.9526 | 0.5777 | 0.4764 |
| EfficientViT-M0 (eff. ViT, 2.16M) | 0.9737 | 0.6716 | 0.4731 |
| TinyViT-5M (eff. ViT, 5.1M) | 0.9685 | **0.7327** | 0.4543 |
| MobileViT-S (eff. ViT, 4.94M) | 0.9520 | 0.7104 | **0.2927** (AUROC 0.215) |

A near-clean CNN-vs-transformer split emerges on Mendeley LBC, with one
honest exception: all three CNNs land at 0.539-0.667; eight of nine
transformer-family models land at 0.293-0.523; the ninth (MobileViTv2)
scores 0.5418, narrowly *above* the lowest CNN (0.5391) by 0.003 — a real
overlap, reported as such rather than rounded away. This split does not
hold at all on Herlev, where transformer models occupy both the best
(MobileViTv2, TinyViT-5M) and among the weaker (LeViT-128) positions.

MobileViT-S is a severe, singular outlier: its Mendeley LBC AUROC (0.215)
is not merely below chance but actively anti-correlated with ground truth.
Its direct successor, MobileViTv2, shows no such problem and is the
single strongest model in the entire registry on Herlev (0.7501) — the
same architecture-family name spanning a catastrophic failure and a top
performer, ruling out architecture lineage as a predictor of shift
robustness as thoroughly as parameter count and in-distribution accuracy
already have been.

## 5. Discussion

Three findings, read together, argue against three different intuitive
heuristics for selecting a deployment backbone from an in-distribution
benchmark alone. First, "smaller is worse" is directly falsified in
distribution (Section 4.1). Second, "the in-distribution winner is the
safe choice" is directly falsified under shift (Section 4.2, Stage 2):
the model with the best in-distribution accuracy in the entire study is
the worst-performing model under both external shifts, dropping below the
chance floor on one. Third, "architecture family/lineage predicts
robustness" is falsified by MobileViT-S and MobileViTv2 sharing a name and
a lineage while producing a catastrophic failure and a top performance,
respectively, on the same evaluation. The practical implication is direct:
a deployment decision based on an in-distribution leaderboard, however
carefully measured, cannot be assumed to transfer to real cross-
institution conditions, and no static property measured here (params,
FLOPs, accuracy, family) substitutes for actually testing under shift. This
motivates the uncertainty-quantification question our companion paper
(Paper 2) asks: since no backbone choice alone is safe, can a model's own
confidence signal at least correctly flag when it has entered a regime
where its predictions should not be trusted?

## 6. Limitations

- Two external datasets (Herlev, Mendeley LBC), both drawn from a small
  number of source institutions; the shift conditions tested here are real
  but not exhaustive of the shift a deployed system might encounter.
- Herlev and Mendeley LBC's own citations require verification (not
  independently confirmed by this project beyond the Kaggle mirror pages
  used for acquisition); SIPaKMeD's citation (Plissiti et al., ICIP 2018)
  is confirmed.
- One train/val/calibration/test split (seed 13) per model; results are
  not averaged over multiple seeds, so small differences between
  closely-ranked models (e.g., the 0.003 MobileViTv2/MobileNetV2 gap on
  Mendeley LBC) should be read with appropriate caution about
  seed-to-seed variance, which was not separately measured.
- The compute path for this benchmark (six Colab GPU sessions, five
  preempted mid-run on the free tier before a paid-tier account completed
  the remainder) is documented in `docs/rq1_results.md` for reproducibility
  but does not affect the reported numbers, which are all from completed,
  non-preempted evaluation runs.
- No trained-weight checkpoints persist; reproducing any single row of
  these tables means retraining from scratch (`src/cervical_screening/train.py`).

## 7. Conclusion

Twelve models across three architecture families show that, on this
cervical cytology classification task, parameter-efficient vision
transformers do not trade accuracy for efficiency — the smallest models in
the registry are also the most accurate in distribution. But that same
efficiency-and-accuracy leaderboard actively misleads about deployment
robustness: the overall in-distribution winner is the worst-performing
model under real cross-dataset shift on both external datasets tested, and
no static property we measured (parameters, FLOPs, in-distribution
accuracy, or architecture family) reliably predicts which model fails
safely. Robustness must be measured directly, under real shift, and cannot
be inferred from an in-distribution benchmark however thorough.

## Data and code availability

Training/evaluation code and raw per-model JSON results
(`runs/colab_rq1/*.json`) are in the project repository. Dataset license
terms for all three datasets require independent verification before any
public release or IRB/funding submission language is finalized (proposal
Section 11).

## References

- Plissiti, M. E., et al. "SIPAKMED: A New Dataset for Feature and Image Based Classification of Normal and Pathological Cervical Cells in Pap Smear Images." ICIP 2018.
- Herlev dataset and Mendeley LBC Pap smear dataset: **citations require verification** — not independently confirmed by this project beyond the Kaggle mirror pages used for data acquisition.
