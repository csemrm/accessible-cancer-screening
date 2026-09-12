# RQ1 baseline benchmark + first cross-dataset shift result

All 12 models in `src/cervical_screening/models/registry.py` trained on
SIPaKMeD (4,049 native-resolution crops; 60/15/10/15 train/val/calibration/test
split, seed 13), full 30-epoch budget with early stopping (patience 7), on a
Colab T4 GPU. Raw per-model results: `runs/colab_rq1/*.json`.

## Efficiency/accuracy benchmark (proposal RQ1/H1)

| Model | Balanced accuracy | Macro F1 | AUROC | Params | CPU/GPU latency |
|---|---|---|---|---|---|
| **efficientvit_m0** | **0.9654** | **0.9656** | 0.9985 | **2,157,401** | 17.56ms |
| efficientvit_b0 | 0.9623 | 0.9624 | 0.9983 | 2,135,413 | 6.97ms |
| deit_tiny_patch16_224 | 0.9606 | 0.9607 | 0.9974 | 5,525,381 | 5.17ms |
| swin_tiny_patch4_window7_224 | 0.9605 | 0.9606 | 0.9965 | 27,523,199 | 11.36ms |
| tiny_vit_5m_224 | 0.9587 | 0.9589 | 0.9979 | 5,073,369 | 9.57ms |
| mobilevit_s | 0.9570 | 0.9575 | 0.9979 | 4,940,837 | 9.32ms |
| resnet50 | 0.9542 | 0.9540 | 0.9966 | 23,518,277 | 5.95ms |
| levit_128 | 0.9458 | 0.9456 | 0.9946 | 8,447,786 | 12.81ms |
| mobilevitv2_100 | 0.9387 | 0.9401 | 0.9966 | 4,391,406 | 9.90ms |
| efficientnet_lite0 | 0.9375 | 0.9376 | 0.9944 | 3,377,413 | 5.51ms |
| vit_small_patch16_224 | 0.9374 | 0.9380 | 0.9960 | 21,667,589 | 5.39ms |
| mobilenetv2_100 | 0.9357 | 0.9365 | 0.9952 | 2,230,277 | 7.01ms |

**Headline finding (H1 supported):** the two smallest models in the entire
registry (EfficientViT-M0 and -B0, ~2.1M params each) are also the two
*best-performing*, beating every full-size baseline including ResNet50
(23.5M params) and Swin-Tiny (27.5M params) by 1-3 points of balanced
accuracy. "Lightweight" did not cost accuracy here — it's not yet clear
whether that holds beyond this one dataset/split, but on this benchmark the
efficiency-accuracy trade-off the proposal is built around is not a
trade-off at all.

LeViT-128 and EfficientViT-B0's training runs showed visibly noisier
validation loss than the others (see raw epoch logs) — worth a closer look
if LeViT is carried forward, since it may indicate learning-rate sensitivity
on a dataset this small rather than a fluke.

## First cross-dataset domain-shift result (proposal RQ4/H4)

DeiT-Tiny (a top-3 in-distribution model) evaluated zero-shot, with no
retraining or calibration, on Herlev and Mendeley LBC after harmonizing all
three datasets onto a shared normal/abnormal label (see
`docs/cross_dataset_harmonization.md`):

| Evaluation | Balanced accuracy | Macro F1 | AUROC |
|---|---|---|---|
| SIPaKMeD (in-distribution) | 0.9504 | 0.9510 | 0.9972 |
| Herlev (zero-shot) | 0.7151 | 0.7307 | 0.7582 |
| Mendeley LBC (zero-shot) | **0.5230** | 0.4801 | 0.5150 |

**This is the central "trustworthy AI" finding the cross-dataset design
exists to produce:** a model that looks excellent in-distribution
(95%+ balanced accuracy, 99.7% AUROC) degrades to **chance-level performance
on Mendeley LBC** (52.3% balanced accuracy on a binary task) under real
cross-institution domain shift — with nothing in the current pipeline able
to detect or flag this failure. This is exactly the scenario the proposal's
uncertainty/calibration/selective-prediction work (RQ2, S8/S18) is meant to
address: without it, this system would confidently misclassify roughly half
of Mendeley LBC's cases with no indication anything was wrong.

### Second model, and a complication: in-distribution accuracy does not predict shift robustness

Running the identical eval against EfficientViT-M0 — the overall best
in-distribution model (97.37% balanced accuracy on this run, AUROC 0.9985,
even higher than its original 96.54% result) — did not just confirm the
collapse, it made the finding more interesting:

| Evaluation | DeiT-Tiny | EfficientViT-M0 |
|---|---|---|
| SIPaKMeD (in-distribution) | 0.9504 bal. acc | **0.9737** bal. acc |
| Herlev (zero-shot) | 0.7151 bal. acc | **0.6716** bal. acc |
| Mendeley LBC (zero-shot) | 0.5230 bal. acc | **0.4731** bal. acc |

EfficientViT-M0 is the *better* in-distribution model but the *worse*
model under domain shift on both shift datasets — on Mendeley LBC its
balanced accuracy (0.4731) is actually **below** the 0.50 chance floor for
this binary task. Two models this close in architecture family and
parameter count producing this reversal is a real, specific caution against
using in-distribution accuracy as a proxy for deployment robustness: the
smaller/more-accurate-in-distribution model here may simply be fitting
SIPaKMeD's particular staining/scanner characteristics more tightly, at the
cost of generalization — exactly the failure mode "resource-efficient"
should not be assumed to avoid for free.

### Third and fourth models: the pattern is EfficientViT-M0 specifically, not ViT-vs-CNN

Adding ResNet50 (the CNN baseline) and Swin-Tiny (the full-size ViT) completes
one model from each of the three architecture families in the RQ1 registry:

| Model (family) | SIPaKMeD (in-dist.) | Herlev (zero-shot) | Mendeley LBC (zero-shot) |
|---|---|---|---|
| EfficientViT-M0 (efficient ViT) | **0.9737** | 0.6716 | **0.4731** (below chance) |
| Swin-Tiny (full ViT) | 0.9590 | **0.7282** | 0.5192 |
| DeiT-Tiny (efficient ViT) | 0.9504 | 0.7151 | 0.5230 |
| ResNet50 (CNN) | 0.9556 | 0.6838 | **0.6082** |

This refines, rather than just confirms, the earlier two-model finding. It
is not a clean CNN-vs-transformer split — on Herlev, the two transformer
models (Swin-Tiny, DeiT-Tiny) actually beat ResNet50. What holds
consistently across both shift datasets is narrower and sharper:
**EfficientViT-M0, the single most parameter-efficient model in the entire
registry and the best in-distribution performer, is the worst performer
under domain shift on both external datasets** — and the only one to drop
below the chance floor. ResNet50, the CNN baseline, is uniquely robust on
Mendeley LBC specifically (0.6082, clearly above chance), despite not being
the best in-distribution model on any of the four.

Read together with the RQ1 efficiency table, this is the caution the
proposal's "trustworthy AI" framing exists to surface: the same
2.16M-parameter model that makes the strongest efficiency argument in
Section 1 makes the weakest robustness argument in Section 2. Neither
table alone tells the deployment-relevant story — a system that treated
EfficientViT-M0's in-distribution numbers as sufficient evidence of
readiness would have picked the model that fails worst in the field.

### Fifth through twelfth models: the full registry, a near-clean CNN-vs-transformer split, and an extreme outlier

All 12 registry models are now tested (ViT-Small's Mendeley LBC number
required a second attempt after a session death lost it mid-fetch; its
final retrained numbers are included below and are consistent with its
first, partially-lost run — Herlev 0.7049 vs. the earlier 0.7194, well
within normal retraining variance):

| Model (family, params) | SIPaKMeD (in-dist.) | Herlev (zero-shot) | Mendeley LBC (zero-shot) |
|---|---|---|---|
| EfficientNet-Lite0 (CNN, 3.38M) | 0.9455 | 0.5970 | **0.6674** |
| ResNet50 (CNN, 23.5M) | 0.9556 | 0.6838 | **0.6082** |
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

**A near-clean CNN-vs-transformer split emerges on Mendeley LBC — with one
honest exception right at the boundary.** All three CNNs (EfficientNet-Lite0,
ResNet50, MobileNetV2) land at 0.539–0.667. Eight of the nine ViT-family
models land at 0.293–0.523, below every CNN. The ninth, MobileViTv2, scores
0.5418 — narrowly *above* the lowest CNN (MobileNetV2, 0.5391) by 0.003.
That is a real overlap, not zero, and it should be stated as such rather
than rounded away: the pattern is "CNNs cluster high; 8 of 9 ViT-family
models cluster low-to-catastrophic; one ViT-family model sits right at the
boundary" — a strong tendency with one genuine exception, not a perfect
partition. This did not hold on Herlev at all, where transformer models
(MobileViTv2, TinyViT-5M) lead the entire field and the weakest performer
(LeViT-128) is itself a transformer.

**MobileViT-S is a severe, singular outlier — and it is the specific
architecture family this proposal's own design (Section 9) is built
around.** Its Mendeley LBC AUROC (0.215) is not just below the 0.50 chance
floor every other below-chance model sits just under — it is far below it,
meaning its predictions are *anti-correlated* with ground truth on this
dataset, not merely uninformative. Its own successor, MobileViTv2, shows no
such problem (0.5418, plus the best Herlev score in the whole registry,
0.7501) — the same "MobileViT" family name spans a catastrophic failure and
a strong performer, ruling out architecture lineage/naming as a predictor
of shift robustness as thoroughly as parameter count has already been
ruled out (and as thoroughly as CNN-vs-transformer status has been, aside
from the one boundary exception noted above).

**LeViT-128 and TinyViT-5M** both drop below chance on Mendeley LBC
(0.4764, 0.4543) while simultaneously ranking among the *best* models on
Herlev (0.5777 is the field's worst on Herlev for LeViT, while TinyViT's
0.7327 is second-best) — reinforcing that "robust" and "fragile" are
properties of a specific model-dataset pair, not portable properties of a
model. LeViT's result is consistent with the training instability
(val-loss spikes) already flagged for it in the original RQ1 benchmark.

## Compute path and its problems (for anyone reproducing this)

Getting this benchmark to run end-to-end took six Colab GPU sessions across
free-tier and early paid-tier attempts:

1. Kaggle Kernels were tried first — blocked by exhausted weekly GPU quota
   and a P100/torch compute-capability mismatch (see
   `docs/dataset_provenance.md` and `src/cervical_screening/device.py`).
2. [googlecolab/google-colab-cli](https://github.com/googlecolab/google-colab-cli)
   was used instead — see `docs/colab_cli_pipeline.md`. Five free-tier
   sessions were preempted mid-run (lifetimes ranging 3-65 minutes, no
   discernible pattern), losing all in-progress training each time; only
   completed per-model results were recoverable, since a session's
   `/content/` filesystem — checkpoints included — dies with it.
3. The sixth session, run under an account with an existing Pay-As-You-Go
   compute-unit balance (245 units) rather than the pure free tier, ran
   the full download-crop-train-eval chain plus three more models without
   preemption. Whether that was the compute-unit balance genuinely granting
   priority access, or coincidental better free-tier availability, wasn't
   directly verified — but it's the working configuration if this needs
   reproducing.

Given this, any future run on this pipeline should budget for possible
mid-run preemption: keep each `colab exec` call scoped to one model (or a
handful of fast ones) rather than one long loop, and treat every checkpoint
as needing to be pulled off the VM (`colab download`) before the session
might die, not after the whole benchmark finishes.

The follow-on shift-eval sessions (EfficientViT-M0; ResNet50+Swin-Tiny;
EfficientViT-B0+MobileNetV2; LeViT-128 through EfficientNet-Lite0 in one
session; ViT-Small, twice, after the first attempt's session died mid-fetch
of its Mendeley LBC result) mostly ran to completion under the same
paid-tier account, each downloading and cropping SIPaKMeD fresh (~1-5 min)
rather than relying on any cross-session persistence, since no session's
`/content/` survives it ending. One retry additionally hit a backend
"too many concurrent assignments" error from two stale orphaned sessions
that neither `colab stop` nor waiting ~15 minutes cleared — resolved only
by continuing to retry every 30s until the backend released the slot on
its own, well over an hour later. Checkpoints still are not saved anywhere
outside the ephemeral VM — retraining is currently the only way to re-run
a shift eval on a given model.
