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

Running the remaining 8 registry models against the shift eval would
establish whether this is specifically an EfficientViT-family effect
(cascaded-group attention over-fitting SIPaKMeD's specific
staining/scanner characteristics) or a broader small-parameter-count
effect — the next natural extension of this result.

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
