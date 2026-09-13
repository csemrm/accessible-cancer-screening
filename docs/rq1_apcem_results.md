# RQ1: third external shift dataset (APCEM-Enhanced Cervical Cytology)

All 12 models in `src/cervical_screening/models/registry.py`, each trained
fresh on SIPaKMeD (4,049 crops, seed 13, identical protocol to
`docs/rq1_results.md`) and evaluated zero-shot, with no retraining or
calibration, on the APCEM-Enhanced Cervical Cytology Pap Smear Image
Dataset (4,253 images; 6-class Bethesda taxonomy harmonized to binary
normal/abnormal via `APCEM_TO_BINARY` in
`src/cervical_screening/data/harmonize_labels.py` — NILM normal, all of
ASCUS/ASCUH/LSIL/HSIL/SSC abnormal). Raw per-model results:
`runs/colab_rq1_apcem/*.json`.

This is APCEM's first use in the project as a *third, independent*
domain-shift test, alongside the existing Herlev and Mendeley LBC results
in `docs/rq1_results.md`. It differs from those two in acquisition
pipeline, institution, and label granularity, so it stresses the
in-distribution-accuracy-does-not-predict-shift-robustness finding under a
meaningfully different shift than either prior dataset.

## Results, sorted by APCEM AUROC

| Model | SIPaKMeD (in-dist.) bal. acc | APCEM bal. acc | APCEM AUROC | Params |
|---|---|---|---|---|
| **mobilenetv2_100** | 0.9623 | **0.6158** | **0.6528** | 2,230,277 |
| efficientnet_lite0 | 0.9309 | 0.5393 | 0.5701 | 3,377,413 |
| levit_128 | 0.9001 | 0.5263 | 0.5376 | 8,447,786 |
| swin_tiny_patch4_window7_224 | 0.9330 | 0.4724 | 0.3082 | 27,523,199 |
| efficientvit_b0 | 0.9638 | 0.4481 | 0.3206 | 2,135,413 |
| deit_tiny_patch16_224 | 0.9589 | 0.4469 | 0.3799 | 5,525,381 |
| mobilevitv2_100 | 0.9570 | 0.4290 | 0.4120 | 4,391,406 |
| tiny_vit_5m_224 | 0.9670 | 0.4100 | 0.3681 | 5,073,369 |
| efficientvit_m0 | 0.9621 | 0.3730 | 0.3407 | 2,157,401 |
| mobilevit_s | 0.9687 | 0.3621 | 0.3161 | 4,940,837 |
| vit_small_patch16_224 | 0.9538 | 0.3444 | 0.2555 | 21,667,589 |
| resnet50 | 0.9639 | 0.3339 | 0.2929 | 23,518,277 |

Chance-level balanced accuracy on this binary task is 0.50; chance-level
AUROC is 0.50.

## Findings

**Every model degrades severely; 11 of 12 fall to or below chance.**
In-distribution balanced accuracy sits in a narrow 0.90-0.97 band across
all 12 architectures — the models are near-indistinguishable on SIPaKMeD.
Under APCEM shift that narrow band explodes into a 0.33-0.62 spread, and
all but one model land at or under the 0.50 chance floor for balanced
accuracy, several (Swin-Tiny, EfficientViT-B0, ViT-Small, ResNet-50) with
AUROC *below* 0.35 — meaningfully worse than a coin flip, i.e.
systematically anti-correlated with the true label under this particular
shift.

**MobileNetV2 is the only model that generalizes, and it is not the
in-distribution favorite.** MobileNetV2-100 is mid-pack in-distribution
(0.9623, ranked 5th of 12) but the clear best performer on APCEM by a wide
margin (0.6158 balanced accuracy / 0.6528 AUROC — the only model with AUROC
meaningfully above chance). This directly extends the RQ1/RQ4 finding
that in-distribution accuracy does not predict domain-shift robustness:
here it is not just non-predictive, it is actively misleading, since three
of the four *worst* shift performers (EfficientViT-B0 at 0.9638, TinyViT-5M
at 0.9670, MobileViT-S at 0.9687, ResNet-50 at 0.9639) are among the
*best* in-distribution models.

**No architecture family is reliably robust.** The registry spans five
lineages — CNN (ResNet-50, MobileNetV2, EfficientNet-Lite0), plain ViT
(DeiT-Tiny, ViT-Small), windowed/hierarchical attention (Swin-Tiny),
mobile hybrid CNN-transformer (MobileViT-S, MobileViTv2-100), and
efficient-attention (EfficientViT-M0, EfficientViT-B0, LeViT-128,
TinyViT-5M) — and every one of them has at least one member in the bottom
half of the APCEM ranking. The CNN family is the most obvious case: it
contains both the best model on APCEM (MobileNetV2) and one of the worst
(ResNet-50, 2nd-to-last by AUROC), so "CNN vs. transformer" is not itself
an explanation. Family membership predicts almost nothing about shift
behavior on this dataset; if anything, the two smallest and cheapest
models tested (MobileNetV2 at 2.2M params, EfficientNet-Lite0 at 3.4M)
occupy the top two slots, while three of the four largest models
(Swin-Tiny 27.5M, ResNet-50 23.5M, ViT-Small 21.7M) are all in the bottom
half — weak evidence that smaller capacity may resist overfitting to
SIPaKMeD-specific acquisition artifacts, though this is a two-shift-dataset
pattern (also loosely visible in the Herlev/Mendeley results), not yet a
reliable rule, and it is contradicted by EfficientViT-B0 (2.1M params,
smallest model tested, 2nd-worst AUROC).

**Consistency with the two prior shift datasets.** DeiT-Tiny and
EfficientViT-M0 — the two models directly compared on Herlev and Mendeley
LBC in `docs/rq1_results.md` — again both collapse on APCEM (0.4469 and
0.3730 balanced accuracy respectively), and again EfficientViT-M0 is the
worse of the two under shift despite being the stronger in-distribution
model on this run (0.9621 vs. 0.9589) — the same reversal reported for
Herlev and Mendeley LBC, now replicated on a third, independently sourced
dataset.

## Implication for the paper

This closes the "future work" gap flagged in `papers/paper1.tex`'s
Threats-to-Validity section around relying on only two shift datasets: the
central claim — in-distribution accuracy does not predict domain-shift
robustness, and no tested efficiency mechanism reliably confers shift
robustness — now holds across three independently acquired external
datasets (Herlev, Mendeley LBC, APCEM), each with different acquisition
pipelines and label granularity, evaluated on all 12 models rather than
just the 2 originally spot-checked on Herlev/Mendeley.
