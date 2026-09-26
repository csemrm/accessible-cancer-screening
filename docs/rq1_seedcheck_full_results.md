# RQ1: full-registry seed-variance check (all 12 models, 3 seeds)

`papers/paper1.tex`'s Threats to Validity section originally measured
seed-to-seed variance for only the two closest-ranked pairs flagged by the
Stage 4/5 tables (MobileViTv2 vs. MobileNetV2 on Mendeley LBC;
EfficientViT-B0 vs. DeiT-Tiny on APCEM), at 2 extra seeds each
(`runs/colab_rq1_seedcheck/`, `runs/colab_rq1_apcem_seedcheck/`). This
document extends that check to all 12 models in the registry, retrained at
seeds 17 and 29 and evaluated zero-shot on all three external shift
datasets (Herlev, Mendeley LBC, APCEM) from the same checkpoint — one
training run per model per seed rather than three
(`colab_kernel/rq1_seedcheck/run_one_full_seed.py`). Combined with the
original seed-13 numbers already reported in `docs/rq1_results.md` and
`docs/rq1_apcem_results.md`, every model now has 3 independent seeds on
every dataset. Raw per-model-per-seed results:
`runs/colab_rq1_seedcheck_full/*.json`.

Architecture families, as used throughout paper 1: **CNN** = ResNet-50,
MobileNetV2, EfficientNet-Lite0; **Transformer** (plain ViT + efficient-ViT
hybrids) = the other 9 models.

## Finding 1: the "near-clean CNN-vs-transformer split" on Mendeley LBC is itself seed-fragile

At seed 13, Mendeley LBC looked like a near-clean split: all 3 CNNs landed
at 0.539–0.667, and only 1 of 9 transformer models (MobileViTv2, 0.5418)
exceeded the weakest CNN (MobileNetV2, 0.5391) — by 0.003, already flagged
in the paper as noise. Extending this check to all 12 models at 2 more
seeds shows the split itself, not just that one close call, does not
reproduce:

| Seed | Weakest CNN | Transformer models exceeding it | Largest margin above the CNN floor |
|---|---|---|---|
| 13 | MobileNetV2 (0.539) | 1 / 9 | +0.003 (MobileViTv2) |
| 17 | ResNet-50 (0.499) | 3 / 9 | +0.098 (EfficientViT-B0) |
| 29 | MobileNetV2 (0.494) | 4 / 9 | +0.113 (EfficientViT-B0) |

Under seeds 17 and 29, a third to almost half of the transformer-family
models score above the weakest CNN, by margins an order of magnitude
larger than the single 0.003 gap the paper already flags. The "near-clean
split" description of Mendeley LBC at seed 13 (`papers/paper1.tex`,
Section 4.2 Stage 4) is a property of that specific seed, not a
reproducible property of the dataset or the model families.

## Finding 2: Herlev's inverse pattern (transformers beating CNNs) is seed-stable

By contrast, Herlev's pattern — most transformer models outscoring the
CNNs — replicates cleanly: 8/9, 7/9, and 8/9 transformer models exceed the
weakest CNN at seeds 13, 17, and 29 respectively. Unlike the Mendeley
split, this is not an artifact of one seed.

## Finding 3: MobileNetV2's APCEM win is seed-stable; EfficientViT-M0's poster-child reversal is not

MobileNetV2 is the best-performing model on APCEM at every seed tested
(rank 1/12, 2/12, 1/12 at seeds 13, 17, 29) — a genuinely reproducible
result, not a single-seed artifact.

EfficientViT-M0 is the paper's running example of "the in-distribution
winner is the shift-robustness loser" (Section 4.1, Stage 2–3;
Section 6/Conclusion). That specific pairing does not hold up across
seeds:

| Seed | In-distribution rank (/12) | Herlev shift rank (/12) | Mendeley LBC shift rank (/12) |
|---|---|---|---|
| 13 | 2 | 10 | 10 |
| 17 | 10 | 7 | 7 |
| 29 | 1 | 5 | 4 |

At seed 13, EfficientViT-M0 is (virtually tied for) the in-distribution
winner and one of the worst shift performers on both external datasets —
exactly the reversal the paper illustrates. At seed 29, it is the outright
in-distribution winner *and* comfortably mid-pack (rank 4–5/12) under
shift on both datasets — no reversal at all. At seed 17 it isn't even the
in-distribution winner (rank 10/12). The specific model used to illustrate
"in-distribution winner becomes shift loser" is a seed-13-specific
example, not a reproducible property of EfficientViT-M0 itself.

## Implication for the paper

This does not weaken the paper's central, higher-level claim — if
anything it reinforces it. The core thesis is that no static model
property (in-distribution accuracy, parameter count, architecture family)
reliably predicts shift robustness. A seed-fragile poster-child example
and a seed-fragile "near-clean family split" are both further instances of
exactly that unreliability, not counterexamples to it. What this check
changes is which *specific* claims should be presented as illustrative
(true of one seed, useful for intuition) versus load-bearing (true across
retraining, safe to state as a general finding):

- **Safe to state generally:** no static property predicts shift
  robustness (now doubly true — not just across models, but across seeds
  of the same model); MobileNetV2's strong APCEM performance; the
  Herlev transformers-beat-CNNs pattern; the CNN family's own internal
  split on APCEM (MobileNetV2 best, ResNet-50 near-worst, both seed-13
  facts already independently reconfirmed as reproducible).
- **Should be presented as one-seed illustrations, not general
  findings:** the Mendeley LBC "near-clean CNN split" once EfficientViT-B0
  and other transformer models are considered beyond the single flagged
  pair; the EfficientViT-M0 in-distribution-winner-becomes-shift-loser
  narrative specifically (the *general* point it illustrates — that
  in-distribution accuracy doesn't predict shift robustness — remains
  supported by the aggregate 12-model tables regardless of which single
  model best dramatizes it).
