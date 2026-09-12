# RQ3 results: is relevancy propagation actually more faithful than attention or Grad-CAM?

## Setup

Proposal S17/RQ3 compares three explainability methods for cervical cytology
classification: **relevancy propagation** (Chefer, Gur & Wolf, CVPR 2021 --
the primary method), **attention rollout** (Abnar & Zuidema, 2020 -- the
"raw attention" baseline), and **Grad-CAM adapted for ViT** (Selvaraju et
al., 2017, applied to a transformer block's output tokens -- the other
baseline). H3 asks whether relevancy propagation shows higher
deletion/insertion faithfulness and higher agreement with expert-marked
regions than the two baselines.

**Backbone: DeiT-Tiny, not EfficientViT-M0.** Relevancy propagation and
attention rollout are both defined for standard multi-head self-attention
transformer blocks. RQ1's winning backbone, EfficientViT-M0, uses a hybrid
conv + cascaded-group-attention design with no established propagation rule
for either method. Rather than force an ad hoc extension onto an
architecture these methods weren't designed for, RQ3 uses DeiT-Tiny --
proposal S13's "efficient ViT" baseline, a standard timm `VisionTransformer`,
0.9606 balanced accuracy in RQ1 (vs. EfficientViT-M0's 0.9654) -- retrained
fresh to 0.9x+ validation accuracy (best val loss 0.0804 after 29 epochs,
early-stopped) for this comparison.

**Metrics:**
- **Deletion AUC** (Petsiuk et al., RISE, BMVC 2018): progressively replace the most-relevant 16x16 patches (in the method's own ranking) with a heavily Gaussian-blurred version of the image, track the predicted probability of the model's own predicted class. A faithful explanation causes a fast drop, so **lower is better**.
- **Insertion AUC**: the reverse -- start fully blurred, reveal the most-relevant patches first. A faithful explanation causes a fast rise, so **higher is better**.
- **Nucleus concentration ratio**: SIPaKMeD ships real expert-marked nucleus boundary polygons per cell (not a proxy). This is the fraction of the heatmap's total relevance mass falling inside the nucleus region, divided by the nucleus's share of the image area. **1.0 is exactly chance level; higher means the heatmap concentrates more than chance in the nucleus** -- clinically the right place to concentrate, since nuclear morphology (enlargement, hyperchromasia, irregular contour) is the primary diagnostic feature in Bethesda-system cytology.

All three metrics were computed on the same 50 SIPaKMeD test-set images (10
per class, stratified, fixed seed), using each method's heatmap for the
model's own predicted class, all matched to their real nucleus polygon
(50/50 matched).

## Results

| Method | Deletion AUC (lower better) | Insertion AUC (higher better) | Nucleus concentration (higher better, 1.0 = chance) |
|---|---|---|---|
| **Relevancy propagation** | **0.734** | **0.970** | **2.714** |
| Grad-CAM (ViT-adapted) | 0.818 | 0.877 | 1.472 |
| Attention rollout | 0.825 | 0.936 | 1.177 |

## Findings

**H3 supported, cleanly, on all three metrics.** Relevancy propagation wins
every comparison: it causes the fastest probability collapse under deletion
(0.734 vs. 0.818-0.825), the fastest probability rise under insertion (0.970
vs. 0.877-0.936), and by far the strongest concentration of relevance inside
the actual nucleus boundary (2.71x chance level, vs. attention rollout's
1.18x -- barely above chance -- and Grad-CAM's 1.47x). Unlike RQ2's more
qualified findings (where the best method changed depending on which shift
condition was being tested), this result is consistent across all three
independent faithfulness/localization measures.

**This makes mechanistic sense, not just measurement luck.** Attention
rollout is class-agnostic -- it never looks at which class the model
predicted, only at where attention flows structurally. Deletion/insertion
faithfulness is inherently a class-discriminative test (does removing a
patch hurt this specific prediction), so a class-agnostic method has an
built-in disadvantage there; its near-chance nucleus concentration (1.18x)
suggests DeiT-Tiny's raw attention, in the aggregate rollout sense, spreads
fairly evenly across the cell rather than committing to any particular
structure. Grad-CAM is class-discriminative but only reflects one block's
channel-gradient signal at fairly coarse spatial resolution (necessarily
the second-to-last block here, not the last -- see the implementation note
below); relevancy propagation is also class-discriminative but integrates
the gradient-weighted attention signal across *all twelve* transformer
blocks via the same rollout math, which plausibly explains why it captures
both diagnostically relevant (nucleus-concentrated) and behaviorally
faithful (deletion/insertion-sensitive) structure better than either
baseline alone.

## Status against the RQ3 hypothesis (proposal S18)

- **H3** (relevancy propagation more faithful and more expert-region-agreeing than the baselines): **supported**, on deletion AUC, insertion AUC, and nucleus concentration, without qualification.

## Implementation notes and a real bug caught before it reached results

- **Grad-CAM's target layer had to be the second-to-last transformer block, not the last.** DeiT-Tiny pools only the CLS token for classification (`global_pool="token"`). Verified directly (not assumed): every patch token's gradient at the *last* block's output was exactly zero, because nothing downstream of that block mixes information back into patch positions -- the final LayerNorm and head only read the CLS token. The second-to-last block's patch tokens still receive real, non-constant gradient (the final block's attention mixes them into CLS), so that's the target layer used. This is the standard fix for CLS-pooled ViTs in the Grad-CAM-for-transformers literature, caught here via a local CPU smoke test (`smoke_test_rq3.py`) before spending any Colab GPU time -- the bug produced a silently all-zero heatmap rather than an error, which a smoke test asserting non-degenerate output range caught immediately.
- **A device-mismatch bug reached the real GPU run and had to be fixed mid-session**: the nucleus mask was rasterized on CPU (via PIL) but heatmaps live on the GPU once `model`/`image` are CUDA tensors, which the local CPU-only smoke test couldn't have caught (both tensors were CPU there). Fixed by moving the mask to the heatmap's device before the elementwise multiply; the eval re-ran cleanly afterward with no need to retrain.
- All three explanation methods and the faithfulness/nucleus-overlap metrics live in `src/cervical_screening/explainability/`; `colab_kernel/rq3/` holds the two-script Colab pipeline (`setup.py` trains DeiT-Tiny and builds the nucleus-polygon manifest, `run_explainability_eval.py` runs the comparison).
- Nucleus polygons come from SIPaKMeD's own per-cell `.dat` boundary annotations (already used for cropping), now also persisted in crop-local pixel coordinates via `crop_sipakmed.py --nucleus-manifest` -- real expert annotations, not a derived proxy.
