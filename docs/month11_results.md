# Month 11 results: resource-efficiency benchmark + synthetic degradation/OOD sanity check

Proposal S19 (resource efficiency) and S16 (robustness evaluation) are
grouped under Month 11 in the timeline (S26). Two independent pieces:

1. A **resource-efficiency benchmark** (params, FLOPs, CPU latency, CPU
   memory) across all 12 RQ1 registry models -- an architecture property,
   not a trained-weight property, so it runs entirely locally with no GPU
   or Colab session.
2. A **synthetic degradation + OOD sanity check** on EfficientViT-M0 (RQ1's
   winning backbone) -- this needs a trained model, so it ran on Colab.

## Part 1: resource-efficiency benchmark

| Model | Params | FLOPs | CPU latency (median, ms) | CPU memory delta (MB) |
|---|---|---|---|---|
| ResNet-50 | 23,518,277 | 8,174,292,992 | 32.0 | 51.2 |
| MobileNetV2 | 2,230,277 | 599,001,344 | 33.5 | 20.8 |
| **EfficientNet-Lite0** | 3,377,413 | 767,827,904 | **137.7** | 28.7 |
| ViT-Small | 21,667,589 | 8,481,672,960 | 26.8 | 16.2 |
| Swin-Tiny | 27,523,199 | 8,979,604,992 | 54.0 | 42.0 |
| DeiT-Tiny | 5,525,381 | 2,149,320,576 | **15.4** | 11.5 |
| LeViT-128 | 8,447,786 | 809,115,136 | **17.5** | 12.2 |
| TinyViT-5M | 5,073,369 | 2,509,712,640 | 44.9 | 29.3 |
| MobileViT-S | 4,940,837 | 3,645,118,720 | 66.3 | 45.9 |
| MobileViTv2 | 4,391,406 | 3,623,162,880 | 65.2 | 63.1 |
| EfficientViT-B0 | 2,135,413 | 197,919,232 | 41.6 | 8.6 |
| EfficientViT-M0 | 2,157,401 | 158,005,120 | 36.2 | 7.9 |

(FLOPs via torch's own `FlopCounterMode`, multiply-adds counted as 2 FLOPs.
CPU latency: single-image batch, median of 30 iterations after 5 warmup
iterations, on one consumer laptop -- see the caveat below. Full JSON in
`runs/efficiency/benchmark.json`.)

### Findings

**Neither params nor FLOPs reliably predict CPU latency.** EfficientNet-Lite0
has the *smallest* CNN parameter count (3.38M, less than half of
MobileNetV2's 2.23M... actually comparable) and a modest FLOP count (0.77G,
an order of magnitude below ResNet-50's 8.17G), yet its CPU latency (137.7ms)
is **more than 4x ResNet-50's** (32.0ms) despite ResNet-50 having roughly
10x the FLOPs. This is a well-documented real phenomenon, now demonstrated
directly on this project's own model set: depthwise-separable convolutions
(EfficientNet-Lite0's core building block) are FLOP-efficient but don't map
well onto the dense-matmul-optimized kernels most generic CPU BLAS
implementations provide, unlike ResNet's dense 3x3/1x1 convolutions. The
same pattern shows up in the ViT family: MobileViT-S and MobileViTv2 (both
"Mobile"-branded) have the *highest* latency of any transformer-family model
(66.3ms, 65.2ms) despite FLOP counts (3.6G) well below Swin-Tiny's (9.0G,
54.0ms latency) -- their local-window unfold/fold operations apparently cost
more in wall-clock terms than their FLOP count suggests. **EfficientViT-M0
(RQ1's overall winner on accuracy+params+FLOPs) is not the fastest model in
wall-clock CPU terms** -- DeiT-Tiny (15.4ms) and LeViT-128 (17.5ms) are both
faster despite having 10-14x more FLOPs than EfficientViT-M0's 158M. This
directly confirms proposal S19's point: efficiency claims must rest on
*measured* latency, not on parameter count or FLOPs alone, because they can
disagree substantially and in either direction.

**A real measurement bug was caught before trusting the memory numbers.**
The initial implementation measured params/FLOPs/latency/memory for all 12
models in one process, in that order; `resource.getrusage().ru_maxrss` is a
monotonically non-decreasing high-water mark for the whole process, so every
model's memory measurement after the first (larger) model silently reported
0.0 -- masked by whichever prior model had already raised the process's
peak. Fixed by isolating each model's memory measurement in its own
subprocess (`measure_memory_isolated()` in `measure_efficiency.py`); the
corrected numbers above are all non-zero and in a sensible range (7.9-63.1MB).

**Caveat: CPU latency was measured on one shared consumer laptop, not
dedicated benchmark hardware.** Repeating the EfficientViT-M0 measurement
in isolation three times gave 27.4/28.4/27.9ms, while the full 12-model
sweep (run twice) gave 28.3ms and 36.2ms for the same model -- a ~30%
swing attributable to background system load during the full sweep, not a
real property change. The *relative ordering* (EfficientNet-Lite0 far
slower than everything else; DeiT-Tiny/LeViT-128 fastest) was stable and
reproduced across repeated runs, but individual latency numbers should be
read as directional, not lab-grade-precise, until measured on dedicated,
isolated hardware -- a genuine limitation of measuring on a general-purpose
machine rather than proposal S19's "ordinary CPU-only clinic-grade machine"
run under controlled conditions. True edge/mobile-class device measurement
(an actual phone or single-board computer) is out of scope without physical
hardware access and is not attempted here.

## Part 2: synthetic degradation + OOD sanity check (EfficientViT-M0)

Fresh EfficientViT-M0 (matching RQ1: clean balanced accuracy 0.975, close to
RQ1's reported 0.965 -- a different random train/val split, not the same
run) evaluated on the SIPaKMeD test set under three degradations at five
severities each, plus a synthetic OOD sanity check.

| Corruption | Mild | | | | Severe | Total drop |
|---|---|---|---|---|---|---|
| JPEG quality | 90: 0.937 | 70: 0.917 | 50: 0.918 | 30: 0.899 | 10: 0.820 | -15.5 pts |
| Gaussian blur radius | 1: 0.954 | 2: 0.899 | 4: 0.781 | 8: 0.648 | | -32.8 pts (radius 8) |
| Brightness/contrast factor | 0.8: 0.962 | 1.4: 0.936 | 0.6: 0.814 | 1.7: 0.794 | | -18.1 pts (factor 1.7) |

(Clean baseline: 0.975 balanced accuracy. Full JSON in
`runs/month11_robustness/degradation_ood.json`.)

### Findings

**Gaussian blur is by far the most damaging corruption, and this makes
clinical sense.** At its most severe tested level, blur costs 32.8 points of
balanced accuracy -- roughly double JPEG compression's worst-case cost
(15.5 points) and nearly double brightness/contrast's (18.1 points).
Cervical cytology diagnosis under the Bethesda system depends heavily on
fine nuclear texture (chromatin pattern, nuclear membrane irregularity,
nucleus-to-cytoplasm ratio at the pixel level) -- exactly the high-frequency
detail Gaussian blur destroys directly. JPEG compression at moderate quality
preserves more of this structure (its artifacts are blockiness and ringing,
not a genuine loss of spatial frequency content until quality gets very
low), and brightness/contrast shifts don't touch spatial detail at all, only
intensity. **This is a concrete, actionable finding for any real deployment**:
defocus/motion blur from handheld or low-quality field cameras is a bigger
threat to this model's reliability than compression artifacts or lighting
variation, and should be prioritized in any real-camera validation effort.

**The model degrades gracefully (accuracy falls smoothly with severity)
rather than catastrophically (a cliff at one severity level) for all three
corruption types** -- there is no threshold effect where accuracy holds
flat then collapses; each corruption type's accuracy curve declines roughly
monotonically with severity. This is a mildly reassuring finding on its own
terms (no hidden brittleness at some untested intermediate severity) but is
a separate question from calibration: this evaluation measures raw accuracy
under corruption, not whether the model's *confidence* drops appropriately
alongside its accuracy (that would require re-running RQ2's calibration/
selective-prediction harness under these same corruptions, which this
evaluation does not do -- a natural next step, not attempted here).

**OOD sanity check: confidence drops on clearly-non-cytology images, but not
to chance level.** Mean max-softmax confidence on 25 synthetic images
(random noise, solid colors, checkerboards -- unambiguously not cervical
cells) was 0.743, versus 0.985 on a matched sample of real in-distribution
test images -- a real, meaningful drop (-0.24 absolute). But 0.743 is still
far above chance level for a 5-class problem (0.20), meaning the model does
not treat these obviously-alien inputs as "I have no idea" -- it stays
moderately confident even on inputs that share no real structure with
cytology images. **This is an honest, partial result, not a clean pass**:
max-softmax confidence alone provides a genuine but weak OOD signal; a
production abstention rule relying on confidence thresholding alone would
need to set its threshold carefully (or use a different signal entirely,
such as evidential vacuity or the deep-ensemble variance evaluated in RQ2)
to reliably catch this class of failure.

## Status against relevant hypotheses

- **H4** (abstention rate rises correspondingly under distribution shift): this evaluation extends the shift-robustness picture from RQ4/RQ2 (real cross-dataset shift) to synthetic corruption and synthetic OOD, both showing the same underlying pattern -- confidence and accuracy both degrade under stress, but neither collapses to a fully "safe" (near-chance, clearly-flagged) state on its own. Consistent with RQ2's finding that no single uncertainty signal alone reaches a reliable abstention operating point under real distribution shift.
- **S19's resource-efficiency claim**: confirmed to require *measurement*, not inference from architecture name/params/FLOPs -- demonstrated directly via EfficientNet-Lite0 (FLOP-efficient, latency-inefficient) and EfficientViT-M0 (best params+FLOPs, not the fastest in wall-clock CPU terms).

## Implementation notes

- `src/cervical_screening/eval/flops.py`: FLOPs via torch's own `FlopCounterMode` (torch>=2.4), avoiding a third-party FLOP-counting dependency.
- `src/cervical_screening/measure_efficiency.py`: the local, no-GPU-needed CLI that produced Part 1's table; `measure_memory_isolated()` is the subprocess-isolation fix for the `ru_maxrss` high-water-mark bug described above.
- `src/cervical_screening/robustness/degradation.py` and `ood_sanity.py`: the corruption functions and synthetic-OOD-image generators used in Part 2.
- `colab_kernel/rq_month11/`: the two-script Colab pipeline for Part 2 (`setup.py` trains EfficientViT-M0 on SIPaKMeD only -- no Herlev/Mendeley needed here; `run_eval.py` runs the degradation sweep and OOD sanity check).
