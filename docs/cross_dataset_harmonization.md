# Cross-dataset domain-shift evaluation design (proposal S16 / RQ4)

## Why binary, not fine-grained, cross-dataset transfer

SIPaKMeD (5 cytomorphology classes), Herlev (7 dysplasia-grade classes), and
Mendeley LBC (4 Bethesda-system classes) do not share a common fine-grained
taxonomy. Nothing in any of the three datasets' documentation establishes
that, say, SIPaKMeD's "Dyskeratotic" corresponds to Herlev's
"severe_dysplastic" rather than "moderate_dysplastic" — treating them as
equivalent would be an unstated assumption baked into the evaluation, not a
finding. The only label space all three datasets support without that leap
is the coarse **normal vs. abnormal** split every one of them already
implies. See `src/cervical_screening/data/harmonize_labels.py` for the exact
per-class mapping.

## Evaluation design

A model trained on SIPaKMeD's native 5-class task (`train.py`) is evaluated
zero-shot on Herlev and Mendeley LBC by summing its 5-class softmax output
into the two binary buckets (`eval/cross_dataset.py`) — no retraining, no
calibration on the shift set, since the whole point is measuring what
happens when the input distribution changes and nothing else does. The same
grouping is applied to SIPaKMeD's own held-out test split to get the
in-distribution binary reference number; the gap between that and the
zero-shot numbers on Herlev/Mendeley LBC is the actual domain-shift result
this evaluation exists to produce, not either number in isolation.

## Status

- SIPaKMeD: acquired and validated (4,049 crops, matches paper).
- Herlev: acquired and validated — 917 total files across 7 classes
  (`normal_columnar`, `normal_intermediate`, `normal_superficiel`,
  `light_dysplastic`, `moderate_dysplastic`, `severe_dysplastic`,
  `carcinoma_in_situ`), exactly matching Plissiti et al.'s reported count.
- Mendeley LBC: downloading (2.06GB). Folder names in
  `harmonize_labels.MENDELEY_LBC_TO_BINARY` are provisional — confirm the
  exact extracted directory names before running `make_shift_manifest.py`
  against it.
