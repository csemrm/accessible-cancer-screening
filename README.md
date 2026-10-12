# Trustworthy, Resource-Efficient Vision Transformers for Cervical Cytology Screening

Research code for the flagship project under the program *"Trustworthy and
Resource-Efficient AI for Early Cancer Screening in Rural and Medically
Underserved Clinical Settings."* See `01_accessible_cancer_screening.md` for
the original proposal and the published proposal artifact (linked in project
notes) for the full literature-grounded revision — gap analysis, architecture
design, research questions/hypotheses, experimental design, and the
national-importance evidence review.

## Status (Months 1–11 complete)

All five research questions and the Month 11 resource-efficiency/robustness
benchmark have real, completed results (not projections). Each phase's full
writeup lives in `docs/`:

| Phase | What it covers | Backbone | Writeup |
|---|---|---|---|
| RQ1 | Efficiency/accuracy benchmark, 12 CNN/ViT/efficient-ViT models | all 12 | `docs/rq1_results.md` |
| RQ4 | Cross-dataset domain shift (SIPaKMeD → Herlev, Mendeley LBC), same 12 models | all 12 | `docs/rq1_results.md` (combined with RQ1) |
| RQ2 | Uncertainty/calibration: temp-scaled softmax vs. MC Dropout vs. evidential vs. deep ensemble | EfficientViT-M0 (RQ1's winner) | `docs/rq2_results.md` |
| RQ3 | Explainability: relevancy propagation vs. attention rollout vs. Grad-CAM | DeiT-Tiny (pure ViT — see note below) | `docs/rq3_results.md` |
| RQ5 | System integration: does composing calibration/explainability/selective-prediction degrade anything? | DeiT-Tiny | `docs/rq5_results.md` |
| Month 11 | Resource-efficiency (params/FLOPs/CPU latency/memory, all 12 models) + synthetic degradation/OOD sanity check | all 12 (efficiency) / EfficientViT-M0 (degradation+OOD) | `docs/month11_results.md` |

**Why RQ3/RQ5 use DeiT-Tiny instead of RQ1's winning EfficientViT-M0:**
relevancy propagation and attention rollout are both defined for standard
multi-head self-attention transformer blocks; EfficientViT-M0's hybrid conv +
cascaded-group-attention design has no established propagation rule for
either method. DeiT-Tiny (0.9606 balanced accuracy in RQ1, vs.
EfficientViT-M0's 0.9654) is the closest standard-ViT baseline in the
registry, used for RQ3/RQ5 rather than forcing an ad hoc extension onto an
architecture these methods weren't designed for. This is stated as a
limitation in each affected writeup, not hidden.

Remaining timeline items (Month 12): Paper 3 drafting, code/data release
preparation, national-importance evidence refresh.

## Manuscripts and the JMI paper

`papers/` holds the merged Journal of Medical Imaging manuscript (`papers/paper_jmi.tex`,
build with `latexmk -pdf paper_jmi.tex`) and its supplementary tables and figures
(`papers/paper_jmi_supp.tex`, Tables S1-S10 and Figures S1-S6). The merged paper combines Papers 1
and 3 and reports its limitations explicitly, including that the original
SIPaKMeD split is at the cell level (87.8% of test cells share a source cluster
image with a training cell) and that two of the three external datasets contain
multi-cell fields of view and one is algorithmically enhanced (APCEM).

### Cluster-grouped re-run (Kaggle)

`kaggle_kernel/cluster_split_rerun/` re-runs the 12-model benchmark and the
three external evaluations with every SIPaKMeD cluster image kept inside one
split, with the same recipe and seed (13), and additionally saves per-image
probabilities, checkpoints, software/hardware versions and per-epoch timing.
It attaches these Kaggle datasets: `marinaeplissiti/sipakmed`,
`yuvrajsinhachowdhury/herlev-dataset`, `blank1508/mendeley-lbc-cervical-cancer`,
and a private copy of APCEM (`csemrm/apcem-cervical-cytology`, see below).

```
kaggle kernels push -p kaggle_kernel/cluster_split_rerun --accelerator NvidiaTeslaT4
```

Outputs are written to `/kaggle/working/out/` on Kaggle. Because the output
also contains thousands of crop images, download only what you need with a
larger page size to avoid API rate limits:

```
kaggle kernels output <owner>/sipakmed-cluster-split-rerun -p out --page-size 200 \
  --file-pattern "(results\.json|env\.json|split_stats\.json|preds/.*\.npz)"
```

### Datasets (not redistributed here)

| Dataset | Source | Licence note |
|---|---|---|
| SIPaKMeD | Kaggle `marinaeplissiti/sipakmed` (original host blocks scripted download) | Kaggle lists "other"; confirm terms with the dataset authors before redistributing derived data |
| Herlev | Kaggle `yuvrajsinhachowdhury/herlev-dataset` | Check the source terms |
| Mendeley LBC | Kaggle `blank1508/mendeley-lbc-cervical-cancer` (Hussain et al., Data in Brief 2020) | Check the source terms |
| APCEM v2 | Mendeley Data, DOI 10.17632/snzb5g6ddy.2 (CC BY 4.0) | The Mendeley file host returns HTTP 403 to Kaggle kernels, so download it locally and upload your own private dataset |

No dataset files are committed. The image-content differences between the
datasets (single-cell crops versus fields of view) are described in the JMI
manuscript.

### Software versions

`requirements.txt` gives minimum versions only. The exact versions used by the
original Colab runs were not recorded. The cluster-grouped Kaggle run writes
its exact Python, PyTorch, timm, CUDA and GPU versions to `out/env.json`; the
copy for the reported run is `runs/kaggle_cluster_split/env.json` (Python 3.13.15,
torch 2.11.0+cu128, timm 1.0.29, numpy 2.1.3, CUDA 12.8, Tesla T4). That folder also
holds the per-image predictions, results, split manifest and `cluster_analysis.json`;
re-create the analysis with `kaggle_kernel/cluster_split_rerun/analyze.py`.

## Layout

```
src/cervical_screening/
  data/
    crop_sipakmed.py       # native-resolution single-cell crops from raw SIPaKMeD (+ optional nucleus-polygon manifest for RQ3)
    make_splits.py          # stratified train/val/calibration/test manifest
    harmonize_labels.py      # SIPaKMeD/Herlev/Mendeley LBC -> shared normal/abnormal binary labels for cross-dataset shift eval
    make_shift_manifest.py    # CLI wrapper for scan_binary_manifest
    dataset.py                 # torch Dataset reading the manifest
    transforms.py                # ImageNet-normalized train/eval transforms
  models/
    registry.py                  # RQ1 baseline registry (CNN / full ViT / efficient ViT), all timm tags verified to exist with pretrained weights (exact timm version not recorded)
  eval/
    metrics.py                     # sensitivity/specificity/F1/AUROC/AUPRC/balanced accuracy
    efficiency.py                    # measured params/CPU latency/RSS delta (see isolation caveat in the docstring)
    flops.py                          # FLOPs via torch's own FlopCounterMode
    cross_dataset.py                   # 5-class -> binary probability grouping for shift eval
  uncertainty/                          # RQ2: calibration, conformal prediction, evidential DL, MC Dropout, selective prediction
    calibration.py, conformal.py, evidential.py, mc_dropout.py, selective.py, train_variants.py
  explainability/                        # RQ3: relevancy propagation, attention rollout, Grad-CAM, faithfulness, nucleus overlap
    relevancy_propagation.py, attention_rollout.py, gradcam.py, faithfulness.py, nucleus_overlap.py, vit_hooks.py, rollout.py, heatmap.py
  integration/                            # RQ5: combined selective-prediction signal, pipeline latency
    combined_selective.py, pipeline_latency.py
  robustness/                             # Month 11: synthetic degradation + OOD sanity check
    degradation.py, ood_sanity.py
  train.py                                 # RQ1 baseline training + evaluation entrypoint
  measure_efficiency.py                     # Month 11 local (no-GPU) efficiency benchmark CLI

colab_kernel/
  run_pipeline.py       # self-contained RQ1/RQ4 Colab script
  rq2/, rq3/, rq5/, rq_month11/    # per-phase Colab setup + eval scripts (see docs/colab_cli_pipeline.md)

runs/            # saved JSON results per phase (colab_rq1 + apcem + seedchecks, colab_rq2, colab_rq3, colab_rq5, efficiency, month11_robustness)
papers/          # draft papers 1-3 and the merged JMI manuscript (paper_jmi.tex)
kaggle_kernel/   # original Kaggle kernel and cluster_split_rerun/ (cluster-grouped re-run)
docs/            # full writeups, one per phase, plus dataset provenance and Colab-pipeline notes
```

## Setup

```
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
```

Requires a working `~/.kaggle/kaggle.json` for dataset acquisition
(SIPaKMeD, Herlev, Mendeley LBC — see `docs/dataset_provenance.md`).

## Reproducing a phase

Every phase after RQ1 was actually run on Google Colab (via
[`google-colab-cli`](https://github.com/googlecolab/google-colab-cli), not
the Colab notebook UI) rather than locally, since training needs a GPU.
`docs/colab_cli_pipeline.md` covers the exact CLI workflow and its quirks.
The general pattern, using RQ2 as an example:

```
colab new -s <session> --gpu T4
colab upload -s <session> colab_kernel/run_pipeline.py /content/run_pipeline.py
# zip+upload src/cervical_screening, upload kaggle.json, unzip on the VM — see docs/colab_cli_pipeline.md
colab exec -s <session> -f colab_kernel/rq2/setup.py --timeout 1800
colab exec -s <session> -f colab_kernel/rq2/common_eval.py --timeout 300
colab exec -s <session> -f colab_kernel/rq2/run_baseline.py --timeout 1800
# ...then run_mc_dropout.py, run_evidential.py, 5x run_ensemble_member.py, run_ensemble_eval.py
```

Each `runs/colab_*/*.json` file is the exact result of one such run, saved
immediately after it completed.

### RQ1 baseline benchmark (12 CNN/ViT/efficient-ViT models)

```
python -m cervical_screening.data.crop_sipakmed \
  --raw-root data/raw/sipakmed_full --out-root data/processed/sipakmed_crops

python -m cervical_screening.data.make_splits \
  --crops-root data/processed/sipakmed_crops --out-csv data/processed/sipakmed_splits.csv

python -m cervical_screening.train \
  --model mobilevit_s --manifest data/processed/sipakmed_splits.csv --out-dir runs/mobilevit_s
```

Repeat the last command once per entry in `MODEL_REGISTRY`
(`src/cervical_screening/models/registry.py`) to reproduce the full
efficiency/accuracy baseline table.

### Month 11 resource-efficiency benchmark (no GPU needed)

```
python -m cervical_screening.measure_efficiency --out runs/efficiency/benchmark.json
```

Runs entirely locally: params, FLOPs, and CPU latency/memory are
architecture properties, not trained-weight properties.

## Known limitations (see each doc for full detail)

- Public cytology datasets are small (900–4,000 images) and drawn from a handful of institutions/countries.
- RQ3/RQ5 use a different backbone (DeiT-Tiny) than RQ1's winner (EfficientViT-M0) — see the table above.
- CPU latency in the Month 11 benchmark was measured on one shared consumer laptop, not dedicated hardware (`docs/month11_results.md` quantifies the resulting run-to-run noise).
- No trained-weight checkpoints or per-image predictions are persisted for the original Colab runs — every Colab session trained from scratch, since `/content/` is ephemeral. `runs/*.json` files are the durable record of each result. The Kaggle cluster-grouped re-run saves both on Kaggle.
- The original SIPaKMeD split is at the cell level; cells from the same source cluster image can fall in both training and test sets (see the JMI manuscript). The cluster-grouped re-run addresses this for one split.
- Mendeley LBC and APCEM images are multi-cell fields of view (APCEM is also algorithmically enhanced), unlike the single-cell training data, so they are content-shift stress tests rather than controlled cross-institution tests.
