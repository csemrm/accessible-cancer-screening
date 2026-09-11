# Running the pipeline via the Colab CLI

[`google-colab-cli`](https://github.com/googlecolab/google-colab-cli) —
Google's own terminal tool for Colab — turned out to be the best path once
Kaggle's weekly GPU quota was exhausted and its P100 pool proved incompatible
with the pinned torch build (see `docs/dataset_provenance.md`). Unlike the
interactive `notebooks/colab_pipeline.ipynb` fallback (which needs a browser
and a manual `kaggle.json` upload each session), this drives a real Colab GPU
VM entirely from the terminal, with no browser step at all.

## One-time setup

```
pip install google-colab-cli
colab new -s test --gpu T4
colab upload -s test ~/.kaggle/kaggle.json /root/.kaggle/kaggle.json
```

Authentication worked out of the box against this machine's existing GCP
Application Default Credentials (`--auth adc`, the CLI's default) — no
separate OAuth step was needed here, though a machine without ADC configured
may see one.

## Verifying the hardware before spending a run on it

```
echo "import torch; print(torch.cuda.get_device_capability(0))" | colab exec -s test
```

T4 reports `(7, 5)` — inside this torch build's supported `sm_70+` range,
unlike Kaggle's P100 (`sm_60`, see the compute-capability fallback in
`src/cervical_screening/device.py`).

## Running the pipeline

`colab_kernel/run_pipeline.py` mirrors `kaggle_kernel/run_pipeline.py`'s crop
→ split → train logic, adapted for this environment: it downloads all three
datasets itself via the `kaggle` CLI (already present on the Colab VM image;
only the credentials file needs uploading), and — since a real, compatible
GPU removes the time-budget risk that forced the Kaggle kernel down to a
10-epoch fast pass — runs the full 30-epoch/patience-7 budget, then chains
directly into the cross-dataset shift evaluation (S16/RQ4) using whichever
model scored best in-distribution.

```
colab exec -s test -f colab_kernel/run_pipeline.py --timeout 14400
colab download -s test /content/runs/results.json ./runs/colab_rq1/results.json
colab download -s test /content/runs/cross_dataset_shift_results.json ./runs/colab_rq1/cross_dataset_shift_results.json
colab stop -s test
```

`colab exec`'s `--timeout` defaults to **30 seconds** — nowhere near enough
for a run that downloads ~8.6GB of datasets before training even starts.
The first attempt hit exactly this: it printed `downloading datasets...`
and then died with `TimeoutError: Timeout waiting for output` once nothing
more had streamed back within 30s. Pass an explicit `--timeout` sized to the
whole run, not just the parts of it that produce frequent output.

## Status

Kicked off with the corrected timeout; not yet complete as of this commit.
See `runs/colab_rq1/` once results land.
