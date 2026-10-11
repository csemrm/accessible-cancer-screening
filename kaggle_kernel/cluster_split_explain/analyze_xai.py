"""Paired cluster-bootstrap intervals for the explainability comparison.

usage: python analyze_xai.py <dir with *_per_image.npz> <output.json>

For each backbone and metric, the paired per-cell difference between relevancy propagation and each baseline
(attention rollout, Grad-CAM) is summarised by its mean, a 95% cluster-bootstrap interval (resampling source cluster
images, 2,000 resamples), and the fraction of cells on which relevancy propagation is better. Marginal means with their
own cluster-bootstrap intervals are also reported.
"""
import json
import sys

import numpy as np

D, OUT = sys.argv[1].rstrip("/") + "/", sys.argv[2]
rng = np.random.default_rng(0)
NB = 2000
BACKBONES = ["deit_tiny_patch16_224", "vit_small_patch16_224"]
METHODS = ["relevancy_propagation", "attention_rollout", "gradcam"]
METRICS = {"deletion": "lower", "insertion": "higher", "concentration": "higher"}

res = {}
for b in BACKBONES:
    z = np.load(D + f"{b}_per_image.npz", allow_pickle=True)
    key = np.array([f"{int(l)}|{c}" for l, c in zip(z["label"], z["cluster"])])
    uniq, inv = np.unique(key, return_inverse=True)
    members = [np.where(inv == i)[0] for i in range(len(uniq))]
    picks = [rng.integers(0, len(members), len(members)) for _ in range(NB)]
    idxs = [np.concatenate([members[i] for i in p]) for p in picks]
    out = {"n_cells": int(len(key)), "n_clusters": int(len(uniq)), "accuracy": float((z["label"] == z["pred"]).mean())}
    for metric, better in METRICS.items():
        arr = {m: z[f"{m}_{metric}"] for m in METHODS}
        out[metric] = {"means": {}, "paired": {}}
        for m in METHODS:
            boots = [arr[m][ix].mean() for ix in idxs]
            out[metric]["means"][m] = {"mean": float(arr[m].mean()), "ci": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]}
        for base in ("attention_rollout", "gradcam"):
            diff = arr["relevancy_propagation"] - arr[base]
            boots = [diff[ix].mean() for ix in idxs]
            wins = (diff < 0) if better == "lower" else (diff > 0)
            out[metric]["paired"][base] = {"mean_diff": float(diff.mean()), "ci": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
                                           "fraction_relevancy_better": float(wins.mean()), "better_when": better}
    res[b] = out
json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps({b: {m: v["paired"] for m, v in o.items() if isinstance(v, dict) and "paired" in v} for b, o in res.items()}, indent=1))
