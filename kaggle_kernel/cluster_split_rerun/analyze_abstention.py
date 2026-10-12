"""Cluster-grouped selective prediction and paired model comparisons from the saved per-image predictions.

usage: python analyze_abstention.py <run dir with out/ (preds, manifest)> <output.json>

1. Combined confidence + conformal-set-size abstention for the cluster-grouped DeiT-Tiny model, with risk-coverage AUC
   (mean risk over coverage levels k/n) for confidence only, standard split-conformal set size only, and the rank-averaged
   combination, in distribution (5-class) and on Herlev, Mendeley LBC and APCEM (binary). Paired bootstrap intervals for
   (combined - confidence): cluster bootstrap in distribution, image bootstrap externally. Also coverage, average set
   size and the fraction of empty sets for the standard threshold.
2. Paired cluster/image-bootstrap differences in balanced accuracy between the in-distribution leader and every other model.
"""
import csv
import json
import sys

import numpy as np

D, OUT = sys.argv[1].rstrip("/") + "/", sys.argv[2]
rng = np.random.default_rng(0)
NB = 2000  # matches analyze.py's resample count, for consistency across all supplementary tables
ABN = [0, 1]
MODELS = ["resnet50", "mobilenetv2_100", "efficientnet_lite0", "vit_small_patch16_224", "swin_tiny_patch4_window7_224",
          "deit_tiny_patch16_224", "levit_128", "tiny_vit_5m_224", "mobilevit_s", "mobilevitv2_100", "efficientvit_b0",
          "efficientvit_m0"]

# test-set clusters in npz order
keys = []
with open(D + "sipakmed_cluster_split.csv") as f:
    for r in csv.DictReader(f):
        if r["split"] == "test":
            keys.append(r["class_name"] + "|" + r["cluster_id"])
uniq, inv = np.unique(np.array(keys), return_inverse=True)
members = [np.where(inv == i)[0] for i in range(len(uniq))]


def rank_avg(v):
    order = np.argsort(v, kind="mergesort"); r = np.empty(len(v)); i = 0
    sv = v[order]
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2.0; i = j + 1
    return r / (len(v) - 1)


def rc_auc(signal, correct, tie_perms=20):
    n = len(signal)
    vals = []
    for _ in range(tie_perms if len(np.unique(signal)) < n else 1):
        noise = rng.permutation(n)
        order = np.lexsort((noise, -signal))
        err = 1 - correct[order]
        vals.append(np.mean(np.cumsum(err) / np.arange(1, n + 1)))
    return float(np.mean(vals))


def temp_fit(p, y):
    lp = np.log(np.clip(p, 1e-12, 1)); best, bt = 1e9, 1.0
    for T in np.linspace(0.5, 5, 91):
        q = np.exp(lp / T); q /= q.sum(1, keepdims=True)
        nll = -np.log(np.clip(q[np.arange(len(y)), y], 1e-12, 1)).mean()
        if nll < best:
            best, bt = nll, T
    return bt


def temp(p, T):
    q = np.exp(np.log(np.clip(p, 1e-12, 1)) / T); return q / q.sum(1, keepdims=True)


res = {}
z = np.load(D + "preds/deit_tiny_patch16_224.npz")
T = temp_fit(z["calibration_probs"], z["calibration_labels"])
cal_p, cal_y = temp(z["calibration_probs"], T), z["calibration_labels"]
sc = 1 - cal_p[np.arange(len(cal_y)), cal_y]
n = len(sc); alpha = 0.1
thr = float(np.quantile(sc, min(1.0, np.ceil((n + 1) * (1 - alpha)) / n), method="higher"))
res["temperature"] = float(T); res["threshold_standard"] = thr; res["n_cal"] = int(n)

conds = {}
tp, ty = temp(z["test_probs"], T), z["test_labels"]
sets = tp >= (1 - thr)
size = sets.sum(1)
res["in_dist_conformal"] = {"coverage": float(sets[np.arange(len(ty)), ty].mean()), "avg_set_size": float(size.mean()),
                            "fraction_empty_sets": float((size == 0).mean()), "fraction_singleton": float((size == 1).mean()),
                            "fraction_size_ge2": float((size >= 2).mean())}
conds["in_distribution"] = dict(conf=tp.max(1), size=size, correct=(tp.argmax(1) == ty).astype(float), groups=members)
for name in ("herlev", "mendeley_lbc", "apcem"):
    p5 = temp(z[f"{name}_probs5"], T); ye = z[f"{name}_labels"]
    pab = p5[:, ABN].sum(1)
    s5 = (p5 >= (1 - thr)).sum(1)
    conds[name] = dict(conf=np.maximum(pab, 1 - pab), size=s5, correct=((pab > 0.5).astype(int) == ye).astype(float), groups=None)
    res[f"{name}_conformal"] = {"avg_set_size": float(s5.mean()), "fraction_empty_sets": float((s5 == 0).mean())}


def signals(c):
    conf, size = c["conf"], c["size"]
    comb = (rank_avg(conf) + rank_avg(-size.astype(float))) / 2
    return {"confidence": conf, "set_size": -size.astype(float), "combined": comb}


sel = {}
for cname, c in conds.items():
    sg = signals(c); out = {k: rc_auc(v, c["correct"]) for k, v in sg.items()}
    d_boot = []
    n_ = len(c["correct"])
    for _ in range(NB):
        if c["groups"] is not None:
            pick = rng.integers(0, len(c["groups"]), len(c["groups"]))
            ix = np.concatenate([c["groups"][i] for i in pick])
        else:
            ix = rng.integers(0, n_, n_)
        a = rc_auc(sg["combined"][ix], c["correct"][ix], tie_perms=1)
        b = rc_auc(sg["confidence"][ix], c["correct"][ix], tie_perms=1)
        d_boot.append(a - b)
    out["combined_minus_confidence"] = {"mean": float(out["combined"] - out["confidence"]),
                                        "ci": [float(np.percentile(d_boot, 2.5)), float(np.percentile(d_boot, 97.5))]}
    sel[cname] = out
res["risk_coverage_auc"] = sel

# ---- paired comparisons of balanced accuracy
def ba(y, p, k):
    return float(np.mean([(p[y == c] == c).mean() for c in range(k) if (y == c).any()]))


pairs = {}
leader = "mobilevitv2_100"
ya = np.load(D + f"preds/{leader}.npz")
for split, k in (("in_dist", 5), ("herlev", 2), ("mendeley_lbc", 2), ("apcem", 2)):
    def getp(m):
        zz = np.load(D + f"preds/{m}.npz")
        if split == "in_dist":
            return zz["test_labels"], zz["test_probs"].argmax(1)
        p5 = zz[f"{split}_probs5"]
        return zz[f"{split}_labels"], (p5[:, ABN].sum(1) > 0.5).astype(int)
    y0, p0 = getp(leader)
    # leader for external splits = best model on that split
    if split != "in_dist":
        best = max(MODELS, key=lambda m: ba(*getp(m), 2))
        y0, p0 = getp(best); lead = best
    else:
        lead = leader
    res_split = {"leader": lead, "leader_ba": ba(y0, p0, k), "vs": {}}
    nloc = len(y0)
    picks = [rng.integers(0, len(members), len(members)) for _ in range(NB)] if split == "in_dist" else [rng.integers(0, nloc, nloc) for _ in range(NB)]
    idxs = [np.concatenate([members[i] for i in p]) for p in picks] if split == "in_dist" else picks
    for m in MODELS:
        if m == lead:
            continue
        y1, p1 = getp(m)
        diffs = [ba(y0[ix], p0[ix], k) - ba(y1[ix], p1[ix], k) for ix in idxs]
        res_split["vs"][m] = {"diff": ba(y0, p0, k) - ba(y1, p1, k), "ci": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]}
    pairs[split] = res_split
res["paired_vs_leader"] = pairs
json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps({"conformal": res["in_dist_conformal"], "rc_auc": res["risk_coverage_auc"]}, indent=1))
