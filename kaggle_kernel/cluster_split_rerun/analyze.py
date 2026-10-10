"""Bootstrap CIs (cluster bootstrap for in-distribution, image bootstrap for external), sensitivity/specificity,
standard vs bootstrap-averaged split-conformal thresholds and selective-prediction operating points (DeiT-Tiny) from the
per-image predictions saved by run.py."""
import csv, json
import numpy as np
from collections import defaultdict

import sys
# usage: python analyze.py <dir with results/preds/manifest, ending in /> <output.json>
D = sys.argv[1]
OUT = sys.argv[2]
rng = np.random.default_rng(0)
NB = 2000
ABN = [0, 1]  # Dyskeratotic, Koilocytotic
NOR = [2, 3, 4]
MODELS = ["resnet50", "mobilenetv2_100", "efficientnet_lite0", "vit_small_patch16_224", "swin_tiny_patch4_window7_224",
          "deit_tiny_patch16_224", "levit_128", "tiny_vit_5m_224", "mobilevit_s", "mobilevitv2_100", "efficientvit_b0",
          "efficientvit_m0"]

# ----- test-split cluster ids in manifest order (== npz test order)
test_clusters, test_cls = [], []
with open(D + "sipakmed_cluster_split.csv") as f:
    for r in csv.DictReader(f):
        if r["split"] == "test":
            test_clusters.append(r["class_name"] + "|" + r["cluster_id"]); test_cls.append(int(r["class_idx"]))
test_clusters = np.array(test_clusters)
uniq, inv = np.unique(test_clusters, return_inverse=True)
members = [np.where(inv == i)[0] for i in range(len(uniq))]
print("test cells", len(test_clusters), "test clusters", len(uniq))


def bal_acc(y, p, k):
    recs = []
    for c in range(k):
        m = y == c
        if m.any():
            recs.append((p[m] == c).mean())
    return float(np.mean(recs))


def auroc_bin(y, s):
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg]); ranks = allv.argsort().argsort() + 1.0
    # average ties
    order = np.argsort(allv); sv = allv[order]; r = np.empty(len(allv)); i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2 + 1; i = j + 1
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def ci(vals):
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


ext_paths = json.load(open(D + "external_paths.json"))
res = {}
for m in MODELS:
    z = np.load(D + f"preds/{m}.npz")
    out = {}
    # in-distribution test (5-class), cluster bootstrap
    y = z["test_labels"]; p5 = z["test_probs"]; pred = p5.argmax(1)
    pt = bal_acc(y, pred, 5)
    boots_c, boots_i = [], []
    for _ in range(NB):
        pick = rng.integers(0, len(members), len(members))
        idx = np.concatenate([members[i] for i in pick])
        boots_c.append(bal_acc(y[idx], pred[idx], 5))
        idx2 = rng.integers(0, len(y), len(y))
        boots_i.append(bal_acc(y[idx2], pred[idx2], 5))
    out["in_dist_5class"] = {"bal_acc": pt, "ci_cluster": ci(boots_c), "ci_image": ci(boots_i)}
    # binary in-dist (abnormal vs normal) too
    yb = np.isin(y, ABN).astype(int); pb = p5[:, ABN].sum(1)
    out["in_dist_binary"] = {"bal_acc": bal_acc(yb, (pb > 0.5).astype(int), 2), "auroc": auroc_bin(yb, pb)}
    # external (binary), image bootstrap
    for name in ("herlev", "mendeley_lbc", "apcem"):
        p5e = z[f"{name}_probs5"]; ye = z[f"{name}_labels"]
        pab = p5e[:, ABN].sum(1); pe = (pab > 0.5).astype(int)
        ba, au = bal_acc(ye, pe, 2), auroc_bin(ye, pab)
        bb, ba_u = [], []
        for _ in range(NB):
            idx = rng.integers(0, len(ye), len(ye))
            bb.append(bal_acc(ye[idx], pe[idx], 2)); ba_u.append(auroc_bin(ye[idx], pab[idx]))
        sens = float((pe[ye == 1] == 1).mean()); spec = float((pe[ye == 0] == 0).mean())
        out[name] = {"bal_acc": ba, "ci": ci(bb), "auroc": au, "auroc_ci": ci(ba_u), "sensitivity": sens, "specificity": spec,
                     "n": int(len(ye))}
    res[m] = out

# ----- operating points, DeiT-Tiny: standard split conformal + selective prediction
z = np.load(D + "preds/deit_tiny_patch16_224.npz")


def fit_temperature(p, y):
    lp = np.log(np.clip(p, 1e-12, 1))
    best, bt = 1e9, 1.0
    for T in np.linspace(0.5, 5, 91):
        q = np.exp(lp / T); q /= q.sum(1, keepdims=True)
        nll = -np.log(np.clip(q[np.arange(len(y)), y], 1e-12, 1)).mean()
        if nll < best:
            best, bt = nll, T
    return bt


def temp(p, T):
    q = np.exp(np.log(np.clip(p, 1e-12, 1)) / T); return q / q.sum(1, keepdims=True)


T = fit_temperature(z["calibration_probs"], z["calibration_labels"])
cal_p = temp(z["calibration_probs"], T); cal_y = z["calibration_labels"]
scores = 1 - cal_p[np.arange(len(cal_y)), cal_y]
alpha = 0.1; n = len(scores)
qlev = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
thr_std = float(np.quantile(scores, qlev, method="higher"))
thr_boot = float(np.mean([np.quantile(scores[rng.choice(n, max(10, int(round(0.8 * n))), replace=True)],
                                      min(1.0, np.ceil((max(10, int(round(0.8 * n))) + 1) * (1 - alpha)) / max(10, int(round(0.8 * n)))),
                                      method="higher") for _ in range(200)]))
tp = temp(z["test_probs"], T); ty = z["test_labels"]
op = {"temperature": float(T), "n_cal": int(n), "thr_standard": thr_std, "thr_bootstrap_avg": thr_boot}
for nm, thr in (("standard", thr_std), ("bootstrap_avg", thr_boot)):
    sets = tp >= (1 - thr)
    op[f"conformal_{nm}"] = {"coverage": float(sets[np.arange(len(ty)), ty].mean()), "avg_set_size": float(sets.sum(1).mean())}


def risk_cov(conf, correct, cov):
    k = int(round(cov * len(conf))); order = np.argsort(-conf)[:k]
    return float(1 - correct[order].mean())


conf = tp.max(1); correct = (tp.argmax(1) == ty)
op["in_dist_selective"] = {f"error_at_{int(c*100)}pct_coverage": risk_cov(conf, correct, c) for c in (1.0, 0.9, 0.8, 0.7)}
# binary external operating points from confidence-only
ext_op = {}
for name in ("herlev", "mendeley_lbc", "apcem"):
    p5e = temp(z[f"{name}_probs5"], T); ye = z[f"{name}_labels"]
    pab = p5e[:, ABN].sum(1); conf_b = np.maximum(pab, 1 - pab); corr = ((pab > 0.5).astype(int) == ye)
    ext_op[name] = {f"error_at_{int(c*100)}pct_coverage": risk_cov(conf_b, corr, c) for c in (1.0, 0.9, 0.8, 0.7)}
op["external_selective_binary_confidence_only"] = ext_op
res["_deit_operating_points"] = op
json.dump(res, open(OUT, "w"), indent=1)
print("done")
