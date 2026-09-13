"""Generates all paper figures directly from this project's own JSON results
(runs/colab_rq1 numbers hardcoded from docs/rq1_results.md, runs/colab_rq2/*.json,
runs/colab_rq3, runs/colab_rq5, runs/month11_robustness -- all real, no synthetic
or illustrative data)."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 9, "figure.dpi": 150})

# ---------------------------------------------------------------------------
# Paper 1, Figure 1: params vs in-distribution balanced accuracy, by family
# ---------------------------------------------------------------------------
rq1 = [
    ("EfficientViT-M0", 2157401, 0.9654, "eff"),
    ("EfficientViT-B0", 2135413, 0.9623, "eff"),
    ("DeiT-Tiny", 5525381, 0.9606, "eff"),
    ("Swin-Tiny", 27523199, 0.9605, "full"),
    ("TinyViT-5M", 5073369, 0.9587, "eff"),
    ("MobileViT-S", 4940837, 0.9570, "eff"),
    ("ResNet-50", 23518277, 0.9542, "cnn"),
    ("LeViT-128", 8447786, 0.9458, "eff"),
    ("MobileViTv2", 4391406, 0.9387, "eff"),
    ("EfficientNet-Lite0", 3377413, 0.9375, "cnn"),
    ("ViT-Small", 21667589, 0.9374, "full"),
    ("MobileNetV2", 2230277, 0.9357, "cnn"),
]
colors = {"cnn": "#1b6ca8", "full": "#8a2be2", "eff": "#d1495b"}
labels_seen = set()
fig, ax = plt.subplots(figsize=(5.2, 3.6))
for name, params, acc, fam in rq1:
    lbl = {"cnn": "CNN", "full": "Full ViT", "eff": "Efficient ViT"}[fam]
    ax.scatter(params, acc, c=colors[fam], s=45, label=lbl if lbl not in labels_seen else None,
               edgecolors="black", linewidths=0.4, zorder=3)
    labels_seen.add(lbl)
    dx = 1.15 if name not in ("EfficientViT-M0",) else 1.0
    ax.annotate(name, (params, acc), fontsize=6.3, xytext=(4, 3), textcoords="offset points")
ax.set_xscale("log")
ax.set_xlabel("Parameters (log scale)")
ax.set_ylabel("In-distribution balanced accuracy")
ax.set_ylim(0.930, 0.972)
ax.legend(loc="lower right", fontsize=7.5, framealpha=0.9)
ax.grid(alpha=0.25, linewidth=0.5)
fig.tight_layout()
fig.savefig("figs/p1_fig1_params_vs_accuracy.pdf")
plt.close(fig)

# ---------------------------------------------------------------------------
# Paper 1, Figure 2: Mendeley LBC balanced accuracy, sorted, by family
# ---------------------------------------------------------------------------
rq4 = [
    ("EfficientNet-Lite0", 0.6674, "cnn"),
    ("ResNet-50", 0.6082, "cnn"),
    ("MobileViTv2", 0.5418, "eff"),
    ("MobileNetV2", 0.5391, "cnn"),
    ("DeiT-Tiny", 0.5230, "eff"),
    ("EfficientViT-B0", 0.5217, "eff"),
    ("Swin-Tiny", 0.5192, "full"),
    ("ViT-Small", 0.5084, "full"),
    ("LeViT-128", 0.4764, "eff"),
    ("EfficientViT-M0", 0.4731, "eff"),
    ("TinyViT-5M", 0.4543, "eff"),
    ("MobileViT-S", 0.2927, "eff"),
]
fig, ax = plt.subplots(figsize=(5.2, 3.6))
names = [r[0] for r in rq4]
vals = [r[1] for r in rq4]
bar_colors = [colors[r[2]] for r in rq4]
y_pos = range(len(names))
ax.barh(y_pos, vals, color=bar_colors, edgecolor="black", linewidth=0.4)
ax.axvline(0.5, color="black", linestyle="--", linewidth=1, label="Chance level")
ax.set_yticks(y_pos)
ax.set_yticklabels(names, fontsize=7.5)
ax.invert_yaxis()
ax.set_xlabel("Balanced accuracy on Mendeley LBC (zero-shot)")
handles = [plt.Rectangle((0, 0), 1, 1, color=colors[k]) for k in ("cnn", "full", "eff")]
ax.legend(handles + [plt.Line2D([0], [0], color="black", linestyle="--")],
          ["CNN", "Full ViT", "Efficient ViT", "Chance level"], loc="lower right", fontsize=7)
ax.grid(alpha=0.25, linewidth=0.5, axis="x")
fig.tight_layout()
fig.savefig("figs/p1_fig2_mendeley_shift.pdf")
plt.close(fig)

# ---------------------------------------------------------------------------
# Paper 1, Figure 3: APCEM AUROC, sorted, by family (3rd independent shift dataset)
# ---------------------------------------------------------------------------
apcem = [
    ("MobileNetV2", 0.6528, "cnn"),
    ("EfficientNet-Lite0", 0.5701, "cnn"),
    ("LeViT-128", 0.5376, "eff"),
    ("MobileViTv2", 0.4120, "eff"),
    ("DeiT-Tiny", 0.3799, "eff"),
    ("TinyViT-5M", 0.3681, "eff"),
    ("EfficientViT-M0", 0.3407, "eff"),
    ("Swin-Tiny", 0.3082, "full"),
    ("MobileViT-S", 0.3161, "eff"),
    ("EfficientViT-B0", 0.3206, "eff"),
    ("ResNet-50", 0.2929, "cnn"),
    ("ViT-Small", 0.2555, "full"),
]
apcem.sort(key=lambda r: r[1], reverse=True)
fig, ax = plt.subplots(figsize=(5.2, 3.6))
names = [r[0] for r in apcem]
vals = [r[1] for r in apcem]
bar_colors = [colors[r[2]] for r in apcem]
y_pos = range(len(names))
ax.barh(y_pos, vals, color=bar_colors, edgecolor="black", linewidth=0.4)
ax.axvline(0.5, color="black", linestyle="--", linewidth=1, label="Chance level")
ax.set_yticks(y_pos)
ax.set_yticklabels(names, fontsize=7.5)
ax.invert_yaxis()
ax.set_xlabel("AUROC on APCEM (zero-shot)")
handles = [plt.Rectangle((0, 0), 1, 1, color=colors[k]) for k in ("cnn", "full", "eff")]
ax.legend(handles + [plt.Line2D([0], [0], color="black", linestyle="--")],
          ["CNN", "Full ViT", "Efficient ViT", "Chance level"], loc="lower right", fontsize=7)
ax.grid(alpha=0.25, linewidth=0.5, axis="x")
fig.tight_layout()
fig.savefig("figs/p1_fig3_apcem_shift.pdf")
plt.close(fig)

# ---------------------------------------------------------------------------
# Paper 2, Figure 1: risk-coverage AUC per method, 3 conditions (own y-scales)
# ---------------------------------------------------------------------------
methods = ["Baseline", "MC Dropout", "Evidential", "Deep ensemble"]
method_colors = ["#888888", "#1b6ca8", "#d1495b", "#2a9d5c"]
rc_in_dist = [0.002446328287491706, 0.001959038339648868, 0.004150150019020782, 0.0019976426168283093]
rc_herlev = [0.193, 0.203, 0.167, 0.147]
rc_mendeley = [0.515, 0.285, 0.476, 0.473]

fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.8))
for ax, data, title in zip(axes, [rc_in_dist, rc_herlev, rc_mendeley],
                            ["In-distribution", "Herlev (moderate shift)", "Mendeley LBC (severe shift)"]):
    ax.bar(methods, data, color=method_colors, edgecolor="black", linewidth=0.4)
    ax.set_title(title, fontsize=8.5)
    ax.set_ylabel("Risk-coverage AUC\n(lower better)", fontsize=7.5)
    ax.tick_params(axis="x", labelsize=6.3, rotation=20)
    ax.grid(alpha=0.25, linewidth=0.5, axis="y")
fig.tight_layout()
fig.savefig("figs/p2_fig1_risk_coverage.pdf")
plt.close(fig)

# ---------------------------------------------------------------------------
# Paper 3, Figure 1: explainability metrics, 3 methods
# ---------------------------------------------------------------------------
exp_methods = ["Relevancy\npropagation", "Grad-CAM", "Attention\nrollout"]
deletion = [0.734, 0.818, 0.825]
insertion = [0.970, 0.877, 0.936]
nucleus = [2.714, 1.472, 1.177]
exp_colors = ["#2a9d5c", "#1b6ca8", "#d1495b"]

fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.8))
axes[0].bar(exp_methods, deletion, color=exp_colors, edgecolor="black", linewidth=0.4)
axes[0].set_title("Deletion AUC\n(lower better)", fontsize=8)
axes[1].bar(exp_methods, insertion, color=exp_colors, edgecolor="black", linewidth=0.4)
axes[1].set_title("Insertion AUC\n(higher better)", fontsize=8)
axes[2].bar(exp_methods, nucleus, color=exp_colors, edgecolor="black", linewidth=0.4)
axes[2].axhline(1.0, color="black", linestyle="--", linewidth=1)
axes[2].set_title("Nucleus concentration\n(1.0 = chance)", fontsize=8)
for ax in axes:
    ax.tick_params(axis="x", labelsize=6.5)
    ax.grid(alpha=0.25, linewidth=0.5, axis="y")
fig.tight_layout()
fig.savefig("figs/p3_fig1_explainability_metrics.pdf")
plt.close(fig)

# ---------------------------------------------------------------------------
# Paper 3, Figure 2: degradation curves, real data from month11 JSON
# ---------------------------------------------------------------------------
deg = json.load(open("../runs/month11_robustness/degradation_ood.json"))["degradation"]
clean_acc = deg["clean"]["balanced_accuracy"]

jpeg_q = [90, 70, 50, 30, 10]
jpeg_acc = [deg["jpeg_compression"][str(q)]["balanced_accuracy"] for q in jpeg_q]
blur_r = [0.0, 1.0, 2.0, 4.0, 8.0]
blur_acc = [deg["gaussian_blur"][str(r) if r != 0.0 else "0.0"]["balanced_accuracy"] for r in blur_r]
bc_f = [1.0, 0.8, 0.6, 1.4, 1.7]
bc_acc_raw = {f: deg["brightness_contrast"][str(f)]["balanced_accuracy"] for f in [1.0, 0.8, 0.6, 1.4, 1.7]}

fig, axes = plt.subplots(1, 3, figsize=(7.8, 2.9), sharey=True)
axes[0].plot(jpeg_q, jpeg_acc, marker="o", color="#1b6ca8")
axes[0].invert_xaxis()
axes[0].set_xlabel("JPEG quality\n(lower = more severe)", fontsize=7.5)
axes[0].set_title("JPEG compression", fontsize=8.5)

axes[1].plot(blur_r, blur_acc, marker="o", color="#d1495b")
axes[1].set_xlabel("Gaussian blur radius\n(higher = more severe)", fontsize=7.5)
axes[1].set_title("Gaussian blur", fontsize=8.5)

bc_order = [1.0, 0.8, 0.6, 1.4, 1.7]
bc_severity = [0, 1, 2, 1, 2]  # distance from 1.0 as a rough severity proxy, dark vs bright shown separately
axes[2].plot([1.0, 0.8, 0.6], [bc_acc_raw[1.0], bc_acc_raw[0.8], bc_acc_raw[0.6]],
             marker="o", color="#8a2be2", label="Darker")
axes[2].plot([1.0, 1.4, 1.7], [bc_acc_raw[1.0], bc_acc_raw[1.4], bc_acc_raw[1.7]],
             marker="s", color="#e07b39", label="Brighter")
axes[2].set_xlabel("Brightness/contrast factor\n(1.0 = clean)", fontsize=7.5)
axes[2].set_title("Brightness/contrast", fontsize=8.5)
axes[2].legend(fontsize=6.5, loc="lower center")

axes[0].set_ylabel("Balanced accuracy")
for ax in axes:
    ax.axhline(clean_acc, color="gray", linestyle=":", linewidth=1)
    ax.grid(alpha=0.25, linewidth=0.5)
    ax.tick_params(labelsize=7)
fig.tight_layout()
fig.savefig("figs/p3_fig2_degradation_curves.pdf")
plt.close(fig)

# ---------------------------------------------------------------------------
# Paper 3, Figure 3: RQ5 combined-selective bar chart
# ---------------------------------------------------------------------------
conditions = ["In-distribution", "Herlev\n(moderate)", "Mendeley LBC\n(severe)"]
conf_only = [0.0050, 0.1190, 0.3244]
set_only = [0.0834, 0.3281, 0.4395]
combined = [0.0173, 0.1793, 0.2855]

fig, ax = plt.subplots(figsize=(5.2, 3.2))
x = range(len(conditions))
w = 0.25
ax.bar([i - w for i in x], conf_only, width=w, label="Confidence only", color="#1b6ca8", edgecolor="black", linewidth=0.4)
ax.bar(list(x), set_only, width=w, label="Conformal set size only", color="#d1495b", edgecolor="black", linewidth=0.4)
ax.bar([i + w for i in x], combined, width=w, label="Combined (rank-averaged)", color="#2a9d5c", edgecolor="black", linewidth=0.4)
ax.set_xticks(list(x))
ax.set_xticklabels(conditions, fontsize=8)
ax.set_ylabel("Risk-coverage AUC (lower better)")
ax.legend(fontsize=7.5)
ax.grid(alpha=0.25, linewidth=0.5, axis="y")
fig.tight_layout()
fig.savefig("figs/p3_fig3_combined_selective.pdf")
plt.close(fig)

print("all figures written to figs/")
