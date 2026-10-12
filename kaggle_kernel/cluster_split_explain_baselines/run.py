"""Explainability re-run with a perturbation-baseline sensitivity check on the cluster-grouped checkpoints.

Same cells, methods, and checkpoints as kaggle_kernel/cluster_split_explain/run.py, but deletion/insertion AUC is computed
under three replacement baselines (heavily blurred image, per-image mean colour, and the constant dataset-mean colour,
i.e. zero after ImageNet normalisation) so that the ranking of explanation methods can be checked for dependence on the
baseline. Heatmaps are computed once per cell. The representative-cell figure is chosen after a prepass over all test
cells (first correct cell of every class and the first three errors), so all five classes appear, with larger panels.
"""

from __future__ import annotations

import csv
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

hits = glob.glob("/kaggle/input/**/explainability/relevancy_propagation.py", recursive=True)
assert hits, "cervical_screening package files not found under /kaggle/input"
pkg_root = Path(hits[0]).parent.parent
link_dir = Path("/kaggle/working/pkg")
link_dir.mkdir(parents=True, exist_ok=True)
if not (link_dir / "cervical_screening").exists():
    (link_dir / "cervical_screening").symlink_to(pkg_root, target_is_directory=True)
sys.path.insert(0, str(link_dir))

from PIL import Image  # noqa: E402

import cervical_screening.explainability.faithfulness as FA  # noqa: E402
from cervical_screening.data.crop_sipakmed import CLASSES, crop_class  # noqa: E402
from cervical_screening.data.transforms import build_transform  # noqa: E402
from cervical_screening.explainability.attention_rollout import attention_rollout  # noqa: E402
from cervical_screening.explainability.faithfulness import deletion_auc, insertion_auc  # noqa: E402
from cervical_screening.explainability.gradcam import vit_gradcam  # noqa: E402
from cervical_screening.explainability.nucleus_overlap import concentration_ratio, rasterize_nucleus_mask  # noqa: E402
from cervical_screening.explainability.relevancy_propagation import relevancy_propagation  # noqa: E402
from cervical_screening.models.registry import build_model  # noqa: E402

WORK = Path("/kaggle/working")
OUT = WORK / "out_xai_baselines"
OUT.mkdir(parents=True, exist_ok=True)
INPUT_SIZE, GRID, N_STEPS = 224, 14, 20
BACKBONES = ["deit_tiny_patch16_224", "vit_small_patch16_224"]
METHODS = ["relevancy_propagation", "attention_rollout", "gradcam"]

_ORIG_BLUR = FA._gaussian_blur
FILLS = {
    "blur": _ORIG_BLUR,
    "mean_color": lambda img, *a, **k: img.mean(dim=(2, 3), keepdim=True).expand_as(img).contiguous(),
    "constant_zero": lambda img, *a, **k: torch.zeros_like(img),
}


def find(pattern: str) -> Path:
    hits = glob.glob(f"/kaggle/input/**/{pattern}", recursive=True)
    assert hits, f"{pattern} not found under /kaggle/input"
    return Path(hits[0])


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise SystemExit("no CUDA device")
    (OUT / "env.json").write_text(json.dumps({"torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
                                              "n_steps": N_STEPS, "baselines": list(FILLS)}, indent=2))

    cands = [Path(h).parent for h in glob.glob("/kaggle/input/**/im_Dyskeratotic", recursive=True)]
    cands = [c for c in cands if (c / "im_Dyskeratotic" / "im_Dyskeratotic").exists()]
    assert cands, "raw SIPaKMeD layout not found"
    raw_root = cands[0]
    crops_root = WORK / "crops"
    nucleus_rows: list[dict] = []
    counts = {c: crop_class(raw_root, crops_root, c, margin=4.0, nucleus_rows=nucleus_rows) for c in CLASSES}
    assert sum(counts.values()) == 4049
    nucleus = {}
    for row in nucleus_rows:
        polys = json.loads(row["nucleus_polygons"]) if isinstance(row["nucleus_polygons"], str) else row["nucleus_polygons"]
        if polys:
            nucleus[str(row["crop_path"])] = {"w": int(row["crop_width"]), "h": int(row["crop_height"]), "polys": polys}

    manifest = find("sipakmed_cluster_split.csv")
    tests = []
    with open(manifest, newline="") as f:
        for r in csv.DictReader(f):
            if r["split"] == "test":
                tests.append((f'{r["class_name"]}/{Path(r["path"]).name}', int(r["class_idx"]), r["cluster_id"]))
    sel = [t for t in tests if t[0] in nucleus]
    print("test cells with nucleus polygon:", len(sel), flush=True)

    tf = build_transform(INPUT_SIZE, train=False)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for name in BACKBONES:
        model = build_model(name, pretrained=False)
        model.load_state_dict(torch.load(find(f"{name}.pt"), map_location="cpu"))
        model = model.to(device).eval()
        npt = getattr(model, "num_prefix_tokens", 1)

        # prepass: predictions for every cell, then choose the figure cells
        preds = []
        for rel, label, cluster in sel:
            x = tf(Image.open(crops_root / rel).convert("RGB")).unsqueeze(0).to(device)
            with torch.no_grad():
                preds.append(int(model(x).argmax(-1).item()))
        lab, prd = np.array([t[1] for t in sel]), np.array(preds)
        picks = [int(np.where((lab == c) & (prd == c))[0][0]) for c in range(5) if ((lab == c) & (prd == c)).any()]
        picks += [int(i) for i in np.where(lab != prd)[0][:3]]
        keep = {}

        rec = {m: {k: {b: [] for b in FILLS} for k in ("deletion", "insertion")} for m in METHODS}
        conc = {m: [] for m in METHODS}
        t0 = time.time()
        for i, (rel, label, cluster) in enumerate(sel):
            x = tf(Image.open(crops_root / rel).convert("RGB")).unsqueeze(0).to(device)
            pred = preds[i]
            maps = {
                "attention_rollout": attention_rollout(model, x, grid_size=GRID, num_prefix_tokens=npt)[0],
                "relevancy_propagation": relevancy_propagation(model, x, target_class=pred, grid_size=GRID, num_prefix_tokens=npt)[0],
                "gradcam": vit_gradcam(model, x, target_class=pred, grid_size=GRID, num_prefix_tokens=npt)[0],
            }
            n = nucleus[rel]
            mask = rasterize_nucleus_mask(n["polys"], n["w"], n["h"], INPUT_SIZE).to(device)
            for m, h in maps.items():
                conc[m].append(concentration_ratio(h, mask))
                for b, fill in FILLS.items():
                    FA._gaussian_blur = fill
                    rec[m]["deletion"][b].append(deletion_auc(model, x, h, target_class=pred, n_steps=N_STEPS))
                    rec[m]["insertion"][b].append(insertion_auc(model, x, h, target_class=pred, n_steps=N_STEPS))
                FA._gaussian_blur = _ORIG_BLUR
            if i in picks:
                keep[i] = {m: h.detach().cpu().numpy() for m, h in maps.items()}
            if (i + 1) % 50 == 0:
                print(f"[{name}] {i + 1}/{len(sel)} ({time.time() - t0:.0f}s)", flush=True)

        arrays = {f"{m}_concentration": np.array(conc[m]) for m in METHODS}
        for m in METHODS:
            for k in ("deletion", "insertion"):
                for b in FILLS:
                    arrays[f"{m}_{k}_{b}"] = np.array(rec[m][k][b])
        arrays.update(label=lab, pred=prd, cluster=np.array([t[2] for t in sel]), path=np.array([t[0] for t in sel]))
        np.savez_compressed(OUT / f"{name}_per_image.npz", **arrays)
        summary = {m: {**{"concentration": float(np.mean(conc[m]))},
                       **{f"{k}_{b}": float(np.mean(rec[m][k][b])) for k in ("deletion", "insertion") for b in FILLS}} for m in METHODS}
        (OUT / f"{name}_summary.json").write_text(json.dumps(summary, indent=2))
        print(name, json.dumps(summary), flush=True)

        # larger representative figure
        fig, ax = plt.subplots(len(picks), 5, figsize=(13, 2.9 * len(picks)))
        ax = np.atleast_2d(ax)
        titles = ["Cell and nucleus outline", "Attention rollout", "Grad-CAM", "Relevancy propagation", ""]
        for r, p in enumerate(picks):
            rel = sel[p][0]
            img = np.array(Image.open(crops_root / rel).convert("RGB").resize((INPUT_SIZE, INPUT_SIZE)))
            n = nucleus[rel]
            mk = rasterize_nucleus_mask(n["polys"], n["w"], n["h"], INPUT_SIZE).numpy()
            ax[r, 0].imshow(img); ax[r, 0].contour(mk, levels=[0.5], colors="lime", linewidths=1.6)
            for c, m in enumerate(["attention_rollout", "gradcam", "relevancy_propagation"], start=1):
                ax[r, c].imshow(img); ax[r, c].imshow(keep[p][m], cmap="jet", alpha=0.45, vmin=0, vmax=1)
            ax[r, 4].axis("off")
            ax[r, 4].text(0.02, 0.5, f"true: {CLASSES[lab[p]][3:]}\npred: {CLASSES[prd[p]][3:]}", fontsize=11, va="center")
            for c in range(4):
                ax[r, c].set_xticks([]); ax[r, c].set_yticks([])
                if r == 0:
                    ax[r, c].set_title(titles[c], fontsize=12)
        fig.tight_layout()
        fig.savefig(OUT / f"examples_{name}.png", dpi=130)
        plt.close(fig)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
