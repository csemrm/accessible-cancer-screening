"""Regenerates only the representative explainability figure (examples_<backbone>.png) at higher resolution and
larger label/title text, for the JMI submission's Figure 1. Same checkpoints and same cell-selection rule as
kaggle_kernel/cluster_split_explain (first correctly classified cell of each class, first 3 misclassified cells,
among test cells with an expert nucleus polygon), but skips the deletion/insertion/concentration metric
recomputation entirely -- those numbers are unchanged and already reported in the paper, so this kernel only
needs a single forward pass per cell (to find the same picks) plus the three heatmap methods on the handful of
picked cells. This is not a new experiment, only a presentation fix requested by peer review (larger figure,
dpi 150 -> 300, title/label font roughly 30% larger).
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

# --- import the project package from the attached private dataset. Kaggle extracts the zipped package folder to the
# dataset root, so the package files sit directly under /kaggle/input/<slug>/ ; link that root as `cervical_screening`.
hits = glob.glob("/kaggle/input/**/explainability/relevancy_propagation.py", recursive=True)
assert hits, "cervical_screening package files not found under /kaggle/input"
pkg_root = Path(hits[0]).parent.parent
link_dir = Path("/kaggle/working/pkg")
link_dir.mkdir(parents=True, exist_ok=True)
if not (link_dir / "cervical_screening").exists():
    (link_dir / "cervical_screening").symlink_to(pkg_root, target_is_directory=True)
sys.path.insert(0, str(link_dir))

from PIL import Image  # noqa: E402

from cervical_screening.data.crop_sipakmed import CLASSES, crop_class  # noqa: E402
from cervical_screening.data.transforms import build_transform  # noqa: E402
from cervical_screening.explainability.attention_rollout import attention_rollout  # noqa: E402
from cervical_screening.explainability.gradcam import vit_gradcam  # noqa: E402
from cervical_screening.explainability.nucleus_overlap import rasterize_nucleus_mask  # noqa: E402
from cervical_screening.explainability.relevancy_propagation import relevancy_propagation  # noqa: E402
from cervical_screening.models.registry import build_model  # noqa: E402

WORK = Path("/kaggle/working")
OUT = WORK / "out_xai"
OUT.mkdir(parents=True, exist_ok=True)
INPUT_SIZE, GRID = 224, 14
BACKBONES = ["deit_tiny_patch16_224", "vit_small_patch16_224"]


def find(pattern: str) -> Path:
    hits = glob.glob(f"/kaggle/input/**/{pattern}", recursive=True)
    assert hits, f"{pattern} not found under /kaggle/input"
    return Path(hits[0])


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise SystemExit("no CUDA device")
    env = {"torch": torch.__version__, "gpu": torch.cuda.get_device_name(0)}
    (OUT / "env.json").write_text(json.dumps(env, indent=2))

    # --- re-crop with nucleus polygons (same cropping code as the training run)
    # the earlier kernel's output also has crops/im_Dyskeratotic, so require the raw layout <root>/<class>/<class>/*.bmp
    cands = [Path(h).parent for h in glob.glob("/kaggle/input/**/im_Dyskeratotic", recursive=True)]
    cands = [c for c in cands if (c / "im_Dyskeratotic" / "im_Dyskeratotic").exists()]
    assert cands, "raw SIPaKMeD layout not found"
    raw_root = cands[0]
    crops_root = WORK / "crops"
    nucleus_rows: list[dict] = []
    counts = {c: crop_class(raw_root, crops_root, c, margin=4.0, nucleus_rows=nucleus_rows) for c in CLASSES}
    print("crop counts:", counts, flush=True)
    assert sum(counts.values()) == 4049
    nucleus = {}
    for row in nucleus_rows:
        polys = json.loads(row["nucleus_polygons"]) if isinstance(row["nucleus_polygons"], str) else row["nucleus_polygons"]
        if polys:
            nucleus[str(row["crop_path"])] = {"w": int(row["crop_width"]), "h": int(row["crop_height"]), "polys": polys}
    print("crops with nucleus polygons:", len(nucleus), flush=True)

    # --- cluster-grouped test split from the earlier run, re-mapped to the new crop paths
    manifest = find("sipakmed_cluster_split.csv")
    tests = []
    with open(manifest, newline="") as f:
        for r in csv.DictReader(f):
            if r["split"] == "test":
                rel = f'{r["class_name"]}/{Path(r["path"]).name}'
                tests.append((rel, int(r["class_idx"]), r["cluster_id"]))
    print("test cells:", len(tests), flush=True)
    sel = [t for t in tests if t[0] in nucleus]
    print("test cells with nucleus polygon:", len(sel), flush=True)

    tf = build_transform(INPUT_SIZE, train=False)
    for name in BACKBONES:
        ckpt = find(f"{name}.pt")
        model = build_model(name, pretrained=False)
        model.load_state_dict(torch.load(ckpt, map_location="cpu"))
        model = model.to(device).eval()
        npt = getattr(model, "num_prefix_tokens", 1)
        meta = {"path": [], "label": [], "pred": [], "cluster": []}
        t0 = time.time()
        for i, (rel, label, cluster) in enumerate(sel):
            pil = Image.open(crops_root / rel).convert("RGB")
            x = tf(pil).unsqueeze(0).to(device)
            with torch.no_grad():
                pred = int(model(x).argmax(-1).item())
            meta["path"].append(rel); meta["label"].append(label); meta["pred"].append(pred); meta["cluster"].append(cluster)
            if (i + 1) % 100 == 0:
                print(f"[{name}] pred pass {i + 1}/{len(sel)} ({time.time() - t0:.0f}s)", flush=True)
        print(name, "accuracy:", float((np.array(meta["pred"]) == np.array(meta["label"])).mean()), flush=True)

        # --- representative figure: first correct cell of each class + first three misclassified cells
        lab, prd = np.array(meta["label"]), np.array(meta["pred"])
        picks = [int(np.where((lab == c) & (prd == c))[0][0]) for c in range(5) if ((lab == c) & (prd == c)).any()]
        picks += [int(i) for i in np.where(lab != prd)[0][:3]]
        keep_maps = {}
        for p in picks:
            rel = meta["path"][p]
            pil = Image.open(crops_root / rel).convert("RGB")
            x = tf(pil).unsqueeze(0).to(device)
            pred = meta["pred"][p]
            maps = {
                "attention_rollout": attention_rollout(model, x, grid_size=GRID, num_prefix_tokens=npt)[0],
                "relevancy_propagation": relevancy_propagation(model, x, target_class=pred, grid_size=GRID,
                                                               num_prefix_tokens=npt)[0],
                "gradcam": vit_gradcam(model, x, target_class=pred, grid_size=GRID, num_prefix_tokens=npt)[0],
            }
            keep_maps[p] = {m: h.detach().cpu().numpy() for m, h in maps.items()}
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(len(picks), 5, figsize=(13, 2.8 * len(picks)))
        ax = np.atleast_2d(ax)
        titles = ["Cell + nucleus", "Attention rollout", "Grad-CAM", "Relevancy propagation", "Predicted"]
        for r, p in enumerate(picks):
            rel = meta["path"][p]
            img = np.array(Image.open(crops_root / rel).convert("RGB").resize((INPUT_SIZE, INPUT_SIZE)))
            n = nucleus[rel]
            mk = rasterize_nucleus_mask(n["polys"], n["w"], n["h"], INPUT_SIZE).numpy()
            ax[r, 0].imshow(img); ax[r, 0].contour(mk, levels=[0.5], colors="lime", linewidths=1.6)
            for c, m in enumerate(["attention_rollout", "gradcam", "relevancy_propagation"], start=1):
                ax[r, c].imshow(img); ax[r, c].imshow(keep_maps[p][m], cmap="jet", alpha=0.45)
            ax[r, 4].axis("off")
            ax[r, 4].text(0.02, 0.5, f"true: {CLASSES[meta['label'][p]][3:]}\npred: {CLASSES[meta['pred'][p]][3:]}", fontsize=12,
                          va="center")
            for c in range(4):
                ax[r, c].set_xticks([]); ax[r, c].set_yticks([])
                if r == 0:
                    ax[r, c].set_title(titles[c], fontsize=13)
        fig.suptitle(name, fontsize=14)
        fig.tight_layout()
        fig.savefig(OUT / f"examples_{name}.png", dpi=300)
        plt.close(fig)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
