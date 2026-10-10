"""Explainability re-run on the cluster-grouped checkpoints (DeiT-Tiny and ViT-Small).

Uses the checkpoints saved by kaggle_kernel/cluster_split_rerun/run.py (attached as a kernel source) and ALL test cells
of the cluster-grouped split that have an expert nucleus polygon (the original run used 50 cells). For every cell and
each of three methods (relevancy propagation, attention rollout, ViT Grad-CAM) it saves deletion AUC, insertion AUC and
nucleus concentration ratio, so paired bootstrap intervals and paired tests can be computed offline. It also saves a
figure of representative correctly and incorrectly classified cells with all three heatmaps.
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

# --- import the project package from the attached private dataset
pkg = glob.glob("/kaggle/input/*/cervical_screening/__init__.py") or glob.glob(
    "/kaggle/input/**/cervical_screening/__init__.py", recursive=True)
assert pkg, "cervical_screening package not found under /kaggle/input"
sys.path.insert(0, str(Path(pkg[0]).parent.parent))

from PIL import Image  # noqa: E402

from cervical_screening.data.crop_sipakmed import CLASSES, crop_class  # noqa: E402
from cervical_screening.data.transforms import build_transform  # noqa: E402
from cervical_screening.explainability.attention_rollout import attention_rollout  # noqa: E402
from cervical_screening.explainability.faithfulness import deletion_auc, insertion_auc  # noqa: E402
from cervical_screening.explainability.gradcam import vit_gradcam  # noqa: E402
from cervical_screening.explainability.nucleus_overlap import concentration_ratio, rasterize_nucleus_mask  # noqa: E402
from cervical_screening.explainability.relevancy_propagation import relevancy_propagation  # noqa: E402
from cervical_screening.models.registry import build_model  # noqa: E402

WORK = Path("/kaggle/working")
OUT = WORK / "out_xai"
OUT.mkdir(parents=True, exist_ok=True)
INPUT_SIZE, GRID = 224, 14
BACKBONES = ["deit_tiny_patch16_224", "vit_small_patch16_224"]
METHODS = ["relevancy_propagation", "attention_rollout", "gradcam"]
N_STEPS = 20


def find(pattern: str) -> Path:
    hits = glob.glob(f"/kaggle/input/**/{pattern}", recursive=True)
    assert hits, f"{pattern} not found under /kaggle/input"
    return Path(hits[0])


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise SystemExit("no CUDA device")
    env = {"torch": torch.__version__, "gpu": torch.cuda.get_device_name(0), "n_steps": N_STEPS}
    (OUT / "env.json").write_text(json.dumps(env, indent=2))

    # --- re-crop with nucleus polygons (same cropping code as the training run)
    raw_root = find("im_Dyskeratotic").parent
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
        rec = {m: {"deletion": [], "insertion": [], "concentration": []} for m in METHODS}
        meta = {"path": [], "label": [], "pred": [], "cluster": []}
        keep_maps = {}
        t0 = time.time()
        for i, (rel, label, cluster) in enumerate(sel):
            pil = Image.open(crops_root / rel).convert("RGB")
            x = tf(pil).unsqueeze(0).to(device)
            with torch.no_grad():
                pred = int(model(x).argmax(-1).item())
            maps = {
                "attention_rollout": attention_rollout(model, x, grid_size=GRID, num_prefix_tokens=npt)[0],
                "relevancy_propagation": relevancy_propagation(model, x, target_class=pred, grid_size=GRID,
                                                               num_prefix_tokens=npt)[0],
                "gradcam": vit_gradcam(model, x, target_class=pred, grid_size=GRID, num_prefix_tokens=npt)[0],
            }
            n = nucleus[rel]
            mask = rasterize_nucleus_mask(n["polys"], n["w"], n["h"], INPUT_SIZE).to(device)
            for m, h in maps.items():
                rec[m]["deletion"].append(deletion_auc(model, x, h, target_class=pred, n_steps=N_STEPS))
                rec[m]["insertion"].append(insertion_auc(model, x, h, target_class=pred, n_steps=N_STEPS))
                rec[m]["concentration"].append(concentration_ratio(h, mask))
            meta["path"].append(rel); meta["label"].append(label); meta["pred"].append(pred); meta["cluster"].append(cluster)
            if len(keep_maps) < 400:  # keep heatmaps only for candidate figure cells
                keep_maps[i] = {m: h.detach().cpu().numpy() for m, h in maps.items()}
            if (i + 1) % 50 == 0:
                print(f"[{name}] {i + 1}/{len(sel)} ({time.time() - t0:.0f}s)", flush=True)

        arrays = {f"{m}_{k}": np.array(v) for m, d in rec.items() for k, v in d.items()}
        arrays.update(label=np.array(meta["label"]), pred=np.array(meta["pred"]), cluster=np.array(meta["cluster"]),
                      path=np.array(meta["path"]))
        np.savez_compressed(OUT / f"{name}_per_image.npz", **arrays)
        summary = {m: {k: float(np.mean(v)) for k, v in d.items()} for m, d in rec.items()}
        summary["n"] = len(sel); summary["accuracy"] = float((np.array(meta["pred"]) == np.array(meta["label"])).mean())
        (OUT / f"{name}_summary.json").write_text(json.dumps(summary, indent=2))
        print(name, json.dumps(summary), flush=True)

        # --- representative figure: first correct cell of each class + first three misclassified cells
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        lab, prd = np.array(meta["label"]), np.array(meta["pred"])
        picks = [int(np.where((lab == c) & (prd == c))[0][0]) for c in range(5) if ((lab == c) & (prd == c)).any()]
        picks += [int(i) for i in np.where(lab != prd)[0][:3]]
        picks = [p for p in picks if p in keep_maps]
        fig, ax = plt.subplots(len(picks), 5, figsize=(10, 2.1 * len(picks)))
        ax = np.atleast_2d(ax)
        titles = ["Cell + nucleus", "Attention rollout", "Grad-CAM", "Relevancy propagation", "Predicted"]
        for r, p in enumerate(picks):
            rel = meta["path"][p]
            img = np.array(Image.open(crops_root / rel).convert("RGB").resize((INPUT_SIZE, INPUT_SIZE)))
            n = nucleus[rel]
            mk = rasterize_nucleus_mask(n["polys"], n["w"], n["h"], INPUT_SIZE).numpy()
            ax[r, 0].imshow(img); ax[r, 0].contour(mk, levels=[0.5], colors="lime", linewidths=1.2)
            for c, m in enumerate(["attention_rollout", "gradcam", "relevancy_propagation"], start=1):
                ax[r, c].imshow(img); ax[r, c].imshow(keep_maps[p][m], cmap="jet", alpha=0.45)
            ax[r, 4].axis("off")
            ax[r, 4].text(0.02, 0.5, f"true: {CLASSES[meta['label'][p]][3:]}\npred: {CLASSES[meta['pred'][p]][3:]}", fontsize=8,
                          va="center")
            for c in range(4):
                ax[r, c].set_xticks([]); ax[r, c].set_yticks([])
                if r == 0:
                    ax[r, c].set_title(titles[c], fontsize=9)
        fig.suptitle(name, fontsize=10)
        fig.tight_layout()
        fig.savefig(OUT / f"examples_{name}.png", dpi=150)
        plt.close(fig)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
