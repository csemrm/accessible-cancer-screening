"""Cluster-grouped re-run of the RQ1 benchmark + cross-dataset evaluation (seed 13).

Why: the original split shuffles SIPaKMeD cells at the cell level, so ~88% of
test cells share a source cluster image (field of view) with a training cell.
This kernel keeps every cluster image inside exactly one of
train/val/calibration/test, with the same recipe, hyperparameters, and seed as
the original run, and additionally saves what the original run did not keep:

  * per-image softmax probabilities for the in-distribution test, validation and
    calibration splits and for all three external datasets (-> bootstrap CIs,
    per-class sensitivity/specificity, confusion matrices, conformal re-runs),
  * model checkpoints (-> explainability re-runs),
  * software/hardware versions and per-epoch wall-clock time.

Everything is written to /kaggle/working/out. Results are saved after every
model so a timeout loses at most one model.
"""

from __future__ import annotations

import csv
import glob
import hashlib
import json
import platform
import random
import sys
import time
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

WORK = Path("/kaggle/working")
OUT = WORK / "out"
SEED = 13
EPOCHS, BATCH_SIZE, LR, PATIENCE = 30, 32, 3e-4, 7
FRACTIONS = (0.60, 0.15, 0.10, 0.15)

CLASSES = ["im_Dyskeratotic", "im_Koilocytotic", "im_Metaplastic", "im_Parabasal", "im_Superficial-Intermediate"]
NORMAL, ABNORMAL = "normal", "abnormal"
SIPAKMED_TO_BINARY = {"im_Parabasal": NORMAL, "im_Metaplastic": NORMAL, "im_Superficial-Intermediate": NORMAL,
                      "im_Koilocytotic": ABNORMAL, "im_Dyskeratotic": ABNORMAL}
HERLEV_TO_BINARY = {"normal_columnar": NORMAL, "normal_intermediate": NORMAL, "normal_superficiel": NORMAL,
                    "light_dysplastic": ABNORMAL, "moderate_dysplastic": ABNORMAL,
                    "severe_dysplastic": ABNORMAL, "carcinoma_in_situ": ABNORMAL}
MENDELEY_LBC_TO_BINARY = {"Negative for Intraepithelial malignancy": NORMAL,
                          "Low squamous intra-epithelial lesion": ABNORMAL,
                          "High squamous intra-epithelial lesion": ABNORMAL,
                          "Squamous cell carcinoma": ABNORMAL}
APCEM_TO_BINARY = {"NILM": NORMAL, "ASCUS": ABNORMAL, "ASCH": ABNORMAL, "LSIL": ABNORMAL, "HSIL": ABNORMAL,
                   "SSC": ABNORMAL}
APCEM_URL = ("https://data.mendeley.com/public-files/datasets/snzb5g6ddy/files/"
             "3ce39216-8d7f-4016-817f-a3a773180e98/file_downloaded")
APCEM_SHA256 = "0e6a049c830dc094e944a01bff70c447832f2b3e3f79f5b8f3f7104c4fede154"
EXPECTED_N = {"sipakmed": 4049, "herlev": 917, "mendeley_lbc": 962, "apcem": 4253}

NUM_CLASSES = 5
MODEL_REGISTRY = {
    "resnet50": "resnet50.a1_in1k",
    "mobilenetv2_100": "mobilenetv2_100.ra_in1k",
    "efficientnet_lite0": "efficientnet_lite0.ra_in1k",
    "vit_small_patch16_224": "vit_small_patch16_224.augreg_in1k",
    "swin_tiny_patch4_window7_224": "swin_tiny_patch4_window7_224.ms_in1k",
    "deit_tiny_patch16_224": "deit_tiny_patch16_224.fb_in1k",
    "levit_128": "levit_128.fb_dist_in1k",
    "tiny_vit_5m_224": "tiny_vit_5m_224.in1k",
    "mobilevit_s": "mobilevit_s.cvnets_in1k",
    "mobilevitv2_100": "mobilevitv2_100.cvnets_in1k",
    "efficientvit_b0": "efficientvit_b0.r224_in1k",
    "efficientvit_m0": "efficientvit_m0.r224_in1k",
}
IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


# ---------------------------------------------------------------------------
# data acquisition
# ---------------------------------------------------------------------------

def find_dir(pattern: str) -> Path:
    hits = glob.glob(f"/kaggle/input/*/{pattern}") or glob.glob(f"/kaggle/input/**/{pattern}", recursive=True)
    if not hits:
        raise FileNotFoundError(f"{pattern} not found under /kaggle/input (is the dataset attached?)")
    return Path(hits[0])


def download_apcem(dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    zip_path = dest / "apcem.zip"
    if not zip_path.exists():
        urllib.request.urlretrieve(APCEM_URL, zip_path)
    digest = hashlib.sha256(zip_path.read_bytes()).hexdigest()
    assert digest == APCEM_SHA256, f"APCEM zip hash mismatch: {digest}"
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(dest / "extracted")
    return dest / "extracted"


def scan_binary_manifest(root: Path, class_to_binary: dict, extensions=(".jpg", ".jpeg", ".png", ".bmp")):
    rows = []
    for class_name, binary_label in class_to_binary.items():
        class_dirs = [root / class_name] if (root / class_name).exists() else []
        if not class_dirs:
            class_dirs = [d for d in root.glob(f"**/{class_name}") if d.is_dir()]
        if not class_dirs:
            raise FileNotFoundError(f"class directory not found under {root}: {class_name}")
        images = [p for d in class_dirs for p in d.rglob("*") if p.suffix.lower() in extensions]
        if not images:
            raise FileNotFoundError(f"no images for {class_name}")
        rows.extend((p, binary_label) for p in sorted(images))
    return rows


def read_polygon(path: Path):
    pts = []
    with open(path, newline="") as f:
        for row in csv.reader(f):
            if len(row) >= 2:
                pts.append((float(row[0]), float(row[1])))
    return pts


def crop_all(raw_root: Path, out_root: Path, margin: float = 4.0) -> dict:
    counts = {}
    for class_dir in CLASSES:
        src_dir = raw_root / class_dir / class_dir
        out_dir = out_root / class_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        n = 0
        for bmp_path in sorted(src_dir.glob("*.bmp")):
            cluster_id = bmp_path.stem
            cyt_files = sorted(src_dir.glob(f"{cluster_id}_cyt*.dat"))
            if not cyt_files:
                continue
            image = Image.open(bmp_path).convert("RGB")
            w, h = image.size
            for cyt_path in cyt_files:
                cell_idx = cyt_path.stem.split("_cyt")[-1]
                pts = read_polygon(cyt_path)
                if len(pts) < 3:
                    continue
                xs, ys = [p[0] for p in pts], [p[1] for p in pts]
                left, top = max(0, int(min(xs) - margin)), max(0, int(min(ys) - margin))
                right, bottom = min(w, int(max(xs) + margin)), min(h, int(max(ys) + margin))
                if right <= left or bottom <= top:
                    continue
                image.crop((left, top, right, bottom)).save(out_dir / f"{cluster_id}_{cell_idx}.png")
                n += 1
        counts[class_dir] = n
    return counts


# ---------------------------------------------------------------------------
# cluster-grouped split
# ---------------------------------------------------------------------------

def cluster_of(path: Path) -> str:
    return path.stem.split("_")[0]


def grouped_split(paths, fractions, seed):
    """Whole cluster images are assigned to one split, filling train, val,
    calibration and test in order until each reaches its cumulative cell target
    (so realised fractions can overshoot by at most one cluster)."""
    clusters = defaultdict(list)
    for p in paths:
        clusters[cluster_of(p)].append(p)
    ids = sorted(clusters)
    random.Random(seed).shuffle(ids)
    n = len(paths)
    cum = np.cumsum(fractions) * n
    names = ["train", "val", "calibration", "test"]
    splits = {k: [] for k in names}
    s, filled = 0, 0
    for cid in ids:
        splits[names[s]].extend(clusters[cid])
        filled += len(clusters[cid])
        if s < 3 and filled >= cum[s]:
            s += 1
    return splits


def build_manifest(crops_root: Path, out_csv: Path):
    rows, stats = [], {}
    for class_idx, class_name in enumerate(CLASSES):
        images = sorted((crops_root / class_name).glob("*.png"))
        assert images, class_name
        splits = grouped_split(images, FRACTIONS, SEED + class_idx)
        for split_name, ps in splits.items():
            assert ps, f"empty {split_name} split for {class_name}"
            rows.extend((str(p), class_name, class_idx, split_name, cluster_of(p)) for p in ps)
        stats[class_name] = {k: len(v) for k, v in splits.items()}
    # no cluster image may appear in two splits
    owner = {}
    for path, cls, _, split, cid in rows:
        key = (cls, cid)
        assert owner.setdefault(key, split) == split, f"cluster {key} leaks across splits"
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "class_name", "class_idx", "split", "cluster_id"])
        w.writerows(rows)
    return stats


# ---------------------------------------------------------------------------
# datasets / model
# ---------------------------------------------------------------------------

def build_transform(size: int, train: bool):
    if train:
        return transforms.Compose([
            transforms.Resize((size, size)), transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20), transforms.ColorJitter(brightness=0.15, contrast=0.15),
            transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    return transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor(),
                               transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


class ManifestDataset(Dataset):
    def __init__(self, manifest_csv: Path, split: str, tf):
        self.tf, self.samples = tf, []
        with open(manifest_csv, newline="") as f:
            for r in csv.DictReader(f):
                if r["split"] == split:
                    self.samples.append((Path(r["path"]), int(r["class_idx"])))
        assert self.samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        p, y = self.samples[i]
        return self.tf(Image.open(p).convert("RGB")), y


class BinaryDataset(Dataset):
    def __init__(self, rows, tf):
        self.rows, self.tf = rows, tf

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        p, lab = self.rows[i]
        return self.tf(Image.open(p).convert("RGB")), 0 if lab == NORMAL else 1


def build_model(name: str, pretrained: bool = True) -> nn.Module:
    return timm.create_model(MODEL_REGISTRY[name], pretrained=pretrained, num_classes=NUM_CLASSES)


def input_size_for(name: str) -> int:
    cfg = timm.get_pretrained_cfg(MODEL_REGISTRY[name])
    return cfg.input_size[-1] if cfg is not None and cfg.input_size else 224


def select_device() -> torch.device:
    if not torch.cuda.is_available():
        return torch.device("cpu")
    try:
        x = torch.randn(8, 8, device="cuda")
        (x @ x).sum().item()
    except RuntimeError as e:
        print(f"CUDA op failed ({e})", flush=True)
        return torch.device("cpu")
    return torch.device("cuda")


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

_ABN = [i for i, c in enumerate(CLASSES) if SIPAKMED_TO_BINARY[c] == ABNORMAL]
_NOR = [i for i, c in enumerate(CLASSES) if SIPAKMED_TO_BINARY[c] == NORMAL]


def to_binary(p5: np.ndarray) -> np.ndarray:
    return np.stack([p5[:, _NOR].sum(axis=1), p5[:, _ABN].sum(axis=1)], axis=1)


def per_class_sens_spec(y_true, y_pred, k):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(k)))
    tot = cm.sum()
    out = []
    for c in range(k):
        tp = cm[c, c]; fn = cm[c].sum() - tp; fp = cm[:, c].sum() - tp; tn = tot - tp - fn - fp
        out.append({"sensitivity": float(tp / (tp + fn)) if tp + fn else None,
                    "specificity": float(tn / (tn + fp)) if tn + fp else None})
    return cm.tolist(), out


def compute_metrics(y_true, y_prob) -> dict:
    k = y_prob.shape[1]
    y_pred = y_prob.argmax(axis=1)
    cm, per_class = per_class_sens_spec(y_true, y_pred, k)
    m = {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall_sensitivity": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": cm, "per_class": per_class,
    }
    try:
        oh = np.eye(k)[y_true]
        m["macro_auroc"] = float(roc_auc_score(oh, y_prob, average="macro", multi_class="ovr"))
        m["macro_auprc"] = float(average_precision_score(oh, y_prob, average="macro"))
        if k == 2:
            m["auroc_binary"] = float(roc_auc_score(y_true, y_prob[:, 1]))
    except ValueError:
        pass
    return m


# ---------------------------------------------------------------------------
# train / predict
# ---------------------------------------------------------------------------

def run_epoch(model, loader, device, optimizer=None) -> float:
    train = optimizer is not None
    model.train(train)
    crit = nn.CrossEntropyLoss()
    tot, n = 0.0, 0
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            loss = crit(model(x), y)
            if train:
                optimizer.zero_grad(); loss.backward(); optimizer.step()
            tot += loss.item() * x.size(0); n += x.size(0)
    return tot / n


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    probs, labels = [], []
    for x, y in loader:
        probs.append(torch.softmax(model(x.to(device)), dim=1).cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(probs), np.concatenate(labels)


def train_and_eval(name, manifest_csv, device, ext_sets):
    torch.manual_seed(SEED)  # same (minimal) seeding as the original pipeline
    size = input_size_for(name)
    ds = {s: ManifestDataset(manifest_csv, s, build_transform(size, s == "train"))
          for s in ("train", "val", "calibration", "test")}
    loaders = {s: DataLoader(d, batch_size=BATCH_SIZE, shuffle=(s == "train"), num_workers=2) for s, d in ds.items()}
    model = build_model(name).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    best, best_state, bad, history = float("inf"), None, 0, []
    t_start = time.time()
    for epoch in range(EPOCHS):
        t0 = time.time()
        tr = run_epoch(model, loaders["train"], device, opt)
        va = run_epoch(model, loaders["val"], device)
        sched.step()
        history.append({"epoch": epoch, "train_loss": tr, "val_loss": va, "seconds": time.time() - t0})
        print(f"[{name}] epoch {epoch:03d} train={tr:.4f} val={va:.4f} ({history[-1]['seconds']:.1f}s)", flush=True)
        if va < best:
            best, bad = va, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                print(f"[{name}] early stop at epoch {epoch}", flush=True)
                break
    model.load_state_dict(best_state)
    torch.save(model.state_dict(), OUT / "checkpoints" / f"{name}.pt")

    arrays, metrics = {}, {}
    for split in ("val", "calibration", "test"):
        p, y = predict(model, loaders[split], device)
        arrays[f"{split}_probs"], arrays[f"{split}_labels"] = p, y
    metrics["in_distribution_test"] = compute_metrics(arrays["test_labels"], arrays["test_probs"])
    tf = build_transform(size, False)
    for ext_name, rows in ext_sets.items():
        loader = DataLoader(BinaryDataset(rows, tf), batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
        p5, y = predict(model, loader, device)
        arrays[f"{ext_name}_probs5"], arrays[f"{ext_name}_labels"] = p5, y
        metrics[ext_name] = compute_metrics(y, to_binary(p5))
    np.savez_compressed(OUT / "preds" / f"{name}.npz", **arrays)
    return {"model": name, "seed": SEED, "best_val_loss": best, "epochs_run": len(history),
            "train_seconds": time.time() - t_start, "history": history, "metrics": metrics,
            "n": {s: len(d) for s, d in ds.items()}}


def main() -> None:
    for sub in ("checkpoints", "preds"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)
    device = select_device()
    env = {"python": sys.version, "platform": platform.platform(), "torch": torch.__version__, "timm": timm.__version__,
           "numpy": np.__version__, "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
           "device": str(device), "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
           "split": "cluster-grouped", "seed": SEED, "epochs": EPOCHS, "batch_size": BATCH_SIZE, "lr": LR}
    (OUT / "env.json").write_text(json.dumps(env, indent=2))
    print(json.dumps(env, indent=2), flush=True)
    if device.type != "cuda":
        raise SystemExit("no usable CUDA device: refusing to train 12 models on CPU")

    raw_root = find_dir("im_Dyskeratotic").parent
    crops_root = WORK / "crops"
    counts = crop_all(raw_root, crops_root)
    print("crop counts:", counts, flush=True)
    assert sum(counts.values()) == EXPECTED_N["sipakmed"], counts

    manifest_csv = OUT / "sipakmed_cluster_split.csv"
    split_stats = build_manifest(crops_root, manifest_csv)
    (OUT / "split_stats.json").write_text(json.dumps(split_stats, indent=2))
    print("cluster-grouped split sizes per class:", json.dumps(split_stats), flush=True)

    ext_sets = {
        "herlev": scan_binary_manifest(find_dir("normal_columnar").parents[1], HERLEV_TO_BINARY),
        "mendeley_lbc": scan_binary_manifest(find_dir("Negative for Intraepithelial malignancy").parent,
                                              MENDELEY_LBC_TO_BINARY),
        "apcem": scan_binary_manifest(download_apcem(WORK / "apcem"), APCEM_TO_BINARY),
    }
    for k, rows in ext_sets.items():
        print(k, len(rows), flush=True)
        assert len(rows) == EXPECTED_N[k], (k, len(rows))
    (OUT / "external_paths.json").write_text(json.dumps(
        {k: [[str(p), lab] for p, lab in rows] for k, rows in ext_sets.items()}))

    results = []
    for name in MODEL_REGISTRY:
        print(f"=== {name} ===", flush=True)
        try:
            results.append(train_and_eval(name, manifest_csv, device, ext_sets))
        except Exception as e:  # keep going; record the failure
            results.append({"model": name, "error": repr(e)})
            print(f"[{name}] FAILED: {e!r}", flush=True)
        (OUT / "results.json").write_text(json.dumps(results, indent=2))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
