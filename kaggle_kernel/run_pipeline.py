"""Self-contained Kaggle Kernel script: crop SIPaKMeD at native resolution,
build splits, and train the RQ1 baseline registry — all on Kaggle's own GPU
so we never pull the 6.4GB raw dataset down to a local machine.

Mirrors the logic in src/cervical_screening/ (crop_sipakmed.py, make_splits.py,
dataset.py, models/registry.py, train.py) but inlined into one file since
Kaggle scripts run without the rest of this repo available. Keep the two in
sync by hand if either changes.
"""

from __future__ import annotations

import csv
import glob
import json
import random
import time
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

# ---------------------------------------------------------------------------
# 0. Locate the mounted dataset — don't hardcode the /kaggle/input/<slug>/
#    folder name, since it depends on how the dataset was attached.
# ---------------------------------------------------------------------------

CLASSES = [
    "im_Dyskeratotic",
    "im_Koilocytotic",
    "im_Metaplastic",
    "im_Parabasal",
    "im_Superficial-Intermediate",
]


def find_raw_root() -> Path:
    candidates = glob.glob("/kaggle/input/*/im_Dyskeratotic")
    if not candidates:
        candidates = glob.glob("/kaggle/input/**/im_Dyskeratotic", recursive=True)
    if not candidates:
        raise FileNotFoundError(
            "could not find im_Dyskeratotic under /kaggle/input — "
            "check the dataset was attached to this kernel"
        )
    return Path(candidates[0]).parent


# ---------------------------------------------------------------------------
# 1. Crop native-resolution single cells from cytoplasm polygon bounding boxes
# ---------------------------------------------------------------------------

def read_polygon(path: Path) -> list[tuple[float, float]]:
    points = []
    with open(path, newline="") as f:
        for row in csv.reader(f):
            if len(row) < 2:
                continue
            points.append((float(row[0]), float(row[1])))
    return points


def bbox(points, margin: float):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return int(min(xs) - margin), int(min(ys) - margin), int(max(xs) + margin), int(max(ys) + margin)


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
                points = read_polygon(cyt_path)
                if len(points) < 3:
                    continue
                left, top, right, bottom = bbox(points, margin)
                left, top = max(0, left), max(0, top)
                right, bottom = min(w, right), min(h, bottom)
                if right <= left or bottom <= top:
                    continue
                image.crop((left, top, right, bottom)).save(out_dir / f"{cluster_id}_{cell_idx}.png")
                n += 1
        counts[class_dir] = n
    return counts


# ---------------------------------------------------------------------------
# 2. Stratified train/val/calibration/test split
# ---------------------------------------------------------------------------

def stratified_split(paths, fractions, seed):
    rng = random.Random(seed)
    shuffled = paths[:]
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = round(n * fractions[0])
    n_val = round(n * fractions[1])
    n_cal = round(n * fractions[2])
    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train:n_train + n_val],
        "calibration": shuffled[n_train + n_val:n_train + n_val + n_cal],
        "test": shuffled[n_train + n_val + n_cal:],
    }


def build_manifest(crops_root: Path, out_csv: Path, seed: int = 13,
                    fractions=(0.60, 0.15, 0.10, 0.15)) -> None:
    rows = []
    for class_idx, class_name in enumerate(CLASSES):
        images = sorted((crops_root / class_name).glob("*.png"))
        if not images:
            raise FileNotFoundError(f"no crops found under {crops_root / class_name}")
        splits = stratified_split(images, fractions, seed=seed + class_idx)
        for split_name, split_paths in splits.items():
            for p in split_paths:
                rows.append((str(p), class_name, class_idx, split_name))

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "class_name", "class_idx", "split"])
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# 3. Dataset / transforms
# ---------------------------------------------------------------------------

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ManifestDataset(Dataset):
    def __init__(self, manifest_csv: Path, split: str, transform):
        self.transform = transform
        self.samples = []
        with open(manifest_csv, newline="") as f:
            for row in csv.DictReader(f):
                if row["split"] == split:
                    self.samples.append((Path(row["path"]), int(row["class_idx"])))
        if not self.samples:
            raise ValueError(f"no rows for split={split!r}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        return self.transform(image), label


def build_transform(input_size: int, train: bool):
    if train:
        return transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


# ---------------------------------------------------------------------------
# 4. Model registry (same tags as src/cervical_screening/models/registry.py)
# ---------------------------------------------------------------------------

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


def build_model(name: str) -> nn.Module:
    tag = MODEL_REGISTRY[name]
    return timm.create_model(tag, pretrained=True, num_classes=NUM_CLASSES)


def input_size_for(name: str) -> int:
    tag = MODEL_REGISTRY[name]
    cfg = timm.get_pretrained_cfg(tag)
    return cfg.input_size[-1] if cfg is not None and cfg.input_size else 224


# ---------------------------------------------------------------------------
# 5. Metrics + efficiency
# ---------------------------------------------------------------------------

def specificity_per_class(y_true, y_pred, num_classes):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    total = cm.sum()
    spec = np.zeros(num_classes)
    for c in range(num_classes):
        tp = cm[c, c]
        fn = cm[c, :].sum() - tp
        fp = cm[:, c].sum() - tp
        tn = total - tp - fn - fp
        spec[c] = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    return spec


def compute_metrics(y_true, y_prob):
    num_classes = y_prob.shape[1]
    y_pred = y_prob.argmax(axis=1)
    metrics = {
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall_sensitivity": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_specificity": float(np.nanmean(specificity_per_class(y_true, y_pred, num_classes))),
    }
    try:
        y_onehot = np.eye(num_classes)[y_true]
        metrics["macro_auroc"] = roc_auc_score(y_onehot, y_prob, average="macro", multi_class="ovr")
        metrics["macro_auprc"] = average_precision_score(y_onehot, y_prob, average="macro")
    except ValueError:
        metrics["macro_auroc"] = float("nan")
        metrics["macro_auprc"] = float("nan")
    return metrics


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def measure_cpu_latency(model, input_size, n_warmup=5, n_iters=20):
    model = model.eval().to("cpu")
    x = torch.randn(1, 3, input_size, input_size)
    for _ in range(n_warmup):
        model(x)
    times = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        model(x)
        times.append(time.perf_counter() - t0)
    times.sort()
    return {"median_latency_ms": times[len(times) // 2] * 1000}


# ---------------------------------------------------------------------------
# 6. Train / eval loop
# ---------------------------------------------------------------------------

def run_epoch(model, loader, device, optimizer=None):
    train_mode = optimizer is not None
    model.train(train_mode)
    criterion = nn.CrossEntropyLoss()
    total_loss, n = 0.0, 0
    with torch.set_grad_enabled(train_mode):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * images.size(0)
            n += images.size(0)
    return total_loss / n


@torch.no_grad()
def predict_probs(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []
    for images, labels in loader:
        logits = model(images.to(device))
        all_probs.append(torch.softmax(logits, dim=1).cpu().numpy())
        all_labels.append(labels.numpy())
    return np.concatenate(all_probs), np.concatenate(all_labels)


def train_one(model_name: str, manifest_csv: Path, device, epochs: int, batch_size: int,
              lr: float, patience: int, seed: int) -> dict:
    torch.manual_seed(seed)
    input_size = input_size_for(model_name)

    train_ds = ManifestDataset(manifest_csv, "train", build_transform(input_size, True))
    val_ds = ManifestDataset(manifest_csv, "val", build_transform(input_size, False))
    test_ds = ManifestDataset(manifest_csv, "test", build_transform(input_size, False))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    model = build_model(model_name).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")
    best_state = None
    bad_epochs = 0

    for epoch in range(epochs):
        t0 = time.time()
        train_loss = run_epoch(model, train_loader, device, optimizer)
        val_loss = run_epoch(model, val_loader, device)
        scheduler.step()
        print(f"[{model_name}] epoch {epoch:03d} train={train_loss:.4f} val={val_loss:.4f} "
              f"({time.time() - t0:.1f}s)", flush=True)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"[{model_name}] early stop at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    y_prob, y_true = predict_probs(model, test_loader, device)
    metrics = compute_metrics(y_true, y_prob)
    efficiency = {
        "num_parameters": count_parameters(model),
        "input_size": input_size,
        "cpu_latency": measure_cpu_latency(model, input_size),
    }
    return {
        "model": model_name,
        "best_val_loss": best_val_loss,
        "test_metrics": metrics,
        "efficiency": efficiency,
        "n_train": len(train_ds), "n_val": len(val_ds), "n_test": len(test_ds),
    }


def select_device() -> torch.device:
    """Kaggle's GPU pool can hand out an older card (e.g. Tesla P100, compute
    capability sm_60) that the pre-installed torch build no longer has kernel
    support for (recent Kaggle images ship torch built only for sm_70+) —
    that combination doesn't raise until the first real CUDA op runs, so
    check compatibility up front and fall back to CPU rather than crash
    every single model in the registry one by one."""
    if not torch.cuda.is_available():
        return torch.device("cpu")

    major, minor = torch.cuda.get_device_capability(0)
    cap_str = f"sm_{major}{minor}"
    supported = torch.cuda.get_arch_list()
    if cap_str not in supported:
        name = torch.cuda.get_device_name(0)
        print(f"GPU {name} ({cap_str}) not supported by this torch build "
              f"(supports {supported}) — falling back to CPU")
        return torch.device("cpu")
    return torch.device("cuda")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    work = Path("/kaggle/working")
    raw_root = find_raw_root()
    print("raw_root:", raw_root)

    crops_root = work / "crops"
    print("cropping...")
    counts = crop_all(raw_root, crops_root)
    print("crop counts:", counts, "total:", sum(counts.values()))

    manifest_csv = work / "sipakmed_splits.csv"
    build_manifest(crops_root, manifest_csv)
    print("wrote manifest:", manifest_csv)

    device = select_device()
    print("device:", device)

    all_results = []
    for model_name in MODEL_REGISTRY:
        print(f"=== training {model_name} ===", flush=True)
        try:
            # Kept deliberately short: no way to cancel a running Kaggle kernel
            # or retrieve partial output before it finishes, and 12 models x 30
            # epochs on a CPU fallback risks running past the session limit
            # with nothing recoverable. This is a fast first-pass benchmark to
            # validate the pipeline end-to-end; extend epochs once GPU access
            # (or a confirmed compute-capability match) is available.
            result = train_one(model_name, manifest_csv, device, epochs=10, batch_size=32,
                                lr=3e-4, patience=4, seed=13)
        except Exception as e:  # keep going even if one model in the registry misbehaves
            result = {"model": model_name, "error": repr(e)}
            print(f"[{model_name}] FAILED: {e!r}")
        all_results.append(result)
        with open(work / "results.json", "w") as f:
            json.dump(all_results, f, indent=2)

    print(json.dumps(all_results, indent=2))


if __name__ == "__main__":
    main()
