"""Headless Colab-CLI pipeline: RQ1 baseline benchmark + cross-dataset shift
eval (proposal S12/S13, S16/RQ4), run via `colab exec -f run_pipeline.py` on
a provisioned GPU VM.

This is the `colab run`/`colab exec` counterpart to kaggle_kernel/run_pipeline.py
(same crop/split/train/metrics logic) — separated because the two run in
different environments (no /kaggle/input mount here; datasets are pulled via
the kaggle CLI, which must already be authenticated on the VM — see
docs/colab_cli_pipeline.md) and because Colab's T4 (sm_75) doesn't need the
CPU fallback the Kaggle P100 pool forced, so this runs the full epoch budget.
"""

from __future__ import annotations

import csv
import json
import random
import subprocess
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

DATA_ROOT = Path("/content/data")
RUNS_ROOT = Path("/content/runs")

CLASSES = [
    "im_Dyskeratotic",
    "im_Koilocytotic",
    "im_Metaplastic",
    "im_Parabasal",
    "im_Superficial-Intermediate",
]

NORMAL, ABNORMAL = "normal", "abnormal"
SIPAKMED_TO_BINARY = {
    "im_Parabasal": NORMAL, "im_Metaplastic": NORMAL, "im_Superficial-Intermediate": NORMAL,
    "im_Koilocytotic": ABNORMAL, "im_Dyskeratotic": ABNORMAL,
}
HERLEV_TO_BINARY = {
    "normal_columnar": NORMAL, "normal_intermediate": NORMAL, "normal_superficiel": NORMAL,
    "light_dysplastic": ABNORMAL, "moderate_dysplastic": ABNORMAL,
    "severe_dysplastic": ABNORMAL, "carcinoma_in_situ": ABNORMAL,
}
MENDELEY_LBC_TO_BINARY = {
    "Negative for Intraepithelial malignancy": NORMAL,
    "Low squamous intra-epithelial lesion": ABNORMAL,
    "High squamous intra-epithelial lesion": ABNORMAL,
    "Squamous cell carcinoma": ABNORMAL,
}

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
# 0. Dataset acquisition
# ---------------------------------------------------------------------------

def kaggle_download(ref: str, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    subprocess.run(["kaggle", "datasets", "download", "-d", ref, "-p", str(dest), "--unzip"],
                    check=True)


def download_all() -> None:
    kaggle_download("marinaeplissiti/sipakmed", DATA_ROOT / "raw/sipakmed_full")
    kaggle_download("yuvrajsinhachowdhury/herlev-dataset", DATA_ROOT / "raw/herlev")
    kaggle_download("blank1508/mendeley-lbc-cervical-cancer", DATA_ROOT / "raw/mendeley_lbc")


# ---------------------------------------------------------------------------
# 1. Crop SIPaKMeD at native resolution
# ---------------------------------------------------------------------------

def read_polygon(path: Path) -> list[tuple[float, float]]:
    points = []
    with open(path, newline="") as f:
        for row in csv.reader(f):
            if len(row) >= 2:
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
# 2. Splits / manifest
# ---------------------------------------------------------------------------

def stratified_split(paths, fractions, seed):
    rng = random.Random(seed)
    shuffled = paths[:]
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train, n_val, n_cal = round(n * fractions[0]), round(n * fractions[1]), round(n * fractions[2])
    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train:n_train + n_val],
        "calibration": shuffled[n_train + n_val:n_train + n_val + n_cal],
        "test": shuffled[n_train + n_val + n_cal:],
    }


def build_manifest(crops_root: Path, out_csv: Path, seed=13, fractions=(0.60, 0.15, 0.10, 0.15)) -> None:
    rows = []
    for class_idx, class_name in enumerate(CLASSES):
        images = sorted((crops_root / class_name).glob("*.png"))
        if not images:
            raise FileNotFoundError(f"no crops under {crops_root / class_name}")
        splits = stratified_split(images, fractions, seed=seed + class_idx)
        for split_name, split_paths in splits.items():
            rows.extend((str(p), class_name, class_idx, split_name) for p in split_paths)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "class_name", "class_idx", "split"])
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# 3. Model / device / transforms / datasets
# ---------------------------------------------------------------------------

def build_model(name: str, pretrained: bool = True) -> nn.Module:
    return timm.create_model(MODEL_REGISTRY[name], pretrained=pretrained, num_classes=NUM_CLASSES)


def input_size_for(name: str) -> int:
    cfg = timm.get_pretrained_cfg(MODEL_REGISTRY[name])
    return cfg.input_size[-1] if cfg is not None and cfg.input_size else 224


def select_device() -> torch.device:
    if not torch.cuda.is_available():
        return torch.device("cpu")
    major, minor = torch.cuda.get_device_capability(0)
    cap_str = f"sm_{major}{minor}"
    if cap_str not in torch.cuda.get_arch_list():
        print(f"GPU {torch.cuda.get_device_name(0)} ({cap_str}) unsupported — falling back to CPU")
        return torch.device("cpu")
    return torch.device("cuda")


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
        return self.transform(Image.open(path).convert("RGB")), label


class BinaryDataset(Dataset):
    LABEL_TO_IDX = {NORMAL: 0, ABNORMAL: 1}

    def __init__(self, rows: list[tuple[Path, str]], transform):
        self.rows = rows
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        path, label = self.rows[idx]
        return self.transform(Image.open(path).convert("RGB")), self.LABEL_TO_IDX[label]


def scan_binary_manifest(root: Path, class_to_binary: dict[str, str],
                          extensions=(".jpg", ".jpeg", ".png", ".bmp")) -> list[tuple[Path, str]]:
    rows: list[tuple[Path, str]] = []
    for class_name, binary_label in class_to_binary.items():
        class_dirs = [root / class_name] if (root / class_name).exists() else []
        if not class_dirs:
            class_dirs = [d for d in root.glob(f"**/{class_name}") if d.is_dir()]
        if not class_dirs:
            raise FileNotFoundError(f"class directory not found under {root}: {class_name}")
        images = [p for class_dir in class_dirs
                  for p in class_dir.rglob("*") if p.suffix.lower() in extensions]
        if not images:
            raise FileNotFoundError(f"no images found for {class_name}")
        rows.extend((p, binary_label) for p in images)
    return rows


# ---------------------------------------------------------------------------
# 4. Metrics / efficiency
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


def compute_metrics(y_true, y_prob) -> dict:
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
def measure_latency(model, input_size, device, n_warmup=10, n_iters=30) -> dict:
    model = model.eval().to(device)
    x = torch.randn(1, 3, input_size, input_size, device=device)
    for _ in range(n_warmup):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    times = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    times.sort()
    return {"median_latency_ms": times[len(times) // 2] * 1000, "device": str(device)}


# ---------------------------------------------------------------------------
# 5. Train / eval
# ---------------------------------------------------------------------------

def run_epoch(model, loader, device, optimizer=None) -> float:
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

    best_val_loss, best_state, bad_epochs = float("inf"), None, 0
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

    ckpt_dir = RUNS_ROOT / model_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt_dir / "model.pt")

    y_prob, y_true = predict_probs(model, test_loader, device)
    metrics = compute_metrics(y_true, y_prob)
    efficiency = {
        "num_parameters": count_parameters(model),
        "input_size": input_size,
        "latency": measure_latency(model, input_size, device),
    }
    return {
        "model": model_name, "best_val_loss": best_val_loss, "test_metrics": metrics,
        "efficiency": efficiency, "n_train": len(train_ds), "n_val": len(val_ds), "n_test": len(test_ds),
    }


# ---------------------------------------------------------------------------
# 6. Cross-dataset shift eval (binary grouping of the 5-class head)
# ---------------------------------------------------------------------------

_ABNORMAL_IDX = [i for i, c in enumerate(CLASSES) if SIPAKMED_TO_BINARY[c] == ABNORMAL]
_NORMAL_IDX = [i for i, c in enumerate(CLASSES) if SIPAKMED_TO_BINARY[c] == NORMAL]


def group_to_binary_probs(probs_5class: np.ndarray) -> np.ndarray:
    p_normal = probs_5class[:, _NORMAL_IDX].sum(axis=1)
    p_abnormal = probs_5class[:, _ABNORMAL_IDX].sum(axis=1)
    return np.stack([p_normal, p_abnormal], axis=1)


@torch.no_grad()
def predict_binary(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []
    for images, labels in loader:
        probs_5 = torch.softmax(model(images.to(device)), dim=1).cpu().numpy()
        all_probs.append(group_to_binary_probs(probs_5))
        all_labels.append(labels.numpy())
    return np.concatenate(all_probs), np.concatenate(all_labels)


def run_cross_dataset_eval(best_model_name: str) -> dict:
    device = select_device()
    input_size = input_size_for(best_model_name)
    model = build_model(best_model_name, pretrained=False)
    model.load_state_dict(torch.load(RUNS_ROOT / best_model_name / "model.pt", map_location="cpu"))
    model = model.to(device)

    herlev_rows = scan_binary_manifest(DATA_ROOT / "raw/herlev/Herlev Dataset", HERLEV_TO_BINARY)
    mendeley_rows = scan_binary_manifest(DATA_ROOT / "raw/mendeley_lbc", MENDELEY_LBC_TO_BINARY)
    print("herlev:", len(herlev_rows), "mendeley_lbc:", len(mendeley_rows))

    eval_tf = build_transform(input_size, train=False)
    shift_results = {}
    for name, rows in [("herlev", herlev_rows), ("mendeley_lbc", mendeley_rows)]:
        loader = DataLoader(BinaryDataset(rows, eval_tf), batch_size=32, shuffle=False, num_workers=2)
        y_prob, y_true = predict_binary(model, loader, device)
        shift_results[name] = compute_metrics(y_true, y_prob)
        print(name, shift_results[name])

    return {"model": best_model_name, "shift_results": shift_results}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    print("downloading datasets...", flush=True)
    download_all()

    crops_root = DATA_ROOT / "processed/sipakmed_crops"
    counts = crop_all(DATA_ROOT / "raw/sipakmed_full", crops_root)
    print("crop counts:", counts, "total:", sum(counts.values()), flush=True)
    assert sum(counts.values()) == 4049, "expected 4049 crops matching Plissiti et al."

    manifest_csv = DATA_ROOT / "processed/sipakmed_splits.csv"
    build_manifest(crops_root, manifest_csv)

    device = select_device()
    print("device:", device, "-", torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU", flush=True)

    all_results = []
    for model_name in MODEL_REGISTRY:
        print(f"=== training {model_name} ===", flush=True)
        try:
            result = train_one(model_name, manifest_csv, device, epochs=30, batch_size=32,
                                lr=3e-4, patience=7, seed=13)
        except Exception as e:
            result = {"model": model_name, "error": repr(e)}
            print(f"[{model_name}] FAILED: {e!r}")
        all_results.append(result)
        RUNS_ROOT.mkdir(parents=True, exist_ok=True)
        with open(RUNS_ROOT / "results.json", "w") as f:
            json.dump(all_results, f, indent=2)

    best = max((r for r in all_results if "error" not in r),
               key=lambda r: r["test_metrics"]["macro_f1"], default=None)
    if best is not None:
        print(f"best in-distribution model: {best['model']}", flush=True)
        shift = run_cross_dataset_eval(best["model"])
        with open(RUNS_ROOT / "cross_dataset_shift_results.json", "w") as f:
            json.dump(shift, f, indent=2)

    print(json.dumps(all_results, indent=2))


if __name__ == "__main__":
    main()
