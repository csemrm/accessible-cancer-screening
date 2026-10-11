"""Uncertainty-quantification comparison (paper2's RQ) rerun under the cluster-grouped
split (same split as kaggle_kernel/cluster_split_rerun, SPLIT_SEED=13 fixed), rather than
the original cell-level split paper2.tex used.

Backbone: EfficientViT-M0 (the registry's in-distribution winner and, per the cluster-grouped
benchmark, a model whose own seed-13 external failure did not reproduce at other seeds --
exactly the kind of single-run claim this uncertainty comparison itself needs to survive).

Four methods, each its own cheapest native confidence signal: baseline (temperature-scaled
max-softmax), MC Dropout (predictive entropy), evidential deep learning (1 - vacuity), deep
ensemble (negative member-variance). Evaluated in-distribution (cluster-grouped test cells)
and zero-shot on Herlev, Mendeley LBC, APCEM.

Significance test: MC Dropout is retrained twice more (seeds 17, 29, cluster split fixed)
and paired-bootstrap-tested against the ensemble on Mendeley LBC and APCEM, mirroring
paper2's original significance-test design (which found the reported severe-shift advantage
did not reproduce on the cell-level split).

Everything written to /kaggle/working/out, after every method so a timeout loses at most one.
"""

from __future__ import annotations

import csv
import glob
import json
import platform
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

WORK = Path("/kaggle/working")
OUT = WORK / "out"
SPLIT_SEED = 13  # identical cluster-grouped split to cluster_split_rerun's seed-13 run
EPOCHS, BATCH_SIZE, LR, PATIENCE = 30, 32, 3e-4, 7
FRACTIONS = (0.60, 0.15, 0.10, 0.15)
MODEL_NAME = "efficientvit_m0"
MODEL_TAG = "efficientvit_m0.r224_in1k"
NUM_CLASSES = 5
N_PASSES = 20
DROPOUT_RATE = 0.3
ANNEALING_EPOCHS = 10
ENSEMBLE_SEEDS = [1, 2, 3, 4, 5]
SIG_TEST_SEEDS = [17, 29]  # MC Dropout retrains for the significance test

CLASSES = ["im_Dyskeratotic", "im_Koilocytotic", "im_Metaplastic", "im_Parabasal", "im_Superficial-Intermediate"]
NORMAL, ABNORMAL = "normal", "abnormal"
HERLEV_TO_BINARY = {"normal_columnar": NORMAL, "normal_intermediate": NORMAL, "normal_superficiel": NORMAL,
                    "light_dysplastic": ABNORMAL, "moderate_dysplastic": ABNORMAL,
                    "severe_dysplastic": ABNORMAL, "carcinoma_in_situ": ABNORMAL}
MENDELEY_LBC_TO_BINARY = {"Negative for Intraepithelial malignancy": NORMAL,
                          "Low squamous intra-epithelial lesion": ABNORMAL,
                          "High squamous intra-epithelial lesion": ABNORMAL,
                          "Squamous cell carcinoma": ABNORMAL}
# See kaggle_kernel/cluster_split_rerun/run.py for provenance of the APCEM private dataset mirror.
APCEM_TO_BINARY = {"NILM": NORMAL, "ASCUS": ABNORMAL, "ASCH": ABNORMAL, "LSIL": ABNORMAL, "HSIL": ABNORMAL,
                   "SCC": ABNORMAL}
EXPECTED_N = {"sipakmed": 4049, "herlev": 917, "mendeley_lbc": 962, "apcem": 4253}
IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
_ABN = [0, 1]  # Dyskeratotic, Koilocytotic
_NOR = [2, 3, 4]


# ---------------------------------------------------------------------------
# data acquisition (identical to cluster_split_rerun/run.py)
# ---------------------------------------------------------------------------

def find_dir(pattern: str) -> Path:
    hits = glob.glob(f"/kaggle/input/*/{pattern}") or glob.glob(f"/kaggle/input/**/{pattern}", recursive=True)
    if not hits:
        raise FileNotFoundError(f"{pattern} not found under /kaggle/input (is the dataset attached?)")
    return Path(hits[0])


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


def cluster_of(path: Path) -> str:
    return path.stem.split("_")[0]


def grouped_split(paths, fractions, seed):
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
        splits = grouped_split(images, FRACTIONS, SPLIT_SEED + class_idx)
        for split_name, ps in splits.items():
            assert ps, f"empty {split_name} split for {class_name}"
            rows.extend((str(p), class_name, class_idx, split_name, cluster_of(p)) for p in ps)
        stats[class_name] = {k: len(v) for k, v in splits.items()}
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


def build_model(pretrained: bool = True) -> nn.Module:
    return timm.create_model(MODEL_TAG, pretrained=pretrained, num_classes=NUM_CLASSES)


def input_size_for() -> int:
    cfg = timm.get_pretrained_cfg(MODEL_TAG)
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


def set_dropout_rate(model: nn.Module, p: float) -> int:
    n = 0
    for module in model.modules():
        if isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            module.p = p
            n += 1
    return n


def enable_dropout(model: nn.Module) -> None:
    model.eval()
    for module in model.modules():
        if isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            module.train()


# ---------------------------------------------------------------------------
# evidential deep learning (Sensoy et al., NeurIPS 2018) -- identical formulas
# to src/cervical_screening/uncertainty/evidential.py
# ---------------------------------------------------------------------------

def logits_to_evidence(logits):
    return F.softplus(logits)


def evidence_to_alpha(evidence):
    return evidence + 1.0


def dirichlet_strength(alpha):
    return alpha.sum(dim=1)


def expected_probability(alpha):
    return alpha / dirichlet_strength(alpha).unsqueeze(1)


def vacuity(alpha, num_classes):
    return num_classes / dirichlet_strength(alpha)


def _kl_dirichlet_to_uniform(alpha):
    num_classes = alpha.shape[1]
    ones = torch.ones_like(alpha)
    sum_alpha = alpha.sum(dim=1, keepdim=True)
    term1 = torch.lgamma(sum_alpha.squeeze(1)) - torch.lgamma(ones.sum(dim=1))
    term2 = torch.lgamma(alpha).sum(dim=1) - torch.lgamma(ones).sum(dim=1)
    term3 = ((alpha - ones) * (torch.digamma(alpha) - torch.digamma(sum_alpha))).sum(dim=1)
    return term1 - term2 + term3


def evidential_loss(logits, labels, epoch, num_classes, annealing_epochs=10):
    evidence = logits_to_evidence(logits)
    alpha = evidence_to_alpha(evidence)
    sum_alpha = dirichlet_strength(alpha).unsqueeze(1)
    y_onehot = F.one_hot(labels, num_classes).float()
    ce_loss = (y_onehot * (torch.digamma(sum_alpha) - torch.digamma(alpha))).sum(dim=1)
    alpha_tilde = y_onehot + (1.0 - y_onehot) * alpha
    kl = _kl_dirichlet_to_uniform(alpha_tilde)
    annealing_coef = min(1.0, epoch / annealing_epochs)
    return (ce_loss + annealing_coef * kl).mean()


# ---------------------------------------------------------------------------
# calibration / selective / significance (identical to src/cervical_screening/uncertainty/*)
# ---------------------------------------------------------------------------

def softmax_np(logits):
    z = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(z)
    return exp / exp.sum(axis=1, keepdims=True)


def expected_calibration_error(probs, labels, n_bins=15):
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correct = (predictions == labels).astype(np.float64)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece, n = 0.0, len(labels)
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        in_bin = (confidences > lo) & (confidences <= hi) if lo > 0 else (confidences >= lo) & (confidences <= hi)
        if not in_bin.any():
            continue
        ece += (in_bin.sum() / n) * abs(correct[in_bin].mean() - confidences[in_bin].mean())
    return float(ece)


def negative_log_likelihood(probs, labels, eps=1e-12):
    true_class_probs = probs[np.arange(len(labels)), labels]
    return float(-np.mean(np.log(np.clip(true_class_probs, eps, 1.0))))


def fit_temperature(logits, labels, t_min=0.1, t_max=5.0, n_steps=200):
    candidates = np.linspace(t_min, t_max, n_steps)
    best_t, best_nll = 1.0, float("inf")
    for t in candidates:
        nll = negative_log_likelihood(softmax_np(logits / t), labels)
        if nll < best_nll:
            best_nll, best_t = nll, t
    return float(best_t)


def apply_temperature(logits, temperature):
    return softmax_np(logits / temperature)


def risk_coverage_auc(confidence, correct):
    order = np.argsort(-confidence)
    correct_sorted = correct[order].astype(np.float64)
    n = len(correct)
    cum_correct = np.cumsum(correct_sorted)
    coverage = np.arange(1, n + 1) / n
    risk = 1.0 - (cum_correct / np.arange(1, n + 1))
    return float(np.sum((risk[1:] + risk[:-1]) / 2.0 * np.diff(coverage)))


def selective_accuracy_at_coverage(confidence, correct, target_coverage):
    n = len(correct)
    k = max(1, int(round(target_coverage * n)))
    order = np.argsort(-confidence)
    return float(correct[order[:k]].mean())


def selective_report(confidence, correct):
    return {
        "risk_coverage_auc": risk_coverage_auc(confidence, correct),
        "selective_accuracy_at_0.80": selective_accuracy_at_coverage(confidence, correct, 0.80),
        "selective_accuracy_at_0.90": selective_accuracy_at_coverage(confidence, correct, 0.90),
        "full_coverage_accuracy": float(correct.mean()),
    }


def paired_bootstrap_rc_auc_test(confidence_a, correct_a, confidence_b, correct_b, n_boot=2000, seed=13):
    assert len(confidence_a) == len(correct_a) == len(confidence_b) == len(correct_b)
    n = len(confidence_a)
    rng = np.random.default_rng(seed)
    observed_a = risk_coverage_auc(confidence_a, correct_a)
    observed_b = risk_coverage_auc(confidence_b, correct_b)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        diffs[i] = risk_coverage_auc(confidence_b[idx], correct_b[idx]) - risk_coverage_auc(confidence_a[idx], correct_a[idx])
    ci_low, ci_high = np.percentile(diffs, [2.5, 97.5])
    p_a_better = float((diffs <= 0).mean())
    return {
        "n_samples": n, "n_bootstrap": n_boot,
        "method_a_rc_auc": observed_a, "method_b_rc_auc": observed_b,
        "observed_diff_b_minus_a": observed_b - observed_a,
        "bootstrap_ci_95": [float(ci_low), float(ci_high)],
        "p_value_a_better_one_sided": p_a_better,
    }


def to_binary(p5: np.ndarray) -> np.ndarray:
    return np.stack([p5[:, _NOR].sum(axis=1), p5[:, _ABN].sum(axis=1)], axis=1)


# ---------------------------------------------------------------------------
# train / predict
# ---------------------------------------------------------------------------

def _make_loaders(manifest_csv, input_size, batch_size=BATCH_SIZE, num_workers=2):
    train_ds = ManifestDataset(manifest_csv, "train", build_transform(input_size, True))
    val_ds = ManifestDataset(manifest_csv, "val", build_transform(input_size, False))
    return (DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers),
            DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers))


def train_generic(manifest_csv, device, train_seed, loss_fn, dropout_rate=None, tag=""):
    """loss_fn(logits, labels, epoch) -> scalar loss. Shared training loop for
    baseline (CE), evidential, and MC Dropout (CE + forced nonzero dropout)."""
    torch.manual_seed(train_seed)
    input_size = input_size_for()
    train_loader, val_loader = _make_loaders(manifest_csv, input_size)
    model = build_model().to(device)
    if dropout_rate is not None:
        n_changed = set_dropout_rate(model, dropout_rate)
        print(f"[{tag}] set dropout rate={dropout_rate} on {n_changed} module(s)", flush=True)
        if n_changed == 0:
            raise RuntimeError(f"{MODEL_NAME} has no nn.Dropout modules -- MC Dropout would be degenerate.")
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    best_val_loss, best_state, bad_epochs = float("inf"), None, 0
    for epoch in range(EPOCHS):
        t0 = time.time()
        model.train()
        train_loss, n = 0.0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            loss = loss_fn(model(images), labels, epoch)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            train_loss += loss.item() * images.size(0); n += images.size(0)
        train_loss /= n
        scheduler.step()
        model.eval()
        val_loss, n = 0.0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                loss = loss_fn(model(images), labels, epoch)
                val_loss += loss.item() * images.size(0); n += images.size(0)
        val_loss /= n
        print(f"[{tag}] epoch {epoch:03d} train={train_loss:.4f} val={val_loss:.4f} ({time.time()-t0:.1f}s)", flush=True)
        if val_loss < best_val_loss:
            best_val_loss, bad_epochs = val_loss, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE:
                print(f"[{tag}] early stop at epoch {epoch}", flush=True)
                break
    model.load_state_dict(best_state)
    return model, best_val_loss


@torch.no_grad()
def predict_baseline(model, loader, device, temperature=None):
    model.eval()
    all_logits, all_labels = [], []
    for images, labels in loader:
        all_logits.append(model(images.to(device)).cpu().numpy())
        all_labels.append(labels.numpy())
    logits = np.concatenate(all_logits); labels = np.concatenate(all_labels)
    probs = apply_temperature(logits, temperature) if temperature is not None else softmax_np(logits)
    return probs, labels, logits


@torch.no_grad()
def predict_evidential(model, loader, device):
    model.eval()
    all_probs, all_vac, all_labels = [], [], []
    for images, labels in loader:
        logits = model(images.to(device))
        alpha = evidence_to_alpha(logits_to_evidence(logits))
        all_probs.append(expected_probability(alpha).cpu().numpy())
        all_vac.append(vacuity(alpha, NUM_CLASSES).cpu().numpy())
        all_labels.append(labels.numpy())
    return np.concatenate(all_probs), np.concatenate(all_vac), np.concatenate(all_labels)


def predict_mc_dropout(model, loader, device, n_passes=N_PASSES):
    all_mean_probs, all_entropy, all_labels = [], [], []
    for images, labels in loader:
        enable_dropout(model)
        passes = []
        with torch.no_grad():
            for _ in range(n_passes):
                passes.append(torch.softmax(model(images.to(device)), dim=1))
        mean_probs = torch.stack(passes, dim=0).mean(dim=0)
        entropy = -(mean_probs * torch.log(mean_probs.clamp_min(1e-12))).sum(dim=1)
        all_mean_probs.append(mean_probs.cpu().numpy())
        all_entropy.append(entropy.cpu().numpy())
        all_labels.append(labels.numpy())
    return np.concatenate(all_mean_probs), np.concatenate(all_entropy), np.concatenate(all_labels)


@torch.no_grad()
def predict_ensemble(models, loader, device):
    for m in models:
        m.eval()
    per_member = [[] for _ in models]
    all_labels = []
    for images, labels in loader:
        images = images.to(device)
        for i, m in enumerate(models):
            per_member[i].append(torch.softmax(m(images), dim=1).cpu().numpy())
        all_labels.append(labels.numpy())
    member_probs = np.stack([np.concatenate(p) for p in per_member], axis=0)
    labels = np.concatenate(all_labels)
    mean_probs = member_probs.mean(axis=0)
    variance = member_probs.var(axis=0).sum(axis=1)
    return mean_probs, variance, labels


def evaluate_method(method_name, predict_fn, loaders, ext_sets_binary):
    """predict_fn(loader) -> (probs5_or_probs, confidence, labels). For in-dist
    (5-class), confidence derives from the 5-class probs directly. For external
    binary sets, predict_fn is called on the 5-class model output and we convert
    to binary ourselves, consistent with cluster_split_rerun/run.py's protocol."""
    test_probs, test_conf, test_labels = predict_fn(loaders["test"])
    test_correct = (test_probs.argmax(axis=1) == test_labels).astype(float)
    result = {
        "method": method_name,
        "in_distribution": {
            "ece": expected_calibration_error(test_probs, test_labels),
            "selective": selective_report(test_conf, test_correct),
        },
        "external": {},
    }
    for name, loader in ext_sets_binary.items():
        probs5, conf, binary_labels = predict_fn(loader)
        binary_probs = to_binary(probs5)
        binary_correct = (binary_probs.argmax(axis=1) == binary_labels).astype(float)
        result["external"][name] = {
            "ece": expected_calibration_error(binary_probs, binary_labels),
            "selective": selective_report(conf, binary_correct),
            "n": int(len(binary_labels)),
            "_confidence": conf.tolist(),
            "_correct": binary_correct.tolist(),
        }
    return result


def main():
    (OUT / "checkpoints").mkdir(parents=True, exist_ok=True)
    device = select_device()
    env = {"python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
           "timm": timm.__version__, "numpy": np.__version__, "cuda": torch.version.cuda,
           "cudnn": torch.backends.cudnn.version(), "device": str(device),
           "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
           "model": MODEL_NAME, "split": "cluster-grouped", "split_seed": SPLIT_SEED,
           "epochs": EPOCHS, "batch_size": BATCH_SIZE, "lr": LR}
    (OUT / "env.json").write_text(json.dumps(env, indent=2))
    print(json.dumps(env, indent=2), flush=True)
    if device.type != "cuda":
        raise SystemExit("no usable CUDA device: refusing to train on CPU")

    raw_root = find_dir("im_Dyskeratotic").parent
    crops_root = WORK / "crops"
    counts = crop_all(raw_root, crops_root)
    print("crop counts:", counts, flush=True)
    assert sum(counts.values()) == EXPECTED_N["sipakmed"], counts

    manifest_csv = OUT / "sipakmed_cluster_split.csv"
    split_stats = build_manifest(crops_root, manifest_csv)
    (OUT / "split_stats.json").write_text(json.dumps(split_stats, indent=2))
    print("cluster-grouped split sizes per class:", json.dumps(split_stats), flush=True)

    ext_rows = {
        "herlev": scan_binary_manifest(find_dir("normal_columnar").parents[1], HERLEV_TO_BINARY),
        "mendeley_lbc": scan_binary_manifest(find_dir("Negative for Intraepithelial malignancy").parent,
                                              MENDELEY_LBC_TO_BINARY),
        "apcem": scan_binary_manifest(find_dir("NILM").parent, APCEM_TO_BINARY),
    }
    for k, rows in ext_rows.items():
        print(k, len(rows), flush=True)
        assert len(rows) == EXPECTED_N[k], (k, len(rows))

    input_size = input_size_for()
    eval_tf = build_transform(input_size, False)
    test_loader = DataLoader(ManifestDataset(manifest_csv, "test", eval_tf), batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=2)
    ext_loaders = {k: DataLoader(BinaryDataset(rows, eval_tf), batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
                   for k, rows in ext_rows.items()}
    loaders = {"test": test_loader}

    all_results = {}

    # ---------------- method 1/4: baseline + temperature scaling ----------------
    print("=== baseline ===", flush=True)
    baseline_model, baseline_val_loss = train_generic(
        manifest_csv, device, train_seed=SPLIT_SEED,
        loss_fn=lambda logits, labels, epoch: nn.functional.cross_entropy(logits, labels), tag="baseline")
    torch.save(baseline_model.state_dict(), OUT / "checkpoints" / "baseline.pt")
    cal_loader = DataLoader(ManifestDataset(manifest_csv, "calibration", eval_tf), batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=2)
    _, cal_labels, cal_logits = predict_baseline(baseline_model, cal_loader, device)
    T = fit_temperature(cal_logits, cal_labels)
    print("fitted temperature:", T, flush=True)

    def baseline_predict_fn(loader):
        probs, labels, _ = predict_baseline(baseline_model, loader, device, temperature=T)
        return probs, probs.max(axis=1), labels

    all_results["baseline_temp_scaled"] = evaluate_method("baseline_temp_scaled", baseline_predict_fn, loaders, ext_loaders)
    all_results["baseline_temp_scaled"]["fitted_temperature"] = T
    all_results["baseline_temp_scaled"]["best_val_loss"] = baseline_val_loss
    (OUT / "results.json").write_text(json.dumps(all_results, indent=2))

    # ---------------- method 2/4: MC Dropout ----------------
    print("=== mc_dropout ===", flush=True)
    mc_model, mc_val_loss = train_generic(
        manifest_csv, device, train_seed=SPLIT_SEED,
        loss_fn=lambda logits, labels, epoch: nn.functional.cross_entropy(logits, labels),
        dropout_rate=DROPOUT_RATE, tag="mc_dropout")
    torch.save(mc_model.state_dict(), OUT / "checkpoints" / "mc_dropout_seed13.pt")

    def mc_predict_fn(loader):
        mean_probs, entropy, labels = predict_mc_dropout(mc_model, loader, device)
        return mean_probs, -entropy, labels

    all_results["mc_dropout"] = evaluate_method("mc_dropout", mc_predict_fn, loaders, ext_loaders)
    all_results["mc_dropout"]["best_val_loss"] = mc_val_loss
    (OUT / "results.json").write_text(json.dumps(all_results, indent=2))

    # ---------------- method 3/4: evidential ----------------
    print("=== evidential ===", flush=True)
    evid_model, evid_val_loss = train_generic(
        manifest_csv, device, train_seed=SPLIT_SEED,
        loss_fn=lambda logits, labels, epoch: evidential_loss(logits, labels, epoch, NUM_CLASSES, ANNEALING_EPOCHS),
        tag="evidential")
    torch.save(evid_model.state_dict(), OUT / "checkpoints" / "evidential.pt")

    def evid_predict_fn(loader):
        probs, vac, labels = predict_evidential(evid_model, loader, device)
        return probs, 1.0 - vac, labels

    all_results["evidential"] = evaluate_method("evidential", evid_predict_fn, loaders, ext_loaders)
    all_results["evidential"]["best_val_loss"] = evid_val_loss
    (OUT / "results.json").write_text(json.dumps(all_results, indent=2))

    # ---------------- method 4/4: deep ensemble (5 members) ----------------
    print("=== deep_ensemble ===", flush=True)
    ensemble_models = []
    for i, es in enumerate(ENSEMBLE_SEEDS):
        print(f"-- ensemble member {i+1}/5, seed={es} --", flush=True)
        m, _ = train_generic(manifest_csv, device, train_seed=es,
                              loss_fn=lambda logits, labels, epoch: nn.functional.cross_entropy(logits, labels),
                              tag=f"ensemble_member{i+1}")
        torch.save(m.state_dict(), OUT / "checkpoints" / f"ensemble_member{i+1}_seed{es}.pt")
        ensemble_models.append(m)

    def ens_predict_fn(loader):
        mean_probs, variance, labels = predict_ensemble(ensemble_models, loader, device)
        return mean_probs, -variance, labels

    all_results["deep_ensemble"] = evaluate_method("deep_ensemble", ens_predict_fn, loaders, ext_loaders)
    (OUT / "results.json").write_text(json.dumps(all_results, indent=2))

    # ---------------- significance test: retrain MC Dropout at 2 more seeds ----------------
    print("=== significance test: MC Dropout retrains ===", flush=True)
    sig_results = {}
    for seed in SIG_TEST_SEEDS:
        print(f"-- MC Dropout retrain, seed={seed} --", flush=True)
        m, _ = train_generic(manifest_csv, device, train_seed=seed,
                              loss_fn=lambda logits, labels, epoch: nn.functional.cross_entropy(logits, labels),
                              dropout_rate=DROPOUT_RATE, tag=f"mc_dropout_seed{seed}")
        torch.save(m.state_dict(), OUT / "checkpoints" / f"mc_dropout_seed{seed}.pt")

        retrain_result = {}
        for name, loader in ext_loaders.items():
            mean_probs, entropy, labels = predict_mc_dropout(m, loader, device)
            binary_probs = to_binary(mean_probs)
            correct = (binary_probs.argmax(axis=1) == labels).astype(float)
            confidence = -entropy
            retrain_result[name] = {
                "rc_auc": risk_coverage_auc(confidence, correct),
                "_confidence": confidence.tolist(),
                "_correct": correct.tolist(),
            }
        sig_results[f"mc_dropout_seed{seed}"] = retrain_result
        (OUT / "sig_test_retrains.json").write_text(json.dumps(sig_results, indent=2))

    # paired bootstrap: each MC Dropout retrain vs the (seed-13-trained) ensemble, per external dataset
    sig_tests = {}
    for seed_key, retrain_result in sig_results.items():
        sig_tests[seed_key] = {}
        for name in ext_loaders:
            mc_conf = np.array(retrain_result[name]["_confidence"])
            mc_correct = np.array(retrain_result[name]["_correct"])
            ens_conf = np.array(all_results["deep_ensemble"]["external"][name]["_confidence"])
            ens_correct = np.array(all_results["deep_ensemble"]["external"][name]["_correct"])
            sig_tests[seed_key][name] = paired_bootstrap_rc_auc_test(mc_conf, mc_correct, ens_conf, ens_correct,
                                                                       n_boot=2000, seed=13)
    # also test the original (seed-13) MC Dropout vs ensemble, for the full seed-13/17/29 picture
    sig_tests["mc_dropout_seed13"] = {}
    for name in ext_loaders:
        mc_conf = np.array(all_results["mc_dropout"]["external"][name]["_confidence"])
        mc_correct = np.array(all_results["mc_dropout"]["external"][name]["_correct"])
        ens_conf = np.array(all_results["deep_ensemble"]["external"][name]["_confidence"])
        ens_correct = np.array(all_results["deep_ensemble"]["external"][name]["_correct"])
        sig_tests["mc_dropout_seed13"][name] = paired_bootstrap_rc_auc_test(mc_conf, mc_correct, ens_conf, ens_correct,
                                                                              n_boot=2000, seed=13)
    (OUT / "significance_tests.json").write_text(json.dumps(sig_tests, indent=2))
    print(json.dumps(sig_tests, indent=2), flush=True)

    print("DONE", flush=True)


if __name__ == "__main__":
    main()
