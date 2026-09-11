"""Train and evaluate one baseline from the RQ1 registry (proposal S12/S13).

Usage:
    python -m cervical_screening.train \
        --model mobilevit_s --manifest data/processed/sipakmed_splits.csv \
        --out-dir runs/mobilevit_s --epochs 30
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from cervical_screening.data.dataset import SipakmedManifestDataset
from cervical_screening.data.transforms import build_transform
from cervical_screening.eval.efficiency import (
    count_parameters,
    measure_cpu_latency,
    measure_peak_rss_delta_mb,
)
from cervical_screening.eval.metrics import compute_metrics
from cervical_screening.models.registry import build_model, input_size_for


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
def predict_probs(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_probs, all_labels = [], []
    for images, labels in loader:
        images = images.to(device)
        logits = model(images)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(labels.numpy())
    return np.concatenate(all_probs), np.concatenate(all_labels)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=7,
                         help="early-stop if val loss doesn't improve for this many epochs")
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_size = input_size_for(args.model)

    train_ds = SipakmedManifestDataset(args.manifest, "train", build_transform(input_size, train=True))
    val_ds = SipakmedManifestDataset(args.manifest, "val", build_transform(input_size, train=False))
    test_ds = SipakmedManifestDataset(args.manifest, "test", build_transform(input_size, train=False))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = build_model(args.model).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0
    history = []

    for epoch in range(args.epochs):
        t0 = time.time()
        train_loss = run_epoch(model, train_loader, device, optimizer)
        val_loss = run_epoch(model, val_loader, device, optimizer=None)
        scheduler.step()

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                         "seconds": time.time() - t0})
        print(f"[{args.model}] epoch {epoch:03d}  train_loss={train_loss:.4f}  "
              f"val_loss={val_loss:.4f}  ({time.time() - t0:.1f}s)")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"early stopping at epoch {epoch} (no val improvement for {args.patience} epochs)")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), args.out_dir / "model.pt")

    y_prob, y_true = predict_probs(model, test_loader, device)
    metrics = compute_metrics(y_true, y_prob)

    # RSS delta must be measured before measure_cpu_latency's own warmup runs —
    # once torch's CPU caching allocator has warmed up in this process, a later
    # forward pass reuses existing buffers and the delta reads as ~0.
    peak_rss_delta_mb = measure_peak_rss_delta_mb(model, input_size)
    efficiency = {
        "num_parameters": count_parameters(model),
        "input_size": input_size,
        "peak_rss_delta_mb": peak_rss_delta_mb,
        "cpu_latency": measure_cpu_latency(model, input_size),
    }

    results = {
        "model": args.model,
        "best_val_loss": best_val_loss,
        "test_metrics": metrics,
        "efficiency": efficiency,
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "n_test": len(test_ds),
    }

    with open(args.out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    with open(args.out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
