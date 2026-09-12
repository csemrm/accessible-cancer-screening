"""Training loops for the four RQ2 uncertainty methods (proposal S18),
backbone held fixed (EfficientViT-M0 by default — RQ1's best in-distribution
model, and the one shown in S16/RQ4 to fail silently under shift), so the
comparison isolates the uncertainty method rather than architecture choice.

Each variant returns the trained model plus whatever extra state its
uncertainty computation needs (a fitted temperature, an ensemble of
checkpoints, nothing extra for evidential/MC-Dropout since their
uncertainty comes straight from the model's own forward pass).
"""

from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from cervical_screening.data.dataset import SipakmedManifestDataset
from cervical_screening.data.transforms import build_transform
from cervical_screening.models.registry import build_model, input_size_for
from cervical_screening.uncertainty import calibration
from cervical_screening.uncertainty.evidential import (
    evidential_loss,
    evidence_to_alpha,
    expected_probability,
    logits_to_evidence,
    vacuity,
)
from cervical_screening.uncertainty.mc_dropout import mc_dropout_predict, set_dropout_rate


def _make_loaders(manifest_csv, input_size, batch_size, num_workers):
    train_ds = SipakmedManifestDataset(manifest_csv, "train", build_transform(input_size, train=True))
    val_ds = SipakmedManifestDataset(manifest_csv, "val", build_transform(input_size, train=False))
    return (
        train_ds, val_ds,
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
    )


@torch.no_grad()
def _softmax_logits_on_loader(model, loader, device):
    model.eval()
    all_logits, all_labels = [], []
    for images, labels in loader:
        logits = model(images.to(device))
        all_logits.append(logits.cpu().numpy())
        all_labels.append(labels.numpy())
    return np.concatenate(all_logits), np.concatenate(all_labels)


def train_baseline(model_name: str, manifest_csv, device, epochs=30, batch_size=32,
                    lr=3e-4, patience=7, seed=13, num_workers=2) -> dict:
    """Standard cross-entropy training. Produces logits the caller can feed
    either raw (softmax) or through temperature scaling — this one function
    covers both the 'plain softmax' and 'temperature-scaled softmax'
    conditions in the RQ2 comparison, since they share the same model."""
    torch.manual_seed(seed)
    input_size = input_size_for(model_name)
    train_ds, val_ds, train_loader, val_loader = _make_loaders(manifest_csv, input_size, batch_size, num_workers)

    model = build_model(model_name).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    best_val_loss, best_state, bad_epochs = float("inf"), None, 0
    for epoch in range(epochs):
        t0 = time.time()
        model.train()
        train_loss, n = 0.0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
            n += images.size(0)
        train_loss /= n
        scheduler.step()

        model.eval()
        val_loss, n = 0.0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                loss = criterion(logits, labels)
                val_loss += loss.item() * images.size(0)
                n += images.size(0)
        val_loss /= n

        print(f"[baseline:{model_name}] epoch {epoch:03d} train={train_loss:.4f} val={val_loss:.4f} "
              f"({time.time() - t0:.1f}s)", flush=True)

        if val_loss < best_val_loss:
            best_val_loss, bad_epochs = val_loss, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"[baseline:{model_name}] early stop at epoch {epoch}", flush=True)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return {"model": model, "input_size": input_size, "best_val_loss": best_val_loss}


def train_mc_dropout(model_name: str, manifest_csv, device, dropout_rate=0.2, epochs=30,
                      batch_size=32, lr=3e-4, patience=7, seed=13, num_workers=2) -> dict:
    """Same loop as train_baseline, but with a real (non-zero) dropout rate
    injected BEFORE training starts. See mc_dropout.set_dropout_rate's
    docstring for why this can't be patched in only at inference time."""
    torch.manual_seed(seed)
    input_size = input_size_for(model_name)
    train_ds, val_ds, train_loader, val_loader = _make_loaders(manifest_csv, input_size, batch_size, num_workers)

    model = build_model(model_name).to(device)
    n_changed = set_dropout_rate(model, dropout_rate)
    print(f"[mc_dropout:{model_name}] set dropout rate={dropout_rate} on {n_changed} module(s)", flush=True)
    if n_changed == 0:
        raise RuntimeError(
            f"{model_name} has no nn.Dropout modules -- MC Dropout would be degenerate. "
            "Pick a different backbone or add an explicit dropout layer."
        )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    best_val_loss, best_state, bad_epochs = float("inf"), None, 0
    for epoch in range(epochs):
        t0 = time.time()
        model.train()
        train_loss, n = 0.0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
            n += images.size(0)
        train_loss /= n
        scheduler.step()

        model.eval()
        val_loss, n = 0.0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                loss = criterion(logits, labels)
                val_loss += loss.item() * images.size(0)
                n += images.size(0)
        val_loss /= n

        print(f"[mc_dropout:{model_name}] epoch {epoch:03d} train={train_loss:.4f} val={val_loss:.4f} "
              f"({time.time() - t0:.1f}s)", flush=True)

        if val_loss < best_val_loss:
            best_val_loss, bad_epochs = val_loss, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"[mc_dropout:{model_name}] early stop at epoch {epoch}", flush=True)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return {"model": model, "input_size": input_size, "best_val_loss": best_val_loss, "dropout_rate": dropout_rate}


def train_evidential(model_name: str, manifest_csv, device, num_classes=5, annealing_epochs=10,
                      epochs=30, batch_size=32, lr=3e-4, patience=7, seed=13, num_workers=2) -> dict:
    """Same architecture and optimizer schedule as train_baseline, but with
    the evidential loss (Sensoy et al.) in place of cross-entropy, and
    validation tracked by evidential loss too (comparing a CE val-loss
    against an evidential val-loss to pick 'best epoch' would not be
    apples-to-apples)."""
    torch.manual_seed(seed)
    input_size = input_size_for(model_name)
    train_ds, val_ds, train_loader, val_loader = _make_loaders(manifest_csv, input_size, batch_size, num_workers)

    model = build_model(model_name).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss, best_state, bad_epochs = float("inf"), None, 0
    for epoch in range(epochs):
        t0 = time.time()
        model.train()
        train_loss, n = 0.0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = evidential_loss(logits, labels, epoch, num_classes, annealing_epochs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
            n += images.size(0)
        train_loss /= n
        scheduler.step()

        model.eval()
        val_loss, n = 0.0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                loss = evidential_loss(logits, labels, epoch, num_classes, annealing_epochs)
                val_loss += loss.item() * images.size(0)
                n += images.size(0)
        val_loss /= n

        print(f"[evidential:{model_name}] epoch {epoch:03d} train={train_loss:.4f} val={val_loss:.4f} "
              f"({time.time() - t0:.1f}s)", flush=True)

        if val_loss < best_val_loss:
            best_val_loss, bad_epochs = val_loss, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"[evidential:{model_name}] early stop at epoch {epoch}", flush=True)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return {"model": model, "input_size": input_size, "best_val_loss": best_val_loss}


# ---------------------------------------------------------------------------
# Prediction helpers — each returns (probs, labels) on a given manifest split
# or an explicit (paths, labels) list, in a form calibration.py/selective.py/
# conformal.py can consume directly.
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_baseline(model, loader, device, temperature: float | None = None):
    model.eval()
    all_logits, all_labels = [], []
    for images, labels in loader:
        logits = model(images.to(device))
        all_logits.append(logits.cpu().numpy())
        all_labels.append(labels.numpy())
    logits = np.concatenate(all_logits)
    labels = np.concatenate(all_labels)
    if temperature is not None:
        probs = calibration.apply_temperature(logits, temperature)
    else:
        probs = calibration.softmax(logits)
    return probs, labels, logits


@torch.no_grad()
def predict_evidential(model, loader, device, num_classes: int):
    model.eval()
    all_probs, all_vacuity, all_labels = [], [], []
    for images, labels in loader:
        logits = model(images.to(device))
        evidence = logits_to_evidence(logits)
        alpha = evidence_to_alpha(evidence)
        all_probs.append(expected_probability(alpha).cpu().numpy())
        all_vacuity.append(vacuity(alpha, num_classes).cpu().numpy())
        all_labels.append(labels.numpy())
    return np.concatenate(all_probs), np.concatenate(all_vacuity), np.concatenate(all_labels)


def predict_mc_dropout(model, loader, device, n_passes: int = 20):
    all_mean_probs, all_entropy, all_labels = [], [], []
    for images, labels in loader:
        all_probs = mc_dropout_predict(model, images.to(device), n_passes=n_passes)
        mean_probs = all_probs.mean(dim=0)
        entropy = -(mean_probs * torch.log(mean_probs.clamp_min(1e-12))).sum(dim=1)
        all_mean_probs.append(mean_probs.cpu().numpy())
        all_entropy.append(entropy.cpu().numpy())
        all_labels.append(labels.numpy())
    return np.concatenate(all_mean_probs), np.concatenate(all_entropy), np.concatenate(all_labels)


@torch.no_grad()
def predict_ensemble(models: list, loader, device):
    """Averages softmax probabilities across ensemble members. Also returns
    the per-member spread (variance of the predicted probability for the
    predicted class) as the ensemble's uncertainty signal, since deep
    ensembles' value specifically comes from members disagreeing more under
    distribution shift (Lakshminarayanan et al.)."""
    for m in models:
        m.eval()
    all_member_probs = []  # (n_members, n_samples, n_classes) built incrementally
    all_labels = []
    per_batch_member_probs = [[] for _ in models]
    for images, labels in loader:
        images = images.to(device)
        for i, m in enumerate(models):
            logits = m(images)
            per_batch_member_probs[i].append(torch.softmax(logits, dim=1).cpu().numpy())
        all_labels.append(labels.numpy())
    member_probs = np.stack([np.concatenate(p) for p in per_batch_member_probs], axis=0)
    labels = np.concatenate(all_labels)
    mean_probs = member_probs.mean(axis=0)
    variance = member_probs.var(axis=0).sum(axis=1)  # total variance across classes, per sample
    return mean_probs, variance, labels
