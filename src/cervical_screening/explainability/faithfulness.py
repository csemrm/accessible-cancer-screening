"""Deletion/insertion faithfulness (Petsiuk et al., RISE, BMVC 2018), applied
at patch granularity (the model's own 16x16 tokenization) rather than raw
pixels, since patch-level relevance is what all three RQ3 explanation
methods actually produce.

Deletion: start from the original image, progressively replace the
most-relevant remaining patches with a heavily blurred version (not black or
mean-color, to avoid a trivially out-of-distribution baseline the model can
key off of) and track target-class probability -- a faithful explanation
causes a fast drop, so a LOWER area under this curve is better.

Insertion: the reverse -- start fully blurred, progressively reveal the
most-relevant patches first, track target-class probability rising -- a
faithful explanation causes a fast rise, so a HIGHER area under this curve
is better.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def _gaussian_blur(image: torch.Tensor, kernel_size: int = 51, sigma: float = 15.0) -> torch.Tensor:
    channels = image.shape[1]
    coords = torch.arange(kernel_size, dtype=image.dtype, device=image.device) - kernel_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = (g / g.sum()).reshape(1, 1, -1)
    kernel_1d_h = g.reshape(1, 1, 1, kernel_size).expand(channels, 1, 1, kernel_size)
    kernel_1d_v = g.reshape(1, 1, kernel_size, 1).expand(channels, 1, kernel_size, 1)
    pad = kernel_size // 2
    blurred = F.conv2d(F.pad(image, (pad, pad, 0, 0), mode="reflect"), kernel_1d_h, groups=channels)
    blurred = F.conv2d(F.pad(blurred, (0, 0, pad, pad), mode="reflect"), kernel_1d_v, groups=channels)
    return blurred


def _patch_order(heatmap: torch.Tensor, patch_size: int) -> np.ndarray:
    """heatmap: (H, W). Returns patch indices (row-major over the patch grid)
    sorted by descending mean relevance."""
    H, W = heatmap.shape
    grid_h, grid_w = H // patch_size, W // patch_size
    pooled = heatmap[: grid_h * patch_size, : grid_w * patch_size].reshape(
        grid_h, patch_size, grid_w, patch_size
    ).mean(dim=(1, 3))
    return torch.argsort(pooled.flatten(), descending=True).cpu().numpy()


def _apply_patches(image: torch.Tensor, fill: torch.Tensor, patch_size: int,
                    grid_w: int, patch_indices: np.ndarray) -> torch.Tensor:
    """Replace the given patch indices in `image` with the corresponding
    patches from `fill`. Returns a new tensor; does not mutate inputs."""
    out = image.clone()
    for idx in patch_indices:
        r, c = divmod(int(idx), grid_w)
        rs, cs = r * patch_size, c * patch_size
        out[:, :, rs:rs + patch_size, cs:cs + patch_size] = fill[:, :, rs:rs + patch_size, cs:cs + patch_size]
    return out


@torch.no_grad()
def _auc(fractions: np.ndarray, probs: np.ndarray) -> float:
    return float(np.sum((probs[1:] + probs[:-1]) / 2.0 * np.diff(fractions)))


@torch.no_grad()
def deletion_auc(model: torch.nn.Module, image: torch.Tensor, heatmap: torch.Tensor,
                  target_class: int, patch_size: int = 16, n_steps: int = 20) -> float:
    """image: (1, C, H, W). heatmap: (H, W). Returns the deletion AUC (lower = more faithful)."""
    model.eval()
    grid_w = image.shape[-1] // patch_size
    order = _patch_order(heatmap, patch_size)
    blurred = _gaussian_blur(image)

    chunk = max(1, len(order) // n_steps)
    fractions, probs = [0.0], [_target_prob(model, image, target_class)]
    for step_end in range(chunk, len(order) + chunk, chunk):
        deleted = order[:step_end]
        perturbed = _apply_patches(image, blurred, patch_size, grid_w, deleted)
        fractions.append(len(deleted) / len(order))
        probs.append(_target_prob(model, perturbed, target_class))

    return _auc(np.array(fractions), np.array(probs))


@torch.no_grad()
def insertion_auc(model: torch.nn.Module, image: torch.Tensor, heatmap: torch.Tensor,
                   target_class: int, patch_size: int = 16, n_steps: int = 20) -> float:
    """image: (1, C, H, W). heatmap: (H, W). Returns the insertion AUC (higher = more faithful)."""
    model.eval()
    grid_w = image.shape[-1] // patch_size
    order = _patch_order(heatmap, patch_size)
    blurred = _gaussian_blur(image)

    chunk = max(1, len(order) // n_steps)
    fractions = [0.0]
    probs = [_target_prob(model, blurred, target_class)]
    for step_end in range(chunk, len(order) + chunk, chunk):
        inserted = order[:step_end]
        perturbed = _apply_patches(blurred, image, patch_size, grid_w, inserted)
        fractions.append(len(inserted) / len(order))
        probs.append(_target_prob(model, perturbed, target_class))

    return _auc(np.array(fractions), np.array(probs))


@torch.no_grad()
def _target_prob(model: torch.nn.Module, image: torch.Tensor, target_class: int) -> float:
    probs = torch.softmax(model(image), dim=-1)
    return float(probs[0, target_class].item())
