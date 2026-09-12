"""Shared utility: a (num_patches,) relevance vector -> a (H, W) heatmap in
[0, 1], upsampled from the patch grid to image resolution."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def patches_to_heatmap(patch_relevance: torch.Tensor, grid_size: int, image_size: int) -> torch.Tensor:
    """patch_relevance: (B, num_patches) with num_patches == grid_size**2.
    Returns (B, image_size, image_size), each sample min-max normalized to [0, 1]."""
    B = patch_relevance.shape[0]
    grid = patch_relevance.reshape(B, 1, grid_size, grid_size)
    up = F.interpolate(grid, size=(image_size, image_size), mode="bilinear", align_corners=False)
    up = up.reshape(B, image_size, image_size)
    mins = up.amin(dim=(1, 2), keepdim=True)
    maxs = up.amax(dim=(1, 2), keepdim=True)
    return (up - mins) / (maxs - mins).clamp_min(1e-12)
