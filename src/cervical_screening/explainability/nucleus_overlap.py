"""Localization-fidelity metric standing in for proposal S17's "overlap with
nucleus-segmentation proxies": SIPaKMeD ships real expert-marked nucleus
boundaries (not a proxy) via its per-cell polygon annotations, so this
compares each heatmap's relevance mass against the actual nucleus region
rather than a derived approximation.
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image, ImageDraw


def rasterize_nucleus_mask(polygons_crop_local: list[list[list[float]]], crop_width: int, crop_height: int,
                            input_size: int) -> torch.Tensor:
    """polygons_crop_local: nucleus polygon(s) in native crop-pixel coordinates
    (as written by crop_sipakmed.py's --nucleus-manifest). Scales to the
    model's (input_size, input_size) grid with the same independent x/y
    resize build_transform uses (plain Resize, no aspect-ratio preservation),
    so the mask lines up with heatmaps computed at that resolution."""
    mask_img = Image.new("L", (input_size, input_size), 0)
    draw = ImageDraw.Draw(mask_img)
    sx, sy = input_size / crop_width, input_size / crop_height
    for polygon in polygons_crop_local:
        scaled = [(x * sx, y * sy) for x, y in polygon]
        if len(scaled) >= 3:
            draw.polygon(scaled, fill=1)
    return torch.from_numpy(np.array(mask_img, dtype=np.float32))


def concentration_ratio(heatmap: torch.Tensor, nucleus_mask: torch.Tensor) -> float:
    """> 1 means the heatmap concentrates relevance inside the nucleus more
    than chance (relative to the nucleus's share of the image area); 1.0 is
    exactly chance-level; < 1 means relevance is pushed away from the nucleus."""
    total_mass = heatmap.sum().clamp_min(1e-12)
    mass_in_nucleus = (heatmap * nucleus_mask).sum() / total_mass
    area_fraction = nucleus_mask.mean().clamp_min(1e-12)
    return float((mass_in_nucleus / area_fraction).item())
