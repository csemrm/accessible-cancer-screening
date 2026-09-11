"""Model registry for the RQ1 efficiency/accuracy baseline benchmark (proposal S13).

Every entry is a timm tag verified to exist with pretrained ImageNet weights
in the pinned timm version (see requirements.txt) — checked interactively
with ``timm.list_models(..., pretrained=True)`` rather than assumed from
memory, since timm's naming scheme has changed across versions.
"""

from __future__ import annotations

import timm
import torch.nn as nn

NUM_CLASSES = 5

# name -> timm tag. Grouped to match the baseline categories in the proposal:
# CNN floor, full-size ViT ceiling, and the efficient-architecture family the
# proposed backbone (Section 9) must be compared against.
MODEL_REGISTRY: dict[str, str] = {
    # CNN baselines
    "resnet50": "resnet50.a1_in1k",
    "mobilenetv2_100": "mobilenetv2_100.ra_in1k",
    "efficientnet_lite0": "efficientnet_lite0.ra_in1k",
    # full-size ViT baselines
    "vit_small_patch16_224": "vit_small_patch16_224.augreg_in1k",
    "swin_tiny_patch4_window7_224": "swin_tiny_patch4_window7_224.ms_in1k",
    # efficient ViT family
    "deit_tiny_patch16_224": "deit_tiny_patch16_224.fb_in1k",
    "levit_128": "levit_128.fb_dist_in1k",
    "tiny_vit_5m_224": "tiny_vit_5m_224.in1k",
    "mobilevit_s": "mobilevit_s.cvnets_in1k",
    "mobilevitv2_100": "mobilevitv2_100.cvnets_in1k",
    "efficientvit_b0": "efficientvit_b0.r224_in1k",
    "efficientvit_m0": "efficientvit_m0.r224_in1k",
}


def build_model(name: str, pretrained: bool = True, num_classes: int = NUM_CLASSES) -> nn.Module:
    if name not in MODEL_REGISTRY:
        raise KeyError(f"unknown model {name!r}; available: {sorted(MODEL_REGISTRY)}")
    tag = MODEL_REGISTRY[name]
    return timm.create_model(tag, pretrained=pretrained, num_classes=num_classes)


def input_size_for(name: str) -> int:
    """Native training resolution timm's pretrained config recommends for this tag."""
    tag = MODEL_REGISTRY[name]
    cfg = timm.get_pretrained_cfg(tag)
    return cfg.input_size[-1] if cfg is not None and cfg.input_size else 224
