"""Raw attention rollout (Abnar & Zuidema, 2020) -- the "raw attention"
comparison baseline from proposal S17, as distinct from relevancy
propagation (the primary method) and Grad-CAM (the other baseline)."""

from __future__ import annotations

import torch

from cervical_screening.explainability.heatmap import patches_to_heatmap
from cervical_screening.explainability.rollout import cls_relevance_map, rollout
from cervical_screening.explainability.vit_hooks import AttentionCapture


@torch.no_grad()
def attention_rollout(model: torch.nn.Module, image: torch.Tensor, grid_size: int,
                       num_prefix_tokens: int = 1) -> torch.Tensor:
    """image: (B, C, H, W) already normalized/resized to the model's input size.
    Returns (B, H, W) heatmap in [0, 1]."""
    model.eval()
    with AttentionCapture(model):
        model(image)
        layer_matrices = [attn.mean(dim=1) for attn in _last_captured(model)]  # mean over heads -> (B, N, N)

    joint = rollout(layer_matrices)
    patch_relevance = cls_relevance_map(joint, num_prefix_tokens)
    return patches_to_heatmap(patch_relevance, grid_size, image.shape[-1])


def _last_captured(model: torch.nn.Module) -> list[torch.Tensor]:
    return [block.attn._captured_attn for block in model.blocks]
