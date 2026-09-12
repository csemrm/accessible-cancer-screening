"""Relevancy propagation (Chefer, Gur & Wolf, "Transformer Interpretability
Beyond Attention Visualization", CVPR 2021) -- the primary explainability
method under proposal S17/RQ3.

Per block, the camera matrix is grad(attn) * attn, averaged over heads and
clamped at zero (only attention the gradient says increases the target
class's logit contributes), then chained across blocks with the same
rollout math as plain attention rollout. This is what distinguishes it from
raw attention rollout: the propagated relevance is class-discriminative
(computed per target class via backprop) rather than a fixed property of
the attention maps alone.
"""

from __future__ import annotations

import torch

from cervical_screening.explainability.heatmap import patches_to_heatmap
from cervical_screening.explainability.rollout import cls_relevance_map, rollout
from cervical_screening.explainability.vit_hooks import AttentionCapture


def relevancy_propagation(model: torch.nn.Module, image: torch.Tensor, target_class: int,
                           grid_size: int, num_prefix_tokens: int = 1) -> torch.Tensor:
    """image: (1, C, H, W). Returns (1, H, W) heatmap in [0, 1]."""
    model.eval()
    with AttentionCapture(model) as cap:
        logits = model(image)
        model.zero_grad(set_to_none=True)
        logits[0, target_class].backward()

        camera_matrices = []
        for attn in cap.captured:
            grad = attn.grad
            cam = (grad * attn).clamp_min(0.0).mean(dim=1)  # mean over heads -> (B, N, N)
            camera_matrices.append(cam)

    with torch.no_grad():
        joint = rollout(camera_matrices)
        patch_relevance = cls_relevance_map(joint, num_prefix_tokens)
        return patches_to_heatmap(patch_relevance, grid_size, image.shape[-1])
