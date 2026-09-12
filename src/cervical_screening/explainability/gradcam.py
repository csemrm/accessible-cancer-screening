"""Grad-CAM adapted for ViT (Selvaraju et al., 2017, applied to a
transformer block's output tokens instead of a CNN's last conv feature map) --
the second comparison baseline from proposal S17.

Target layer is the SECOND-TO-LAST transformer block's output, not the last:
DeiT-Tiny pools only the CLS token for classification (`global_pool="token"`),
so after the final block nothing downstream mixes information back into the
patch-token positions, and their gradient there is identically zero
(verified empirically -- every patch token's gradient at the last block's
output was exactly 0, while the second-to-last block's patch gradients vary
normally, since the final block's attention still mixes them into CLS).
This is the standard fix used for CLS-pooled ViTs in the Grad-CAM-for-
transformers literature, not specific to this codebase."""

from __future__ import annotations

import torch

from cervical_screening.explainability.heatmap import patches_to_heatmap


def vit_gradcam(model: torch.nn.Module, image: torch.Tensor, target_class: int,
                grid_size: int, num_prefix_tokens: int = 1, target_block: int = -2) -> torch.Tensor:
    """image: (1, C, H, W). Returns (1, H, W) heatmap in [0, 1]."""
    model.eval()
    activations = {}

    def save_activation(_module, _inp, output):
        output.retain_grad()
        activations["value"] = output

    handle = model.blocks[target_block].register_forward_hook(save_activation)
    try:
        logits = model(image)
        model.zero_grad(set_to_none=True)
        logits[0, target_class].backward()
    finally:
        handle.remove()

    with torch.no_grad():
        tokens = activations["value"]              # (B, N, C)
        grad = tokens.grad                          # (B, N, C)
        patch_tokens = tokens[:, num_prefix_tokens:, :]
        patch_grad = grad[:, num_prefix_tokens:, :]

        channel_weights = patch_grad.mean(dim=1, keepdim=True)           # (B, 1, C) -- GAP over tokens
        cam = (channel_weights * patch_tokens).sum(dim=-1).clamp_min(0)  # (B, N_patches)
        return patches_to_heatmap(cam, grid_size, image.shape[-1])
