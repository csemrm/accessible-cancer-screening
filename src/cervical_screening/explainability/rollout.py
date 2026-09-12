"""Shared rollout math for attention_rollout.py and relevancy_propagation.py.

Both methods reduce to: build one (N, N) matrix per transformer block, add
the identity to account for the residual connection around attention,
row-normalize, then chain-multiply across blocks (Abnar & Zuidema, 2020).
They differ only in what goes into the per-block matrix before this shared
step (raw softmax attention vs. gradient-weighted attention).
"""

from __future__ import annotations

import torch


def rollout(layer_matrices: list[torch.Tensor]) -> torch.Tensor:
    """layer_matrices: list of (B, N, N) tensors, one per block, in block order.
    Returns (B, N, N) joint attention after chaining all blocks."""
    batch_size, num_tokens, _ = layer_matrices[0].shape
    eye = torch.eye(num_tokens, device=layer_matrices[0].device).expand(batch_size, num_tokens, num_tokens)

    augmented = [m + eye for m in layer_matrices]
    normalized = [m / m.sum(dim=-1, keepdim=True).clamp_min(1e-12) for m in augmented]

    joint = normalized[0]
    for m in normalized[1:]:
        joint = m @ joint
    return joint


def cls_relevance_map(joint_attention: torch.Tensor, num_prefix_tokens: int = 1) -> torch.Tensor:
    """Extract the CLS token's row (relevance over patch tokens) and drop the
    prefix tokens (CLS itself, plus any register tokens) from the columns.
    Returns (B, num_patches)."""
    cls_row = joint_attention[:, 0, :]
    return cls_row[:, num_prefix_tokens:]
