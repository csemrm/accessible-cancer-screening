"""Shared attention-capture hook for a timm VisionTransformer (DeiT-Tiny).

timm's fused-attention path (F.scaled_dot_product_attention) never
materializes the attention matrix, so both attention rollout and relevancy
propagation need the manual softmax path with the attention tensor retained
(and, for relevancy propagation, its gradient). This monkeypatches each
block's Attention.forward for the duration of a `with` block and restores
the original afterward, rather than permanently modifying the model.
"""

from __future__ import annotations

import types

import torch


def _manual_forward(self, x, attn_mask=None, is_causal=False):
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    q, k = self.q_norm(q), self.k_norm(k)

    q = q * self.scale
    attn = q @ k.transpose(-2, -1)
    attn = attn.softmax(dim=-1)
    if attn.requires_grad:
        attn.retain_grad()
    self._captured_attn = attn  # (B, heads, N, N), softmaxed, grad-retained if applicable

    attn_dropped = self.attn_drop(attn)
    x = attn_dropped @ v
    x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
    x = self.norm(x)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x


class AttentionCapture:
    """Patches every transformer block's attention module to record its
    softmaxed attention matrix (grad-retained) into `captured` during
    __enter__/__exit__, restoring the original forward on exit."""

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self._originals: list[tuple[torch.nn.Module, "types.MethodType"]] = []

    def __enter__(self) -> "AttentionCapture":
        for block in self.model.blocks:
            attn_module = block.attn
            attn_module.fused_attn = False
            self._originals.append((attn_module, attn_module.forward))
            attn_module.forward = types.MethodType(_manual_forward, attn_module)
        return self

    def __exit__(self, *exc_info) -> None:
        for attn_module, original_forward in self._originals:
            attn_module.forward = original_forward
            if hasattr(attn_module, "_captured_attn"):
                del attn_module._captured_attn
        self._originals.clear()

    @property
    def captured(self) -> list[torch.Tensor]:
        """One (B, heads, N, N) tensor per block, in block order."""
        return [block.attn._captured_attn for block in self.model.blocks]
