"""MC Dropout (Gal & Ghahramani, ICML 2016) — proposal S18/RQ2 baseline.

Keeps dropout layers active at inference and averages multiple stochastic
forward passes. This is the multi-pass baseline evidential deep learning
(evidential.py) is being tested against precisely because MC Dropout
multiplies inference cost by the number of passes, which is the cost
evidential uncertainty exists to avoid for edge/point-of-care deployment
(proposal S19).
"""

from __future__ import annotations

import torch
import torch.nn as nn


def set_dropout_rate(model: nn.Module, p: float) -> int:
    """Overrides every Dropout submodule's rate. Required before training,
    not just before inference: EfficientViT-M0's timm implementation ships
    exactly one Dropout module (in the classifier head) at p=0.0 by default
    -- with that default, MC Dropout passes are numerically identical and
    the whole method is degenerate. MC Dropout's Bayesian-approximation
    justification (Gal & Ghahramani) requires dropout to have been present
    *during training*, so this must be set before the MC-Dropout model
    variant is trained, not patched in only at evaluation time. Returns the
    number of modules changed, so a return of 0 is a loud signal something
    is wrong rather than a silent no-op."""
    n_changed = 0
    for module in model.modules():
        if isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            module.p = p
            n_changed += 1
    return n_changed


def enable_dropout(model: nn.Module) -> None:
    """Set the model to eval() (fixes BatchNorm running stats, disables
    non-dropout stochastic layers) while forcing every Dropout submodule
    back into train mode, so only dropout stays stochastic across passes."""
    model.eval()
    for module in model.modules():
        if isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            module.train()


@torch.no_grad()
def mc_dropout_predict(model: nn.Module, images: torch.Tensor, n_passes: int = 20) -> torch.Tensor:
    """Returns (n_passes, batch, n_classes) softmax probabilities. Caller
    reduces this how it needs — mean for the point prediction, variance or
    predictive entropy of the mean for uncertainty."""
    enable_dropout(model)
    all_probs = []
    for _ in range(n_passes):
        logits = model(images)
        all_probs.append(torch.softmax(logits, dim=1))
    return torch.stack(all_probs, dim=0)


def predictive_entropy(mean_probs: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Entropy of the mean prediction — captures both aleatoric and
    epistemic uncertainty combined, the standard MC-Dropout uncertainty
    signal used for selective prediction."""
    return -(mean_probs * torch.log(mean_probs.clamp_min(eps))).sum(dim=1)


def mutual_information(all_probs: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """BALD-style mutual information: entropy of the mean minus the mean of
    the per-pass entropies. Isolates epistemic uncertainty specifically
    (high when passes disagree with each other, not just when the mean
    prediction itself is spread across classes)."""
    mean_probs = all_probs.mean(dim=0)
    entropy_of_mean = predictive_entropy(mean_probs, eps)
    mean_of_entropies = -(all_probs * torch.log(all_probs.clamp_min(eps))).sum(dim=2).mean(dim=0)
    return entropy_of_mean - mean_of_entropies
