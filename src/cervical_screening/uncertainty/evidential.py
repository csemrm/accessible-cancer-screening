"""Evidential deep learning (Sensoy, Kaplan & Kandemir, NeurIPS 2018),
proposal S9/S18 — a single-forward-pass alternative to MC Dropout and deep
ensembles. The network's logits are treated as non-negative "evidence" for
each class via a softplus activation; evidence + 1 gives the concentration
parameters of a Dirichlet distribution over the class-probability simplex,
whose spread directly gives an uncertainty ("vacuity") estimate without any
repeated forward passes — the property that makes it worth testing here at
all, since MC Dropout/ensembles multiply inference cost by 5-30x.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def logits_to_evidence(logits: torch.Tensor) -> torch.Tensor:
    """Non-negative evidence per class. Softplus rather than ReLU so evidence
    stays differentiable and non-zero everywhere (a ReLU'd class gets exactly
    zero gradient once its evidence is negative, stalling training for it)."""
    return F.softplus(logits)


def evidence_to_alpha(evidence: torch.Tensor) -> torch.Tensor:
    return evidence + 1.0


def dirichlet_strength(alpha: torch.Tensor) -> torch.Tensor:
    return alpha.sum(dim=1)


def expected_probability(alpha: torch.Tensor) -> torch.Tensor:
    return alpha / dirichlet_strength(alpha).unsqueeze(1)


def vacuity(alpha: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Epistemic uncertainty: num_classes / Dirichlet strength. Approaches 0
    as evidence accumulates for some class, approaches 1 (maximum) when the
    Dirichlet is uniform (alpha_k=1 for all k, i.e. zero evidence anywhere)."""
    return num_classes / dirichlet_strength(alpha)


def _kl_dirichlet_to_uniform(alpha: torch.Tensor) -> torch.Tensor:
    """KL[Dir(alpha) || Dir(1,...,1)], closed form (Sensoy et al. Eq. 5)."""
    num_classes = alpha.shape[1]
    ones = torch.ones_like(alpha)
    sum_alpha = alpha.sum(dim=1, keepdim=True)

    term1 = torch.lgamma(sum_alpha.squeeze(1)) - torch.lgamma(ones.sum(dim=1))
    term2 = torch.lgamma(alpha).sum(dim=1) - torch.lgamma(ones).sum(dim=1)
    term3 = ((alpha - ones) * (torch.digamma(alpha) - torch.digamma(sum_alpha))).sum(dim=1)

    kl = term1 - term2 + term3
    return kl


def evidential_loss(logits: torch.Tensor, labels: torch.Tensor, epoch: int,
                     num_classes: int, annealing_epochs: int = 10) -> torch.Tensor:
    """Type II maximum-likelihood loss plus an annealed KL penalty that
    shrinks evidence for classes the model did NOT predict correctly,
    following Sensoy et al. The KL term is computed on alpha_tilde (evidence
    kept only for the wrong classes, correct-class evidence zeroed out) so
    it never penalizes the network for being confidently right."""
    evidence = logits_to_evidence(logits)
    alpha = evidence_to_alpha(evidence)
    sum_alpha = dirichlet_strength(alpha).unsqueeze(1)

    y_onehot = F.one_hot(labels, num_classes).float()

    # Bayes-risk cross-entropy: E_{p~Dir(alpha)}[-sum_k y_k log p_k]
    ce_loss = (y_onehot * (torch.digamma(sum_alpha) - torch.digamma(alpha))).sum(dim=1)

    alpha_tilde = y_onehot + (1.0 - y_onehot) * alpha
    kl = _kl_dirichlet_to_uniform(alpha_tilde)

    annealing_coef = min(1.0, epoch / annealing_epochs)
    return (ce_loss + annealing_coef * kl).mean()
