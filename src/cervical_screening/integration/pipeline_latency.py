"""CPU latency cost of trustworthiness (proposal S19/RQ5): how much slower is
one prediction when it also produces a relevancy-propagation explanation,
compared to a bare forward pass. Temperature scaling and conformal
prediction-set construction are both O(1) elementwise operations on a
logit/probability vector already in hand (microseconds) and are not
separately profiled here -- the only stage with a real, measurable cost is
the explanation's extra backward pass and rollout across all transformer
blocks.

CPU, not GPU: proposal S19's efficiency claim is about the no-GPU-assumed
rural/clinic deployment case, not a GPU benchmark.
"""

from __future__ import annotations

import time

import torch

from cervical_screening.explainability.relevancy_propagation import relevancy_propagation


def measure_bare_forward_latency(model: torch.nn.Module, input_size: int,
                                  n_warmup: int = 5, n_iters: int = 30) -> dict:
    model = model.eval().to("cpu")
    x = torch.randn(1, 3, input_size, input_size)

    with torch.no_grad():
        for _ in range(n_warmup):
            model(x)

        times = []
        for _ in range(n_iters):
            start = time.perf_counter()
            model(x)
            times.append(time.perf_counter() - start)

    times.sort()
    return {"median_latency_ms": times[len(times) // 2] * 1000}


def measure_explained_latency(model: torch.nn.Module, input_size: int, grid_size: int,
                               num_prefix_tokens: int = 1, n_warmup: int = 5, n_iters: int = 30) -> dict:
    """Forward pass + relevancy-propagation heatmap for the model's own
    predicted class, end to end -- the full cost of returning a prediction
    that also comes with an explanation."""
    model = model.eval().to("cpu")
    x = torch.randn(1, 3, input_size, input_size)

    def _predict_and_explain():
        with torch.no_grad():
            pred_class = int(model(x).argmax(dim=-1).item())
        relevancy_propagation(model, x, target_class=pred_class, grid_size=grid_size,
                               num_prefix_tokens=num_prefix_tokens)

    for _ in range(n_warmup):
        _predict_and_explain()

    times = []
    for _ in range(n_iters):
        start = time.perf_counter()
        _predict_and_explain()
        times.append(time.perf_counter() - start)

    times.sort()
    return {"median_latency_ms": times[len(times) // 2] * 1000}
