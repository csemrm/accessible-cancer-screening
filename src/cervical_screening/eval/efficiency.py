"""Efficiency measurements for proposal S19 — the 'lightweight' claim must be
measured (params, FLOPs, latency, memory), not asserted from an architecture's
name or parameter count alone.
"""

from __future__ import annotations

import time

import torch
import torch.nn as nn


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def measure_cpu_latency(model: nn.Module, input_size: int, batch_size: int = 1,
                         n_warmup: int = 5, n_iters: int = 30) -> dict:
    """Wall-clock CPU inference latency — the realistic no-GPU-assumed rural/
    clinic-deployment case (proposal S19), not a GPU benchmark."""
    model = model.eval().to("cpu")
    x = torch.randn(batch_size, 3, input_size, input_size)

    for _ in range(n_warmup):
        model(x)

    times = []
    for _ in range(n_iters):
        start = time.perf_counter()
        model(x)
        times.append(time.perf_counter() - start)

    times.sort()
    return {
        "batch_size": batch_size,
        "median_latency_ms": times[len(times) // 2] * 1000,
        "p90_latency_ms": times[int(len(times) * 0.9)] * 1000,
    }


@torch.no_grad()
def measure_peak_rss_delta_mb(model: nn.Module, input_size: int, batch_size: int = 1) -> float:
    """Process peak-RSS delta around a forward pass, as a CPU-inference memory
    proxy. Deliberately not tracemalloc: torch's C++-allocated tensors are
    invisible to Python's allocator, so tracemalloc would silently under-report.
    RSS delta is coarser (it includes any other process activity in the window)
    but at least measures real memory the OS actually committed.

    CALLER MUST ISOLATE EACH MODEL IN ITS OWN PROCESS. ru_maxrss is a
    monotonically non-decreasing high-water mark for the whole process
    lifetime, not a live delta -- calling this (or any other measurement that
    runs a forward pass) on a second model in the same process after a first,
    larger model already ran will silently return 0.0, because the second
    model's forward pass never exceeds the peak the first one already
    established. measure_efficiency.py's measure_memory_isolated() spawns a
    fresh subprocess per model specifically to avoid this; do not call this
    function directly in a loop over multiple models in one process."""
    import resource

    model = model.eval().to("cpu")
    x = torch.randn(batch_size, 3, input_size, input_size)

    before_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    model(x)
    after_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    # ru_maxrss is KB on Linux, bytes on macOS.
    import platform
    divisor = 1024.0 if platform.system() == "Darwin" else 1.0
    return max(0.0, (after_kb - before_kb) / divisor / 1024.0)
