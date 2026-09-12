"""FLOPs measurement for proposal S19 -- params/FLOPs/memory/latency must all
be measured together, since parameter count alone can hide a high-FLOPs
design (e.g. a small number of very wide, expensive layers).

Uses torch's own FlopCounterMode (torch>=2.4) rather than adding a
third-party dependency (fvcore/thop/ptflops) for one measurement.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.flop_counter import FlopCounterMode


@torch.no_grad()
def count_flops(model: nn.Module, input_size: int, batch_size: int = 1) -> int:
    """Total FLOPs (multiply-adds counted as 2 FLOPs, torch's own convention)
    for one forward pass at the given input size."""
    model = model.eval()
    x = torch.randn(batch_size, 3, input_size, input_size)
    with FlopCounterMode(display=False) as counter:
        model(x)
    return int(counter.get_total_flops())
