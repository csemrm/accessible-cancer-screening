"""Resource-efficiency benchmark across the full RQ1 registry (proposal S19).

Params, FLOPs, CPU inference latency, and CPU memory delta are all
architecture properties (they don't depend on trained weight VALUES, only on
the model's structure and input size), so this runs untrained
(pretrained=False) models entirely locally on CPU -- no GPU, no Colab
session, no dataset download needed.

Usage:
    python -m cervical_screening.measure_efficiency --out runs/efficiency/benchmark.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from cervical_screening.eval.efficiency import count_parameters, measure_cpu_latency
from cervical_screening.eval.flops import count_flops
from cervical_screening.models.registry import MODEL_REGISTRY, build_model, input_size_for

_MEMORY_WORKER = """
import resource, sys
sys.path.insert(0, {src_path!r})
import torch
from cervical_screening.models.registry import build_model, input_size_for
model = build_model({name!r}, pretrained=False).eval()
input_size = input_size_for({name!r})
before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
x = torch.randn(1, 3, input_size, input_size)
with torch.no_grad():
    model(x)
after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
import platform
divisor = 1024.0 if platform.system() == "Darwin" else 1.0
print(max(0.0, (after - before) / divisor / 1024.0))
"""


def measure_memory_isolated(name: str, src_path: str) -> float:
    """Runs the memory measurement in a fresh subprocess, one per model.
    ru_maxrss is a monotonically non-decreasing high-water mark for the whole
    process, not a live delta -- measuring several models' FLOPs/latency
    first in the same process (as an earlier version of this script did)
    means every later model's "delta" is masked by the highest peak any
    prior model already established, silently producing 0.0 for everything
    after the first. Isolating each model in its own subprocess is what
    actually fixes this, not just reordering within one process."""
    out = subprocess.run([sys.executable, "-c", _MEMORY_WORKER.format(name=name, src_path=src_path)],
                          capture_output=True, text=True, check=True)
    return float(out.stdout.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    src_path = str(Path(__file__).resolve().parent.parent)
    results = {}
    for name in MODEL_REGISTRY:
        print(f"measuring {name}...", flush=True)
        model = build_model(name, pretrained=False)
        input_size = input_size_for(name)

        params = count_parameters(model)
        flops = count_flops(model, input_size)
        latency = measure_cpu_latency(model, input_size)
        memory_mb = measure_memory_isolated(name, src_path)

        results[name] = {
            "input_size": input_size,
            "params": params,
            "flops": flops,
            "cpu_latency_ms": latency,
            "cpu_memory_delta_mb": memory_mb,
        }
        print(f"  params={params:,} flops={flops:,} "
              f"median_latency_ms={latency['median_latency_ms']:.1f} "
              f"memory_delta_mb={memory_mb:.1f}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {len(results)} model results to {args.out}", flush=True)


if __name__ == "__main__":
    main()
