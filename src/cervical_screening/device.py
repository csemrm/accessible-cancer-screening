"""Device selection with a compute-capability sanity check.

Cloud GPU pools (Kaggle's in particular) can hand out an older card whose
compute capability the installed torch build no longer ships kernels for —
e.g. a Tesla P100 (sm_60) against a torch build compiled only for sm_70+.
torch.cuda.is_available() returns True in that case; the failure only
surfaces on the first real CUDA op, as an opaque "no kernel image is
available for execution on the device" error. Check compatibility up front
instead and fall back to CPU.
"""

from __future__ import annotations

import torch


def select_device() -> torch.device:
    if not torch.cuda.is_available():
        return torch.device("cpu")

    major, minor = torch.cuda.get_device_capability(0)
    cap_str = f"sm_{major}{minor}"
    supported = torch.cuda.get_arch_list()
    if cap_str not in supported:
        name = torch.cuda.get_device_name(0)
        print(f"GPU {name} ({cap_str}) not supported by this torch build "
              f"(supports {supported}) — falling back to CPU")
        return torch.device("cpu")
    return torch.device("cuda")
