"""Device selection with a compute-capability sanity check.

Cloud GPU pools (Kaggle's in particular) can hand out an older card whose
compute capability the installed torch build no longer ships kernels for —
e.g. a Tesla P100 (sm_60) against a torch build compiled only for sm_70+.
torch.cuda.is_available() returns True in that case; the failure only
surfaces on the first real CUDA op, as an opaque "no kernel image is
available for execution on the device" error. Check compatibility up front
instead and fall back to CPU.

The check is an actual tiny CUDA op, not a static arch-list string match.
An earlier version compared torch.cuda.get_device_capability() against
torch.cuda.get_arch_list() as an exact string (e.g. "sm_89"), which is too
strict: PTX forward compatibility means a build with kernels for sm_86 can
still run correctly on an sm_89 device (a newer point-revision within the
same architecture generation), so an exact-match check produced a false
"unsupported" verdict and fell back to CPU on a real Colab L4 session,
even though a real matmul and a real model forward pass both succeeded on
that exact GPU when tested directly. Running the real op ourselves avoids
needing to encode CUDA's forward-compatibility rules by hand.
"""

from __future__ import annotations

import torch


def select_device() -> torch.device:
    if not torch.cuda.is_available():
        return torch.device("cpu")

    try:
        x = torch.randn(8, 8, device="cuda")
        (x @ x).sum().item()
    except RuntimeError as e:
        name = torch.cuda.get_device_name(0)
        cap = torch.cuda.get_device_capability(0)
        print(f"GPU {name} (compute {cap}) failed a real CUDA op "
              f"({e}) — falling back to CPU")
        return torch.device("cpu")
    return torch.device("cuda")
