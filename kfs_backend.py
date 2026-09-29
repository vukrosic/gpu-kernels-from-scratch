"""Run the same Triton code on an NVIDIA GPU or on any laptop.

With CUDA and Triton installed (Colab, a Linux GPU box): real Triton, DEVICE = "cuda".
Anywhere else (Mac, Windows, CPU-only Linux): minitriton, a small CPU stand-in that runs
your kernel one program at a time. It checks correctness, not speed.

Force the stand-in with KFS_CPU=1. On Linux without a GPU, TRITON_INTERPRET=1 runs real
Triton's own interpreter instead.
"""
import os
import sys

import torch

_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)


def _pick():
    if os.environ.get("KFS_CPU") == "1":
        return None
    try:
        import triton  # noqa: F401
    except ImportError:
        return None
    if torch.cuda.is_available():
        return "cuda"
    if os.environ.get("TRITON_INTERPRET") == "1":
        return "cpu"
    return None


DEVICE = _pick()
if DEVICE:
    import triton

    where = torch.cuda.get_device_name() if DEVICE == "cuda" else "cpu (TRITON_INTERPRET=1)"
    BACKEND = f"triton {triton.__version__} on {where}"
else:
    import minitriton
    import minitriton.language

    sys.modules["triton"] = minitriton
    sys.modules["triton.language"] = minitriton.language
    DEVICE = "cpu"
    BACKEND = "minitriton on cpu (checks correctness, not speed)"
