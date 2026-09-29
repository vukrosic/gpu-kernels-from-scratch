"""minitriton: a tiny CPU stand-in for Triton, so the course runs on any laptop.

Your kernel code stays real Triton code. `kfs_backend` swaps this module in as `triton`
only when there is no NVIDIA GPU. It runs every program of the grid one after another,
in a shuffled order (a GPU promises no order either). It checks correctness, not speed.
"""
import inspect
import random

import numpy as np

from . import language
from .language import IllegalMemoryAccess, Pointer, cdiv

__version__ = "minitriton"


def next_power_of_2(n):
    return 1 if n <= 1 else 1 << (int(n) - 1).bit_length()


def _as_pointer(v, name):
    try:
        import torch
    except ImportError:
        torch = None
    if torch is not None and isinstance(v, torch.Tensor):
        st = v.untyped_storage()
        whole = torch.empty(0, dtype=v.dtype).set_(st, 0, (st.nbytes() // v.element_size(),))
        return Pointer(whole.numpy(), v.storage_offset(), v.numel(), name)
    if isinstance(v, np.ndarray):
        if not v.flags.c_contiguous:
            raise ValueError(f"{name}: pass a contiguous NumPy array (np.ascontiguousarray)")
        return Pointer(v.reshape(-1), 0, v.size, name)
    return v


_LAUNCH_OPTIONS = {"num_warps", "num_stages", "num_ctas", "maxnreg"}  # GPU scheduling hints; nothing to do on a CPU


class JITFunction:
    def __init__(self, fn):
        self.fn = fn
        self.params = list(inspect.signature(fn).parameters)
        self.__name__ = fn.__name__

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            names = self.params
            args = [_as_pointer(a, names[i] if i < len(names) else f"arg{i}") for i, a in enumerate(args)]
            kwargs = {k: _as_pointer(v, k) for k, v in kwargs.items() if k not in _LAUNCH_OPTIONS}
            meta = {**dict(zip(names, args)), **kwargs}  # what a grid lambda sees as META
            g = grid(meta) if callable(grid) else grid
            g = tuple(int(x) for x in (g if isinstance(g, (tuple, list)) else (g,)))
            g = (g + (1, 1, 1))[:3]
            pids = [(i, j, k) for k in range(g[2]) for j in range(g[1]) for i in range(g[0])]
            random.shuffle(pids)
            language._state["grid"] = g
            with np.errstate(all="ignore"):  # a GPU makes inf and NaN without a warning; so does this
                for pid in pids:
                    language._state["pid"] = pid
                    self.fn(*args, **kwargs)

        return launch

    def __call__(self, *a, **k):
        raise RuntimeError(f"{self.__name__} is a kernel: launch it with {self.__name__}[grid](...)")


def jit(fn=None, **_options):
    if fn is None:
        return lambda f: JITFunction(f)
    return JITFunction(fn)


class Config:
    """One set of compile-time constants for triton.autotune to try, plus GPU scheduling hints."""

    def __init__(self, kwargs, num_warps=4, num_stages=3, **_options):
        self.kwargs = dict(kwargs)
        self.num_warps = num_warps
        self.num_stages = num_stages

    def __repr__(self):
        return ", ".join(f"{k}: {v}" for k, v in self.kwargs.items()) + f", num_warps: {self.num_warps}, num_stages: {self.num_stages}"


class Autotuner:
    """A kernel with several Configs.

    On a GPU, triton.autotune times every config the first time it sees new values of `key`
    and keeps the fastest. A CPU can't tell you which is fastest on a GPU, so minitriton
    runs the first config. check.py tests every config, one at a time, by setting `.configs`.
    """

    def __init__(self, fn, configs, key):
        self.fn = fn
        self.configs = list(configs)
        self.keys = list(key)
        self.cache = {}
        self.best_config = None
        self.__name__ = fn.__name__

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            clash = set(kwargs) & set(self.configs[0].kwargs)
            if clash:
                raise ValueError(f"{self.__name__}: {', '.join(sorted(clash))} come from the autotune configs; don't pass them")
            self.best_config = self.configs[0]
            self.fn[grid](*args, **kwargs, **self.best_config.kwargs)

        return launch

    def __call__(self, *a, **k):
        raise RuntimeError(f"{self.__name__} is a kernel: launch it with {self.__name__}[grid](...)")


def autotune(configs, key, **_options):
    def wrap(fn):
        return Autotuner(fn if isinstance(fn, JITFunction) else JITFunction(fn), configs, key)

    return wrap


class _Testing:
    @staticmethod
    def do_bench(*_a, **_k):
        raise RuntimeError("Benchmarks need a real NVIDIA GPU. Run bench.py on Colab (free T4) or your own GPU.")


testing = _Testing()
