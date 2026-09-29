"""Checks your Part 3 matmul kernels against PyTorch.

    python check.py 9                 # exercise 9
    python check.py 10 11
    python check.py all
    python check.py all --solutions   # check the reference solutions instead

Inputs are float16 and the output is float16, added up in float32, like a real model's
matmuls. B is scaled by 1/sqrt(K) so every output is about the size of 1 at any K.
The cases include sizes that are not multiples of the tile (100 x 70 @ 70 x 90), a vector
times a matrix, an A that is a slice of a wider matrix, and B = W.t(): nn.Linear computes
x @ W.t(), and W.t() is stored column after column, not row after row.
The guards, the "never written" marker and the output lines are in ../kfs_check.py.
"""
import contextlib
import io
import sys

from backend import DEVICE

import torch
from kfs_check import Guarded, main, run_case

F16 = torch.float16

# M, N, K, layout: "" row after row, "slice" A is a slice of a wider matrix, "W.t()" B = W.t()
TILE_CASES = [(16, 16, 16, ""), (64, 64, 64, ""), (100, 90, 70, ""), (1, 300, 1, ""), (33, 47, 128, ""),
              (100, 90, 50, "slice"), (100, 90, 50, "W.t()")]
MM_CASES = [(16, 16, 16, ""), (64, 64, 64, ""), (128, 128, 128, ""), (100, 90, 70, ""), (1, 256, 257, ""),
            (300, 200, 1000, ""), (256, 256, 512, ""), (100, 90, 64, "slice"), (100, 90, 64, "W.t()")]
BIG_CASES = [(16, 16, 16, ""), (128, 128, 128, ""), (300, 200, 100, ""), (520, 130, 64, ""), (1, 256, 257, ""),
             (100, 90, 64, "slice"), (100, 90, 64, "W.t()")]


def operands(M, N, K, layout, seed):
    gen = torch.Generator().manual_seed(seed)
    a = torch.randn(M, K, generator=gen)
    w = torch.randn(N, K, generator=gen) / K ** 0.5
    A = Guarded((M, K), F16, a, row_pad=16 if layout == "slice" else 0)
    if layout == "W.t()":
        W = Guarded((N, K), F16, w)
        return A, W, W.t.t(), {"a": A, "w": W}
    B = Guarded((K, N), F16, w.t().contiguous())
    return A, B, B.t, {"a": A, "b": B}


def label(M, N, K, layout):
    note = {"": "", "slice": f"  A rows {K + 16} apart", "W.t()": "  B = W.t()"}[layout]
    return f"({M}x{K}) @ ({K}x{N}){note}".ljust(40)


def check_cases(cases, seed0):
    def checker(fn):
        ok = True
        for i, (M, N, K, layout) in enumerate(cases):
            A, _, b, inputs = operands(M, N, K, layout, seed0 + i)
            out = Guarded((M, N), F16)
            want = A.t.double().cpu() @ b.double().cpu()
            ok &= run_case(label(M, N, K, layout), lambda: fn(A.t, b, out=out.t), want, out, inputs)
        return ok

    return checker


def check_every_config(fn):
    """Runs the cases once per autotune config: one line per config, the failing lines if any."""
    kernel = sys.modules[fn.__module__].matmul_kernel
    configs = list(kernel.configs)
    ok = True
    try:
        for cfg in configs:
            kernel.configs, kernel.cache = [cfg], {}
            name = "  ".join(f"{k}={v}" for k, v in cfg.kwargs.items() if k != "GROUP_M").ljust(40)
            buf = io.StringIO()
            try:
                with contextlib.redirect_stdout(buf):
                    passed = check_cases(BIG_CASES, 60)(fn)
            except Exception:  # StopChecking after an ERROR line
                if "OutOfResources" in buf.getvalue():  # needs more on-chip memory than this GPU has; autotune skips it too
                    print(f"  {name}  SKIP  too big for this GPU")
                    continue
                print(f"  {name}  FAIL\n" + "\n".join("    " + l.strip() for l in buf.getvalue().splitlines() if "PASS" not in l))
                print("  (other configs skipped)")
                return False
            print(f"  {name}  PASS" if passed else f"  {name}  FAIL")
            if not passed:
                print("\n".join("    " + l.strip() for l in buf.getvalue().splitlines() if "PASS" not in l))
            ok &= passed
    finally:
        kernel.configs, kernel.cache = configs, {}
    return ok


EXERCISES = {
    "9": ("ex9_one_tile", [("matmul_small_k", check_cases(TILE_CASES, 0))]),
    "10": ("ex10_matmul", [("matmul", check_cases(MM_CASES, 20))]),
    "11": ("ex11_spot_the_bug", [("matmul_a", check_cases(MM_CASES, 20)), ("matmul_b", check_cases(MM_CASES, 20)),
                                 ("matmul_c", check_cases(MM_CASES, 20))]),
    "12": ("ex12_grouped_autotune", [("matmul", check_every_config)]),
}

if __name__ == "__main__":
    main(EXERCISES)
