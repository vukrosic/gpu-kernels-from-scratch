"""Checks your Part 2 kernels against PyTorch.

    python check.py 5                 # exercise 5
    python check.py 5 6
    python check.py all
    python check.py all --solutions   # check the reference solutions instead

The cases include sizes that are not powers of 2, rows that sit further apart than
their width, and values chosen to break a careless kernel: one logit of 100 (exp(100)
overflows float32) and a float16 row with a 300 in it (300 * 300 overflows float16).
The guards, the "never written" marker and the output lines are in ../kfs_check.py.
"""
from backend import DEVICE

import torch
from kfs_check import Guarded, main, run_case

EPS = 1e-6


def rows(M, N, dtype=torch.float32, pad=0, seed=0, scale=1.0, plant=None):
    """An (M, N) input. plant=(col, value) puts one big value in every row."""
    gen = torch.Generator().manual_seed(seed)
    v = torch.randn(M, N, generator=gen) * scale
    if plant:
        v[:, plant[0]] = plant[1]
    return Guarded((M, N), dtype, v, row_pad=pad)


def label(M, N, note=""):
    return f"{M}x{N}{note}".ljust(34)


SOFTMAX_CASES = [
    # M, N, dtype, pad, scale, plant, note
    (1, 1, torch.float32, 0, 1.0, None, ""),
    (4, 1000, torch.float32, 0, 1.0, None, ""),
    (64, 33, torch.float32, 0, 1.0, None, ""),
    (3, 1024, torch.float32, 0, 1.0, None, ""),
    (5, 4097, torch.float32, 0, 1.0, None, ""),
    (4, 1000, torch.float32, 5, 1.0, None, " (rows 1005 apart)"),
    (4, 1000, torch.float32, 0, 4.0, (7, 100.0), " one logit = 100"),
    (8, 1000, torch.float16, 0, 1.0, None, " float16"),
]

RMS_CASES = [
    (1, 1, torch.float32, 0, 1.0, None, ""),
    (4, 1000, torch.float32, 0, 1.0, None, ""),
    (64, 33, torch.float32, 0, 1.0, None, ""),
    (5, 4097, torch.float32, 0, 1.0, None, ""),
    (4, 1000, torch.float32, 5, 1.0, None, " (rows 1005 apart)"),
    (8, 4096, torch.float16, 0, 1.0, None, " float16"),
    (4, 1000, torch.float16, 0, 30.0, (3, 300.0), " float16, one value = 300"),
]

LONG_CASES = [
    (4, 1000, torch.float32, 0, 1.0, None, ""),
    (3, 5000, torch.float32, 0, 1.0, None, ""),
    (2, 100_000, torch.float32, 0, 1.0, None, ""),
    (2, 3000, torch.float32, 7, 1.0, None, " (rows 3007 apart)"),
    (2, 100_000, torch.float16, 0, 30.0, (3, 300.0), " float16, one value = 300"),
]


def check_softmax(fn):
    ok = True
    for i, (M, N, dt, pad, scale, plant, note) in enumerate(SOFTMAX_CASES):
        x = rows(M, N, dt, pad, seed=i, scale=scale, plant=plant)
        out = Guarded((M, N), dt)
        want = torch.softmax(x.t.float(), dim=1)
        ok &= run_case(label(M, N, note), lambda: fn(x.t, out=out.t), want, out, {"x": x})
    return ok


def rms_ref(x, w):
    x = x.float()
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * w.float()


def check_rms(cases, seed0):
    def checker(fn):
        ok = True
        for i, (M, N, dt, pad, scale, plant, note) in enumerate(cases):
            x = rows(M, N, dt, pad, seed=seed0 + i, scale=scale, plant=plant)
            gen = torch.Generator().manual_seed(seed0 + 100 + i)
            w = Guarded((N,), dt, 1 + 0.1 * torch.randn(N, generator=gen))
            out = Guarded((M, N), dt)
            want = rms_ref(x.t, w.t)
            ok &= run_case(label(M, N, note), lambda: fn(x.t, w.t, EPS, out=out.t), want, out, {"x": x, "w": w})
        return ok

    return checker


EXERCISES = {
    "5": ("ex5_softmax", [("softmax", check_softmax)]),
    "6": ("ex6_rmsnorm", [("rmsnorm", check_rms(RMS_CASES, 20))]),
    "7": ("ex7_spot_the_bug", [("softmax_a", check_softmax), ("softmax_b", check_softmax), ("rmsnorm_c", check_rms(RMS_CASES, 20))]),
    "8": ("ex8_long_rows", [("rmsnorm_long", check_rms(LONG_CASES, 40))]),
}

if __name__ == "__main__":
    main(EXERCISES)
