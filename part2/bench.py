"""Measures your Part 2 kernels on an NVIDIA GPU and appends to bench.csv.

    python bench.py                # your ex5 softmax and ex6 RMSNorm
    python bench.py --solutions    # the reference solutions
    python bench.py --peak 320     # your GPU's memory bandwidth (GB/s) if it isn't listed

Needs an NVIDIA GPU (Colab's free T4 works). On a laptop, use check.py.

Softmax and RMSNorm are memory-bound: every number is read once and written once, and
the arithmetic is nearly free. "bytes" counts what each version actually moves:

  naive softmax, 5 PyTorch kernels (max, subtract, exp, sum, divide):
      reads 5MN + 2M numbers, writes 3MN + 2M        (about 8 trips over the matrix)
  eager RMSNorm, x * rsqrt(mean(x^2) + eps) * w, 6 kernels:
      reads 4MN + ..., writes 3MN + ...              (about 7 trips)
  one fused kernel (yours, torch.softmax): reads MN, writes MN (2 trips)

"best ms" is the fewest bytes the job needs (2 trips) divided by the bandwidth.
"""
import argparse
import csv
import importlib
import os
import sys

from backend import BACKEND, DEVICE

import torch

# Spec-sheet memory bandwidth in GB/s, matched against torch.cuda.get_device_name().
PEAK_GBPS = [
    ("RTX 5090", 1792), ("RTX 5080", 960), ("RTX 4090", 1008), ("RTX 4080", 717), ("RTX 3090", 936),
    ("H100 80GB HBM3", 3350), ("H100 PCIe", 2000), ("A100-SXM4-80GB", 2039), ("A100 80GB PCIe", 1935),
    ("A100-SXM4-40GB", 1555), ("A100-PCIE-40GB", 1555), ("L4", 300), ("T4", 320), ("A10G", 600), ("V100", 900),
]
EPS = 1e-6


def peak_for(name):
    for key, gbps in PEAK_GBPS:
        if key in name:
            return gbps
    return None


def naive_softmax(x):
    x_max = x.max(dim=1)[0]  # read MN, write M
    z = x - x_max[:, None]  # read MN + M, write MN
    num = torch.exp(z)  # read MN, write MN
    den = num.sum(dim=1)  # read MN, write M
    return num / den[:, None]  # read MN + M, write MN


def eager_rmsnorm(x, w):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--solutions", action="store_true")
    ap.add_argument("--m", type=int, default=4096)
    ap.add_argument("--n", type=int, default=4096)
    ap.add_argument("--peak", type=float, help="memory bandwidth of your GPU in GB/s")
    ap.add_argument("--csv", default="bench.csv")
    a = ap.parse_args()
    if DEVICE != "cuda":
        print(f"backend: {BACKEND}\nbench.py measures speed, so it needs an NVIDIA GPU. Try Colab's free T4, or run check.py here.")
        sys.exit(1)
    import triton

    pre = "solutions." if a.solutions else ""
    ex5 = importlib.import_module(pre + "ex5_softmax")
    ex6 = importlib.import_module(pre + "ex6_rmsnorm")

    gpu = torch.cuda.get_device_name()
    peak = a.peak or peak_for(gpu)
    M, N, fp = a.m, a.n, 4
    x = torch.randn(M, N, device="cuda")
    w = torch.randn(N, device="cuda")
    out = torch.empty_like(x)
    two_trips = 2 * M * N * fp

    rows = [
        ("naive softmax, 5 kernels", lambda: naive_softmax(x), (8 * M * N + 4 * M) * fp),
        ("torch.softmax", lambda: torch.softmax(x, dim=1), two_trips),
        ("your softmax (ex5)", lambda: ex5.softmax(x, out=out), two_trips),
        ("eager RMSNorm, 6 kernels", lambda: eager_rmsnorm(x, w), (7 * M * N + 6 * M + N) * fp),
        ("your RMSNorm (ex6)", lambda: ex6.rmsnorm(x, w, EPS, out=out), two_trips + N * fp),
    ]
    try:
        compiled = torch.compile(eager_rmsnorm)
        compiled(x, w)
        rows.insert(4, ("torch.compile RMSNorm", lambda: compiled(x, w), two_trips + N * fp))
    except Exception as e:  # torch.compile is optional here
        print(f"(skipping torch.compile: {type(e).__name__})")

    print(f"{gpu}   spec bandwidth: {peak or '?'} GB/s   {M} x {N} float32   {BACKEND}")
    print(f"{'kernel':<30}{'ms':>8}{'GB/s':>8}{'% peak':>8}{'best ms':>9}")
    records = []
    for name, fn, nbytes in rows:
        ms, _, _ = triton.testing.do_bench(fn, quantiles=[0.5, 0.2, 0.8])
        gbps = nbytes / (ms * 1e-3) / 1e9
        pct = 100 * gbps / peak if peak else float("nan")
        best = two_trips / (peak * 1e9) * 1e3 if peak else float("nan")
        print(f"{name:<30}{ms:>8.3f}{gbps:>8.0f}{pct:>7.0f}%{best:>9.3f}")
        records.append(dict(gpu=gpu, triton=triton.__version__, torch=torch.__version__, kernel=name, n=M * N, dtype="float32",
                            ms=round(ms, 4), bytes_moved=nbytes, gb_per_s=round(gbps, 1), peak_gb_per_s=peak, pct_of_peak=round(pct, 1)))
    new = not os.path.exists(a.csv)
    with open(a.csv, "a", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(records[0]))
        if new:
            wr.writeheader()
        wr.writerows(records)
    print(f"\nappended {len(records)} rows to {a.csv}")


if __name__ == "__main__":
    main()
