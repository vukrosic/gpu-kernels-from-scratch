"""Measures your Part 3 matmuls on an NVIDIA GPU and appends to bench.csv.

    python bench.py                # your ex10 and ex12 kernels
    python bench.py --solutions    # the reference solutions
    python bench.py --peak 120     # your GPU's float16 tensor-core TFLOPS if it isn't listed

Needs an NVIDIA GPU (Colab's free T4 works). On a laptop, use check.py.

A 4096 x 4096 x 4096 float16 matmul is 2 * 4096^3 = 137 billion FLOPs (a multiply and an
add per step). "best ms" is the larger of two limits: the FLOPs at the tensor cores'
peak, and the bytes at the memory bus's peak if every matrix crossed it only once.
For a square matmul this big, the tensor cores are the limit.
"""
import argparse
import csv
import importlib
import os
import sys

from backend import BACKEND, DEVICE

import torch

# Spec-sheet dense float16 tensor TFLOPS with float32 accumulate, and memory GB/s,
# matched against torch.cuda.get_device_name(). Sources: NVIDIA architecture whitepapers and datasheets.
PEAKS = [
    ("RTX 5090", 209.5, 1792), ("RTX 5080", 112.6, 960), ("RTX 4090", 165.2, 1008), ("RTX 4080", 97.5, 717),
    ("RTX 3090", 71.2, 936), ("H100 80GB HBM3", 989.4, 3350), ("H100 PCIe", 756.0, 2000), ("A100", 312.0, 1555),
    ("L4", 121.0, 300), ("T4", 65.0, 320), ("V100", 125.0, 900),
]


def peaks_for(name):
    for key, tflops, gbps in PEAKS:
        if key in name:
            return tflops, gbps
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--solutions", action="store_true")
    ap.add_argument("--size", type=int, default=4096, help="M = N = K")
    ap.add_argument("--peak", type=float, help="float16 tensor-core TFLOPS of your GPU")
    ap.add_argument("--csv", default="bench.csv")
    a = ap.parse_args()
    if DEVICE != "cuda":
        print(f"backend: {BACKEND}\nbench.py measures speed, so it needs an NVIDIA GPU. Try Colab's free T4, or run check.py here.")
        sys.exit(1)
    import triton

    pre = "solutions." if a.solutions else ""
    ex10 = importlib.import_module(pre + "ex10_matmul")
    ex12 = importlib.import_module(pre + "ex12_grouped_autotune")

    gpu = torch.cuda.get_device_name()
    peak, gbps = peaks_for(gpu)
    peak = a.peak or peak
    n = a.size
    x = torch.randn(n, n, device="cuda", dtype=torch.float16)
    w = torch.randn(n, n, device="cuda", dtype=torch.float16) / n ** 0.5
    out = torch.empty(n, n, device="cuda", dtype=torch.float16)
    flops = 2 * n ** 3
    once = 3 * n * n * 2  # A and B read once, C written once, 2 bytes each
    best = max(flops / (peak * 1e12) if peak else 0, once / (gbps * 1e9) if gbps else 0) * 1e3 or float("nan")

    rows = [
        ("torch.matmul (cuBLAS)", lambda: torch.matmul(x, w, out=out)),
        ("your matmul, 64x64 tiles (ex10)", lambda: ex10.matmul(x, w, out=out)),
        ("your matmul, autotuned (ex12)", lambda: ex12.matmul(x, w, out=out)),
    ]
    ex12.matmul(x, w, out=out)  # the first call runs the autotuner
    print(f"{gpu}   spec: {peak or '?'} TFLOPS, {gbps or '?'} GB/s   {n} x {n} x {n} float16   {BACKEND}")
    print(f"{'kernel':<34}{'ms':>8}{'TFLOPS':>8}{'% peak':>8}{'best ms':>9}")
    records = []
    for name, fn in rows:
        ms = triton.testing.do_bench(fn, quantiles=[0.5, 0.2, 0.8])[0]
        tf = flops / (ms * 1e-3) / 1e12
        pct = 100 * tf / peak if peak else float("nan")
        print(f"{name:<34}{ms:>8.3f}{tf:>8.1f}{pct:>7.0f}%{best:>9.3f}")
        records.append(dict(gpu=gpu, triton=triton.__version__, torch=torch.__version__, kernel=name, size=n, dtype="float16",
                            ms=round(ms, 4), tflops=round(tf, 1), peak_tflops=peak, pct_of_peak=round(pct, 1)))
    cfg = getattr(ex12.matmul_kernel, "best_config", None)
    if cfg:
        print(f"\nautotune picked: {cfg}")
    new = not os.path.exists(a.csv)
    with open(a.csv, "a", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(records[0]))
        if new:
            wr.writeheader()
        wr.writerows(records)
    print(f"appended {len(records)} rows to {a.csv}")


if __name__ == "__main__":
    main()
