"""Measures your Part 1 kernels on an NVIDIA GPU and writes bench.csv.

    python bench.py                # your ex1 and ex3
    python bench.py --solutions    # the reference solutions
    python bench.py --peak 320     # set your GPU's memory bandwidth (GB/s) if it isn't listed

Needs an NVIDIA GPU (Colab's free T4 works). On a laptop, use check.py.

Every kernel here is memory-bound, so the number that matters is GB/s: bytes moved per
second, compared with the GPU's spec-sheet bandwidth. "best ms" is bytes / bandwidth,
the time the memory bus alone needs.
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


def peak_for(name):
    for key, gbps in PEAK_GBPS:
        if key in name:
            return gbps
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--solutions", action="store_true")
    ap.add_argument("--n", type=int, default=100_000_000)
    ap.add_argument("--peak", type=float, help="memory bandwidth of your GPU in GB/s")
    ap.add_argument("--csv", default="bench.csv")
    a = ap.parse_args()
    if DEVICE != "cuda":
        print(f"backend: {BACKEND}\nbench.py measures speed, so it needs an NVIDIA GPU. Try Colab's free T4, or run check.py here.")
        sys.exit(1)
    import triton

    pre = "solutions." if a.solutions else ""
    ex1 = importlib.import_module(pre + "ex1_vector_add")
    ex3 = importlib.import_module(pre + "ex3_fused")

    gpu = torch.cuda.get_device_name()
    peak = a.peak or peak_for(gpu)
    n, fp = a.n, 4
    x = torch.randn(n, device="cuda")
    y = torch.randn(n, device="cuda")
    out = torch.empty_like(x)
    s, b = 1.7, -0.3

    def eager():
        return torch.relu(x * s + b)

    rows = [
        ("torch x + y", lambda: torch.add(x, y, out=out), 3 * n * fp),
        ("your add (ex1)", lambda: ex1.add(x, y, out=out), 3 * n * fp),
        ("torch relu(x*s+b), 3 kernels", eager, 6 * n * fp),
        ("your fused (ex3)", lambda: ex3.scale_shift_relu(x, s, b, out=out), 2 * n * fp),
    ]
    try:
        compiled = torch.compile(eager)
        compiled()
        rows.insert(3, ("torch.compile relu(x*s+b)", compiled, 2 * n * fp))
    except Exception as e:  # torch.compile is optional here
        print(f"(skipping torch.compile: {type(e).__name__})")

    print(f"{gpu}   spec bandwidth: {peak or '?'} GB/s   n = {n:,} float32   {BACKEND}")
    print(f"{'kernel':<30}{'ms':>8}{'GB/s':>8}{'% peak':>8}{'best ms':>9}")
    records = []
    for name, fn, nbytes in rows:
        ms, _, _ = triton.testing.do_bench(fn, quantiles=[0.5, 0.2, 0.8])
        gbps = nbytes / (ms * 1e-3) / 1e9
        pct = 100 * gbps / peak if peak else float("nan")
        best = nbytes / (peak * 1e9) * 1e3 if peak else float("nan")
        print(f"{name:<30}{ms:>8.3f}{gbps:>8.0f}{pct:>7.0f}%{best:>9.3f}")
        records.append(dict(gpu=gpu, triton=triton.__version__, torch=torch.__version__, kernel=name, n=n, dtype="float32",
                            ms=round(ms, 4), bytes_moved=nbytes, gb_per_s=round(gbps, 1), peak_gb_per_s=peak, pct_of_peak=round(pct, 1)))
    new = not os.path.exists(a.csv)
    with open(a.csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0]))
        if new:
            w.writeheader()
        w.writerows(records)
    print(f"\nappended {len(records)} rows to {a.csv}")


if __name__ == "__main__":
    main()
