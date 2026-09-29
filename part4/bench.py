"""Measures your Part 4 attention kernels on an NVIDIA GPU and appends to bench.csv.

    python bench.py                  # your ex14 and ex16 kernels
    python bench.py --solutions      # the reference solutions
    python bench.py --n 1024 4096    # other sequence lengths
    python bench.py --peak 120       # your GPU's float16 tensor-core TFLOPS if it isn't listed

Needs an NVIDIA GPU (Colab's free T4 works). On a laptop, use check.py.

Every row runs 16 heads with D = 64. Attention is two matmuls, q @ k^T and p @ v, so it is
4 * heads * N^2 * D FLOPs; causal skips about half of them but is still counted as the
full 4 * heads * N^2 * D here, so its TFLOPS show how much faster the same answer comes.
"extra MB" is the memory a call allocates while it runs (yours writes into a ready out=; the
others also allocate their output, heads * N * D * 2 bytes). The naive version stores the
(N, N) scores and probabilities, so it grows with N^2; at large N it runs out.
"""
import argparse
import csv
import importlib
import os
import sys

from backend import BACKEND, DEVICE

import torch
import torch.nn.functional as F

# Spec-sheet dense float16 tensor TFLOPS with float32 accumulate, and memory GB/s,
# matched against torch.cuda.get_device_name(). Sources: NVIDIA architecture whitepapers and datasheets.
PEAKS = [
    ("RTX 5090", 209.5, 1792), ("RTX 5080", 112.6, 960), ("RTX 4090", 165.2, 1008), ("RTX 4080", 97.5, 717),
    ("RTX 3090", 71.2, 936), ("H100 80GB HBM3", 989.4, 3350), ("H100 PCIe", 756.0, 2000), ("A100", 312.0, 1555),
    ("L4", 121.0, 300), ("T4", 65.0, 320), ("V100", 125.0, 900),
]
HEADS, D = 16, 64


def peaks_for(name):
    for key, tflops, gbps in PEAKS:
        if key in name:
            return tflops, gbps
    return None, None


def naive(q, k, v, causal=False):
    s = q @ k.transpose(1, 2) * D ** -0.5  # (heads, N, N): the wall
    if causal:
        N = q.shape[1]
        s = s.masked_fill(torch.ones(N, N, dtype=torch.bool, device=q.device).triu(1), float("-inf"))
    return torch.softmax(s, dim=-1) @ v


def measure(fn):
    """(median ms, MB allocated beyond what existed before the call), or None if it runs out of memory."""
    import triton

    try:
        fn()
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        fn()
        torch.cuda.synchronize()
        extra = (torch.cuda.max_memory_allocated() - before) / 1e6
        ms = triton.testing.do_bench(fn, quantiles=[0.5, 0.2, 0.8])[0]
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return None
    return ms, extra


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--solutions", action="store_true")
    ap.add_argument("--n", type=int, nargs="+", default=[1024, 4096, 16384], help="sequence lengths")
    ap.add_argument("--peak", type=float, help="float16 tensor-core TFLOPS of your GPU")
    ap.add_argument("--csv", default="bench.csv")
    a = ap.parse_args()
    if DEVICE != "cuda":
        print(f"backend: {BACKEND}\nbench.py measures speed, so it needs an NVIDIA GPU. Try Colab's free T4, or run check.py here.")
        sys.exit(1)
    import triton

    pre = "solutions." if a.solutions else ""
    ex14 = importlib.import_module(pre + "ex14_flash_attention")
    ex16 = importlib.import_module(pre + "ex16_causal")

    gpu = torch.cuda.get_device_name()
    peak, gbps = peaks_for(gpu)
    peak = a.peak or peak
    print(f"{gpu}   spec: {peak or '?'} TFLOPS, {gbps or '?'} GB/s   {HEADS} heads, D = {D}, float16   {BACKEND}")
    print(f"{'N':>6}  {'kernel':<34}{'ms':>9}{'TFLOPS':>8}{'% peak':>8}{'extra MB':>10}")
    records = []
    for n in a.n:
        q, k, v = (torch.randn(HEADS, n, D, device="cuda", dtype=torch.float16) for _ in range(3))
        out = torch.empty_like(q)
        flops = 4 * HEADS * n * n * D
        rows = [
            ("naive: scores in memory", lambda: naive(q, k, v)),
            ("torch SDPA", lambda: F.scaled_dot_product_attention(q, k, v)),
            ("your FlashAttention (ex14)", lambda: ex14.attention(q, k, v, out=out)),
            ("naive causal", lambda: naive(q, k, v, causal=True)),
            ("torch SDPA causal", lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True)),
            ("your causal (ex16)", lambda: ex16.attention_causal(q, k, v, out=out)),
        ]
        for name, fn in rows:
            r = measure(fn)
            if r is None:
                print(f"{n:>6}  {name:<34}{'out of memory':>35}")
                records.append(dict(gpu=gpu, triton=triton.__version__, torch=torch.__version__, kernel=name, n=n,
                                    heads=HEADS, d=D, ms="", tflops="", peak_tflops=peak, pct_of_peak="", extra_mb="OOM"))
                continue
            ms, extra = r
            tf = flops / (ms * 1e-3) / 1e12
            pct = 100 * tf / peak if peak else float("nan")
            print(f"{n:>6}  {name:<34}{ms:>9.3f}{tf:>8.1f}{pct:>7.0f}%{extra:>10.0f}")
            records.append(dict(gpu=gpu, triton=triton.__version__, torch=torch.__version__, kernel=name, n=n, heads=HEADS,
                                d=D, ms=round(ms, 4), tflops=round(tf, 1), peak_tflops=peak, pct_of_peak=round(pct, 1),
                                extra_mb=round(extra)))
        del q, k, v, out
        torch.cuda.empty_cache()
        print()
    new = not os.path.exists(a.csv)
    with open(a.csv, "a", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(records[0]))
        if new:
            wr.writeheader()
        wr.writerows(records)
    print(f"appended {len(records)} rows to {a.csv}")


if __name__ == "__main__":
    main()
