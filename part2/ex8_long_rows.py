"""Exercise 8 (stretch): RMSNorm for rows longer than one block.

A row of 100,000 numbers is too long to hold on the chip at once: in one block it would
spill out of the registers into slow memory. So BLOCK is fixed at 1024 here, and each
program walks along its row in steps of BLOCK:

    for start in tl.range(0, N, BLOCK):
        cols = start + tl.arange(0, BLOCK)
        ...

RMSNorm needs two walks: one to add up the squares, one to write the output. Keep a
float32 block of partial sums (tl.zeros((BLOCK,), dtype=tl.float32)) during the first walk
and reduce it with tl.sum after the loop.

Part 3's matmul walks along K the same way.

You are done when  python check.py 8  prints PASS.
"""
from backend import DEVICE  # noqa: F401

import torch
import triton
import triton.language as tl


@triton.jit
def rmsnorm_long_kernel(x_ptr, w_ptr, out_ptr, N, eps, x_row_stride, out_row_stride, BLOCK: tl.constexpr):
    # TODO
    pass


def rmsnorm_long(x, w, eps=1e-6, out=None):
    if out is None:
        out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    M, N = x.shape
    rmsnorm_long_kernel[(M,)](x, w, out, N, eps, x.stride(0), out.stride(0), BLOCK=1024)
    return out
