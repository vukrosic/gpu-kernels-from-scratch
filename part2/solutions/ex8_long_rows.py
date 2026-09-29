"""Solution to exercise 8."""
from backend import DEVICE  # noqa: F401

import torch
import triton
import triton.language as tl


@triton.jit
def rmsnorm_long_kernel(x_ptr, w_ptr, out_ptr, N, eps, x_row_stride, out_row_stride, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    x_row = x_ptr + row * x_row_stride
    out_row = out_ptr + row * out_row_stride
    acc = tl.zeros((BLOCK,), dtype=tl.float32)  # BLOCK partial sums of squares
    for start in tl.range(0, N, BLOCK):  # walk 1: add up the squares
        cols = start + tl.arange(0, BLOCK)
        x = tl.load(x_row + cols, mask=cols < N, other=0.0).to(tl.float32)
        acc += x * x
    r = tl.rsqrt(tl.sum(acc, axis=0) / N + eps)
    for start in tl.range(0, N, BLOCK):  # walk 2: write the output
        cols = start + tl.arange(0, BLOCK)
        mask = cols < N
        x = tl.load(x_row + cols, mask=mask).to(tl.float32)
        w = tl.load(w_ptr + cols, mask=mask).to(tl.float32)
        tl.store(out_row + cols, x * r * w, mask=mask)


def rmsnorm_long(x, w, eps=1e-6, out=None):
    if out is None:
        out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    M, N = x.shape
    rmsnorm_long_kernel[(M,)](x, w, out, N, eps, x.stride(0), out.stride(0), BLOCK=1024)
    return out
