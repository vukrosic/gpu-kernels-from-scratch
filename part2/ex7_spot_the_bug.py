"""Exercise 7: spot the bug.

Three kernels written the way AI assistants write them: two softmaxes and one RMSNorm.
All three run without an error, pass some cases, and each has exactly one wrong line.

1. Run  python check.py 7  and read which cases PASS and which FAIL.
2. Before reading the code below, predict from those lines alone what went wrong.
3. Find the wrong line in each version, fix it, and run  python check.py 7  until all PASS.

Hints (only if stuck) are in README.md.
"""
from backend import DEVICE  # noqa: F401

import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------- version A: softmax


@triton.jit
def softmax_kernel_a(x_ptr, out_ptr, N, x_row_stride, out_row_stride, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    # Masked lanes load 0 so they stay neutral
    x = tl.load(x_ptr + row * x_row_stride + cols, mask=mask, other=0.0)
    x = x.to(tl.float32)
    # Subtract the max for numerical stability
    x = x - tl.max(x, axis=0)
    num = tl.exp(x)
    den = tl.sum(num, axis=0)
    tl.store(out_ptr + row * out_row_stride + cols, num / den, mask=mask)


def softmax_a(x, out=None):
    if out is None:
        out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    M, N = x.shape
    BLOCK = triton.next_power_of_2(N)
    softmax_kernel_a[(M,)](x, out, N, x.stride(0), out.stride(0), BLOCK=BLOCK)
    return out


# ---------------------------------------------------------------- version B: softmax


@triton.jit
def softmax_kernel_b(x_ptr, out_ptr, N, x_row_stride, out_row_stride, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(x_ptr + row * x_row_stride + cols, mask=mask, other=-float("inf"))
    x = x.to(tl.float32)
    # Compute the softmax numerator and denominator
    num = tl.exp(x)
    den = tl.sum(num, axis=0)
    tl.store(out_ptr + row * out_row_stride + cols, num / den, mask=mask)


def softmax_b(x, out=None):
    if out is None:
        out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    M, N = x.shape
    BLOCK = triton.next_power_of_2(N)
    softmax_kernel_b[(M,)](x, out, N, x.stride(0), out.stride(0), BLOCK=BLOCK)
    return out


# ---------------------------------------------------------------- version C: RMSNorm


@triton.jit
def rmsnorm_kernel_c(x_ptr, w_ptr, out_ptr, N, eps, x_row_stride, out_row_stride, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(x_ptr + row * x_row_stride + cols, mask=mask, other=0.0)
    w = tl.load(w_ptr + cols, mask=mask)
    # Mean of squares over the row
    ms = tl.sum(x * x, axis=0) / N
    y = x.to(tl.float32) * tl.rsqrt(ms + eps) * w.to(tl.float32)
    tl.store(out_ptr + row * out_row_stride + cols, y, mask=mask)


def rmsnorm_c(x, w, eps=1e-6, out=None):
    if out is None:
        out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    M, N = x.shape
    BLOCK = triton.next_power_of_2(N)
    rmsnorm_kernel_c[(M,)](x, w, out, N, eps, x.stride(0), out.stride(0), BLOCK=BLOCK)
    return out
