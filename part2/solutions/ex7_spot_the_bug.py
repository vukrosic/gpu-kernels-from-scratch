"""Solution to exercise 7.

Each version below has its one wrong line fixed; the comment says what changed.
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
    # FIX: 0 is neutral for a sum, not for exp: each masked lane added exp(0 - max) to the sum
    x = tl.load(x_ptr + row * x_row_stride + cols, mask=mask, other=-float("inf"))
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
    # FIX: subtract the max first; exp(100) overflows float32 (its limit is exp(88.7))
    num = tl.exp(x - tl.max(x, axis=0))
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
    # FIX: square in float32; in float16, 300 * 300 = inf (float16 stops at 65,504)
    x = tl.load(x_ptr + row * x_row_stride + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=mask)
    ms = tl.sum(x * x, axis=0) / N
    y = x * tl.rsqrt(ms + eps) * w.to(tl.float32)
    tl.store(out_ptr + row * out_row_stride + cols, y, mask=mask)


def rmsnorm_c(x, w, eps=1e-6, out=None):
    if out is None:
        out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    M, N = x.shape
    BLOCK = triton.next_power_of_2(N)
    rmsnorm_kernel_c[(M,)](x, w, out, N, eps, x.stride(0), out.stride(0), BLOCK=BLOCK)
    return out
