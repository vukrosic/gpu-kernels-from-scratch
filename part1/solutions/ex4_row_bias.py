"""Solution to exercise 4."""
from backend import DEVICE  # noqa: F401

import torch
import triton
import triton.language as tl


@triton.jit
def row_bias_kernel(x_ptr, bias_ptr, out_ptr, N, x_row_stride, out_row_stride, BLOCK: tl.constexpr):
    row = tl.program_id(0)  # one program per row
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(x_ptr + row * x_row_stride + cols, mask=mask)
    b = tl.load(bias_ptr + cols, mask=mask)
    tl.store(out_ptr + row * out_row_stride + cols, x + b, mask=mask)


def add_row_bias(x, bias, out=None):
    if out is None:
        out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    M, N = x.shape
    BLOCK = triton.next_power_of_2(N)
    row_bias_kernel[(M,)](x, bias, out, N, x.stride(0), out.stride(0), BLOCK=BLOCK)
    return out
