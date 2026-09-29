"""Exercise 4 (stretch): add a bias to every row of a matrix.

    out[i, j] = x[i, j] + bias[j]        x: (M, N)   bias: (N,)   out: (M, N)

New idea: a 2D tensor is still one long row of boxes in memory. Row i of x starts at box
i * x.stride(0). Launch one program per row, and let each program cover its whole row:
BLOCK = triton.next_power_of_2(N), with a mask for the extra lanes.

check.py 4 also passes an x that is a slice of a wider matrix, so its rows are NOT N boxes
apart. Use the strides, not N.

You are done when  python check.py 4  prints PASS.
"""
from backend import DEVICE  # noqa: F401

import torch
import triton
import triton.language as tl


@triton.jit
def row_bias_kernel(x_ptr, bias_ptr, out_ptr, N, x_row_stride, out_row_stride, BLOCK: tl.constexpr):
    # TODO
    pass


def add_row_bias(x, bias, out=None):
    if out is None:
        out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    # TODO: pick BLOCK and the grid, launch row_bias_kernel
    return out
