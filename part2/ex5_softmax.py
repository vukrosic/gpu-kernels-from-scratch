"""Exercise 5: softmax of every row, in one kernel.

    out[i, :] = exp(x[i, :] - max(x[i, :])) / sum(exp(x[i, :] - max(x[i, :])))

One program per row, and each program holds its whole row in one block:
BLOCK = triton.next_power_of_2(N), with a mask for the extra lanes.

Two things from the video:
  * the switched-off lanes must never win the max and must add nothing to the sum;
  * subtract the row's max before exp, or a big number overflows to inf.

Fill in TODO 1-4. You are done when

    python check.py 5

prints PASS.
"""
from backend import DEVICE

import torch
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(x_ptr, out_ptr, N, x_row_stride, out_row_stride, BLOCK: tl.constexpr):
    row = tl.program_id(0)  # one program per row
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(x_ptr + row * x_row_stride + cols, mask=mask, other=...)  # TODO 1: what do switched-off lanes read?
    x = x.to(tl.float32)
    x = ...  # TODO 2: subtract the row's max (tl.max(x, axis=0))
    num = ...  # TODO 3: exponentiate (tl.exp)
    den = ...  # TODO 4: the row's sum (tl.sum)
    tl.store(out_ptr + row * out_row_stride + cols, num / den, mask=mask)


def softmax(x, out=None):
    if out is None:
        out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    M, N = x.shape
    BLOCK = triton.next_power_of_2(N)
    softmax_kernel[(M,)](x, out, N, x.stride(0), out.stride(0), BLOCK=BLOCK)
    return out


if __name__ == "__main__":
    x = torch.tensor([[10.0, 11.0, 12.0]], device=DEVICE)
    print(softmax(x))  # expect 0.0900, 0.2447, 0.6652
