"""Exercise 1: your first kernel. Add two vectors.

Fill in the three TODO lines. You are done when

    python check.py 1

prints PASS.

The picture: the numbers sit in one long row of boxes. The grid launches one program per
crate of BLOCK boxes. Program `pid` handles boxes pid*BLOCK up to pid*BLOCK + BLOCK - 1.
The last crate usually hangs past the end of the row, so those lanes must be switched off.
"""
from backend import DEVICE

import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)  # which crate am I?
    offsets = ...  # TODO 1: the BLOCK box numbers this program handles (use pid, BLOCK, tl.arange)
    mask = ...  # TODO 2: True for boxes that exist, False past the end of the row
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)


def add(x, y, out=None):
    if out is None:
        out = torch.empty_like(x)
    n = out.numel()
    BLOCK = 1024
    grid = ...  # TODO 3: how many programs? A tuple with one number (triton.cdiv rounds up)
    add_kernel[grid](x, y, out, n, BLOCK=BLOCK)
    return out


if __name__ == "__main__":
    x = torch.arange(10, dtype=torch.float32, device=DEVICE)
    print(add(x, x))  # expect 0, 2, 4, ... 18
