"""Solution to exercise 3."""
from backend import DEVICE

import torch
import triton
import triton.language as tl


@triton.jit
def scale_shift_relu_kernel(x_ptr, out_ptr, n, scale, shift, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask)  # trip 1: read x
    y = tl.maximum(x * scale + shift, 0.0)  # all three ops happen in registers
    tl.store(out_ptr + offsets, y, mask=mask)  # trip 2: write out


def scale_shift_relu(x, scale, shift, out=None):
    if out is None:
        out = torch.empty_like(x)
    n = out.numel()
    BLOCK = 1024
    grid = (triton.cdiv(n, BLOCK),)
    scale_shift_relu_kernel[grid](x, out, n, scale, shift, BLOCK=BLOCK)
    return out


def scale_shift_relu_torch(x, scale, shift):
    return torch.relu(x * scale + shift)


if __name__ == "__main__":
    x = torch.linspace(-2, 2, 9, device=DEVICE)
    print(scale_shift_relu(x, 2.0, 1.0))
