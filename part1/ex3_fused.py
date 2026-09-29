"""Exercise 3: fuse three operations into one trip.

    out = relu(x * scale + shift)

PyTorch runs this line as three kernels (multiply, add, relu). Each one reads its input
from GPU memory and writes its result back: six trips over the memory bus. Write ONE kernel
that reads x once and writes out once: two trips.

Fill in the kernel body. You are done when  python check.py 3  prints PASS.
On a GPU, then run  python bench.py  and compare your kernel with PyTorch's three.
"""
from backend import DEVICE

import torch
import triton
import triton.language as tl


@triton.jit
def scale_shift_relu_kernel(x_ptr, out_ptr, n, scale, shift, BLOCK: tl.constexpr):
    # TODO: the same pattern as exercise 1: pid, offsets, mask, load x,
    #       compute relu(x * scale + shift) (tl.maximum(v, 0.0) is relu), store.
    pass


def scale_shift_relu(x, scale, shift, out=None):
    if out is None:
        out = torch.empty_like(x)
    n = out.numel()
    BLOCK = 1024
    grid = (triton.cdiv(n, BLOCK),)
    scale_shift_relu_kernel[grid](x, out, n, scale, shift, BLOCK=BLOCK)
    return out


def scale_shift_relu_torch(x, scale, shift):
    return torch.relu(x * scale + shift)  # three kernels, six trips


if __name__ == "__main__":
    x = torch.linspace(-2, 2, 9, device=DEVICE)
    print(scale_shift_relu(x, 2.0, 1.0))  # expect 0, 0, 0, 0, 1, 2, 3, 4, 5
