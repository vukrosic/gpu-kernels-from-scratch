"""Solution to exercise 2: each version with its one line fixed."""
from backend import DEVICE  # noqa: F401

import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------- version A


@triton.jit
def add_kernel_a(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    # Each program processes one block of elements
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)


def add_a(x, y, out=None):
    if out is None:
        out = torch.empty_like(x)
    n = out.numel()
    BLOCK = 256
    # Launch one program per block, rounding UP so the last partial block runs
    grid = (triton.cdiv(n, BLOCK),)
    add_kernel_a[grid](x, y, out, n, BLOCK=BLOCK)
    return out


# ---------------------------------------------------------------- version B


@triton.jit
def add_kernel_b(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    # Compute the offsets for this program's elements
    offsets = pid * BLOCK + tl.arange(0, BLOCK)  # was: pid + ... (programs overlapped)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)


def add_b(x, y, out=None):
    if out is None:
        out = torch.empty_like(x)
    n = out.numel()
    BLOCK = 256
    grid = (triton.cdiv(n, BLOCK),)
    add_kernel_b[grid](x, y, out, n, BLOCK=BLOCK)
    return out


# ---------------------------------------------------------------- version C


@triton.jit
def add_kernel_c(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    # Load inputs (masked to stay in bounds)
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    # Write the result
    tl.store(out_ptr + offsets, x + y, mask=mask)  # was: no mask (wrote past the end)


def add_c(x, y, out=None):
    if out is None:
        out = torch.empty_like(x)
    n = out.numel()
    BLOCK = 256
    grid = (triton.cdiv(n, BLOCK),)
    add_kernel_c[grid](x, y, out, n, BLOCK=BLOCK)
    return out
