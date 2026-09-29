"""Exercise 2: spot the bug.

Three versions of the vector-add kernel, each with the kind of mistake AI assistants make.
All three run without an error. Each has exactly one wrong line.

1. Run  python check.py 2  and read the FAIL lines.
2. Before reading the code below, predict from the FAIL lines alone what went wrong.
   Every version uses BLOCK = 256.
3. Find the wrong line in each version, fix it, and run  python check.py 2  until all PASS.

Hints (only if stuck) are at the bottom of README.md.
"""
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
    # Launch one program per block
    grid = (n // BLOCK,)
    add_kernel_a[grid](x, y, out, n, BLOCK=BLOCK)
    return out


# ---------------------------------------------------------------- version B


@triton.jit
def add_kernel_b(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    # Compute the offsets for this program's elements
    offsets = pid + tl.arange(0, BLOCK)
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
    tl.store(out_ptr + offsets, x + y)


def add_c(x, y, out=None):
    if out is None:
        out = torch.empty_like(x)
    n = out.numel()
    BLOCK = 256
    grid = (triton.cdiv(n, BLOCK),)
    add_kernel_c[grid](x, y, out, n, BLOCK=BLOCK)
    return out
