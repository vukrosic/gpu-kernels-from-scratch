"""Solution to exercise 6."""
from backend import DEVICE

import torch
import triton
import triton.language as tl


@triton.jit
def rmsnorm_kernel(x_ptr, w_ptr, out_ptr, N, eps, x_row_stride, out_row_stride, BLOCK: tl.constexpr):
    row = tl.program_id(0)  # one program per row
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(x_ptr + row * x_row_stride + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=mask).to(tl.float32)
    ms = tl.sum(x * x, axis=0) / N  # switched-off lanes read 0, so they add nothing
    y = x * tl.rsqrt(ms + eps) * w
    tl.store(out_ptr + row * out_row_stride + cols, y, mask=mask)


def rmsnorm(x, w, eps=1e-6, out=None):
    if out is None:
        out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    M, N = x.shape
    BLOCK = triton.next_power_of_2(N)
    rmsnorm_kernel[(M,)](x, w, out, N, eps, x.stride(0), out.stride(0), BLOCK=BLOCK)
    return out


if __name__ == "__main__":
    x = torch.tensor([[3.0, 4.0]], device=DEVICE)
    w = torch.ones(2, device=DEVICE)
    print(rmsnorm(x, w))  # 0.8485, 1.1314
