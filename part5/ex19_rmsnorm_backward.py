"""Exercise 19: the RMSNorm backward kernel.

Forward (given, in kernels.py), for one row of N numbers:

    r = 1 / sqrt(mean(x ** 2) + eps)      saved: one float32 per row, like lse in ex17
    y = x * r * w

Backward: you get dy (same shape as y) and return dx and dw. Your answer to P14:

    g  = dy * w
    dx = r * g - x * r**3 * mean(g * x)
    dw = sum over rows of dy * x * r

dx is one row at a time, like the forward. dw adds up over every row, and many programs
can't all add into the same N numbers (that needs atomics). So each program takes rows
pid, pid + programs, pid + 2 * programs, ..., keeps its own running dw, and writes one row
of dw_part; PyTorch sums those rows. Write the four TODO lines.

    python check.py 19

prints PASS when dx and dw match a float64 reference, including float16 rows with a 300
in them (ex6: 300 * 300 overflows float16, so do the math in float32).
"""
from backend import DEVICE

import torch
import triton
import triton.language as tl
from kernels import rmsnorm_forward


@triton.jit
def rmsnorm_backward_kernel(x_ptr, w_ptr, dy_ptr, rstd_ptr, dx_ptr, dw_part_ptr, M, N,
                            x_row_stride, dy_row_stride, dx_row_stride, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    programs = tl.num_programs(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    dw = tl.zeros((BLOCK,), dtype=tl.float32)  # this program's share of dw, summed over its rows
    for row in range(pid, M, programs):
        x = tl.load(x_ptr + row * x_row_stride + cols, mask=mask, other=0.0).to(tl.float32)
        dy = tl.load(dy_ptr + row * dy_row_stride + cols, mask=mask, other=0.0).to(tl.float32)
        r = tl.load(rstd_ptr + row)
        g = ...  # TODO
        c = ...  # TODO: mean(g * x) over the row; switched-off lanes loaded 0
        dx = ...  # TODO
        tl.store(dx_ptr + row * dx_row_stride + cols, dx, mask=mask)
        dw += ...  # TODO: this row's part of dw
    tl.store(dw_part_ptr + pid * N + cols, dw, mask=mask)


class RMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, eps=1e-6):
        y, rstd = rmsnorm_forward(x, w, eps)
        ctx.save_for_backward(x, w, rstd)
        return y

    @staticmethod
    def backward(ctx, dy):
        x, w, rstd = ctx.saved_tensors
        dy = dy.contiguous()  # autograd may hand over a broadcast view, e.g. the gradient of y.sum()
        M, N = x.shape
        dx = torch.empty((M, N), dtype=x.dtype, device=x.device)
        programs = min(M, 128)  # each program adds up dw over M / 128 rows; no two write the same place
        dw_part = torch.empty((programs, N), dtype=torch.float32, device=x.device)
        rmsnorm_backward_kernel[(programs,)](x, w, dy, rstd, dx, dw_part, M, N,
                                             x.stride(0), dy.stride(0), dx.stride(0),
                                             BLOCK=triton.next_power_of_2(N))
        return dx, dw_part.sum(0).to(w.dtype), None


def rmsnorm(x, w, eps=1e-6):
    return RMSNorm.apply(x, w, eps)


if __name__ == "__main__":
    x = torch.tensor([[3.0, 4.0]], device=DEVICE, requires_grad=True)
    w = torch.ones(2, device=DEVICE, requires_grad=True)
    y = rmsnorm(x, w, 0.0)
    y.backward(torch.tensor([[1.0, 0.0]], device=DEVICE))
    print(x.grad)  # P14: [[0.1810, -0.1358]]
    print(w.grad)  # dy * x * rstd = [0.8485, 0]
