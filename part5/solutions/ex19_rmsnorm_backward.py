"""Solution to exercise 19."""
from backend import DEVICE  # noqa: F401

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
        g = dy * w
        c = tl.sum(g * x, axis=0) / N
        dx = r * g - x * (r * r * r * c)
        tl.store(dx_ptr + row * dx_row_stride + cols, dx, mask=mask)
        dw += dy * x * r
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
