"""Solution to exercise 18."""
from backend import DEVICE  # noqa: F401

import torch
from kernels import matmul


class Matmul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return matmul(a, b)

    @staticmethod
    def backward(ctx, dc):
        a, b = ctx.saved_tensors
        da = matmul(dc, b.t())  # (M, N) @ (N, K): b.t() is a view, the kernel reads its strides
        db = matmul(a.t(), dc)  # (K, M) @ (M, N)
        return da, db


if __name__ == "__main__":
    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float16, device=DEVICE, requires_grad=True)
    b = torch.tensor([[5.0, 6.0], [7.0, 8.0]], dtype=torch.float16, device=DEVICE, requires_grad=True)
    Matmul.apply(a, b).float().sum().backward()
    print(a.grad)  # every dc is 1: da = row sums of b = [[11, 15], [11, 15]]
    print(b.grad)  # db = column sums of a, repeated = [[4, 4], [6, 6]]
