"""Exercise 18: teach PyTorch to train through your matmul.

Your ex10 kernel writes C into a tensor PyTorch made with torch.empty. PyTorch never saw
the math, so C has no grad_fn and no gradient flows back through it:

    RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn

That line is from a real run of `matmul(a, b).float().sum().backward()`. Inside a model it
is worse: the rest of the model still has gradients, so there is no error at all, the
weights before your kernel get grad = None and never change.

torch.autograd.Function is the fix: you give PyTorch a forward and a backward.
forward is done. Write the two lines of backward, using only matmul() from kernels.py.
Your answer to P13 (dA and dB) is the whole exercise.

    python check.py 18

prints PASS when both gradients match a float64 reference, including for a and b that
are slices of wider matrices (their rows sit further apart than their width).
"""
from backend import DEVICE

import torch
from kernels import matmul


class Matmul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)  # the backward needs both inputs
        return matmul(a, b)

    @staticmethod
    def backward(ctx, dc):
        a, b = ctx.saved_tensors
        da = ...  # TODO: (M, K), from dc (M, N) and b (K, N)
        db = ...  # TODO: (K, N), from a (M, K) and dc (M, N)
        return da, db


if __name__ == "__main__":
    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float16, device=DEVICE, requires_grad=True)
    b = torch.tensor([[5.0, 6.0], [7.0, 8.0]], dtype=torch.float16, device=DEVICE, requires_grad=True)
    Matmul.apply(a, b).float().sum().backward()
    print(a.grad)  # every dc is 1: da = row sums of b = [[11, 15], [11, 15]]
    print(b.grad)  # db = column sums of a, repeated = [[4, 4], [6, 6]]
