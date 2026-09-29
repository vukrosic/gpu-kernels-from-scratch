"""Exercise 6: RMSNorm, the normalisation inside Llama and most LLMs since.

    rms[i]    = sqrt(mean(x[i, :] ** 2) + eps)
    out[i, j] = x[i, j] / rms[i] * w[j]          x: (M, N)   w: (N,)   out: (M, N)

Same shape as your softmax: one program per row, the whole row in one block.

Before you write it, predict: x = 300 in float16. What is x * x?
(float16 tops out at 65,504.) check.py 6 includes a float16 row with a 300 in it.

Write the kernel body. tl.rsqrt(v) is 1 / sqrt(v). You are done when

    python check.py 6

prints PASS.
"""
from backend import DEVICE

import torch
import triton
import triton.language as tl


@triton.jit
def rmsnorm_kernel(x_ptr, w_ptr, out_ptr, N, eps, x_row_stride, out_row_stride, BLOCK: tl.constexpr):
    # TODO
    pass


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
    print(rmsnorm(x, w))  # rms = sqrt((9 + 16) / 2) = 3.536, so expect 0.8485, 1.1314
