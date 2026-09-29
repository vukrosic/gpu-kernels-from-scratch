"""Exercise 10: the full matmul.

Now K can be any size, so the program walks along K in steps of BLOCK_K, the way ex8 walked
along a long row:

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        # load a (BLOCK_M, BLOCK_K) block of A and a (BLOCK_K, BLOCK_N) block of B
        acc += tl.dot(a, b)
        # move both blocks of addresses BLOCK_K along K

1. The output tile, acc, stays on the chip for the whole loop and is written once at the end.
2. Move the addresses with the strides: one step along K is BLOCK_K * stride_ak in A and
   BLOCK_K * stride_bk in B.
3. The last step can run past K. Mask it, and load a value that adds nothing to the dot.
4. Add up in float32 and store float16 (acc.to(tl.float16)).

Start from your ex9 kernel. You are done when  python check.py 10  prints PASS.
"""
from backend import DEVICE  # noqa: F401

import torch
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                  stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    # TODO
    pass


def matmul(a, b, out=None):
    M, K = a.shape
    K2, N = b.shape
    assert K == K2
    if out is None:
        out = torch.empty((M, N), dtype=torch.float16, device=a.device)
    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    matmul_kernel[grid](a, b, out, M, N, K,
                        a.stride(0), a.stride(1), b.stride(0), b.stride(1), out.stride(0), out.stride(1),
                        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K)
    return out
