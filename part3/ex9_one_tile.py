"""Exercise 9: one output tile.

C = A @ B, with A of shape (M, K) and B of shape (K, N). Each program computes one
BLOCK_M x BLOCK_N tile of C, so the grid is 2D: program (pid_m, pid_n) writes rows
pid_m * BLOCK_M ... and columns pid_n * BLOCK_N ... of C.

In this exercise K is small (at most 128), so one block holds all of K: the program loads a
(BLOCK_M, BLOCK_K) block of A and a (BLOCK_K, BLOCK_N) block of B, and tl.dot multiplies them.

A 2D block of addresses is a column of offsets plus a row of offsets:
    rm[:, None] * stride_am + rk[None, :] * stride_ak      # shape (BLOCK_M, BLOCK_K)
Use the strides; never assume a row is K numbers long.

Fill in TODO 1-3 (5 lines). You are done when  python check.py 9  prints PASS.
"""
from backend import DEVICE  # noqa: F401

import torch
import triton
import triton.language as tl


@triton.jit
def tile_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)  # which row of tiles
    pid_n = tl.program_id(1)  # which column of tiles
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # rows of C this program writes
    rn = ...  # TODO 1: columns of C this program writes
    rk = tl.arange(0, BLOCK_K)                    # all of K, in one block
    # TODO 2: a (BLOCK_M, BLOCK_K) block of addresses into A, and a (BLOCK_K, BLOCK_N) one into B
    a_ptrs = ...
    b_ptrs = ...
    # TODO 3: load both. Mask every lane outside the matrix, and give switched-off lanes a
    # value that adds nothing to a dot product.
    a = ...
    b = ...
    c = tl.dot(a, b)  # (BLOCK_M, BLOCK_N), float32
    c_ptrs = c_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn
    tl.store(c_ptrs, c, mask=(rm[:, None] < M) & (rn[None, :] < N))


def matmul_small_k(a, b, out=None):
    M, K = a.shape
    K2, N = b.shape
    assert K == K2 and K <= 128, "this version holds all of K in one block"
    if out is None:
        out = torch.empty((M, N), dtype=a.dtype, device=a.device)
    BLOCK_M, BLOCK_N = 32, 32
    BLOCK_K = max(16, triton.next_power_of_2(K))  # tl.dot needs every size >= 16
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    tile_kernel[grid](a, b, out, M, N, K,
                      a.stride(0), a.stride(1), b.stride(0), b.stride(1), out.stride(0), out.stride(1),
                      BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K)
    return out
