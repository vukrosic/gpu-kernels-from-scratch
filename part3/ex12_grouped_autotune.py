"""Exercise 12 (stretch): launch order and autotune.

Two changes turn your ex10 kernel into one that gets close to cuBLAS on a GPU.

1. Launch order. A 4096 x 4096 matmul with 128 x 128 tiles has 32 x 32 = 1024 programs, and
   the GPU runs a few hundred at a time, roughly in pid order. Row by row, the programs
   running together all need different columns of B: all 32 of them. Grouped, GROUP_M rows
   of tiles are walked column by column, so the programs running together share a few rows
   of A and a few columns of B, and re-reads come from the L2 cache instead of memory.

       pid   0  1  2  3 ...        GROUP_M = 2, 4 columns of tiles:
       row by row:  (0,0) (0,1) (0,2) (0,3) (1,0) ...
       grouped:     (0,0) (1,0) (0,1) (1,1) (0,2) ...

   TODO 1: compute pid_m and pid_n from the one program id. The grid is 1D now.

2. Autotune. The best tile size depends on the GPU and the matrix shapes, so the kernel
   lists several configs and triton.autotune times each one the first time it sees a new
   (M, N, K), then keeps the fastest. The tile size is not known until then, so the grid
   is a function of the chosen config: META["BLOCK_M"] and so on.

   TODO 2: the grid lambda.

check.py 12 runs every config, one at a time. You are done when all PASS.
"""
from backend import DEVICE  # noqa: F401

import torch
import triton
import triton.language as tl

CONFIGS = [  # a subset of the Triton matmul tutorial's CUDA configs
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_M": 8}, num_stages=3, num_warps=8),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8}, num_stages=4, num_warps=4),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8}, num_stages=4, num_warps=4),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8}, num_stages=4, num_warps=4),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_K": 32, "GROUP_M": 8}, num_stages=5, num_warps=2),
    triton.Config({"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8}, num_stages=5, num_warps=2),
]


@triton.autotune(configs=CONFIGS, key=["M", "N", "K"])
@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                  stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    # TODO 1: pid_m and pid_n, grouped. Suggested steps:
    #   in_group = programs in one group (GROUP_M rows of tiles)
    #   first_m  = the first row of tiles in this program's group
    #   group_m  = rows of tiles in this group (the last group can be shorter: use min)
    #   then pid_m and pid_n from where pid sits inside its group
    pid_m = ...
    pid_n = ...
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + rm[:, None] * stride_am + rk[None, :] * stride_ak
    b_ptrs = b_ptr + rk[:, None] * stride_bk + rn[None, :] * stride_bn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_left = K - k * BLOCK_K
        a = tl.load(a_ptrs, mask=(rm[:, None] < M) & (rk[None, :] < k_left), other=0.0)
        b = tl.load(b_ptrs, mask=(rk[:, None] < k_left) & (rn[None, :] < N), other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    c_ptrs = c_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn
    tl.store(c_ptrs, acc.to(tl.float16), mask=(rm[:, None] < M) & (rn[None, :] < N))


def matmul(a, b, out=None):
    M, K = a.shape
    K2, N = b.shape
    assert K == K2
    if out is None:
        out = torch.empty((M, N), dtype=torch.float16, device=a.device)
    # The tile size isn't known until autotune picks a config, so the grid is a function of it.
    grid = ...  # TODO 2: lambda META: (number of programs,)
    matmul_kernel[grid](a, b, out, M, N, K,
                        a.stride(0), a.stride(1), b.stride(0), b.stride(1), out.stride(0), out.stride(1))
    return out
