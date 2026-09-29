"""Exercise 17 (stretch): save the logsumexp.

The backward pass (Part 5) needs the softmax probabilities again, e^(score - m) / l for
every score. Storing them is the (N, N) wall again. Storing m and l instead takes 2 numbers
per query, and one number is enough:

    lse = m + log(l) = log(sum of e^score)     so     probability = e^(score - lse)

1. Copy the body of your ex14 kernel into attention_lse_kernel.
2. After the loop, store one float32 lse per query at lse_ptr + z * stride_lz + rm.
   Rows past N don't exist: mask them.

You are done when  python check.py 17  prints PASS for the output and the logsumexp.
"""
from backend import DEVICE  # noqa: F401

import torch
import triton
import triton.language as tl


@triton.jit
def attention_lse_kernel(q_ptr, k_ptr, v_ptr, o_ptr, lse_ptr, N, scale,
                         stride_qz, stride_qn, stride_kz, stride_kn, stride_vz, stride_vn, stride_oz, stride_on,
                         stride_lz, D: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    # TODO: your ex14 kernel, plus one tl.store of m + log(l) per query
    pass


def attention_lse(q, k, v, out=None, lse=None):
    """Returns (out, lse). lse[z, i] = log(sum_j e^(score[i, j])), one float32 per query."""
    Z, N, D = q.shape
    assert k.shape == q.shape and v.shape == q.shape and D in (16, 32, 64, 128)
    assert q.stride(2) == 1 and k.stride(2) == 1 and v.stride(2) == 1, "the last dimension must be contiguous"
    if out is None:
        out = torch.empty((Z, N, D), dtype=torch.float16, device=q.device)
    if lse is None:
        lse = torch.empty((Z, N), dtype=torch.float32, device=q.device)
    BLOCK_M, BLOCK_N = 64, 64
    grid = (triton.cdiv(N, BLOCK_M), Z)
    attention_lse_kernel[grid](q, k, v, out, lse, N, D ** -0.5,
                               q.stride(0), q.stride(1), k.stride(0), k.stride(1), v.stride(0), v.stride(1),
                               out.stride(0), out.stride(1), lse.stride(0), D=D, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N)
    return out, lse
