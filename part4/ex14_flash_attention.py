"""Exercise 14: FlashAttention, forward pass.

out = softmax(q @ k^T / sqrt(D)) @ v, for q, k, v of shape (Z, N, D): Z heads, N tokens.

The plain way writes the (N, N) scores to memory, reads them back for the softmax, writes
the probabilities, and reads them again for @ v. This kernel never writes them anywhere.
Each program takes one block of 64 queries of one head and walks along the keys in
blocks of 64. For each block of keys it makes a (64, 64) block of scores on the chip,
folds it into a running max m, a running sum l and a running output acc, and moves on.
Only the (64, D) output is written, once, at the end.

acc is a sum of e^(score - m) * v, measured against the max so far, exactly like l.
When the max grows, acc must shrink by the same factor as l.

The loads, the dot and the mask are written for you. Fill in the TODO (5 lines).
You are done when  python check.py 14  prints PASS.
"""
from backend import DEVICE  # noqa: F401

import torch
import triton
import triton.language as tl


@triton.jit
def attention_kernel(q_ptr, k_ptr, v_ptr, o_ptr, N, scale,
                     stride_qz, stride_qn, stride_kz, stride_kn, stride_vz, stride_vn, stride_oz, stride_on,
                     D: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)  # which block of queries
    z = tl.program_id(1)  # which head
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rd = tl.arange(0, D)
    q = tl.load(q_ptr + z * stride_qz + rm[:, None] * stride_qn + rd[None, :], mask=rm[:, None] < N, other=0.0)
    m = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)  # each query's largest score so far
    l = tl.zeros((BLOCK_M,), dtype=tl.float32)  # sum of e^(score - m) so far
    acc = tl.zeros((BLOCK_M, D), dtype=tl.float32)  # sum of e^(score - m) * v so far
    for start in range(0, N, BLOCK_N):
        rn = start + tl.arange(0, BLOCK_N)
        kt = tl.load(k_ptr + z * stride_kz + rd[:, None] + rn[None, :] * stride_kn, mask=rn[None, :] < N, other=0.0)
        v = tl.load(v_ptr + z * stride_vz + rn[:, None] * stride_vn + rd[None, :], mask=rn[:, None] < N, other=0.0)
        s = tl.dot(q, kt) * scale  # (BLOCK_M, BLOCK_N) scores: they live on the chip and nowhere else
        s = tl.where(rn[None, :] < N, s, float("-inf"))  # keys past the end get no weight
        # TODO: your ex13 update, for BLOCK_M rows at once (5 lines)
        m_new = ...  # (BLOCK_M,) the max so far of each row
        alpha = ...  # (BLOCK_M,) the factor that moves the old sums to m_new
        p = ...  # (BLOCK_M, BLOCK_N) e^(score - m_new)
        l = ...  # (BLOCK_M,)
        acc = ...  # (BLOCK_M, D) rescale it like l, then add p @ v (tl.dot wants p in float16)
        m = m_new
    out = acc / l[:, None]
    tl.store(o_ptr + z * stride_oz + rm[:, None] * stride_on + rd[None, :], out.to(tl.float16), mask=rm[:, None] < N)


def attention(q, k, v, out=None):
    """softmax(q @ k^T / sqrt(D)) @ v for q, k, v of shape (Z, N, D). Z is batch x heads."""
    Z, N, D = q.shape
    assert k.shape == q.shape and v.shape == q.shape and D in (16, 32, 64, 128)
    assert q.stride(2) == 1 and k.stride(2) == 1 and v.stride(2) == 1, "the last dimension must be contiguous"
    if out is None:
        out = torch.empty((Z, N, D), dtype=torch.float16, device=q.device)
    BLOCK_M, BLOCK_N = 64, 64
    grid = (triton.cdiv(N, BLOCK_M), Z)
    attention_kernel[grid](q, k, v, out, N, D ** -0.5,
                           q.stride(0), q.stride(1), k.stride(0), k.stride(1), v.stride(0), v.stride(1),
                           out.stride(0), out.stride(1), D=D, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N)
    return out
