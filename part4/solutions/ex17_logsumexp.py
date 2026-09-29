"""Solution to exercise 17."""
from backend import DEVICE  # noqa: F401

import torch
import triton
import triton.language as tl


@triton.jit
def attention_lse_kernel(q_ptr, k_ptr, v_ptr, o_ptr, lse_ptr, N, scale,
                         stride_qz, stride_qn, stride_kz, stride_kn, stride_vz, stride_vn, stride_oz, stride_on,
                         stride_lz, D: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    z = tl.program_id(1)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rd = tl.arange(0, D)
    q = tl.load(q_ptr + z * stride_qz + rm[:, None] * stride_qn + rd[None, :], mask=rm[:, None] < N, other=0.0)
    m = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)
    l = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, D), dtype=tl.float32)
    for start in range(0, N, BLOCK_N):
        rn = start + tl.arange(0, BLOCK_N)
        kt = tl.load(k_ptr + z * stride_kz + rd[:, None] + rn[None, :] * stride_kn, mask=rn[None, :] < N, other=0.0)
        v = tl.load(v_ptr + z * stride_vz + rn[:, None] * stride_vn + rd[None, :], mask=rn[:, None] < N, other=0.0)
        s = tl.dot(q, kt) * scale
        s = tl.where(rn[None, :] < N, s, float("-inf"))
        m_new = tl.maximum(m, tl.max(s, axis=1))
        alpha = tl.exp(m - m_new)
        p = tl.exp(s - m_new[:, None])
        l = l * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.float16), v)
        m = m_new
    out = acc / l[:, None]
    tl.store(o_ptr + z * stride_oz + rm[:, None] * stride_on + rd[None, :], out.to(tl.float16), mask=rm[:, None] < N)
    tl.store(lse_ptr + z * stride_lz + rm, m + tl.log(l), mask=rm < N)  # log of the softmax's denominator: m + log(l)


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
