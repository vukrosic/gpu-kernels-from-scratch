"""Exercise 15: spot the bug.

Three FlashAttention kernels written the way AI assistants write them. All three run
without an error, pass some cases, and each has exactly one wrong line.

1. Run  python check.py 15  and read which cases PASS and which FAIL.
2. Before reading the code below, predict from those lines alone what went wrong.
3. Find the wrong line in each version, fix it, and run  python check.py 15  until all PASS.

Hints (only if stuck) are in README.md.
"""
from backend import DEVICE  # noqa: F401

import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------- version A


@triton.jit
def attention_kernel_a(q_ptr, k_ptr, v_ptr, o_ptr, N, scale,
                       stride_qz, stride_qn, stride_kz, stride_kn, stride_vz, stride_vn, stride_oz, stride_on,
                       D: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
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
        # Online softmax: rescale the running sum, then accumulate
        l = l * alpha + tl.sum(p, axis=1)
        acc = acc + tl.dot(p.to(tl.float16), v)
        m = m_new
    out = acc / l[:, None]
    tl.store(o_ptr + z * stride_oz + rm[:, None] * stride_on + rd[None, :], out.to(tl.float16), mask=rm[:, None] < N)


def attention_a(q, k, v, out=None):
    return _launch(attention_kernel_a, q, k, v, out)


# ---------------------------------------------------------------- version B


@triton.jit
def attention_kernel_b(q_ptr, k_ptr, v_ptr, o_ptr, N, scale,
                       stride_qz, stride_qn, stride_kz, stride_kn, stride_vz, stride_vn, stride_oz, stride_on,
                       D: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
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
        # Masked keys and values load 0, so they add nothing
        kt = tl.load(k_ptr + z * stride_kz + rd[:, None] + rn[None, :] * stride_kn, mask=rn[None, :] < N, other=0.0)
        v = tl.load(v_ptr + z * stride_vz + rn[:, None] * stride_vn + rd[None, :], mask=rn[:, None] < N, other=0.0)
        s = tl.dot(q, kt) * scale
        m_new = tl.maximum(m, tl.max(s, axis=1))
        alpha = tl.exp(m - m_new)
        p = tl.exp(s - m_new[:, None])
        l = l * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.float16), v)
        m = m_new
    out = acc / l[:, None]
    tl.store(o_ptr + z * stride_oz + rm[:, None] * stride_on + rd[None, :], out.to(tl.float16), mask=rm[:, None] < N)


def attention_b(q, k, v, out=None):
    return _launch(attention_kernel_b, q, k, v, out)


# ---------------------------------------------------------------- version C


@triton.jit
def attention_kernel_c(q_ptr, k_ptr, v_ptr, o_ptr, N, scale,
                       stride_qz, stride_qn, stride_kz, stride_kn, stride_vz, stride_vn, stride_oz, stride_on,
                       D: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    z = tl.program_id(1)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rd = tl.arange(0, D)
    q = tl.load(q_ptr + z * stride_qz + rm[:, None] * stride_qn + rd[None, :], mask=rm[:, None] < N, other=0.0)
    m = tl.zeros((BLOCK_M,), dtype=tl.float32)  # running max
    l = tl.zeros((BLOCK_M,), dtype=tl.float32)  # running sum
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


def attention_c(q, k, v, out=None):
    return _launch(attention_kernel_c, q, k, v, out)


def _launch(kernel, q, k, v, out):
    Z, N, D = q.shape
    assert k.shape == q.shape and v.shape == q.shape and D in (16, 32, 64, 128)
    assert q.stride(2) == 1 and k.stride(2) == 1 and v.stride(2) == 1, "the last dimension must be contiguous"
    if out is None:
        out = torch.empty((Z, N, D), dtype=torch.float16, device=q.device)
    BLOCK_M, BLOCK_N = 64, 64
    grid = (triton.cdiv(N, BLOCK_M), Z)
    kernel[grid](q, k, v, out, N, D ** -0.5,
                 q.stride(0), q.stride(1), k.stride(0), k.stride(1), v.stride(0), v.stride(1),
                 out.stride(0), out.stride(1), D=D, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N)
    return out
