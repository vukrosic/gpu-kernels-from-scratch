"""The forward kernels from Parts 2-4, ready for Part 5.

These are the reference solutions, not your own files, so a mistake left in an earlier part
doesn't block this one. Each forward also saves what its backward will need:

    matmul(a, b)               ex10: nothing extra, the backward re-reads a and b
    rmsnorm_forward(x, w)      ex6 plus one float32 per row: rstd = 1 / rms
    attention_forward(q, k, v) ex16 (causal) plus ex17: one float32 per query, lse
"""
from backend import DEVICE  # noqa: F401

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------- matmul (ex10)

@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                  stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
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
    """a (M, K) @ b (K, N) -> float16 (M, N). Any strides: a.t() and b.t() work without a copy."""
    M, K = a.shape
    K2, N = b.shape
    assert K == K2, f"inner sizes differ: {tuple(a.shape)} @ {tuple(b.shape)}"
    if out is None:
        out = torch.empty((M, N), dtype=torch.float16, device=a.device)
    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    matmul_kernel[grid](a, b, out, M, N, K,
                        a.stride(0), a.stride(1), b.stride(0), b.stride(1), out.stride(0), out.stride(1),
                        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K)
    return out


# ---------------------------------------------------------------- RMSNorm (ex6), saving rstd

@triton.jit
def rmsnorm_forward_kernel(x_ptr, w_ptr, y_ptr, rstd_ptr, N, eps, x_row_stride, y_row_stride, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(x_ptr + row * x_row_stride + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    rstd = tl.rsqrt(tl.sum(x * x, axis=0) / N + eps)
    tl.store(y_ptr + row * y_row_stride + cols, x * rstd * w, mask=mask)
    tl.store(rstd_ptr + row, rstd)  # one float32 per row, for the backward


def rmsnorm_forward(x, w, eps=1e-6):
    """Returns (y, rstd). y has x's dtype; rstd is float32, one per row."""
    M, N = x.shape
    assert x.stride(1) == 1, "the last dimension must be contiguous"
    y = torch.empty((M, N), dtype=x.dtype, device=x.device)
    rstd = torch.empty((M,), dtype=torch.float32, device=x.device)
    rmsnorm_forward_kernel[(M,)](x, w, y, rstd, N, eps, x.stride(0), y.stride(0), BLOCK=triton.next_power_of_2(N))
    return y, rstd


# ---------------------------------------------------------------- causal attention (ex16 + ex17)

@triton.jit
def attention_forward_kernel(q_ptr, k_ptr, v_ptr, o_ptr, lse_ptr, N, scale,
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
    for start in range(0, (pid_m + 1) * BLOCK_M, BLOCK_N):
        rn = start + tl.arange(0, BLOCK_N)
        kt = tl.load(k_ptr + z * stride_kz + rd[:, None] + rn[None, :] * stride_kn, mask=rn[None, :] < N, other=0.0)
        v = tl.load(v_ptr + z * stride_vz + rn[:, None] * stride_vn + rd[None, :], mask=rn[:, None] < N, other=0.0)
        s = tl.dot(q, kt) * scale
        s = tl.where((rn[None, :] <= rm[:, None]) & (rn[None, :] < N), s, float("-inf"))
        m_new = tl.maximum(m, tl.max(s, axis=1))
        alpha = tl.exp(m - m_new)
        p = tl.exp(s - m_new[:, None])
        l = l * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.float16), v)
        m = m_new
    out = acc / l[:, None]
    tl.store(o_ptr + z * stride_oz + rm[:, None] * stride_on + rd[None, :], out.to(tl.float16), mask=rm[:, None] < N)
    tl.store(lse_ptr + z * stride_lz + rm, m + tl.log(l), mask=rm < N)


def attention_forward(q, k, v):
    """Causal attention. q, k, v: (heads, N, D) float16. Returns (out float16, lse float32 (heads, N))."""
    Z, N, D = q.shape
    assert k.shape == q.shape and v.shape == q.shape and D in (16, 32, 64, 128)
    assert q.stride(2) == 1 and k.stride(2) == 1 and v.stride(2) == 1, "the last dimension must be contiguous"
    out = torch.empty((Z, N, D), dtype=torch.float16, device=q.device)
    lse = torch.empty((Z, N), dtype=torch.float32, device=q.device)
    BLOCK = 64
    attention_forward_kernel[(triton.cdiv(N, BLOCK), Z)](
        q, k, v, out, lse, N, D ** -0.5,
        q.stride(0), q.stride(1), k.stride(0), k.stride(1), v.stride(0), v.stride(1), out.stride(0), out.stride(1),
        lse.stride(0), D=D, BLOCK_M=BLOCK, BLOCK_N=BLOCK)
    return out, lse
