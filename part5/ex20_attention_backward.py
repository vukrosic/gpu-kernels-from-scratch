"""Exercise 20: the attention backward, with no N x N matrix.

The forward (kernels.py) saves q, k, v, the output and lse: one float32 per query (ex17).
The backward needs P, the softmax of every score. Storing P would bring back the wall
from P10 (P15 asks how big it is), so the backward rebuilds each block of P on the chip:

    p = exp(s - lse)          exact, because lse = log of the row's whole sum

Then, for one head, with dO the gradient of the output:

    dV = P^T @ dO
    dP = dO @ V^T
    dS = P * (dP - delta)     delta = rowsum(dO * O), one number per query (given)
    dQ = dS @ K * scale
    dK = dS^T @ Q * scale

attention_dq_kernel below is finished: one program per block of queries, walking the keys.
Read it first. attention_dkdv_kernel is one program per block of keys, walking the queries
that can see them, so everything in it is transposed: st = k @ q^T is s^T, pt is p^T.
That is why it loads q^T and dO^T (qt, do_t) as well as q and dO. Write its five TODO
lines, the same math as the dq kernel with the transposes of P13.

    python check.py 20

prints PASS when dQ, dK and dV match a float64 reference, for N that is not a multiple
of 64 and for several heads.
"""
from backend import DEVICE

import torch
import triton
import triton.language as tl
from kernels import attention_forward


@triton.jit
def attention_dq_kernel(q_ptr, k_ptr, v_ptr, do_ptr, lse_ptr, delta_ptr, dq_ptr, N, scale,
                        stride_z, stride_n, stride_lz, D: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    """One program per block of 64 queries: dQ = dS @ K, over the keys this block can see."""
    pid_m = tl.program_id(0)
    z = tl.program_id(1)
    base = z * stride_z
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rd = tl.arange(0, D)
    q = tl.load(q_ptr + base + rm[:, None] * stride_n + rd[None, :], mask=rm[:, None] < N, other=0.0)
    do = tl.load(do_ptr + base + rm[:, None] * stride_n + rd[None, :], mask=rm[:, None] < N, other=0.0)
    lse = tl.load(lse_ptr + z * stride_lz + rm, mask=rm < N, other=0.0)
    delta = tl.load(delta_ptr + z * stride_lz + rm, mask=rm < N, other=0.0)
    dq = tl.zeros((BLOCK_M, D), dtype=tl.float32)
    for start in range(0, (pid_m + 1) * BLOCK_M, BLOCK_N):
        rn = start + tl.arange(0, BLOCK_N)
        kt = tl.load(k_ptr + base + rd[:, None] + rn[None, :] * stride_n, mask=rn[None, :] < N, other=0.0)
        vt = tl.load(v_ptr + base + rd[:, None] + rn[None, :] * stride_n, mask=rn[None, :] < N, other=0.0)
        k = tl.load(k_ptr + base + rn[:, None] * stride_n + rd[None, :], mask=rn[:, None] < N, other=0.0)
        s = tl.dot(q, kt) * scale
        p = tl.exp(s - lse[:, None])  # the softmax, rebuilt from one saved number per query
        p = tl.where((rn[None, :] <= rm[:, None]) & (rn[None, :] < N), p, 0.0)
        dp = tl.dot(do, vt)
        ds = p * (dp - delta[:, None])
        dq += tl.dot(ds.to(tl.float16), k)
    tl.store(dq_ptr + base + rm[:, None] * stride_n + rd[None, :], (dq * scale).to(tl.float16), mask=rm[:, None] < N)


@triton.jit
def attention_dkdv_kernel(q_ptr, k_ptr, v_ptr, do_ptr, lse_ptr, delta_ptr, dk_ptr, dv_ptr, N, scale,
                          stride_z, stride_n, stride_lz, D: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    """One program per block of 64 keys: dK and dV, over the queries that can see these keys.

    Everything is transposed: rows are keys, columns are queries. pt[j, i] is p[i, j].
    """
    pid_n = tl.program_id(0)
    z = tl.program_id(1)
    base = z * stride_z
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rd = tl.arange(0, D)
    k = tl.load(k_ptr + base + rn[:, None] * stride_n + rd[None, :], mask=rn[:, None] < N, other=0.0)
    v = tl.load(v_ptr + base + rn[:, None] * stride_n + rd[None, :], mask=rn[:, None] < N, other=0.0)
    dk = tl.zeros((BLOCK_N, D), dtype=tl.float32)
    dv = tl.zeros((BLOCK_N, D), dtype=tl.float32)
    for start in range(pid_n * BLOCK_N, N, BLOCK_M):  # causal: query i sees key j only if i >= j
        rm = start + tl.arange(0, BLOCK_M)
        qt = tl.load(q_ptr + base + rd[:, None] + rm[None, :] * stride_n, mask=rm[None, :] < N, other=0.0)
        q = tl.load(q_ptr + base + rm[:, None] * stride_n + rd[None, :], mask=rm[:, None] < N, other=0.0)
        do_t = tl.load(do_ptr + base + rd[:, None] + rm[None, :] * stride_n, mask=rm[None, :] < N, other=0.0)
        do = tl.load(do_ptr + base + rm[:, None] * stride_n + rd[None, :], mask=rm[:, None] < N, other=0.0)
        lse = tl.load(lse_ptr + z * stride_lz + rm, mask=rm < N, other=0.0)
        delta = tl.load(delta_ptr + z * stride_lz + rm, mask=rm < N, other=0.0)
        st = tl.dot(k, qt) * scale
        pt = ...  # TODO: p^T for this block, rebuilt from lse (lse is one per query: a column here)
        pt = tl.where((rm[None, :] >= rn[:, None]) & (rm[None, :] < N), pt, 0.0)  # no looking ahead
        dv += ...  # TODO (tl.dot takes float16: .to(tl.float16))
        dpt = ...  # TODO: dP^T
        dst = ...  # TODO: dS^T
        dk += ...  # TODO (the * scale is applied once, at the store)
    tl.store(dk_ptr + base + rn[:, None] * stride_n + rd[None, :], (dk * scale).to(tl.float16), mask=rn[:, None] < N)
    tl.store(dv_ptr + base + rn[:, None] * stride_n + rd[None, :], dv.to(tl.float16), mask=rn[:, None] < N)


class Attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v):
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        out, lse = attention_forward(q, k, v)
        ctx.save_for_backward(q, k, v, out, lse)  # no N x N matrix: lse is one float32 per query
        return out

    @staticmethod
    def backward(ctx, do):
        q, k, v, out, lse = ctx.saved_tensors
        do = do.contiguous().to(torch.float16)
        Z, N, D = q.shape
        delta = (do.float() * out.float()).sum(-1)  # rowsum(dO * O) = rowsum(dP * P), one per query
        dq, dk, dv = (torch.empty_like(q) for _ in range(3))
        BLOCK = 64
        grid = (triton.cdiv(N, BLOCK), Z)
        attention_dq_kernel[grid](q, k, v, do, lse, delta, dq, N, D ** -0.5, q.stride(0), q.stride(1), lse.stride(0),
                                  D=D, BLOCK_M=BLOCK, BLOCK_N=BLOCK)
        attention_dkdv_kernel[grid](q, k, v, do, lse, delta, dk, dv, N, D ** -0.5, q.stride(0), q.stride(1),
                                    lse.stride(0), D=D, BLOCK_M=BLOCK, BLOCK_N=BLOCK)
        return dq, dk, dv


def attention(q, k, v):
    return Attention.apply(q, k, v)


if __name__ == "__main__":
    torch.manual_seed(0)
    q, k, v = (torch.randn(1, 100, 32, dtype=torch.float16, device=DEVICE, requires_grad=True) for _ in range(3))
    attention(q, k, v).float().pow(2).sum().backward()
    print(q.grad[0, :2, :4])
