"""Exercise 16: causal attention.

In a language model, token i may only look at tokens 0..i: the answer can't depend on
words that come later. So score[i, j] for j > i gets no weight, like a key past the end.

1. Copy the body of your ex14 kernel into causal_attention_kernel.
2. Change the mask so query rm sees key rn only when rn <= rm (and rn < N, as before).
3. Half of the (N, N) scores are now masked. Blocks of keys that start after this
   program's last query are masked completely: stop the loop before them instead of
   computing scores you then throw away. Which block is the last one worth loading?
   (P12 in the README counts how much work this saves.)

You are done when  python check.py 16  prints PASS.
"""
from backend import DEVICE  # noqa: F401

import torch
import triton
import triton.language as tl


@triton.jit
def causal_attention_kernel(q_ptr, k_ptr, v_ptr, o_ptr, N, scale,
                            stride_qz, stride_qn, stride_kz, stride_kn, stride_vz, stride_vn, stride_oz, stride_on,
                            D: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    # TODO: your ex14 kernel with a causal mask and a shorter loop
    pass


def attention_causal(q, k, v, out=None):
    """Query i sees keys 0..i only, as in a language model."""
    Z, N, D = q.shape
    assert k.shape == q.shape and v.shape == q.shape and D in (16, 32, 64, 128)
    assert q.stride(2) == 1 and k.stride(2) == 1 and v.stride(2) == 1, "the last dimension must be contiguous"
    if out is None:
        out = torch.empty((Z, N, D), dtype=torch.float16, device=q.device)
    BLOCK_M, BLOCK_N = 64, 64
    grid = (triton.cdiv(N, BLOCK_M), Z)
    causal_attention_kernel[grid](q, k, v, out, N, D ** -0.5,
                                  q.stride(0), q.stride(1), k.stride(0), k.stride(1), v.stride(0), v.stride(1),
                                  out.stride(0), out.stride(1), D=D, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N)
    return out
