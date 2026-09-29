"""Solution to exercise 5."""
from backend import DEVICE

import torch
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(x_ptr, out_ptr, N, x_row_stride, out_row_stride, BLOCK: tl.constexpr):
    row = tl.program_id(0)  # one program per row
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(x_ptr + row * x_row_stride + cols, mask=mask, other=-float("inf"))
    x = x.to(tl.float32)
    x = x - tl.max(x, axis=0)
    num = tl.exp(x)
    den = tl.sum(num, axis=0)
    tl.store(out_ptr + row * out_row_stride + cols, num / den, mask=mask)


def softmax(x, out=None):
    if out is None:
        out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    M, N = x.shape
    BLOCK = triton.next_power_of_2(N)
    softmax_kernel[(M,)](x, out, N, x.stride(0), out.stride(0), BLOCK=BLOCK)
    return out


if __name__ == "__main__":
    x = torch.tensor([[10.0, 11.0, 12.0]], device=DEVICE)
    print(softmax(x))  # 0.0900, 0.2447, 0.6652
