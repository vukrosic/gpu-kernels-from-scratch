"""minitriton.language: the subset of `triton.language` this course uses, on CPU with NumPy.

A kernel runs once per program. Inside it, every value is a whole block (a NumPy array),
exactly like Triton: `tl.arange(0, 256)` is 256 lanes at once, not a loop.

Memory behaves like a GPU's:
  * reading or writing past the end of a tensor but inside its allocation succeeds silently
    (you read or overwrite a neighbour's numbers), which is why check.py puts guards
    around every tensor;
  * touching memory outside the allocation raises IllegalMemoryAccess, like CUDA's
    "an illegal memory access was encountered";
  * lanes switched off by a mask and loaded without `other=` come back as NaN, because on
    a GPU their value is undefined.
"""
import numpy as np

# A GPU overflows to inf and makes NaN without a warning; so does the stand-in.
np.seterr(over="ignore", invalid="ignore", divide="ignore", under="ignore")

_state = {"pid": (0, 0, 0), "grid": (1, 1, 1)}


class IllegalMemoryAccess(RuntimeError):
    pass


class constexpr:
    """Marks a kernel argument as a compile-time constant (`BLOCK: tl.constexpr`)."""


class dtype:
    def __init__(self, name, np_type):
        self.name = name
        self.np = np_type

    def __repr__(self):
        return f"tl.{self.name}"


float16 = dtype("float16", np.float16)
float32 = dtype("float32", np.float32)
float64 = dtype("float64", np.float64)
int32 = dtype("int32", np.int32)
int64 = dtype("int64", np.int64)
int1 = dtype("int1", np.bool_)


class Block(np.ndarray):
    """A block of lanes. Behaves like a NumPy array, plus Triton's `.to(dtype)`."""

    def to(self, dt):
        return np.asarray(self).astype(dt.np).view(Block)


def _block(a):
    return np.asarray(a).view(Block)


class Pointer:
    """A tensor argument inside a kernel: the start of its memory.

    `mem` is the whole allocation the tensor lives in (a view can sit inside a bigger
    buffer); `base` is where this tensor starts in it, in elements.
    """

    def __init__(self, mem, base, numel, name):
        self.mem = mem
        self.base = int(base)
        self.numel = int(numel)
        self.name = name

    def __add__(self, off):
        return PointerBlock(self, self.base + np.asarray(off, dtype=np.int64))

    __radd__ = __add__

    def __repr__(self):
        return f"<pointer to {self.name}: {self.numel} x {self.mem.dtype}>"


class PointerBlock:
    """A block of addresses: `x_ptr + offsets`."""

    def __init__(self, ptr, idx):
        self.ptr = ptr
        self.idx = idx

    def __add__(self, off):
        return PointerBlock(self.ptr, self.idx + np.asarray(off, dtype=np.int64))

    __radd__ = __add__


def _addresses(p, mask):
    if isinstance(p, Pointer):
        p = PointerBlock(p, np.asarray(p.base, dtype=np.int64))
    if not isinstance(p, PointerBlock):
        raise TypeError(f"tl.load/tl.store need a pointer (a tensor argument plus offsets), got {type(p).__name__}")
    idx = p.idx
    if mask is None:
        m = np.ones(idx.shape, dtype=bool)
    else:
        m = np.broadcast_to(np.asarray(mask, dtype=bool), np.broadcast_shapes(idx.shape, np.shape(mask)))
        idx = np.broadcast_to(idx, m.shape)
    return p.ptr, idx, m


def _bounds(ptr, idx, m, verb):
    live = idx[m]
    bad = live[(live < 0) | (live >= ptr.mem.shape[0])]
    if bad.size:
        i = int(bad[0]) - ptr.base
        pid = _state["pid"][0]
        raise IllegalMemoryAccess(
            f"program {pid} tried to {verb} {ptr.name}[{i}], outside its memory "
            f"({ptr.name} has {ptr.numel} elements). On a GPU: 'an illegal memory access was encountered'."
        )


def load(pointer, mask=None, other=None, **_ignored):
    ptr, idx, m = _addresses(pointer, mask)
    _bounds(ptr, idx, m, "read")
    out = np.empty(idx.shape, dtype=ptr.mem.dtype)
    out[m] = ptr.mem[idx[m]]
    if not m.all():
        if other is None:
            out[~m] = np.nan if np.issubdtype(out.dtype, np.floating) else 0
        else:
            out[~m] = np.broadcast_to(np.asarray(other), m.shape)[~m]
    return _block(out)


def store(pointer, value, mask=None, **_ignored):
    ptr, idx, m = _addresses(pointer, mask)
    _bounds(ptr, idx, m, "write")
    val = np.broadcast_to(np.asarray(value), idx.shape).astype(ptr.mem.dtype)
    ptr.mem[idx[m]] = val[m]


def program_id(axis=0):
    return _state["pid"][axis]


def num_programs(axis=0):
    return _state["grid"][axis]


def arange(start, end):
    n = end - start
    if n <= 0 or n & (n - 1):
        raise ValueError(f"arange's range must be a power of 2, got tl.arange({start}, {end})")
    return _block(np.arange(start, end, dtype=np.int32))


def zeros(shape, dtype=float32):
    return _block(np.zeros(shape, dtype=dtype.np))


def full(shape, value, dtype=float32):
    return _block(np.full(shape, value, dtype=dtype.np))


def maximum(a, b):
    return _block(np.maximum(a, b))


def minimum(a, b):
    return _block(np.minimum(a, b))


def where(cond, a, b):
    return _block(np.where(cond, a, b))


def exp(x):
    return _block(np.exp(x))


def log(x):
    return _block(np.log(x))


def sqrt(x):
    return _block(np.sqrt(x))


def rsqrt(x):
    return _block(1 / np.sqrt(x))


def abs(x):
    return _block(np.abs(x))


def sum(x, axis=None):
    return _block(np.sum(x, axis=axis))


def max(x, axis=None):
    return _block(np.max(x, axis=axis))


def min(x, axis=None):
    return _block(np.min(x, axis=axis))


def dot(a, b, acc=None, **_options):
    """(M, K) @ (K, N) on the tensor cores: float16 in, float32 out. Triton needs M, N, K >= 16."""
    if a is Ellipsis or b is Ellipsis:
        raise TypeError("tl.dot got an ellipsis")
    a, b = np.asarray(a), np.asarray(b)
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError(f"tl.dot needs two 2D blocks, got shapes {a.shape} and {b.shape}")
    if a.shape[1] != b.shape[0]:
        raise ValueError(f"tl.dot: inner sizes differ, {a.shape} @ {b.shape}")
    if min(a.shape + b.shape) < 16:
        raise ValueError(f"tl.dot: every block size must be at least 16 (tensor cores work on 16-wide tiles), got {a.shape} @ {b.shape}")
    r = a.astype(np.float32) @ b.astype(np.float32)
    return _block(r if acc is None else np.asarray(acc) + r)


def cdiv(a, b):
    return (a + b - 1) // b


range = range
static_range = range
