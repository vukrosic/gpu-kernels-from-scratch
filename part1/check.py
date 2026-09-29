"""Checks your Part 1 kernels against PyTorch.

    python check.py 1                 # exercise 1
    python check.py 2 3               # several
    python check.py all
    python check.py all --solutions   # check the reference solutions instead

Every tensor your kernel touches sits inside a bigger buffer with guard values in front
and behind. A GPU does not stop a kernel that writes past the end of a tensor, so this is
how the checker catches it. Output lanes start as NaN, so any box your kernel never
writes shows up as "never written".
"""
import argparse
import importlib
import sys

from backend import BACKEND, DEVICE

import torch

GUARD = 8192
GUARD_VALUE = 1234.5
SIZES = [(1, torch.float32), (1000, torch.float32), (1024, torch.float32), (4097, torch.float32), (100_003, torch.float32), (4097, torch.float16)]
TOL = {torch.float32: (1e-5, 1e-6), torch.float16: (1e-2, 1e-2)}


class StopChecking(Exception):
    """Raised after an ERROR so the other sizes don't repeat the same message."""


class Guarded:
    """A tensor placed in the middle of a buffer full of guard values."""

    def __init__(self, shape, dtype, init=None, row_pad=0):
        rows, cols = (1, shape[0]) if len(shape) == 1 else shape
        span = rows * (cols + row_pad)
        self.buf = torch.full((GUARD + span + GUARD,), GUARD_VALUE, dtype=dtype, device=DEVICE)
        body = self.buf[GUARD : GUARD + span].view(rows, cols + row_pad)[:, :cols]
        self.t = body.reshape(shape) if len(shape) == 1 else body
        if init is None:
            self.t.fill_(float("nan"))
        else:
            self.t.copy_(init)
        self.before = self.t.clone()
        self.span = span

    def outside_writes(self):
        """Indices (relative to the tensor start) of guard values that changed."""
        g = torch.full((GUARD,), GUARD_VALUE, dtype=self.buf.dtype, device=DEVICE)
        front = (self.buf[:GUARD] != g).nonzero().flatten() - GUARD
        back = (self.buf[GUARD + self.span :] != g).nonzero().flatten() + self.span
        return torch.cat([front, back]).tolist()


def ranges(idx, limit=3):
    idx = sorted(idx)
    out, start, prev = [], idx[0], idx[0]
    for i in idx[1:] + [None]:
        if i is not None and i == prev + 1:
            prev = i
            continue
        out.append(f"{start}..{prev}" if prev > start else f"{start}")
        if i is not None:
            start = prev = i
    return ", ".join(out[:limit]) + (" ..." if len(out) > limit else "")


def compare(got, want, dtype):
    rtol, atol = TOL[dtype]
    shape = tuple(want.shape)
    got, want = got.float().flatten().cpu(), want.float().flatten().cpu()
    bad = ~torch.isclose(got, want, rtol=rtol, atol=atol)
    if not bad.any():
        return None
    idx = bad.nonzero().flatten().tolist()
    unwritten = torch.isnan(got[bad])
    n = got.numel()
    if unwritten.all():
        return f"{len(idx)} of {n} never written: {ranges(idx)}"
    i = idx[0]
    where = f"[{i // shape[1]}, {i % shape[1]}]" if len(shape) == 2 else f"index {i}"
    g = got[i].item()
    note = " = a guard value, read from outside the input" if abs(g - GUARD_VALUE) < 1 else ""
    return f"{len(idx)} of {n} wrong: {ranges(idx)}  ({where}: expected {want[i].item():.4g}, got {g:.4g}{note})"


def run_case(label, call, want, out, inputs):
    try:
        ret = call()
    except Exception as e:  # show the first line, the way a terminal would
        msg = str(e).strip().splitlines()[0] if str(e).strip() else ""
        if "ellipsis" in msg.lower():
            msg = "a TODO line is still `...`"
        print(f"  {label}  ERROR {type(e).__name__}: {msg}")
        raise StopChecking
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    if ret is not None and ret.data_ptr() != out.t.data_ptr():
        print(f"  {label}  FAIL  returned a new tensor instead of writing into out=")
        return False
    problems = []
    wrong = compare(out.t, want, out.buf.dtype)
    if wrong:
        problems.append(wrong)
    spill = out.outside_writes()
    if spill:
        where = "past the end of" if min(spill) >= 0 else "outside"
        head = "values OK, but " if not wrong else ""
        problems.append(f"{head}{len(spill)} writes {where} out: {ranges(spill)}")
    for name, g in inputs.items():
        if g.outside_writes() or not torch.equal(g.t, g.before):
            problems.append(f"your kernel wrote into input {name} or next to it")
    if problems:
        print(f"  {label}  FAIL  " + "; ".join(problems))
        return False
    print(f"  {label}  PASS")
    return True


def check_add(fn):
    ok = True
    gen = torch.Generator().manual_seed(0)
    for n, dt in SIZES:
        x = Guarded((n,), dt, torch.randn(n, generator=gen))
        y = Guarded((n,), dt, torch.randn(n, generator=gen))
        out = Guarded((n,), dt)
        want = x.t.float() + y.t.float()
        label = f"n={n:<7} {str(dt)[6:]:<7}"
        ok &= run_case(label, lambda: fn(x.t, y.t, out=out.t), want, out, {"x": x, "y": y})
    return ok


def check_fused(fn):
    ok = True
    gen = torch.Generator().manual_seed(1)
    scale, shift = 1.7, -0.3
    for n, dt in SIZES:
        x = Guarded((n,), dt, torch.randn(n, generator=gen))
        out = Guarded((n,), dt)
        want = torch.relu(x.t.float() * scale + shift)
        label = f"n={n:<7} {str(dt)[6:]:<7}"
        ok &= run_case(label, lambda: fn(x.t, scale, shift, out=out.t), want, out, {"x": x})
    return ok


def check_row_bias(fn):
    ok = True
    gen = torch.Generator().manual_seed(2)
    cases = [(1, 1, 0), (3, 1000, 0), (64, 33, 0), (5, 4097, 0), (4, 1000, 5)]
    for M, N, pad in cases:
        x = Guarded((M, N), torch.float32, torch.randn(M, N, generator=gen), row_pad=pad)
        b = Guarded((N,), torch.float32, torch.randn(N, generator=gen))
        out = Guarded((M, N), torch.float32)
        want = x.t.float() + b.t.float()
        note = f" (x rows {N + pad} apart)" if pad else ""
        label = f"{M}x{N}{note}".ljust(28)
        ok &= run_case(label, lambda: fn(x.t, b.t, out=out.t), want, out, {"x": x, "bias": b})
    return ok


EXERCISES = {
    "1": ("ex1_vector_add", [("add", check_add)]),
    "2": ("ex2_spot_the_bug", [("add_a", check_add), ("add_b", check_add), ("add_c", check_add)]),
    "3": ("ex3_fused", [("scale_shift_relu", check_fused)]),
    "4": ("ex4_row_bias", [("add_row_bias", check_row_bias)]),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("which", nargs="+", help="exercise numbers (1-4) or 'all'")
    ap.add_argument("--solutions", action="store_true", help="check solutions/ instead of your files")
    a = ap.parse_args()
    which = list(EXERCISES) if "all" in a.which else a.which
    print(f"backend: {BACKEND}")
    results = []
    for w in which:
        mod_name, fns = EXERCISES[w]
        mod = importlib.import_module(("solutions." if a.solutions else "") + mod_name)
        for fn_name, checker in fns:
            print(f"\nex{w}  {fn_name}")
            try:
                ok = checker(getattr(mod, fn_name))
            except StopChecking:
                print("  (other sizes skipped)")
                ok = False
            results.append((f"ex{w} {fn_name}", ok))
    print()
    for name, ok in results:
        print(f"{'PASS' if ok else 'FAIL'}  {name}")
    sys.exit(0 if all(ok for _, ok in results) else 1)


if __name__ == "__main__":
    main()
