"""Shared checker for Part 2 onward: guarded tensors, PASS/FAIL lines, the command line.

Every tensor a kernel touches sits inside a bigger buffer with guard values in front and
behind, because a GPU does not stop a kernel that writes past the end of a tensor.
Outputs start filled with UNWRITTEN (-4096), so a box the kernel never wrote shows up as
"never written", and a NaN or inf the kernel did write shows up as a wrong value.
"""
import argparse
import importlib
import sys

from kfs_backend import BACKEND, DEVICE

import torch

GUARD = 8192
GUARD_VALUE = 1234.5
UNWRITTEN = -4096.0  # exact in float16 and float32; no kernel in this course outputs it
TOL = {torch.float32: (1e-5, 1e-6), torch.float16: (1e-2, 1e-5)}


class StopChecking(Exception):
    """Raised after an ERROR so the other cases don't repeat the same message."""


class Guarded:
    """A tensor placed in the middle of a buffer full of guard values.

    row_pad > 0 makes the rows of a 2D tensor sit further apart than its width, like a
    slice of a wider matrix, so a kernel that assumes row r starts at r * N fails.
    """

    def __init__(self, shape, dtype, init=None, row_pad=0):
        cols = shape[-1]
        rows = 1
        for d in shape[:-1]:
            rows *= d
        span = rows * (cols + row_pad)
        self.buf = torch.full((GUARD + span + GUARD,), GUARD_VALUE, dtype=dtype, device=DEVICE)
        body = self.buf[GUARD : GUARD + span].view(rows, cols + row_pad)[:, :cols]
        self.t = body if len(shape) == 2 else body.view(shape)  # (heads, rows, cols) splits the rows: still a view
        if init is None:
            self.t.fill_(UNWRITTEN)
        else:
            self.t.copy_(init)
        self.before = self.t.clone()
        self.span = span
        self.row_pad = row_pad
        self.cols = cols

    def outside_writes(self):
        """Indices (relative to the tensor start) of guard values that changed, including row gaps."""
        g = self.buf.clone()
        g[GUARD : GUARD + self.span].view(-1, self.cols + self.row_pad)[:, : self.cols] = GUARD_VALUE
        changed = (g != torch.tensor(GUARD_VALUE, dtype=g.dtype, device=g.device)).nonzero().flatten() - GUARD
        return changed.tolist()


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


def num(v):
    return "0" if v == 0 else f"{v:.4g}"  # no "-0"


def compare(got, want, tol):
    """None if close enough, else one line: what is wrong and the first place it is wrong."""
    rtol, atol = tol
    got, want = got.float().cpu(), want.float().cpu()
    if got.dim() == 3:  # (heads, rows, columns): describe the first head that is wrong
        heads = (~torch.isclose(got, want, rtol=rtol, atol=atol)).flatten(1).any(1).nonzero().flatten().tolist()
        if not heads:
            return None
        line = compare(got[heads[0]], want[heads[0]], tol)
        if got.shape[0] == 1:
            return line
        more = f" (and {len(heads) - 1} more heads)" if len(heads) > 1 else ""
        return f"head {heads[0]}: {line}{more}"
    two_d = got.dim() == 2
    if not two_d:
        got, want = got[None], want[None]
    bad = ~torch.isclose(got, want, rtol=rtol, atol=atol)
    if not bad.any():
        return None
    n, k = got.numel(), int(bad.sum())
    rows = bad.any(1).nonzero().flatten().tolist()
    r = rows[0]
    cols = bad[r].nonzero().flatten().tolist()
    if bool((got[bad] == UNWRITTEN).all()):
        where = f"rows {ranges(rows)}, columns {ranges(cols)}" if two_d else ranges(cols)
        return f"{k} of {n} never written: {where}"
    c = cols[0]
    g = got[r, c].item()
    at = f"[{r}, {c}]" if two_d else f"index {c}"
    note = " = a guard value, read from outside the input" if abs(g - GUARD_VALUE) < 1 else ""
    in_rows = f" in rows {ranges(rows)}" if two_d else ""
    return f"{k} of {n} wrong{in_rows}  ({at}: expected {num(want[r, c].item())}, got {num(g)}{note})"


def run_case(label, call, want, out, inputs, tol=None):
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
    wrong = compare(out.t, want, tol or TOL[out.buf.dtype])
    if wrong:
        problems.append(wrong)
    spill = out.outside_writes()
    if spill:
        where = "past the end of" if min(spill) >= out.span else "outside"
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


def main(exercises):
    """exercises: {"5": ("ex5_softmax", [("softmax", checker)]), ...}"""
    ap = argparse.ArgumentParser()
    ap.add_argument("which", nargs="+", help=f"exercise numbers ({', '.join(exercises)}) or 'all'")
    ap.add_argument("--solutions", action="store_true", help="check solutions/ instead of your files")
    a = ap.parse_args()
    which = list(exercises) if "all" in a.which else a.which
    for w in which:
        if w not in exercises:
            ap.error(f"no exercise {w} in this part; choose from {', '.join(exercises)} or all")
    print(f"backend: {BACKEND}")
    results = []
    for w in which:
        mod_name, fns = exercises[w]
        mod = importlib.import_module(("solutions." if a.solutions else "") + mod_name)
        for fn_name, checker in fns:
            print(f"\nex{w}  {fn_name}")
            try:
                ok = checker(getattr(mod, fn_name))
            except StopChecking:
                print("  (other cases skipped)")
                ok = False
            results.append((f"ex{w} {fn_name}", ok))
    print()
    for name, ok in results:
        print(f"{'PASS' if ok else 'FAIL'}  {name}")
    sys.exit(0 if all(ok for _, ok in results) else 1)
