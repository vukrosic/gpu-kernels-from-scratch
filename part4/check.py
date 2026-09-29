"""Checks your Part 4 attention kernels against a float64 reference.

    python check.py 13                # exercise 13
    python check.py 14 15
    python check.py all
    python check.py all --solutions   # check the reference solutions instead

q, k and v are (heads, N, D) float16; the output is float16, added up in float32.
The cases include N that is not a multiple of the 64-row block (100, 1000, 257), several
heads, q/k/v that are slices of one packed qkv tensor (rows 3*D apart, as in most models),
and a head where every score is below -100, which is far below anything exp can represent
in float32 (e^-104 is already 0).
The guards, the "never written" marker and the output lines are in ../kfs_check.py.
"""
import numpy as np

from backend import DEVICE  # noqa: F401

import torch
from kfs_check import Guarded, StopChecking, main, run_case

F16 = torch.float16
ATTN_TOL = (1e-2, 2e-3)  # p is rounded to float16 before p @ v, as in every fast attention kernel

# heads, N, D, layout: "" separate tensors, "qkv" slices of one (heads, N, 3*D) tensor, "far" every score below -100
FA_CASES = [(1, 16, 16, ""), (1, 64, 64, ""), (2, 128, 64, ""), (1, 100, 64, ""), (1, 1000, 32, ""),
            (3, 257, 64, ""), (1, 256, 64, "qkv"), (1, 256, 64, "far")]
CAUSAL_CASES = [(1, 16, 16, ""), (1, 64, 64, ""), (2, 128, 64, ""), (1, 100, 64, ""), (1, 1000, 32, ""),
                (3, 257, 64, ""), (1, 256, 64, "qkv")]
LSE_CASES = [(1, 64, 64, ""), (2, 128, 64, ""), (1, 100, 64, ""), (3, 257, 64, ""), (1, 256, 64, "far")]


def operands(Z, N, D, layout, seed):
    gen = torch.Generator().manual_seed(seed)
    q, k, v = (torch.randn(Z, N, D, generator=gen) for _ in range(3))
    if layout == "far":  # q points one way, k the opposite way: scores near -200
        u = torch.ones(D) / D ** 0.5
        q, k = -40 * u + 0.3 * q, 40 * u + 0.3 * k
    if layout == "qkv":
        G = Guarded((Z, N, 3 * D), F16, torch.cat([q, k, v], dim=2))
        return G.t[..., :D], G.t[..., D : 2 * D], G.t[..., 2 * D :], {"qkv": G}
    Q, K, V = (Guarded((Z, N, D), F16, x) for x in (q, k, v))
    return Q.t, K.t, V.t, {"q": Q, "k": K, "v": V}


def reference(q, k, v, causal=False):
    """softmax(q @ k^T / sqrt(D)) @ v in float64, and the logsumexp of every row of scores."""
    q, k, v = (x.double().cpu() for x in (q, k, v))
    s = q @ k.transpose(1, 2) * q.shape[-1] ** -0.5
    if causal:
        N = q.shape[1]
        s = s.masked_fill(torch.ones(N, N, dtype=torch.bool).triu(1), float("-inf"))
    return torch.softmax(s, dim=-1) @ v, torch.logsumexp(s, dim=-1)


def label(Z, N, D, layout):
    note = {"": "", "qkv": "  q, k, v slices of one qkv", "far": "  every score below -100"}[layout]
    return f"heads={Z} N={N} D={D}{note}".ljust(46)


def check_attention(cases, seed0, causal=False):
    def checker(fn):
        ok = True
        for i, (Z, N, D, layout) in enumerate(cases):
            q, k, v, inputs = operands(Z, N, D, layout, seed0 + i)
            out = Guarded((Z, N, D), F16)
            want, _ = reference(q, k, v, causal)
            ok &= run_case(label(Z, N, D, layout), lambda: fn(q, k, v, out=out.t), want, out, inputs, ATTN_TOL)
        return ok

    return checker


def check_lse(fn):
    ok = True
    for i, (Z, N, D, layout) in enumerate(LSE_CASES):
        q, k, v, inputs = operands(Z, N, D, layout, 40 + i)
        out = Guarded((Z, N, D), F16)
        lse = Guarded((Z, N, 1), torch.float32)  # (heads, N) with room to name the row that is wrong
        want, want_lse = reference(q, k, v)
        ok &= run_case(label(Z, N, D, layout), lambda: fn(q, k, v, out=out.t, lse=lse.t[..., 0])[0], want, out,
                       inputs, ATTN_TOL)
        ok &= run_case("  logsumexp".ljust(46), lambda: None, want_lse[..., None], lse, {}, (1e-5, 1e-4))
    return ok


# ---------------------------------------------------------------- exercise 13: NumPy, no kernel

ROW_CASES = [
    ("[1, 3, 2, 5] in blocks of 2", lambda g: np.array([1.0, 3.0, 2.0, 5.0]), 2),
    ("1000 random numbers, blocks of 64", lambda g: g.standard_normal(1000) * 3, 64),
    ("a rising row, blocks of 16", lambda g: np.linspace(-5, 20, 300), 16),
    ("a falling row, blocks of 16", lambda g: np.linspace(20, -5, 300), 16),
    ("numbers near 1000, blocks of 8", lambda g: 1000 + g.standard_normal(100), 8),
    ("blocks of 1", lambda g: g.standard_normal(50), 1),
]


def _call(name, call):
    """Runs the student's function; an exception becomes one ERROR line, as in kfs_check.run_case."""
    try:
        with np.errstate(all="ignore"):
            return call()
    except Exception as e:
        msg = str(e).strip().splitlines()[0] if str(e).strip() else ""
        if "ellipsis" in msg.lower():
            msg = "a TODO line is still `...`"
        print(f"  {name.ljust(40)}  ERROR {type(e).__name__}: {msg}")
        raise StopChecking


def _close(got, want):
    return got is not None and np.isfinite(got).all() and np.allclose(got, want, rtol=1e-9, atol=1e-12)


def check_running(fn):
    ok = True
    for i, (name, make, block) in enumerate(ROW_CASES):
        x = make(np.random.default_rng(i))
        want_m = x.max()
        want_l = np.exp(x - want_m).sum()
        blocks = (x[j : j + block].copy() for j in range(0, len(x), block))  # one pass: a generator can't be rewound
        m, l = _call(name, lambda: fn(blocks))
        if m is Ellipsis or l is Ellipsis:
            print(f"  {name.ljust(40)}  ERROR a TODO line is still `...`")
            raise StopChecking
        problems = []
        if not _close(np.float64(m), want_m):
            problems.append(f"max = {m:.6g}, expected {want_m:.6g}")
        if not _close(np.float64(l), want_l):
            problems.append(f"sum = {l:.6g}, expected {want_l:.6g}")
        print(f"  {name.ljust(40)}  " + ("FAIL  " + "; ".join(problems) if problems else "PASS"))
        ok &= not problems
    return ok


def check_softmax(fn):
    ok = True
    for i, (name, make, block) in enumerate(ROW_CASES):
        x = make(np.random.default_rng(i))
        want = np.exp(x - x.max()) / np.exp(x - x.max()).sum()
        got = np.asarray(_call(name, lambda: fn(x, block=block)), dtype=np.float64)
        if got.shape != want.shape:
            print(f"  {name.ljust(40)}  FAIL  returned shape {got.shape}, expected {want.shape}")
            ok = False
            continue
        bad = ~np.isclose(got, want, rtol=1e-9, atol=1e-15)
        if bad.any():
            j = int(bad.nonzero()[0][0])
            print(f"  {name.ljust(40)}  FAIL  {int(bad.sum())} of {len(x)} wrong  (index {j}: expected {want[j]:.4g}, got {got[j]:.4g})")
            ok = False
        else:
            print(f"  {name.ljust(40)}  PASS")
    return ok


EXERCISES = {
    "13": ("ex13_online_softmax", [("running_max_and_sum", check_running), ("softmax", check_softmax)]),
    "14": ("ex14_flash_attention", [("attention", check_attention(FA_CASES, 0))]),
    "15": ("ex15_spot_the_bug", [("attention_a", check_attention(FA_CASES, 0)),
                                 ("attention_b", check_attention(FA_CASES, 0)),
                                 ("attention_c", check_attention(FA_CASES, 0))]),
    "16": ("ex16_causal", [("attention_causal", check_attention(CAUSAL_CASES, 20, causal=True))]),
    "17": ("ex17_logsumexp", [("attention_lse", check_lse)]),
}

if __name__ == "__main__":
    main(EXERCISES)
