"""Checks your Part 5 backward passes against a float64 reference.

    python check.py 18                # exercise 18
    python check.py 19 20
    python check.py all
    python check.py all --solutions   # check the reference solutions instead

Each case runs your forward and backward through PyTorch's autograd, exactly as training
does, then compares every gradient with float64 PyTorch on the same inputs. The inputs sit
inside guard values (../kfs_check.py), so a backward that writes into an input fails.
"""
from backend import DEVICE

import torch
from kfs_check import TOL, Guarded, StopChecking, compare, main

F16, F32 = torch.float16, torch.float32
GRAD_TOL = {F32: (1e-4, 1e-5), F16: (1e-2, 1e-3)}
ATTN_TOL = (2e-2, 4e-3)  # p is rounded to float16 before each p @ dO, as in every fast attention kernel


def _run(label, call):
    try:
        call()
    except Exception as e:
        msg = str(e).strip().splitlines()[0] if str(e).strip() else ""
        if "ellipsis" in msg.lower() or "dtype('O')" in msg:  # dtype('O'): a block built from `...`
            msg = "a TODO line is still `...`"
        print(f"  {label}  ERROR {type(e).__name__}: {msg}")
        raise StopChecking
    if DEVICE == "cuda":
        torch.cuda.synchronize()


def _grads(label, pairs, inputs, tol):
    """pairs: [(name, got, want)]. One PASS/FAIL line for the case."""
    problems = []
    for name, got, want in pairs:
        if got is None:
            problems.append(f"no gradient reached {name}")
            continue
        if got.shape != want.shape:
            problems.append(f"d{name} has shape {tuple(got.shape)}, expected {tuple(want.shape)}")
            continue
        wrong = compare(got, want, tol or TOL[got.dtype])
        if wrong:
            problems.append(f"d{name}: {wrong}")
    for name, g in inputs.items():
        if g.outside_writes() or not torch.equal(g.t, g.before):
            problems.append(f"your backward wrote into {name} or next to it")
    print(f"  {label}  " + ("FAIL  " + "; ".join(problems) if problems else "PASS"))
    return not problems


def _leaf(G):
    return G.t.requires_grad_()


# ---------------------------------------------------------------- exercise 18: matmul backward

# M, K, N, row_pad: a row_pad > 0 makes a and b slices of wider matrices
MATMUL_CASES = [(16, 16, 16, 0), (64, 64, 64, 0), (128, 96, 80, 0), (100, 70, 130, 0), (257, 33, 65, 0),
                (200, 64, 100, 24)]


def check_matmul(Fn):
    ok = True
    for i, (M, K, N, pad) in enumerate(MATMUL_CASES):
        gen = torch.Generator().manual_seed(180 + i)
        A = Guarded((M, K), F16, torch.randn(M, K, generator=gen), row_pad=pad)
        B = Guarded((K, N), F16, torch.randn(K, N, generator=gen), row_pad=pad)
        dC = Guarded((M, N), F16, torch.randn(M, N, generator=gen))
        a, b = _leaf(A), _leaf(B)
        label = f"({M} x {K}) @ ({K} x {N})" + ("  slices of wider matrices" if pad else "")
        _run(label.ljust(52), lambda: Fn.apply(a, b).backward(dC.t))
        a64, b64, dc64 = (t.detach().double().cpu() for t in (a, b, dC.t))
        ok &= _grads(label.ljust(52), [("a", a.grad, dc64 @ b64.T), ("b", b.grad, a64.T @ dc64)],
                     {"a": A, "b": B, "dC": dC}, TOL[F16])
    return ok


# ---------------------------------------------------------------- exercise 19: RMSNorm backward

# M, N, dtype, note: "slice" rows further apart than N, "300" a float16 row with a 300 in it
RMS_CASES = [(1, 2, F32, ""), (4, 8, F32, ""), (64, 1000, F32, ""), (1000, 257, F32, ""), (300, 64, F16, ""),
             (3, 4096, F16, ""), (200, 96, F32, "slice"), (16, 128, F16, "300")]


def check_rmsnorm(fn):
    ok = True
    for i, (M, N, dt, note) in enumerate(RMS_CASES):
        gen = torch.Generator().manual_seed(190 + i)
        x0 = torch.randn(M, N, generator=gen)
        if note == "300":
            x0[3, 5] = 300.0
        X = Guarded((M, N), dt, x0, row_pad=8 if note == "slice" else 0)
        W = Guarded((N,), dt, 1 + 0.1 * torch.randn(N, generator=gen))
        dY = Guarded((M, N), dt, torch.randn(M, N, generator=gen))
        x, w = _leaf(X), _leaf(W)
        label = f"{M} row{'s' if M > 1 else ''} of {N}, {str(dt)[6:]}" + {"": "", "slice": "  rows further apart than N",
                                                   "300": "  one number is 300"}[note]
        _run(label.ljust(52), lambda: fn(x, w, 1e-6).backward(dY.t))
        x64, w64 = (t.detach().double().cpu().requires_grad_() for t in (x, w))
        y64 = x64 * torch.rsqrt((x64 * x64).mean(1, keepdim=True) + 1e-6) * w64
        y64.backward(dY.t.double().cpu())
        ok &= _grads(label.ljust(52), [("x", x.grad, x64.grad), ("w", w.grad, w64.grad)],
                     {"x": X, "w": W, "dy": dY}, GRAD_TOL[dt])
    return ok


# ---------------------------------------------------------------- exercise 20: attention backward

ATTN_CASES = [(1, 16, 16), (1, 64, 64), (2, 128, 64), (1, 100, 64), (3, 257, 64), (1, 1000, 32)]


def check_attention(fn):
    ok = True
    for i, (Z, N, D) in enumerate(ATTN_CASES):
        gen = torch.Generator().manual_seed(200 + i)
        Q, K, V, dO = (Guarded((Z, N, D), F16, torch.randn(Z, N, D, generator=gen)) for _ in range(4))
        q, k, v = _leaf(Q), _leaf(K), _leaf(V)
        label = f"heads={Z} N={N} D={D}  causal"
        _run(label.ljust(52), lambda: fn(q, k, v).backward(dO.t))
        q64, k64, v64 = (t.detach().double().cpu().requires_grad_() for t in (q, k, v))
        s = q64 @ k64.transpose(1, 2) * D ** -0.5
        s = s.masked_fill(torch.ones(N, N, dtype=torch.bool).triu(1), float("-inf"))
        (torch.softmax(s, dim=-1) @ v64).backward(dO.t.double().cpu())
        ok &= _grads(label.ljust(52), [("q", q.grad, q64.grad), ("k", k.grad, k64.grad), ("v", v.grad, v64.grad)],
                     {"q": Q, "k": K, "v": V, "dO": dO}, ATTN_TOL)
    return ok


EXERCISES = {
    "18": ("ex18_matmul_backward", [("Matmul", check_matmul)]),
    "19": ("ex19_rmsnorm_backward", [("rmsnorm", check_rmsnorm)]),
    "20": ("ex20_attention_backward", [("attention", check_attention)]),
}

if __name__ == "__main__":
    main(EXERCISES)
