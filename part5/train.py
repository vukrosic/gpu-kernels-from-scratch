"""Exercise 21: train a small GPT on your kernels, and on PyTorch, and compare.

    python train.py                     # your ex18-ex20, then PyTorch: same seed, same batches
    python train.py --kernels solutions # the reference solutions instead of your files
    python train.py --steps 100         # a shorter run

The model is a character-level GPT with 2 layers. Every matmul (ex18), every RMSNorm (ex19)
and all of attention (ex20) run on your kernels; the embedding, GELU, residual adds and the
loss stay in PyTorch. The text is the Python of Parts 1-4, so the model learns to write
something that looks like a Triton kernel.

You are done when the two loss columns agree to within 0.1. Each row is the average loss
over 100 steps: single steps don't agree exactly, because your kernels round to float16
where the PyTorch run stays in float32, and two runs that differ in rounding drift apart
the way two seeds would. On minitriton the 1000 steps take about 8 minutes and tokens/s
mean nothing; on a GPU they take seconds, so raise --batch and --context and compare
tokens/s too.
"""
import argparse
import importlib
import math
import os
import time

from backend import BACKEND, DEVICE

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))


def load_text():
    """Every Python file of Parts 1-4: the kernels, the checkers, the benchmarks."""
    root = os.path.dirname(HERE)  # code/
    paths = []
    for part in ("part1", "part2", "part3", "part4"):
        for d, _, files in os.walk(os.path.join(root, part)):
            paths += [os.path.join(d, f) for f in files if f.endswith(".py")]
    return "\n".join(open(p, encoding="utf-8").read() for p in sorted(paths))


class Ops:
    """The four operations the model needs, from your kernels or from PyTorch."""

    def __init__(self, kernels):
        self.kernels = kernels
        if kernels != "torch":
            pre = "solutions." if kernels == "solutions" else ""
            self.Matmul = importlib.import_module(pre + "ex18_matmul_backward").Matmul
            self.rmsnorm_fn = importlib.import_module(pre + "ex19_rmsnorm_backward").rmsnorm
            self.attention_fn = importlib.import_module(pre + "ex20_attention_backward").attention

    def matmul(self, x, w):  # (M, K) float32 @ (K, N) float32 -> float32
        if self.kernels == "torch":
            return x @ w
        return self.Matmul.apply(x.half(), w.half()).float()

    def rmsnorm(self, x, w):  # (M, N) float32
        if self.kernels == "torch":
            return x * torch.rsqrt((x * x).mean(-1, keepdim=True) + 1e-6) * w
        return self.rmsnorm_fn(x, w, 1e-6)

    def attention(self, q, k, v):  # (heads, T, D) float32, causal
        if self.kernels == "torch":
            return F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.attention_fn(q.half(), k.half(), v.half()).float()


class GPT(torch.nn.Module):
    def __init__(self, vocab, d, layers, heads, context):
        super().__init__()
        self.heads, self.context = heads, context
        g = lambda *s: torch.nn.Parameter(torch.randn(*s) * 0.02)  # noqa: E731
        self.tok = g(vocab, d)
        self.pos = g(context, d)
        self.blocks = torch.nn.ModuleList()
        for _ in range(layers):
            b = torch.nn.ParameterDict({
                "n1": torch.nn.Parameter(torch.ones(d)), "qkv": g(d, 3 * d), "proj": g(d, d),
                "n2": torch.nn.Parameter(torch.ones(d)), "up": g(d, 4 * d), "down": g(4 * d, d)})
            self.blocks.append(b)
        self.norm = torch.nn.Parameter(torch.ones(d))
        self.head = g(d, vocab)

    def forward(self, idx, ops):
        B, T = idx.shape
        d, H = self.tok.shape[1], self.heads
        x = (self.tok[idx] + self.pos[:T]).view(B * T, d)
        for b in self.blocks:
            qkv = ops.matmul(ops.rmsnorm(x, b["n1"]), b["qkv"])  # (B*T, 3d)
            q, k, v = (t.reshape(B, T, H, d // H).transpose(1, 2).reshape(B * H, T, d // H)
                       for t in qkv.split(d, dim=1))
            a = ops.attention(q, k, v).reshape(B, H, T, d // H).transpose(1, 2).reshape(B * T, d)
            x = x + ops.matmul(a, b["proj"])
            x = x + ops.matmul(F.gelu(ops.matmul(ops.rmsnorm(x, b["n2"]), b["up"])), b["down"])
        return ops.matmul(ops.rmsnorm(x, self.norm), self.head).view(B, T, -1)


def run(kernels, data, vocab, a):
    torch.manual_seed(a.seed)
    model = GPT(vocab, a.dim, a.layers, a.heads, a.context).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.0)
    ops = Ops(kernels)
    batches = torch.Generator().manual_seed(a.seed + 1)  # both runs see the same batches
    losses, t0 = [], time.time()
    for step in range(a.steps):
        start = torch.randint(0, len(data) - a.context - 1, (a.batch,), generator=batches)
        chunk = torch.stack([data[s : s + a.context + 1] for s in start.tolist()]).to(DEVICE)
        logits = model(chunk[:, :-1], ops)
        loss = F.cross_entropy(logits.reshape(-1, vocab), chunk[:, 1:].reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        for name, p in model.named_parameters():
            if p.grad is None:  # the failure ex18 fixes: no error, the weight just never trains
                raise RuntimeError(f"no gradient reached {name}: is a backward returning None?")
        opt.step()
        losses.append(loss.item())
        if step == 0 and not math.isfinite(losses[0]):
            raise RuntimeError(f"step 0 loss is {losses[0]}")
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    return model, ops, losses, a.steps * a.batch * a.context / (time.time() - t0)


@torch.no_grad()
def sample(model, ops, chars, prompt, n, temperature=0.8):
    gen = torch.Generator().manual_seed(1)  # the same text every run
    idx = torch.tensor([[chars.index(c) for c in prompt]], device=DEVICE)
    for _ in range(n):
        logits = model(idx[:, -model.context :], ops)[0, -1].float().cpu() / temperature
        nxt = torch.multinomial(torch.softmax(logits, dim=-1), 1, generator=gen)
        idx = torch.cat([idx, nxt.view(1, 1).to(DEVICE)], dim=1)
    return "".join(chars[i] for i in idx[0].tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernels", default="yours", choices=["yours", "solutions"])
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--context", type=int, default=64)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--heads", type=int, default=2)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--csv", default=os.path.join(HERE, "losses.csv"))
    a = ap.parse_args()

    text = load_text()
    chars = sorted(set(text))
    data = torch.tensor([chars.index(c) for c in text])
    print(f"backend: {BACKEND}")
    print(f"text: the Python of Parts 1-4, {len(text):,} characters, {len(chars)} different ones")
    print(f"model: {a.layers} layers, d={a.dim}, {a.heads} heads, context {a.context}, batch {a.batch}\n")

    model, ops, mine, tps_mine = run(a.kernels, data, len(chars), a)
    _, _, ref, tps_ref = run("torch", data, len(chars), a)

    print(f"{'steps':>11}  {a.kernels + ' kernels':>17}  {'PyTorch':>9}  {'difference':>10}")
    worst, w = 0.0, 100
    for s in range(0, a.steps, w):
        x, y = (sum(l[s : s + w]) / len(l[s : s + w]) for l in (mine, ref))
        worst = max(worst, abs(x - y))
        print(f"{f'{s}-{min(s + w, a.steps) - 1}':>11}  {x:>17.4f}  {y:>9.4f}  {abs(x - y):>10.4f}")
    with open(a.csv, "w") as f:
        f.write(f"step,{a.kernels},torch\n" + "".join(f"{i},{x:.5f},{y:.5f}\n" for i, (x, y) in enumerate(zip(mine, ref))))
    speed = "" if DEVICE == "cuda" else "  (minitriton: checks correctness, not speed)"
    print(f"\ntokens/s: {tps_mine:,.0f} on {a.kernels} kernels, {tps_ref:,.0f} on PyTorch{speed}")
    print(f"losses saved to {os.path.relpath(a.csv)}")
    print("\nyour model writes:\n" + sample(model, ops, chars, "tl.", 160))
    ok = worst < 0.1
    print(f"\n{'PASS' if ok else 'FAIL'}  largest loss difference {worst:.4f} ({'under' if ok else 'not under'} 0.1)")


if __name__ == "__main__":
    main()
