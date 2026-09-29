# Part 5: Capstone, train on your kernels

Your kernels from Parts 2–4 compute a forward pass. Training also needs the backward pass: the gradient of the loss with respect to every input of every kernel. PyTorch can't work that out for a Triton kernel, because it never sees the math inside it. This part writes the three backward passes a transformer needs, matmul, RMSNorm and attention, and then trains a small GPT on them next to PyTorch.

The tasks come in the order the video gives them. Paper tasks need no computer. Code tasks are finished when `python check.py N` prints PASS.

| Task | Kind | Done when |
| --- | --- | --- |
| P13 Predict the backward | predict on paper | two formulas and one number |
| ex18 Matmul backward | 2 lines | `python check.py 18` → PASS |
| P14 Invent the RMSNorm backward | invent on paper | two numbers and what to save |
| ex19 RMSNorm backward | 4 lines of a kernel | `python check.py 19` → PASS |
| P15 Store P or rebuild it | predict on paper | three numbers |
| ex20 Attention backward | 5 lines of a kernel | `python check.py 20` → PASS |
| ex21 Train | run it | `python train.py` → PASS |

`kernels.py` holds the forward kernels, copied from the reference solutions of Parts 2–4, so a mistake left in an earlier part doesn't block this one. Each forward also saves what its backward needs.

## Why PyTorch can't train your kernel

Your kernel writes its output into a tensor PyTorch made with `torch.empty`. PyTorch records how every tensor was computed so it can run the chain rule backwards, but it never saw what happened inside the kernel. This is a real run of `matmul(a, b).float().sum().backward()` with the ex10 kernel:

```
RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn
```

Inside a model it is worse. The other layers still have gradients, so there is no error: the weights that feed your kernel get `grad = None`, and the optimizer skips them. The loss still goes down, only more slowly, and nothing tells you why.

`torch.autograd.Function` is the fix. You give PyTorch a `forward` and a `backward`; in `forward` you save what `backward` will need with `ctx.save_for_backward`.

## P13 Predict the backward

C = A @ B, with A of shape (M, K) and B of shape (K, N). The backward gets dC, the gradient of the loss with respect to C, shape (M, N).

1. Write dA (shape (M, K)) and dB (shape (K, N)) as matmuls of dC, A and B. Check with A = [[1, 2], [3, 4]], B = [[5, 6], [7, 8]] and loss = the sum of C, so every entry of dC is 1.
2. The forward is 2·M·N·K FLOPs. How many FLOPs is the backward?

<details><summary>Answer</summary>

1. C[i, j] = Σ_k A[i, k] · B[k, j], so A[i, k] affects every C[i, j] in row i, each by B[k, j]. Adding those up: dA[i, k] = Σ_j dC[i, j] · B[k, j], which is **dA = dC @ Bᵀ**. The same argument for B gives **dB = Aᵀ @ dC**. The shapes are the check: (M, N) @ (N, K) is (M, K). In the example, dA = [[11, 15], [11, 15]] (the row sums of B) and dB = [[4, 4], [6, 6]].
2. Two matmuls of the same size: **4·M·N·K**, twice the forward. A training step is forward plus backward, 6·M·N·K, which is where the rule of thumb "training costs 6 × parameters × tokens FLOPs" comes from (Kaplan et al., "Scaling Laws for Neural Language Models", 2020, [arXiv 2001.08361](https://arxiv.org/abs/2001.08361)).

Bᵀ and Aᵀ cost nothing. `b.t()` is the same memory with its two strides swapped, and your ex10 kernel reads the strides, so no copy is made. That is the ex11 version C bug from the other side: a kernel that assumes rows are stored one after another gives wrong gradients here.
</details>

## ex18 Matmul backward

Open `ex18_matmul_backward.py`. `forward` saves a and b and calls `matmul` from `kernels.py`. Write the two lines of `backward` with `matmul` and `.t()`, and run `python check.py 18`. The cases include a and b that are slices of wider matrices, so their rows sit further apart than their width.

## P14 Invent the RMSNorm backward

For one row of N numbers, RMSNorm is

```
r = 1 / sqrt(mean(x²) + eps)
y = x · r · w
```

Every y depends on every x, through r. You get dy; call g = dy · w.

1. **Invent it.** Work out dx. (r depends on x_j: dr/dx_j = −r³ · x_j / N.)
2. Check with x = [3, 4], w = [1, 1], eps = 0 and dy = [1, 0]. What are dx and dw?
3. What does the forward have to save so the backward doesn't recompute the mean?

<details><summary>Answer</summary>

1. y_i = x_i · r · w_i. Changing x_j moves y_j directly, by r · w_j, and moves every y_i through r. Adding up both paths:
   dx_j = r · g_j + Σ_i g_i · x_i · (−r³ · x_j / N) = **r · g_j − x_j · r³ · mean(g · x)**.
2. r = 1 / sqrt(12.5) = 0.28284, r³ = 0.022627, g = [1, 0], mean(g · x) = 1.5.
   dx = [0.28284 − 3 · 0.022627 · 1.5, −4 · 0.022627 · 1.5] = **[0.1810, −0.1358]**.
   dw_j = Σ over rows of dy_j · x_j · r = **[0.8485, 0]**.
3. **r, one `float32` per row**, as ex17 saved lse, one number per query. 4 bytes a row; x itself is saved anyway.
</details>

## ex19 RMSNorm backward

Open `ex19_rmsnorm_backward.py`. dx is one row at a time, like the forward. dw adds up over every row, and thousands of programs can't all add into the same N numbers without atomics. So the kernel runs 128 programs (fewer if there are fewer rows); each takes rows pid, pid + 128, pid + 256, …, keeps its own running dw and writes one row of `dw_part`, and PyTorch adds up the 128 rows. Write the four TODO lines and run `python check.py 19`.

<details><summary>Hint</summary>

Load x and dy with `.to(tl.float32)` first, as in ex6: one case is a `float16` row with a 300 in it. Lanes past N load as 0, so they add nothing to `tl.sum(g * x, axis=0)`; divide by N, not by BLOCK.
</details>

## P15 Store P or rebuild it

The attention backward needs P, the softmax of every score (Part 4). N = 8192, D = 64, one head, `float16`, on an RTX 5080 (112.6 TFLOPS, 960 GB/s).

1. How big is P? How big is lse (ex17), one `float32` per query?
2. Rebuilding P from q, k and lse costs one more q @ kᵀ, 2 · N² · D FLOPs. How long is that at 112.6 TFLOPS?
3. How long does reading a stored P once take at 960 GB/s?

<details><summary>Answer</summary>

1. P is 8192² × 2 bytes = **134.2 MB**, the wall from P10 again. lse is 8192 × 4 bytes = **32.8 KB**, 4,096 times smaller.
2. 8.59 billion FLOPs take **76 µs**.
3. **140 µs**, and writing it in the forward is another 140 µs. Rebuilding P is faster than storing it even before counting the memory. That is why FlashAttention's backward keeps only lse: p = e^(score − lse) is exact, because lse is the log of the row's whole sum.
</details>

## ex20 Attention backward

With dO the gradient of the output, for one head:

```
p     = exp(s - lse)            rebuilt block by block, never stored
dV    = Pᵀ @ dO
dP    = dO @ Vᵀ
dS    = P * (dP - delta)        delta = rowsum(dO * O), one number per query
dQ    = dS @ K * scale
dK    = dSᵀ @ Q * scale
```

delta comes from the softmax: dS_ij = P_ij (dP_ij − Σ_k P_ik dP_ik), and Σ_k P_ik dP_ik = dO_i · O_i, so it needs only dO and the saved output.

Open `ex20_attention_backward.py`. There are two kernels, so that no two programs ever write the same number. `attention_dq_kernel` is finished: one program per block of 64 queries, walking the keys it can see. Read it first. `attention_dkdv_kernel` is one program per block of 64 keys, walking the queries that can see them, so everything in it is transposed: `st = k @ qᵀ` is sᵀ and `pt` is pᵀ. Write its five TODO lines and run `python check.py 20`.

<details><summary>Hint</summary>

The transposes of P13: pᵀ @ dO needs no transpose here, because you already hold pᵀ. lse is one number per query, and queries are the columns of `st`, so it is `lse[None, :]`, and so is delta. `tl.dot` takes `float16`: `pt.to(tl.float16)`.
</details>

## ex21 Train

```bash
python train.py
```

A character-level GPT (2 layers, d = 64, 2 heads, context 64, batch 8) learns to write Triton from the Python of Parts 1–4, 123,020 characters. Every matmul runs on ex18, every RMSNorm on ex19 and all of attention on ex20; the embedding, GELU, residual adds and the loss stay in PyTorch. Then the same model trains again on PyTorch alone, from the same seed on the same batches. It prints the average loss over every 100 steps for both runs.

This is the reference run (`python train.py --kernels solutions` on a Mac through minitriton, 1000 steps, 7 min 57 s):

```
      steps  solutions kernels    PyTorch  difference
       0-99             3.2139     3.2280      0.0142
    100-199             2.6398     2.6461      0.0063
    ...
    800-899             1.6643     1.6936      0.0293
    900-999             1.5786     1.5965      0.0179

PASS  largest loss difference 0.0293 (under 0.1)
```

Step 0 is 4.5376 for both: a model that knows nothing gives each of the 93 characters about 1/93, and ln 93 = 4.53. The runs don't agree step for step, because your kernels round to `float16` where PyTorch stays in `float32`, and two runs that differ in rounding drift apart the way two seeds would. The averages agree, and that is the test. After 1000 steps the model writes lines like `r tl.arange(2)` and `nevice=x.device`.

On minitriton, tokens/s measures nothing. On a GPU, make the model bigger (`--dim 256 --heads 4 --context 256 --batch 32`) and compare the two tokens/s numbers.

## What check.py tests

- Every gradient against `float64` PyTorch on the same inputs, through `autograd`, exactly as training calls it.
- Sizes that are not multiples of the blocks (70, 100, 257, 1000), because a bug past the end hides when they are.
- Inputs that are slices of wider matrices (ex18, ex19), so a backward that assumes rows one after another fails.
- A `float16` RMSNorm row with a 300 in it, and 1000 rows, so each of the 128 programs sums dw over several rows.
- Causal attention with several heads and N = 16 to 1000.
- Guards around every input: a backward must not write into what the forward saved.
