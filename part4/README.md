# Part 4: FlashAttention

Attention is two matmuls with a softmax between them, and you have written all three pieces. Written the plain way, it stores an N × N matrix of scores that nobody needs once the output is done, and that matrix is most of its memory traffic. This part removes it: first a softmax that never sees its whole row, then the FlashAttention forward kernel, then the bugs that hide in it, then causal attention at half the work.

The tasks come in the order the video gives them, and each one has its own chapter in the video. Pause when a task card appears. Paper tasks need no computer. Code tasks are finished when `python check.py N` prints PASS.

| Task | Kind | Done when |
| --- | --- | --- |
| P10 The wall | predict on paper | three numbers |
| P11 Invent the rescale | predict on paper, then invent the fix | two numbers and one idea |
| ex13 Online softmax | fill in 2 lines of NumPy | `python check.py 13` → PASS |
| ex14 FlashAttention forward | fill in 5 update lines | `python check.py 14` → PASS |
| ex15 Spot the bug | diagnose and fix three AI-written kernels | `python check.py 15` → PASS ×3 |
| P12 Causal work | predict on paper | two numbers |
| ex16 Causal attention | change ex14's mask and loop | `python check.py 16` → PASS |
| ex17 Logsumexp (stretch) | save one number per query | `python check.py 17` → PASS |

Numbers below are for an RTX 5080: 112.6 TFLOPS of dense `float16` tensor-core math with `float32` accumulate, 960 GB/s of memory bandwidth and 64 MB of L2 cache (NVIDIA's RTX Blackwell architecture whitepaper).

Every token has a query, a key and a value, each D numbers long (D = 64 here). One head of attention is

```
S = Q @ K.T / sqrt(D)      (N, N)   score of every query for every key
P = softmax of each row of S
O = P @ V                  (N, D)
```

## P10 The wall

N = 8192 tokens, `float16` (2 bytes a number), one head.

1. How big is S?
2. The plain way writes S to memory, reads it back for the softmax, writes P and reads P again for the second matmul. How long do those four trips take at 960 GB/s?
3. How long does the math take at 112.6 TFLOPS? (Two matmuls, each 2 · N² · D FLOPs.)

<details><summary>Answer</summary>

1. 8192² = 67.1 million scores × 2 bytes = **134.2 MB**, for one head.
2. 4 × 134.2 MB = 536.9 MB, which takes **0.559 ms**.
3. 4 · N² · D = 17.18 billion FLOPs, which take **0.153 ms**. The memory trips take 3.7 times longer than the math. Q, K, V and O together are only 4.19 MB.

S and P are almost all the traffic, and they grow with N². With 32 heads, S is 4.29 GB for one layer; at N = 32,768 it is 68.7 GB, more than four times the RTX 5080's 16 GB. So never write S: make a block of scores on the chip, use it, and throw it away. The catch is softmax. It needs the row's max before it can make one weight, and the sum of all the weights before it can divide.
</details>

## P11 Invent the rescale

Keep two numbers while the row arrives in blocks: **m**, the largest score so far, and **l**, the sum of e^(x − m) over everything so far.

1. The row is [1, 3, 2, 5], in blocks of 2. After the first block, what are m and l?
2. **Invent the fix.** The second block brings a new max, 5. Every term in l was measured against the old max. How do you correct l without seeing the first block again?

<details><summary>Answer</summary>

1. m = 3, l = e⁻² + e⁰ = **1.1353**.
2. Every old term e^(x − 3) should be e^(x − 5): each one is off by the same factor, e^(3 − 5) = e⁻². So **one multiplication** fixes all of them: 1.1353 × 0.1353 = 0.1537. Then add the new block's terms, e⁻³ + e⁰: l = **1.2034**, exactly the sum over the whole row measured against 5. The softmax is [0.0152, 0.1125, 0.0414, 0.8310].

```
m_new = max(m, max(block))
l     = l * exp(m - m_new) + sum(exp(block - m_new))
m     = m_new
```

This is the online softmax of Maxim Milakov and Natalia Gimelshein (NVIDIA), "Online normalizer calculation for softmax", 2018, [arXiv 1805.02867](https://arxiv.org/abs/1805.02867).
</details>

## ex13 Online softmax

Open `ex13_online_softmax.py`. It is plain NumPy, no kernel: `running_max_and_sum` gets the row one block at a time, from a generator that can't be rewound, and keeps only m and l. Write the two TODO lines and run `python check.py 13`. `softmax` then uses your m and l.

<details><summary>Hint</summary>

m starts at −∞ and l at 0. One case is 100 numbers near 1000: e^1000 is infinity in any float, so every exponent has to be measured against the max so far, never against 0.
</details>

## ex14 FlashAttention forward

Open `ex14_flash_attention.py`. Each program takes 64 queries of one head (BLOCK_M = 64) and walks along the keys 64 at a time. The loads, the `tl.dot` that makes a 64 × 64 block of scores and the mask for keys past the end are written for you. Fill in the five update lines, the ex13 update for 64 rows at once, and run `python check.py 14`.

Each query keeps its own m and l, and one more running value, `acc`: the sum of e^(score − m) times the value, for the keys so far. It is measured against the max so far, exactly like l, so when the max grows it shrinks by the same factor. After the last block, `acc / l` is the output.

<details><summary>Hint</summary>

Name the factor: `alpha = tl.exp(m - m_new)`, one number per query. `m_new[:, None]` and `alpha[:, None]` turn a row of 64 numbers into a column, so every query uses its own. Round p to `float16` for `tl.dot(p, v)` (the tensor cores take `float16`) and keep `acc` in `float32`.
</details>

Now the traffic: Q, K, V and O cross the memory bus once, 4.2 MB for N = 8192, instead of 536.9 MB. Every block of queries reads K and V again, but for one head they are 2.1 MB, small enough to stay in the 64 MB L2 cache. The math didn't change; only where the numbers travel.

## ex15 Spot the bug

Three FlashAttention kernels written the way AI assistants write them. They all run without an error, pass some cases, and each has one wrong line. Run `python check.py 15`. Look at which cases pass and which fail, and say what went wrong before you read the code. Then fix each version.

<details><summary>Hint for version A</summary>

It passes N = 16 and N = 64 and fails every longer sequence. How many blocks of keys do 16 and 64 make, with BLOCK_N = 64? With one block, can the max ever change?
</details>

<details><summary>Hint for version B</summary>

It passes N = 64, 128 and 256, and fails N = 16, 100 and 257. What do 64, 128 and 256 have in common? What score does a key past the end get, and what weight does softmax give that score? It also passes N = 1000: its outputs are 1.4% too small there, inside the tolerance, which is why the checker tests short sequences too.
</details>

<details><summary>Hint for version C</summary>

It passes every case except the head where every score is below −100, and there every output is NaN. What is m after a block of scores near −200, if it starts at 0? `float32` rounds e^x to 0 below about x = −103.
</details>

## P12 Causal work

In a language model a token may only look at itself and the tokens before it: score (i, j) gets no weight when j > i. N = 4096, in blocks of 64 queries and 64 keys.

1. How many (query block, key block) pairs are there?
2. If every program stops at the diagonal, how many pairs does it compute? How many of those need a mask inside the block?

<details><summary>Answer</summary>

1. 64 × 64 = **4,096**.
2. Query block 0 needs 1 key block, block 1 needs 2, block 63 needs all 64: 1 + 2 + … + 64 = **2,080**, 50.8% of the work. Only the **64** blocks on the diagonal need a mask; every block below it is fully visible, and every block above it is never loaded.
</details>

## ex16 Causal attention

Open `ex16_causal.py` and start from your ex14 kernel. Keep a key when it is at or before the query (and before N), and end the loop at the diagonal instead of at N. Run `python check.py 16`.

<details><summary>Hint</summary>

The last key any query in this block may see is the block's last query, `(pid_m + 1) * BLOCK_M - 1`. The loop bound has to be a plain number, not a block, so use `(pid_m + 1) * BLOCK_M` rather than a `tl.minimum`. Keys past N still need the `rn < N` mask.
</details>

## ex17 Logsumexp (stretch)

The backward pass (Part 5) recomputes the probabilities instead of storing P. For that it needs one number per query: lse = log Σ e^score = m + log(l), so that a probability is e^(score − lse). Open `ex17_logsumexp.py`, start from ex14, and store lse as `float32` at `lse_ptr + z * stride_lz + rm`. Run `python check.py 17`.

On a GPU, `python bench.py` times the plain version (scores in memory), `torch.nn.functional.scaled_dot_product_attention` and your ex14 and ex16 kernels, 16 heads with D = 64 at N = 1024, 4096 and 16384, and appends to `bench.csv`. At N = 16384 the plain version needs 8.59 GB for S and 17.2 GB for S and P together.

## What check.py tests

- N that is not a multiple of the 64-key block (16, 100, 257, 1000), because a bug in the keys past the end hides whenever N is a multiple of 64.
- Sequences of one block (N = 16, 64), where the max never changes, and of up to 16 blocks.
- Several heads, and q, k, v that are slices of one packed (heads, N, 3·D) tensor, as in most models. Use the strides.
- A head where every score is below −100 (about −200), far below what e^x can represent in `float32`.
- `float16` inputs and output, compared with a `float64` reference at rtol 1e-2, atol 2e-3, because p is rounded to `float16` before `p @ v`, as in every fast attention kernel.
- Guards around every tensor, as in Parts 1–3.
