# Part 2: Rows and reductions

In Part 1 every output number needed one input number. Softmax and RMSNorm need a whole row before they can write anything: its max, its sum, its mean square. This part teaches one program per row, reductions (`tl.max`, `tl.sum`), and the two ways a reduction breaks: switched-off lanes that leak into it, and numbers that overflow.

The tasks come in the order the video gives them, and each one has its own chapter in the video. Pause when a task card appears. Paper tasks need no computer. Code tasks are finished when `python check.py N` prints PASS.

| Task | Kind | Done when |
| --- | --- | --- |
| P4 Rows | predict on paper, then invent one step | four answers |
| P5 Overflow | predict on paper, then invent one step | three answers |
| ex5 Softmax | fill in 4 TODO lines | `python check.py 5` → PASS |
| P6 Count the trips | predict on paper | two numbers, then `python bench.py` on a GPU |
| ex6 RMSNorm | write a kernel body | `python check.py 6` → PASS |
| ex7 Spot the bug | diagnose and fix three AI-written kernels | `python check.py 7` → PASS ×3 |
| ex8 Long rows (stretch) | write a kernel with a loop | `python check.py 8` → PASS |

## P4 Rows

A batch has 4096 rows of 1000 scores each (`float32`). One program handles one whole row, and holds it in one block of lanes.

1. How many programs does the grid need?
2. What is BLOCK? It must be a power of 2.
3. How many lanes in each program are switched off by the mask?
4. **Invent the step.** The kernel takes the max of the row. What should the switched-off lanes load, so that they never win the max, and add nothing once you take `exp` and sum?

<details><summary>Answer</summary>

1. **4096**, one per row: `grid = (M,)`.
2. **1024**, the next power of 2 after 1000 (`triton.next_power_of_2(1000)`).
3. **24**: lanes 1000 to 1023.
4. **Minus infinity**: `tl.load(..., mask=mask, other=-float("inf"))`. No number is smaller, so it never wins the max, and `exp(-inf) = 0`, so it adds nothing to the sum. For a plain sum the neutral value is 0. Each reduction has its own neutral value, and a masked lane has to load it.
</details>

## P5 Overflow

Softmax turns a row of scores into probabilities: `exp(x_i) / sum_j exp(x_j)`. The largest `float16` number is 65,504.

1. What is `exp(12)` in `float16`?
2. What is the softmax of `[10, 11, 12]` if you compute it exactly as written, in `float16`?
3. **Invent the step.** Which number can you subtract from every score without changing the softmax, so that no `exp` can overflow? Compute the softmax of `[10, 11, 12]` that way.

<details><summary>Answer</summary>

1. `exp(12) = 162,755`, which is more than 65,504, so it becomes **inf**. The largest safe input is `ln(65,504) = 11.09`.
2. `exp(10) = 22,026` and `exp(11) = 59,874` still fit, but the sum is inf. So the result is `[0, 0, NaN]`, because `22,026 / inf = 0` and `inf / inf = NaN`.
3. **The row's max.** Subtracting any constant `c` from every score multiplies the top and the bottom of the fraction by `exp(-c)`, so it cancels. With `c = max`, the largest input to `exp` is 0, and `exp(0) = 1`. `[10, 11, 12] - 12 = [-2, -1, 0]`, and `exp` of that is `[0.135, 0.368, 1]`, which sums to 1.503. The softmax is **`[0.090, 0.245, 0.665]`**.

`float32` has the same problem further out: its largest number is 3.4 × 10³⁸, so `exp(x)` overflows past `x = 88.7`. One score above 88.7 gives NaN in its own place (`inf / inf`) and 0 everywhere else in the row (`finite / inf`), even in `float32`, and the next layer spreads the NaN.
</details>

## ex5 Softmax

Open `ex5_softmax.py` and fill in TODO 1–4. Run `python check.py 5`.

## P6 Count the trips

Softmax on a 4096 × 4096 `float32` matrix, written as five PyTorch lines:

```python
x_max = x.max(dim=1)[0]
z = x - x_max[:, None]
num = torch.exp(z)
den = num.sum(dim=1)
out = num / den[:, None]
```

Each line is its own kernel. It reads its inputs from memory and writes its result back.

1. How many times does the whole matrix cross between memory and the chip? Count reads and writes; ignore the small per-row vectors. How many times for your one kernel?
2. What is the best possible speedup from fusing?

<details><summary>Answer</summary>

1. Reads: `max` 1, subtract 1, `exp` 1, `sum` 1, divide 1 = 5. Writes: subtract 1, `exp` 1, divide 1 = 3. That is **8 trips** of 67 MB each, 537 MB in total. Your kernel reads the matrix once and writes it once: **2 trips**, 134 MB.
2. At most **4×**, because both versions are limited by the memory bus. At 960 GB/s that is 0.56 ms versus 0.14 ms. The Triton softmax tutorial counts it the same way: 5MN + 2M reads and 3MN + 2M writes. `torch.softmax` is already one fused kernel, so the fair target is to match it. `bench.py` measures all three.
</details>

## ex6 RMSNorm

RMSNorm scales each row so its root-mean-square is 1, then multiplies by a learned weight:

```text
rms[i]    = sqrt(mean(x[i, :] ** 2) + eps)
out[i, j] = x[i, j] / rms[i] * w[j]
```

Llama and most LLMs since use it in place of LayerNorm. Before you write it, predict: `x = 300` in `float16`. What is `x * x`?

<details><summary>Answer to the prediction</summary>

`300 × 300 = 90,000`, which is more than 65,504, so it is **inf**. The mean of the row becomes inf, `rsqrt(inf) = 0`, and the whole row comes out as zeros. So load the row, convert it with `.to(tl.float32)`, and only then square it. Hugging Face's Llama RMSNorm casts its input to `float32` first for the same reason.
</details>

Open `ex6_rmsnorm.py`, write the kernel body, and run `python check.py 6`. The masked lanes should load 0 here, because 0 adds nothing to a sum of squares.

## ex7 Spot the bug

Here are two softmaxes and one RMSNorm, written the way AI assistants write them. They all run without an error, they pass some cases, and each one has one wrong line. Run `python check.py 7`. Look at which cases pass and which fail, and say what went wrong before you read the code. Then fix each version.

<details><summary>Hint for version A</summary>

It passes `1x1`, `3x1024` and `one logit = 100`, and fails everywhere else. Every value is a little too small (0.0001269 where 0.0001287 is expected). Which cases have switched-off lanes? What does a switched-off lane add to the sum? Why is that amount negligible when the max is 100?
</details>

<details><summary>Hint for version B</summary>

Only `one logit = 100` fails, with NaN at column 7. What is `exp(100)` in `float32`?
</details>

<details><summary>Hint for version C</summary>

Only the `float16` row with a 300 in it fails, and the output is 0. You predicted this one in ex6.
</details>

## ex8 Long rows (stretch)

A row of 100,000 numbers is too long to hold on the chip at once. Triton would accept a block of 131,072 (its limit is 2²⁰ elements), but the values would spill out of the registers into slow memory. `ex8_long_rows.py` fixes BLOCK at 1024. Each program walks along its row with `for start in tl.range(0, N, BLOCK)`, once to add up the squares and once to write the output. Part 3's matmul walks along its inner dimension the same way.

## What check.py tests

- Sizes that are not powers of 2 (1000, 33, 4097), because most masking bugs hide at 1024.
- Rows that sit further apart in memory than their width (a slice of a wider matrix). Use the strides.
- Values chosen to break a careless kernel: one logit of 100 (`exp(100)` overflows `float32`), and a `float16` row with a 300 in it (`300²` overflows `float16`).
- Guards around every tensor, as in Part 1. Output starts as −4096, so any box your kernel never wrote shows up as "never written", and a NaN it did write shows up as "got nan".
