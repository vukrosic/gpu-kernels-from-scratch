# Part 3: Matmul

Most of a language model's arithmetic is matrix multiplication. It is also the first kernel in this course that can keep the tensor cores busy instead of waiting on memory, but only if every number loaded from memory gets used many times. This part builds that kernel: one output tile, then the loop along K, then the bugs that hide in it, then choosing the tile size.

The tasks come in the order the video gives them, and each one has its own chapter in the video. Pause when a task card appears. Paper tasks need no computer. Code tasks are finished when `python check.py N` prints PASS.

| Task | Kind | Done when |
| --- | --- | --- |
| P7 Count the work | predict on paper | three numbers |
| P8 Reuse | predict on paper, then invent the tile | two numbers and one idea |
| ex9 One output tile | fill in 5 lines (TODO 1–3) | `python check.py 9` → PASS |
| ex10 The K loop | write a kernel body | `python check.py 10` → PASS |
| ex11 Spot the bug | diagnose and fix three AI-written kernels | `python check.py 11` → PASS ×3 |
| P9 Tile size | predict on paper | three numbers, then `python bench.py` on a GPU |
| ex12 Launch order and autotune (stretch) | fill in 2 TODOs | `python check.py 12` → PASS for every config |

Numbers below are for an RTX 5080: 112.6 TFLOPS of dense `float16` tensor-core math with `float32` accumulate, 960 GB/s of memory bandwidth, 84 SMs, 256 KB of registers and 64 MB of L2 cache (NVIDIA's RTX Blackwell architecture whitepaper).

## P7 Count the work

`C = A @ B` with A, B and C all 4096 × 4096 in `float16` (2 bytes a number). Each number of C is a row of A times a column of B: K multiplies and K adds.

1. How many floating point operations (FLOPs) is the whole matmul?
2. How long do they take at 112.6 TFLOPS?
3. If A and B each cross the memory bus once and C crosses once, how many bytes is that, and how long does it take at 960 GB/s? Which one is the limit?

<details><summary>Answer</summary>

1. 4096³ = 68.7 billion multiply-adds, so **137.4 billion FLOPs** (2 · 4096³).
2. 137.4 × 10⁹ / 112.6 × 10¹² = **1.22 ms**.
3. 3 × 4096² × 2 bytes = **100.7 MB**, which takes **0.105 ms**. The arithmetic takes about 12 times longer than the moving, so the tensor cores are the limit. In Part 1 it was the other way round: vector add spent 350 times longer moving than adding.
</details>

## P8 Reuse

The simplest matmul kernel gives each program one number of C. It loads a row of A and a column of B and adds up their products.

1. How many bytes does it load from memory in total? How long does that take at 960 GB/s?
2. **Invent the fix.** Look at one number of A. How many outputs need it? How could one program use every number it loads many times?

<details><summary>Answer</summary>

1. Each output loads 2 × 4096 numbers = 16 KB. There are 4096² = 16.8 million outputs, so **274.9 GB**, which takes **286 ms**: about 235 times longer than the 1.22 ms of arithmetic. (Caches would catch some of this on a real GPU; the point is how far the simple kernel sits from the limit.)
2. Every output in its row of C needs it: **4096**. Give each program a **tile** of C, say 128 × 128. It loads 128 rows of A and 128 columns of B, a strip along K at a time, and uses every number it loads 128 times. Traffic drops 128×: 1024 tiles × 2 MB = **2.15 GB**, which takes 2.24 ms.

**Arithmetic intensity** is FLOPs per byte loaded. For a BM × BN tile it is BM · BN / (BM + BN) FLOPs per byte (with 2-byte numbers):

| Kernel | FLOPs per byte |
| --- | --- |
| one output per program | 0.5 |
| 64 × 64 tile | 32 |
| 128 × 128 tile | 64 |
| 128 × 256 tile | 85 |
| every matrix read once (4096³) | 1365 |

The GPU's ridge point is 112.6 TFLOPS / 960 GB/s = **117 FLOPs per byte**. Below it the memory bus is the limit, above it the tensor cores. A 128 × 128 tile is still below the ridge if every load came from memory. The rest comes from the 64 MB L2 cache: programs running at the same time that load the same strips mostly hit the cache, and ex12 arranges the launch order so they do.
</details>

## ex9 One output tile

Open `ex9_one_tile.py`. Each program computes one tile of C, so the grid is 2D. K is at most 128 here, so one block holds all of it. Fill in TODO 1–3 and run `python check.py 9`.

A 2D block of addresses is a column of offsets plus a row of offsets: `rm[:, None] * stride_am + rk[None, :] * stride_ak` has shape (BLOCK_M, BLOCK_K). Use the strides; never assume a row of A is K numbers long.

<details><summary>Hint</summary>

The switched-off lanes must load `other=0.0`. A dot product is a sum of products, and 0 is the neutral value for a sum (Part 2). A masked lane loaded without `other=` holds garbage on a GPU (NaN in minitriton), and one garbage number in K spreads into every output of its row or column.
</details>

## ex10 The K loop

Open `ex10_matmul.py` and write the kernel body. The tile walks along K in steps of BLOCK_K, the way ex8 walked along a long row: load a strip of A and a strip of B, `acc += tl.dot(a, b)`, move both blocks of addresses BLOCK_K along K, repeat. Keep `acc` in `float32` (a `float16` sum keeps only about 3 significant digits) and store `acc.to(tl.float16)` once after the loop. Run `python check.py 10`.

## ex11 Spot the bug

Three matmuls written the way AI assistants write them. They all run without an error, pass some cases, and each has one wrong line. Run `python check.py 11`. Look at which cases pass and which fail, and say what went wrong before you read the code. Then fix each version.

<details><summary>Hint for version A</summary>

It passes exactly the cases where K is 64, 128 or 512, and gets every output wrong by thousands when K is 16, 70, 257 or 1000. What do 64, 128 and 512 have in common, given BLOCK_K = 32? What does the last strip along K read when K is 70?
</details>

<details><summary>Hint for version B</summary>

It passes only `(16x16) @ (16x16)` and `B = W.t()`. How many steps along K does the loop take when K = 16? When B = W.t(), how far apart in memory are B[k, n] and B[k + 1, n]?
</details>

<details><summary>Hint for version C</summary>

It passes every case except the two whose matrices are not stored row after row: a slice of a wider matrix, and W.t(). `nn.Linear` computes `x @ W.t()`, so this kernel would fail inside a real model.
</details>

## P9 Tile size

The C tile lives in registers, in `float32`. Each SM has 256 KB of registers (65,536 of 4 bytes).

1. How much of them does a 128 × 128 tile take?
2. A 256 × 256 tile?
3. C is 512 × 512 and the tiles are 256 × 256. How many programs are there, and how many of the 84 SMs get work?

<details><summary>Answer</summary>

1. 128² × 4 bytes = **64 KB**, a quarter.
2. 256² × 4 bytes = **256 KB**, all of them, with nothing left for the strips of A and B, the addresses or a second program. Values spill out to memory and the kernel slows down.
3. **4 programs**: 4 SMs work and 80 sit idle.

So the best tile depends on the GPU and the shapes. Reason your way close, then measure: `@triton.autotune` times each config the first time it sees a new (M, N, K) and keeps the fastest. On a GPU, `python bench.py` prints TFLOPS for `torch.matmul` (cuBLAS), your ex10 and your ex12, next to the spec-sheet peak.
</details>

## ex12 Launch order and autotune (stretch)

Two changes turn ex10 into a kernel that can get close to cuBLAS. The launch order: walk GROUP_M rows of tiles column by column, so the programs running at the same time share rows of A and columns of B in the L2 cache. And `@triton.autotune`, which picks the tile size by timing. `ex12_grouped_autotune.py` explains both; fill in the two TODOs. `python check.py 12` runs every config one at a time. On a GPU, a config too big for its on-chip memory prints SKIP, the same way autotune skips it.

## What check.py tests

- Sizes that are not multiples of the tile or of BLOCK_K (100 × 70 @ 70 × 90, K = 257, K = 1000), because a K-loop bug hides whenever K is a multiple of 32.
- A vector times a matrix (M = 1).
- An A that is a slice of a wider matrix (rows further apart than K), and B = W.t(), stored column after column, the way `nn.Linear` passes it. Use the strides.
- `float16` inputs and output, compared with a `float64` reference. B is scaled by 1/√K so every output is about the size of 1.
- Guards around every tensor, as in Parts 1 and 2. A K loop that runs past the end of B reads the guard values, and its outputs come out wrong by thousands.
