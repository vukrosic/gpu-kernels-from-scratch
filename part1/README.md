# Part 1: Your first kernel

The tasks come in the order the video gives them, and each one has its own chapter in the video. Pause when a task card appears. Paper tasks need no computer. Code tasks are finished when `python check.py N` prints PASS.

| Task | Kind | Done when |
| --- | --- | --- |
| P1 Adding vs moving | predict on paper | you have two numbers, then check them below |
| P2 Crates | predict on paper | three numbers |
| ex1 Vector add | fill in 3 TODO lines | `python check.py 1` → PASS |
| ex2 Spot the bug | diagnose and fix three buggy kernels | `python check.py 2` → PASS ×3 |
| P3 Fusion | predict on paper | two numbers |
| ex3 Fused `relu(x*scale+shift)` | write a kernel body | `python check.py 3` → PASS, then `python bench.py` on a GPU |
| ex4 Row bias (stretch) | write the kernel and the launch | `python check.py 4` → PASS |

## P1 Adding vs moving

The GPU is an RTX 5080: 10,752 cores at 2.62 GHz and 960 GB/s of memory bandwidth. You add two lists of 100 million `float32` numbers into a third list.

1. Every core does one add per clock tick. How long do the 100 million adds take?
2. How many bytes travel between memory and the chip? How long does that take at 960 GB/s?

<details><summary>Answer</summary>

1. 10,752 × 2.62 billion = 28.2 trillion adds per second, so 100 million adds take **3.5 µs**.
2. You read two lists and write one. Each list is 100 million × 4 bytes = 400 MB, so the total is **1.2 GB**. At 960 GB/s that takes **1.25 ms**.

Moving takes about 350 times longer than adding. For a kernel like this, speed means moving fewer bytes, and the adds are nearly free.
</details>

## P2 Crates

There are n = 1000 numbers and BLOCK = 256. Each program handles one crate of 256 boxes.

1. How many programs does the grid need?
2. Which box numbers does the last program cover?
3. How many of those boxes exist?

<details><summary>Answer</summary>

1. **4**. 1000 / 256 = 3.9, rounded **up** (`triton.cdiv(1000, 256)`).
2. **768 to 1023**: `3 * 256 + tl.arange(0, 256)`.
3. **232**: boxes 768 to 999. The mask `offsets < n` switches off the other 24.
</details>

## ex1 Vector add

Open `ex1_vector_add.py` and fill in TODO 1–3. Run `python check.py 1`.

## ex2 Spot the bug

Here are three versions of the add, each with the kind of mistake AI assistants make. They all run without an error, and each has one wrong line. Run `python check.py 2`. From the FAIL lines alone, say what went wrong before you read the code. Then fix each version.

<details><summary>Hint for version A</summary>

The first box that was never written is 768 when n = 1000. What is 768 in crates of 256? Why does n = 1024 pass?
</details>

<details><summary>Hint for version B</summary>

The written boxes stop at 258 when n = 1000. Four programs, each covering 256 boxes, reach only box 258. So where does each program start?
</details>

<details><summary>Hint for version C</summary>

Every value is correct, yet 24 writes landed after the end of `out` (1000..1023). Which of the three memory operations has no mask? On a GPU nothing stops this. You overwrite whatever memory sits after your tensor.
</details>

## P3 Fusion

The line is `out = torch.relu(x * scale + shift)` on 100 million `float32` numbers. PyTorch runs it as three kernels, and each one reads its input from memory and writes its result back.

1. How many bytes do the three kernels move in total? How many does one fused kernel move?
2. What is the best possible speedup from fusing?

<details><summary>Answer</summary>

1. Three kernels × (read 400 MB + write 400 MB) = **2.4 GB**. The fused kernel reads x once and writes out once: **0.8 GB**.
2. At most **3×**, because both versions are limited by the memory bus. At 960 GB/s that is 2.5 ms versus 0.83 ms. `bench.py` measures how close you get. `torch.compile` fuses this line automatically, and it does so by generating a Triton kernel like yours.
</details>

## ex3 Fused kernel

Open `ex3_fused.py` and write the kernel body. Run `python check.py 3`. On a GPU, also run `python bench.py`.

## ex4 Row bias (stretch)

Open `ex4_row_bias.py`. The new idea is strides. A matrix row starts at `row * x.stride(0)`, which is not always `row * N`.

## What check.py does

- It tests sizes that are **not** multiples of the block (1, 1000, 4097, 100003), because most indexing bugs hide at 1024.
- Every tensor sits between guard values, so writes past the end are caught. A real GPU does not catch them.
- Output starts as NaN, so a box your kernel never wrote shows up as "never written".
- On a laptop it runs `minitriton`, a CPU stand-in that runs your real Triton code program by program, in a shuffled order. It checks correctness, not speed. On an NVIDIA GPU it runs real Triton.
