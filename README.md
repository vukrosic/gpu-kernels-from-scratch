# GPU Kernels From Scratch

[Watch the full course on YouTube](https://youtu.be/1BUxizMA0po)

This is the code for the video course. You write GPU kernels in [Triton](https://github.com/triton-lang/triton), from your first vector add to FlashAttention. Each part has tasks, a checker that says PASS or FAIL, and a benchmark.

## Why learn this when AI can type a kernel

AI assistants can type a kernel in seconds. The algorithms that made AI fast came from people who understood the hardware down to the byte: FlashAttention (2022) and FlashAttention-3 (2024), and DeepSeek's FlashMLA and DeepGEMM (2025). Engineers at frontier labs still work these out themselves. When you understand where every byte goes, you can still find better algorithms than an AI will write for you. This course builds that understanding as four skills:

- **See it**: which program touches which numbers, how many bytes move, and what stays on the chip.
- **Predict it**: how many programs run, and the fastest time the hardware allows, before you run anything.
- **Prove it**: `check.py` on sizes and values chosen to break your kernel, with guards around memory.
- **Measure it**: GB/s or TFLOPS against the GPU's spec sheet, and then invent the next step (fuse, tile, go online).

So most tasks are *predict*, *spot the bug*, *measure* and *invent the step*, plus short pieces of real kernel code. The "AI wrote this" kernels are in the course because understanding is what finds their bugs.

## Setup

**Colab (free GPU):** open `colab.ipynb`, choose Runtime → Change runtime type → T4 GPU, and run the cells.

**Your own NVIDIA GPU (Linux or WSL):** run `pip install torch numpy`. Triton comes with PyTorch's CUDA builds.

**Laptop with no NVIDIA GPU (Mac, Windows):** run `pip install torch numpy`. `check.py` then uses `minitriton`, a small CPU stand-in that runs your real Triton code and checks it. Speed needs a GPU (use Colab for `bench.py`).

Then:

```bash
cd part1
python check.py 1     # FAIL until you fill in ex1_vector_add.py
```

## Parts

| Part | You build | Status |
| --- | --- | --- |
| 1 Your first kernel | vector add, spot three AI bugs, a fused kernel, row bias | ready |
| 2 Rows and reductions | softmax, RMSNorm, three AI bugs, long rows | ready |
| 3 Matmul | one output tile, the K loop with `tl.dot`, three AI bugs, grouped launch order and autotune | ready |
| 4 FlashAttention | online softmax, FlashAttention forward, three AI bugs, causal attention, logsumexp | ready |
| 5 Capstone | matmul, RMSNorm and attention backward passes, then a small GPT trained on your kernels next to PyTorch | ready |

## Layout

```text
kfs_backend.py   picks real Triton (GPU) or minitriton (CPU)
minitriton/      the CPU stand-in: runs Triton code with NumPy, GPU-like memory rules
part1/
  README.md      the tasks, with answers folded away
  ex*.py         your exercises
  check.py       PASS / FAIL with the first wrong index
  bench.py       GB/s or TFLOPS against your GPU's spec (GPU only), writes bench.csv
  solutions/     reference solutions
part2/ ...       the same layout for every part
kfs_check.py     the checker shared by Part 2 onward (guards, PASS/FAIL lines)
```

## Validation scope

The CPU stand-in is for learning and correctness checks, not GPU performance validation. GPU benchmark results are not included in this release; run the benchmarks on your own supported NVIDIA GPU. The Colab notebook introduces Parts 1–2; continue with each later part’s README for Parts 3–5.
