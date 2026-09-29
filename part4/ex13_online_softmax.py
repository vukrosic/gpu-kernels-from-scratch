"""Exercise 13: online softmax, in NumPy.

softmax(x)[i] = e^(x[i] - m) / l,   with m = max(x) and l = sum of e^(x[j] - m).

Subtracting the max keeps every e^ at most 1 (e^1000 is inf in any float). The usual way
reads the row twice: once for m, once for l. Here the row arrives in blocks, one at a time,
and you may read each block once. You can't know the max before you have seen the last
block, so you keep a running max m and a running sum l, both measured against the max so far.

When a block brings a bigger max m_new, every term already in l was e^(x - m); it should be
e^(x - m_new). Every one of them is off by the same factor, e^(m - m_new), so one
multiplication fixes all of them at once.

Try it by hand first: the row [1, 3, 2, 5] in blocks of 2 (P11 in the README).

Fill in the two TODO lines. You are done when  python check.py 13  prints PASS.
"""
import numpy as np


def running_max_and_sum(blocks):
    """blocks: the pieces of one row, in order. Returns (m, l) = (max of the row, sum of e^(x - m))."""
    m = -np.inf  # a max of nothing: every real number is bigger
    l = 0.0
    for b in blocks:
        m_new = ...  # TODO 1: the max so far, including this block
        l = ...  # TODO 2: rescale the old sum to m_new, then add this block's terms
        m = m_new
    return m, l


def softmax(x, block=64):
    m, l = running_max_and_sum(x[i : i + block] for i in range(0, len(x), block))
    return np.exp(x - m) / l


if __name__ == "__main__":
    print(running_max_and_sum(iter([np.array([1.0, 3.0]), np.array([2.0, 5.0])])))  # should be (5.0, 1.2034...)
