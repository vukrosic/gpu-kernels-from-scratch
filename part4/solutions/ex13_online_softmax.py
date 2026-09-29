"""Solution to exercise 13."""
import numpy as np


def running_max_and_sum(blocks):
    m = -np.inf
    l = 0.0
    for b in blocks:
        m_new = max(m, b.max())
        l = l * np.exp(m - m_new) + np.exp(b - m_new).sum()  # every old term was measured against the old max
        m = m_new
    return m, l


def softmax(x, block=64):
    m, l = running_max_and_sum(x[i : i + block] for i in range(0, len(x), block))
    return np.exp(x - m) / l


if __name__ == "__main__":
    print(running_max_and_sum(iter([np.array([1.0, 3.0]), np.array([2.0, 5.0])])))  # (5.0, 1.2034...)
