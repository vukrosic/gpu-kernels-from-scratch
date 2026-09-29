# Real Triton on an NVIDIA GPU, the minitriton CPU stand-in anywhere else. See ../kfs_backend.py.
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from kfs_backend import BACKEND, DEVICE  # noqa: E402,F401
