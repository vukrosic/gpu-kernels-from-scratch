# Lets `python solutions/ex18_matmul_backward.py` run directly. See ../../kfs_backend.py.
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, "..", ".."))
sys.path.insert(0, os.path.join(_here, ".."))  # kernels.py
from kfs_backend import BACKEND, DEVICE  # noqa: E402,F401
