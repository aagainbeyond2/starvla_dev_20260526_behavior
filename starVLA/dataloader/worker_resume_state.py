"""Deterministic worker-side randomness for exact dataloader resume."""

import random
from contextlib import contextmanager

import numpy as np
import torch


@contextmanager
def deterministic_sample_rng(seed: int):
    """Temporarily seed Python, NumPy, and Torch CPU RNGs for one sample."""

    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()

    random.seed(seed)
    np.random.seed(seed & 0xFFFFFFFF)
    generator = torch.Generator()
    generator.manual_seed(seed & 0x7FFFFFFFFFFFFFFF)
    torch.set_rng_state(generator.get_state())
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
