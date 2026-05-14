"""Global RNG seeding for reproducibility.

Reproducibility requires every source of randomness to be deterministic.
Setting only ``numpy.random.seed`` is not enough — Python's ``random``,
the hash seed, PyTorch (when present), and ``PYTHONHASHSEED`` for subprocesses
all matter. This module sets them in one call so callers never miss one.

The function tolerates optional backends (PyTorch, TensorFlow): if the package
is not installed, that step is skipped silently. This keeps the dependency
on those frameworks lazy.
"""

from __future__ import annotations

import os
import random

import numpy as np

DEFAULT_SEED = 42


def set_global_seed(seed: int = DEFAULT_SEED, *, deterministic_torch: bool = True) -> None:
    """Set seeds for Python, NumPy, and (when available) PyTorch.

    Args:
        seed: Integer seed value. The same value should produce byte-identical
            artifacts on the same hardware.
        deterministic_torch: When True and PyTorch is installed, force
            deterministic CUDA kernels. Slower but guarantees reproducibility.

    Notes:
        ``PYTHONHASHSEED`` must be set before the interpreter starts to affect
        the current process. Setting it here only affects child processes.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch  # type: ignore[import-not-found]
    except ImportError:
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic_torch:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
