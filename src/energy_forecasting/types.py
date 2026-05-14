"""Shared type aliases and structural protocols.

Centralizing these gives the rest of the package a small, stable vocabulary
for talking about frequencies, metric functions, and the forecaster interface.
Models live in ``models/`` but every model must satisfy :class:`Forecaster`.
"""

from __future__ import annotations

from typing import Callable, Literal, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt
import pandas as pd

# ── Frequencies we natively support (pandas offset aliases) ──────────
# Raw data is at 1-minute resolution; production target is hourly.
# Daily is kept for low-frequency comparisons and slide-deck visuals.
Frequency = Literal["1min", "5min", "15min", "30min", "1h", "1D"]

# ── Aggregation strategies when downsampling ─────────────────────────
Aggregation = Literal["mean", "sum", "median", "first", "last"]

# ── Metric function shape ────────────────────────────────────────────
# A metric takes (y_true, y_pred) arrays of the same shape and returns
# a single scalar. All metrics in ``evaluation.metrics`` follow this.
FloatArray = npt.NDArray[np.float64]
MetricFn = Callable[[FloatArray, FloatArray], float]


@runtime_checkable
class Forecaster(Protocol):
    """The contract every forecasting model satisfies.

    Models are uniform in interface so the training loop, evaluator, and
    inference service can iterate over them without knowing their internals.
    The protocol is structural: implementations need not subclass anything,
    they just need methods with the right names and signatures.
    """

    name: str

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "Forecaster":
        """Fit on training data and return self for chaining."""

    def predict(self, X: pd.DataFrame) -> FloatArray:
        """Return point predictions aligned with ``X.index``."""

    def get_hyperparameters(self) -> dict[str, object]:
        """Return the hyperparameters used to train this instance."""
