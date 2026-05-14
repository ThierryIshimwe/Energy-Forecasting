"""Naive baselines: y_t = y_{t-lag} read from a precomputed lag feature.

Three instances cover the canonical naïves:

- ``NaiveLagBaseline("Global_active_power_lag_1h")`` — persistence (yesterday-
  at-this-hour-ish, since we are at hourly resolution).
- ``NaiveLagBaseline("Global_active_power_lag_24h")`` — seasonal-naïve daily.
- ``NaiveLagBaseline("Global_active_power_lag_168h")`` — seasonal-naïve weekly.

These baselines are the floor every real model must beat. A model that doesn't
beat the appropriate naive isn't doing forecasting — it's modelling noise.

There is no ``fit`` step: the prediction comes directly from a feature column
that the feature pipeline already computed (with the correct ``shift`` to keep
it leakage-safe).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd

from energy_forecasting.exceptions import ModelError
from energy_forecasting.models.base import BaseForecaster

FloatArray = npt.NDArray[np.float64]


class NaiveLagBaseline(BaseForecaster):
    """Predict each ``y_t`` as the value of a precomputed lag feature column.

    The model does not store training data — the lag feature is already
    present in the feature matrix because the feature pipeline put it there.
    This is what makes the baseline trivially leakage-safe: the feature was
    built with the correct ``shift`` and is registered as safe.

    Args:
        lag_column: Name of the lag feature column in ``X`` (e.g.
            ``"Global_active_power_lag_24h"``).
        name: Optional override for the model name (used in logs and the
            leaderboard). Defaults to ``"naive_<lag_column>"``.
    """

    def __init__(self, lag_column: str, *, name: str | None = None) -> None:
        if not lag_column:
            raise ValueError("lag_column must be a non-empty string")
        self.lag_column: str = lag_column
        self.name: str = name or f"naive_{lag_column}"

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "NaiveLagBaseline":
        """No-op training. Verifies that the lag column is present in X."""
        if self.lag_column not in X.columns:
            raise ModelError(
                f"NaiveLagBaseline requires column {self.lag_column!r} in X. "
                f"Available columns include: {list(X.columns)[:8]}…"
            )
        return self

    def predict(self, X: pd.DataFrame) -> FloatArray:
        """Return ``X[lag_column].to_numpy(dtype=float64)``."""
        if self.lag_column not in X.columns:
            raise ModelError(
                f"NaiveLagBaseline.predict requires column {self.lag_column!r} in X."
            )
        return np.asarray(X[self.lag_column].values, dtype=np.float64)

    def get_hyperparameters(self) -> dict[str, object]:
        return {"lag_column": self.lag_column}
