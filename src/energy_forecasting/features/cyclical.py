"""Cyclical (sin/cos) encodings for periodic calendar fields.

Hour-of-day and day-of-week are circular: hour 23 and hour 0 are adjacent in
time but maximally distant as integers. Encoding them as ``sin(2π·x/period)``
and ``cos(2π·x/period)`` gives the model two continuous features whose
Euclidean distance reflects true temporal proximity.

These features are leakage-safe (derived purely from the timestamp).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from energy_forecasting.exceptions import DataValidationError


def build_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return ``hour_sin/cos``, ``day_of_week_sin/cos``, ``month_sin/cos``.

    Args:
        df: Any DataFrame with a DatetimeIndex (values are ignored).

    Returns:
        DataFrame with the same index and six float32 columns.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise DataValidationError(
            f"Cyclical features need DatetimeIndex, got {type(df.index).__name__}"
        )

    idx = df.index
    hour = idx.hour.to_numpy()
    dow = idx.dayofweek.to_numpy()
    month = idx.month.to_numpy()

    out = pd.DataFrame(
        {
            "hour_sin": np.sin(2 * np.pi * hour / 24.0).astype("float32"),
            "hour_cos": np.cos(2 * np.pi * hour / 24.0).astype("float32"),
            "day_of_week_sin": np.sin(2 * np.pi * dow / 7.0).astype("float32"),
            "day_of_week_cos": np.cos(2 * np.pi * dow / 7.0).astype("float32"),
            # Month uses (month - 1) so January=0 and December=11 are adjacent.
            "month_sin": np.sin(2 * np.pi * (month - 1) / 12.0).astype("float32"),
            "month_cos": np.cos(2 * np.pi * (month - 1) / 12.0).astype("float32"),
        },
        index=idx,
    )
    return out
