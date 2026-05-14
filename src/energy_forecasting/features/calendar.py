"""Calendar features derived purely from the DatetimeIndex.

All features here are leakage-safe by construction: they require nothing but
the timestamp to be evaluable, and that timestamp is the prediction time.
"""

from __future__ import annotations

import pandas as pd

from energy_forecasting.exceptions import DataValidationError


def build_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame of calendar features aligned to ``df.index``.

    Produced columns: ``hour``, ``day_of_week``, ``day_of_month``,
    ``day_of_year``, ``month``, ``quarter``, ``is_weekend``.

    Args:
        df: Any DataFrame with a DatetimeIndex (values are ignored).

    Returns:
        DataFrame with the same index and the calendar columns, all int8.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise DataValidationError(
            f"Calendar features need DatetimeIndex, got {type(df.index).__name__}"
        )

    idx = df.index
    out = pd.DataFrame(
        {
            "hour": idx.hour.astype("int8"),
            "day_of_week": idx.dayofweek.astype("int8"),
            "day_of_month": idx.day.astype("int8"),
            "day_of_year": idx.dayofyear.astype("int16"),
            "month": idx.month.astype("int8"),
            "quarter": idx.quarter.astype("int8"),
            "is_weekend": (idx.dayofweek >= 5).astype("int8"),
        },
        index=idx,
    )
    return out
