"""Lag features — the past values of the target.

Naming convention: ``<target>_lag_<N>h`` is the target value N hours before
the current timestamp. The implementation uses ``Series.shift(N)`` so the
first N rows of the resulting feature are NaN — that's correct and expected.
Downstream code drops NaN rows or masks them per-fold.

Leakage discussion: as long as N ≥ 1, the value is observed before the
prediction timestamp, so this is safe. Lag 0 would be the target itself and
is *never* emitted.
"""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from energy_forecasting.exceptions import DataValidationError


def build_lag_features(
    series: pd.Series,
    lag_hours: Iterable[int],
    *,
    target_name: str | None = None,
) -> pd.DataFrame:
    """Return a DataFrame of lagged versions of ``series``.

    Args:
        series: Target series with DatetimeIndex at the working frequency
            (typically hourly).
        lag_hours: Lag sizes to generate, in units of the series frequency.
            Must all be ≥ 1.
        target_name: Used as the column-name prefix. Defaults to ``series.name``
            or ``"target"`` if the series has no name.

    Returns:
        DataFrame indexed like ``series`` with one column per requested lag:
        ``"<target>_lag_<N>h"``.
    """
    lag_list = list(lag_hours)
    if any(n < 1 for n in lag_list):
        raise DataValidationError(
            f"Lag values must be >= 1 (lag 0 = target leakage). Got: {lag_list}"
        )
    if not isinstance(series.index, pd.DatetimeIndex):
        raise DataValidationError(
            f"Lag features need DatetimeIndex, got {type(series.index).__name__}"
        )

    name = target_name or series.name or "target"
    return pd.DataFrame(
        {f"{name}_lag_{n}h": series.shift(n) for n in lag_list},
        index=series.index,
    )
