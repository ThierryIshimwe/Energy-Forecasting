"""Rolling-window statistics over past target values.

Each rolling feature is computed as ``series.shift(1).rolling(window=W).agg()``.
The ``shift(1)`` is the leakage guard: at timestamp t we use values strictly
before t, never t itself.

Naming convention: ``<target>_roll_<stat>_<W>h`` where ``stat`` is mean, std,
min, or max and ``W`` is the window length in hours.
"""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from energy_forecasting.exceptions import DataValidationError

# Which rolling statistics to emit per window.
_ROLLING_STATS: tuple[str, ...] = ("mean", "std", "min", "max")


def build_rolling_features(
    series: pd.Series,
    window_hours: Iterable[int],
    *,
    target_name: str | None = None,
    stats: tuple[str, ...] = _ROLLING_STATS,
) -> pd.DataFrame:
    """Return a DataFrame of rolling-window stats over past values of ``series``.

    Args:
        series: Target series at the working frequency.
        window_hours: Window lengths in periods (typically hours).
        target_name: Column prefix; defaults to ``series.name`` or ``"target"``.
        stats: Which statistics to compute. Default: mean, std, min, max.

    Returns:
        DataFrame indexed like ``series`` with one column per (window, stat).
    """
    windows = list(window_hours)
    if any(w < 2 for w in windows):
        raise DataValidationError(
            f"Rolling windows must be >= 2 (smaller windows are degenerate). Got: {windows}"
        )
    if not isinstance(series.index, pd.DatetimeIndex):
        raise DataValidationError(
            f"Rolling features need DatetimeIndex, got {type(series.index).__name__}"
        )
    unknown_stats = set(stats) - {"mean", "std", "min", "max", "median"}
    if unknown_stats:
        raise DataValidationError(f"Unknown rolling stats: {sorted(unknown_stats)}")

    name = target_name or series.name or "target"
    # The shift(1) guards against leakage: at time t the rolling window covers
    # [t-W, t-1], never including t itself.
    past = series.shift(1)

    out: dict[str, pd.Series] = {}
    for w in windows:
        rolled = past.rolling(window=w, min_periods=w)
        for stat in stats:
            out[f"{name}_roll_{stat}_{w}h"] = getattr(rolled, stat)()
    return pd.DataFrame(out, index=series.index)
