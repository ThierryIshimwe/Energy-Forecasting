"""Frequency conversion (downsampling) for the energy time series.

The raw UCI data is at 1-minute resolution, which is finer than what most
forecasting models want or need. The configured pipeline target is hourly,
which:

- aligns with how electricity markets bid (1-hour blocks);
- attenuates minute-level noise without erasing daily seasonality;
- reduces the working dataset from ~2M to ~35K rows, making cross-validation
  and tuning tractable.

This module handles two column families with different aggregation semantics:

- **Measurement columns** (power, voltage, intensity, sub-metering) →
  aggregated via the user-configured method (typically ``mean``).
- **Indicator columns** (``is_outage_gap``, ``is_originally_missing``) →
  aggregated via ``max`` — an hour is flagged as outage if *any* minute
  inside it was an outage. This is the conservative choice.
"""

from __future__ import annotations

from typing import Literal

import pandas as pd

from energy_forecasting.exceptions import DataValidationError
from energy_forecasting.utils import get_logger

_logger = get_logger(__name__)

AggregationMethod = Literal["mean", "sum", "median", "first", "last"]

# Columns that are 0/1 indicators and must be aggregated with `max`, not mean.
_INDICATOR_COLUMNS: frozenset[str] = frozenset({"is_outage_gap", "is_originally_missing"})


def resample_to_frequency(
    df: pd.DataFrame,
    *,
    frequency: str,
    aggregation: AggregationMethod = "mean",
) -> pd.DataFrame:
    """Downsample ``df`` to the given frequency.

    Args:
        df: Input frame with DatetimeIndex. Numeric measurement columns are
            aggregated by ``aggregation``; indicator columns (recognized by
            name) are aggregated by ``max``.
        frequency: Pandas offset alias (``"1h"``, ``"15min"``, ``"1D"``, ...).
        aggregation: How to aggregate measurement columns.

    Returns:
        New DataFrame at the target frequency. Rows where no source rows
        existed are dropped (e.g., during a multi-hour outage at the very
        edge of the source data).

    Raises:
        DataValidationError: Index is not a DatetimeIndex, no numeric columns
            to aggregate, or unknown aggregation method.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise DataValidationError(
            f"resample_to_frequency requires DatetimeIndex, got {type(df.index).__name__}"
        )

    numeric_cols = [
        c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c not in _INDICATOR_COLUMNS
    ]
    indicator_cols = [c for c in df.columns if c in _INDICATOR_COLUMNS]

    if not numeric_cols and not indicator_cols:
        raise DataValidationError("No numeric or indicator columns available to resample")

    resampler = df.resample(frequency)

    # Pandas wants the agg method named identically. Validate up-front so the
    # error message points at the config field, not at a deep pandas trace.
    if aggregation not in {"mean", "sum", "median", "first", "last"}:
        raise DataValidationError(f"Unknown aggregation method: {aggregation!r}")

    parts: list[pd.DataFrame] = []
    if numeric_cols:
        parts.append(resampler[numeric_cols].agg(aggregation))
    if indicator_cols:
        parts.append(resampler[indicator_cols].max().astype("int8"))

    result = pd.concat(parts, axis=1) if len(parts) > 1 else parts[0]

    # Drop fully-empty rows that can appear at the edges if the source has
    # nothing inside a resample bucket. (`.dropna(how='all')` only on the
    # measurement columns — indicator columns may be 0 and shouldn't kill the row.)
    if numeric_cols:
        empty_mask = result[numeric_cols].isna().all(axis=1)
        if empty_mask.any():
            n_dropped = int(empty_mask.sum())
            _logger.info(f"Dropped {n_dropped:,} empty buckets at frequency {frequency!r}")
            result = result.loc[~empty_mask]

    _logger.info(
        f"Resampled to {frequency!r}: {len(df):,} rows -> {len(result):,} rows, "
        f"aggregation={aggregation!r}"
    )
    return result
