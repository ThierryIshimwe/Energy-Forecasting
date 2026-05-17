"""Missing-value handling for the energy time series.

Policy (from EDA-validated decisions):

1. **Short gaps** — consecutive missing rows up to ``short_gap_max_rows`` — are
   interpolated using **past-only** information (``limit_direction="forward"``).
   These are random isolated sensor blips with no structural meaning; filling
   them avoids destroying long usable spans of data. *Forward-only* is
   essential: a two-sided interpolation would fill a NaN at time t using a
   future observation at t+k, which would leak future information into any
   downstream lag/rolling feature built from this series.

2. **Long gaps** — anything longer — are preserved as NaN and flagged with an
   explicit ``is_outage_gap`` indicator. These represent real acquisition
   outages (the EDA found gaps up to 7,226 consecutive minutes) and should
   be visible to downstream models, not silently imputed.

A second indicator, ``is_originally_missing``, records *any* row that had a
missing value in the raw input. Useful for reproducibility checks and to let
downstream models learn that "this row was imputed" is sometimes a signal.

This module does NOT delete rows. Deletion would break the temporal index.
Long gaps remain as NaN; downstream feature pipelines (``shift``, rolling
windows) propagate NaN safely.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from energy_forecasting.exceptions import DataValidationError
from energy_forecasting.utils import get_logger

_logger = get_logger(__name__)

InterpolationMethod = Literal["time", "linear"]


def _ensure_datetime_index(df: pd.DataFrame) -> None:
    """Raise if ``df`` does not have a sorted DatetimeIndex."""
    if not isinstance(df.index, pd.DatetimeIndex):
        raise DataValidationError(
            f"Expected DatetimeIndex for time-series preprocessing, got {type(df.index).__name__}"
        )
    if not df.index.is_monotonic_increasing:
        raise DataValidationError("DatetimeIndex must be sorted in ascending order")


def summarize_missing_gaps(df: pd.DataFrame) -> pd.DataFrame:
    """Return ``(start, end, length)`` for every consecutive-missing-rows segment.

    A "missing row" is one where at least one numeric column is NaN. Indicator
    columns added by this module are ignored.

    Returns:
        DataFrame with columns ``[start, end, length]``. Empty DataFrame if no
        gaps exist.
    """
    _ensure_datetime_index(df)

    # Only look at the original measurement columns — indicator columns are
    # always defined and would mask real gaps.
    measurement_cols = [c for c in df.columns if c not in ("is_outage_gap", "is_originally_missing")]
    if not measurement_cols:
        return pd.DataFrame(columns=["start", "end", "length"])

    missing_mask = df[measurement_cols].isna().any(axis=1)
    if not missing_mask.any():
        return pd.DataFrame(columns=["start", "end", "length"])

    # Group runs of True/False using a cumulative-sum trick.
    group_ids = (missing_mask != missing_mask.shift(fill_value=False)).cumsum()
    records: list[dict[str, object]] = []
    for _, segment in missing_mask.groupby(group_ids):
        if not bool(segment.iloc[0]):
            continue  # skip the False (present) segments
        records.append(
            {
                "start": segment.index[0],
                "end": segment.index[-1],
                "length": int(segment.sum()),
            }
        )
    return pd.DataFrame.from_records(records)


def apply_missing_value_policy(
    df: pd.DataFrame,
    *,
    short_gap_max_rows: int = 3,
    interpolation_method: InterpolationMethod = "time",
) -> pd.DataFrame:
    """Apply the two-stage missing-value policy and return the cleaned DataFrame.

    Args:
        df: Input frame with DatetimeIndex and numeric measurement columns.
        short_gap_max_rows: Gaps of this length or shorter are interpolated.
        interpolation_method: ``"time"`` (recommended for irregular indices) or
            ``"linear"`` (uniform-spaced).

    Returns:
        A new DataFrame with:

        - Short-gap rows interpolated in-place on the measurement columns.
        - Long-gap rows kept as NaN.
        - ``is_originally_missing`` (0/1) — any row missing in the raw input.
        - ``is_outage_gap`` (0/1) — any row inside a long (above-threshold) gap.

        The original DataFrame is not modified.

    Raises:
        ValueError: ``short_gap_max_rows < 1``.
        DataValidationError: Index is not a sorted DatetimeIndex.
    """
    _ensure_datetime_index(df)
    if short_gap_max_rows < 1:
        raise ValueError("short_gap_max_rows must be >= 1")

    measurement_cols = [c for c in df.columns if c not in ("is_outage_gap", "is_originally_missing")]
    if not measurement_cols:
        raise DataValidationError("No measurement columns to apply missing-value policy to")

    original_missing = df[measurement_cols].isna().any(axis=1)

    # Identify which rows belong to long gaps so we can re-mask them after interpolation.
    group_ids = (original_missing != original_missing.shift(fill_value=False)).cumsum()
    long_gap_mask = pd.Series(False, index=df.index)
    n_short_gaps = 0
    n_long_gaps = 0
    for _, segment in original_missing.groupby(group_ids):
        if not bool(segment.iloc[0]):
            continue
        length = int(segment.sum())
        if length > short_gap_max_rows:
            long_gap_mask.loc[segment.index] = True
            n_long_gaps += 1
        else:
            n_short_gaps += 1

    cleaned = df.copy()
    # PAST-ONLY FILL. Pandas' `interpolate(method="time")` ALWAYS blends both
    # surrounding observations for interior NaNs (`limit_direction` only
    # controls which side of a NaN-run gets filled at the edges) — that would
    # leak future information into y[t] and then into every lag/rolling
    # feature derived from this series. We use ffill() instead so each filled
    # cell is exactly equal to the last past observation, no future blending.
    cleaned[measurement_cols] = cleaned[measurement_cols].ffill(
        limit=short_gap_max_rows
    )
    # Re-mask the long gaps. ffill can spill across short boundaries
    # into long gaps; we explicitly null those out to preserve the policy.
    cleaned.loc[long_gap_mask, measurement_cols] = np.nan

    cleaned["is_originally_missing"] = original_missing.astype("int8")
    cleaned["is_outage_gap"] = long_gap_mask.astype("int8")

    total_missing = int(original_missing.sum())
    remaining_missing = int(cleaned[measurement_cols].isna().any(axis=1).sum())
    _logger.info(
        f"Missing-value policy applied: {total_missing:,} originally missing rows; "
        f"{n_short_gaps:,} short gaps interpolated ({total_missing - remaining_missing:,} rows filled), "
        f"{n_long_gaps:,} long gaps preserved ({remaining_missing:,} rows remain NaN)"
    )
    return cleaned
