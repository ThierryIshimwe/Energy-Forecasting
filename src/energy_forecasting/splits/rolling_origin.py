"""Rolling-origin (sliding-window) cross-validation for time series.

Scheme: at each fold, the *most recent* ``valid_window_days`` of data is the
validation set, immediately preceded by ``train_window_days`` of training
data. Across folds, this entire window slides backward by ``valid_window_days``
each step. The most recent fold ends at the last available timestamp.

Properties:

- **No leakage.** Every fold's training data is strictly older than its
  validation data; an optional ``gap_days`` parameter inserts a buffer to
  prevent leakage via lag/rolling features that span the boundary.
- **Equal-size validation windows.** All folds have the same number of
  validation samples, giving comparable per-fold metrics.
- **Sliding (not expanding) training window.** Training size is held constant
  so model capacity is challenged equally across folds; this is the appropriate
  setting when downstream production retraining is windowed (e.g., "retrain on
  last 12 months").

If your dataset does not span enough days to fit ``n_folds`` folds, a
``ValueError`` is raised immediately — no silent truncation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import pandas as pd

if TYPE_CHECKING:
    from energy_forecasting.config import ForecastConfig


@dataclass(frozen=True, slots=True)
class FoldSpec:
    """Metadata describing one fold of the rolling-origin split.

    Attributes:
        fold_id: 1-indexed fold number; fold 1 is the *earliest* in time,
            fold N is the most recent.
        train_start: First timestamp (inclusive) of the training window.
        train_end:   Last timestamp (inclusive) of the training window.
        valid_start: First timestamp (inclusive) of the validation window.
        valid_end:   Last timestamp (inclusive) of the validation window.
    """

    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    valid_start: pd.Timestamp
    valid_end: pd.Timestamp

    def train_mask(self, index: pd.DatetimeIndex) -> npt.NDArray[np.bool_]:
        """Boolean mask selecting training rows from ``index``."""
        # DatetimeIndex comparisons return ndarray directly.
        return np.asarray((index >= self.train_start) & (index <= self.train_end))

    def valid_mask(self, index: pd.DatetimeIndex) -> npt.NDArray[np.bool_]:
        """Boolean mask selecting validation rows from ``index``."""
        return np.asarray((index >= self.valid_start) & (index <= self.valid_end))


def make_rolling_origin_splits(
    index: pd.DatetimeIndex,
    *,
    config: "ForecastConfig",
) -> list[FoldSpec]:
    """Generate a rolling-origin split plan that fits within ``index``.

    Args:
        index: DatetimeIndex of the feature matrix the model will train on.
            Must be sorted and have at least
            ``train_window_days + n_folds * valid_window_days + gap_days``
            calendar days of span.
        config: ForecastConfig — ``config.splits`` provides the parameters.

    Returns:
        ``list[FoldSpec]`` of length ``config.splits.n_folds``, ordered fold 1
        (earliest in time) → fold N (most recent).

    Raises:
        ValueError: ``index`` is empty, unsorted, or shorter than the required
            span for the configured number of folds.
    """
    if not isinstance(index, pd.DatetimeIndex):
        raise ValueError(f"index must be DatetimeIndex, got {type(index).__name__}")
    if len(index) == 0:
        raise ValueError("index must not be empty")
    if not index.is_monotonic_increasing:
        raise ValueError("index must be sorted ascending")

    s = config.splits
    end_ts = index.max()
    start_ts = index.min()

    required_days = s.train_window_days + s.n_folds * s.valid_window_days + s.gap_days
    available_days = (end_ts - start_ts).total_seconds() / 86400
    if available_days < required_days:
        raise ValueError(
            f"Index spans {available_days:.1f} days but {required_days} are required for "
            f"{s.n_folds} folds (train={s.train_window_days}d, "
            f"valid={s.valid_window_days}d, gap={s.gap_days}d)."
        )

    # Boundaries are inclusive at hourly resolution.
    one_hour = pd.Timedelta(hours=1)

    folds: list[FoldSpec] = []
    for fold_id in range(1, s.n_folds + 1):
        # Number of valid_window_days steps from the most-recent fold (which has folds_back=0).
        folds_back = s.n_folds - fold_id

        valid_end = end_ts - pd.Timedelta(days=folds_back * s.valid_window_days)
        valid_start = valid_end - pd.Timedelta(days=s.valid_window_days) + one_hour
        train_end = valid_start - one_hour - pd.Timedelta(days=s.gap_days)
        train_start = train_end - pd.Timedelta(days=s.train_window_days) + one_hour

        folds.append(
            FoldSpec(
                fold_id=fold_id,
                train_start=train_start,
                train_end=train_end,
                valid_start=valid_start,
                valid_end=valid_end,
            )
        )
    return folds


def splits_to_dataframe(folds: list[FoldSpec], index: pd.DatetimeIndex | None = None) -> pd.DataFrame:
    """Render a list of folds as a tabular summary.

    Args:
        folds: As returned by :func:`make_rolling_origin_splits`.
        index: Optional — if given, include actual row counts per fold's
            train and validation mask (catches issues where a fold's window
            is partly outside the index).

    Returns:
        DataFrame with one row per fold and columns suitable for display.
    """
    rows = []
    for f in folds:
        row = {
            "fold_id": f.fold_id,
            "train_start": f.train_start,
            "train_end": f.train_end,
            "valid_start": f.valid_start,
            "valid_end": f.valid_end,
            "train_days": round((f.train_end - f.train_start).total_seconds() / 86400, 1),
            "valid_days": round((f.valid_end - f.valid_start).total_seconds() / 86400, 1),
        }
        if index is not None:
            row["train_rows"] = int(f.train_mask(index).sum())
            row["valid_rows"] = int(f.valid_mask(index).sum())
        rows.append(row)
    return pd.DataFrame(rows)
