"""Temporal cross-validation: rolling-origin splits.

The only validation scheme in this project — chosen because it is the only
honest evaluation for time-series forecasting (no information from the future
leaks into a fold's training data).

Public API:
    FoldSpec                       — typed metadata for one fold
    make_rolling_origin_splits     — generate the N-fold split plan from a config
    splits_to_dataframe            — render a list of folds as a display table
"""

from energy_forecasting.splits.rolling_origin import (
    FoldSpec,
    make_rolling_origin_splits,
    splits_to_dataframe,
)

__all__ = ["FoldSpec", "make_rolling_origin_splits", "splits_to_dataframe"]
