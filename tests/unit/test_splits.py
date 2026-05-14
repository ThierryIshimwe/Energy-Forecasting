"""Unit tests for the rolling-origin splitter."""

from __future__ import annotations

import pandas as pd
import pytest

from energy_forecasting.config import (
    DEFAULT_CONFIG_PATH,
    ForecastConfig,
    SplitsConfig,
)
from energy_forecasting.splits import (
    FoldSpec,
    make_rolling_origin_splits,
    splits_to_dataframe,
)


@pytest.fixture
def long_index() -> pd.DatetimeIndex:
    """Two years of hourly timestamps — plenty for any plausible split plan."""
    return pd.date_range("2008-01-01", "2009-12-31 23:00", freq="1h")


def test_returns_correct_number_of_folds(
    long_index: pd.DatetimeIndex,
    default_config: ForecastConfig,
) -> None:
    folds = make_rolling_origin_splits(long_index, config=default_config)
    assert len(folds) == default_config.splits.n_folds
    assert all(isinstance(f, FoldSpec) for f in folds)


def test_folds_are_chronologically_ordered(
    long_index: pd.DatetimeIndex,
    default_config: ForecastConfig,
) -> None:
    folds = make_rolling_origin_splits(long_index, config=default_config)
    for a, b in zip(folds, folds[1:]):
        assert a.valid_end < b.valid_end, "fold N+1 should validate later than fold N"


def test_last_fold_valid_ends_at_index_max(
    long_index: pd.DatetimeIndex,
    default_config: ForecastConfig,
) -> None:
    folds = make_rolling_origin_splits(long_index, config=default_config)
    assert folds[-1].valid_end == long_index.max()


def test_train_strictly_before_valid_within_each_fold(
    long_index: pd.DatetimeIndex,
    default_config: ForecastConfig,
) -> None:
    folds = make_rolling_origin_splits(long_index, config=default_config)
    for f in folds:
        assert f.train_end < f.valid_start, f"fold {f.fold_id}: train must end before valid starts"


def test_validation_windows_dont_overlap(
    long_index: pd.DatetimeIndex,
    default_config: ForecastConfig,
) -> None:
    folds = make_rolling_origin_splits(long_index, config=default_config)
    for a, b in zip(folds, folds[1:]):
        assert a.valid_end < b.valid_start, "validation windows must be disjoint"


def test_window_lengths_match_config(
    long_index: pd.DatetimeIndex,
    default_config: ForecastConfig,
) -> None:
    folds = make_rolling_origin_splits(long_index, config=default_config)
    train_days_expected = default_config.splits.train_window_days
    valid_days_expected = default_config.splits.valid_window_days
    for f in folds:
        train_days = (f.train_end - f.train_start).total_seconds() / 86400
        valid_days = (f.valid_end - f.valid_start).total_seconds() / 86400
        # Allow 1-hour rounding tolerance
        assert abs(train_days - train_days_expected) < 0.05
        assert abs(valid_days - valid_days_expected) < 0.05


def test_gap_days_inserts_buffer(long_index: pd.DatetimeIndex) -> None:
    """With gap_days=2, train_end should be 2 days before valid_start."""
    cfg = ForecastConfig.from_yaml(DEFAULT_CONFIG_PATH)
    splits_with_gap = SplitsConfig(
        n_folds=cfg.splits.n_folds,
        train_window_days=cfg.splits.train_window_days,
        valid_window_days=cfg.splits.valid_window_days,
        gap_days=2,
    )
    from dataclasses import replace
    cfg_gap = replace(cfg, splits=splits_with_gap)
    folds = make_rolling_origin_splits(long_index, config=cfg_gap)
    for f in folds:
        gap = (f.valid_start - f.train_end).total_seconds() / 86400
        # 1 hour for the inclusive-boundary + 2 days for the gap
        assert 1.95 < gap < 2.1, f"fold {f.fold_id} gap = {gap:.3f} days"


def test_index_too_short_raises() -> None:
    """A 10-day index can't support 6 folds × 30 valid days."""
    cfg = ForecastConfig.from_yaml(DEFAULT_CONFIG_PATH)
    tiny_index = pd.date_range("2008-01-01", "2008-01-10", freq="1h")
    with pytest.raises(ValueError, match="days but"):
        make_rolling_origin_splits(tiny_index, config=cfg)


def test_train_mask_and_valid_mask_are_disjoint(
    long_index: pd.DatetimeIndex,
    default_config: ForecastConfig,
) -> None:
    folds = make_rolling_origin_splits(long_index, config=default_config)
    for f in folds:
        train = f.train_mask(long_index)
        valid = f.valid_mask(long_index)
        assert not (train & valid).any(), f"fold {f.fold_id}: train and valid overlap"


def test_splits_to_dataframe_includes_row_counts_when_index_given(
    long_index: pd.DatetimeIndex,
    default_config: ForecastConfig,
) -> None:
    folds = make_rolling_origin_splits(long_index, config=default_config)
    df = splits_to_dataframe(folds, index=long_index)
    assert "train_rows" in df.columns
    assert "valid_rows" in df.columns
    assert len(df) == default_config.splits.n_folds
