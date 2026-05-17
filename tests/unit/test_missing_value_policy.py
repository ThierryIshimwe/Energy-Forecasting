"""Regression tests for the missing-value imputation policy.

The critical guarantee tested here: short-gap interpolation must use
**past-only** information (``limit_direction="forward"``). A bilateral or
backward interpolation would fill a NaN at time t with a value derived from
y[t+k], which leaks future information into any downstream lag/rolling
feature built from this series.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from energy_forecasting.preprocessing.missing import apply_missing_value_policy


class TestForwardOnlyInterpolation:
    """Imputation must not pull values from future timestamps."""

    def test_isolated_nan_does_not_use_future_value(self) -> None:
        """A single NaN at time t should NOT be filled with information from
        time t+1. Specifically: if past = 1.0 and future = 100.0, a leaky
        bilateral interpolation would yield ~50.5; a forward-only fill must
        give exactly 1.0 (the last observed value).
        """
        index = pd.date_range("2020-01-01", periods=5, freq="min")
        df = pd.DataFrame({"val": [1.0, np.nan, 100.0, 100.0, 100.0]}, index=index)
        cleaned = apply_missing_value_policy(df, short_gap_max_rows=3,
                                              interpolation_method="time")
        filled = cleaned["val"].iloc[1]
        assert filled == pytest.approx(1.0, abs=1e-9), (
            f"NaN was filled with {filled} — must equal the last past value (1.0). "
            f"Any other value (e.g. ~50.5 from bilateral interpolation) is a "
            f"future-information leak."
        )

    def test_multi_step_nan_run_uses_forward_only(self) -> None:
        """A 2-row NaN run must be carried forward from the past observation,
        not blended with the future observation that follows it.
        """
        index = pd.date_range("2020-01-01", periods=6, freq="min")
        df = pd.DataFrame({"val": [10.0, np.nan, np.nan, 1000.0, 1000.0, 1000.0]},
                          index=index)
        cleaned = apply_missing_value_policy(df, short_gap_max_rows=3,
                                              interpolation_method="time")
        # The two filled cells must be entirely derived from the past
        # observation (10.0). Forward-fill semantics make them both 10.0.
        # A bilateral interpolation would have given ~340 and ~670.
        for i in (1, 2):
            v = cleaned["val"].iloc[i]
            assert v == pytest.approx(10.0, abs=1e-9), (
                f"Cell {i} = {v} — must equal the last past value (10.0). "
                f"Non-trivial value indicates future leakage."
            )

    def test_lag_built_from_imputed_series_does_not_see_future(self) -> None:
        """End-to-end guarantee: a lag_1h feature built on top of the cleaned
        series must not encode information from y[t+k]. The way to check:
        when a NaN at t was imputed, the lag value at t+1 (which equals the
        imputed y[t]) must equal the past observation, not a future-aware
        blend.
        """
        index = pd.date_range("2020-01-01", periods=5, freq="h")
        df = pd.DataFrame({"y": [5.0, np.nan, 50.0, 50.0, 50.0]}, index=index)
        cleaned = apply_missing_value_policy(df, short_gap_max_rows=3,
                                              interpolation_method="time")
        lag = cleaned["y"].shift(1)
        # lag_1h at index 2 == cleaned y at index 1. With forward-only fill,
        # that's 5.0 (past). With bilateral, it would be ~27.5.
        assert lag.iloc[2] == pytest.approx(5.0, abs=1e-9), (
            f"Lag-1 feature at t=2 = {lag.iloc[2]} encodes future info; "
            f"forward-only fill should make it equal to the past (5.0)."
        )
