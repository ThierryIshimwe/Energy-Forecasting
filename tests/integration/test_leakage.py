"""The critical leakage-prevention test suite.

These tests codify the project's core safety guarantee: when
``config.features.strict_leakage_safe`` is True, the feature pipeline cannot
emit features that would cause target leakage.

The columns ``Global_intensity``, ``Voltage``, and ``Sub_metering_1/2/3`` are
all measured at the same timestamp as the target ``Global_active_power``. They
are physical components of the target (``P = V × I`` from Ohm's law; sub-meters
sum into the total). Any model trained with them as features reconstructs the
target from its own components rather than forecasting it — the suite below
makes that mistake impossible to reintroduce silently.

All tests in this file carry the ``leakage`` marker so they can be run
exclusively via ``make leakage-check`` or ``pytest -m leakage``.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import pandas as pd

from energy_forecasting.config import ForecastConfig
from energy_forecasting.exceptions import LeakageError
from energy_forecasting.features import (
    FEATURE_REGISTRY,
    build_features,
    is_safe,
)

pytestmark = pytest.mark.leakage


# ─────────────────────────────────────────────────────────────────────
#  Registry-level guarantees
# ─────────────────────────────────────────────────────────────────────

class TestRegistry:
    """The registry must classify every same-timestamp component of the target as unsafe."""

    UNSAFE_RAW_COLUMNS = (
        "Global_intensity",
        "Voltage",
        "Global_reactive_power",
        "Sub_metering_1",
        "Sub_metering_2",
        "Sub_metering_3",
    )

    @pytest.mark.parametrize("col", UNSAFE_RAW_COLUMNS)
    def test_contemporaneous_raw_columns_marked_unsafe(self, col: str) -> None:
        """Every contemporaneous raw measurement must be flagged unsafe."""
        assert col in FEATURE_REGISTRY, f"{col!r} should be registered"
        assert FEATURE_REGISTRY[col].leakage_safe is False, (
            f"{col!r} is the original leakage culprit — must remain leakage_safe=False"
        )

    def test_lag_features_marked_safe(self) -> None:
        """The lag-feature family must be registered as safe."""
        assert is_safe("Global_active_power_lag_1h") is True
        assert is_safe("Global_active_power_lag_24h") is True
        assert is_safe("Global_active_power_lag_999h") is True  # wildcard match

    def test_calendar_features_marked_safe(self) -> None:
        for col in ("hour", "day_of_week", "month", "is_weekend"):
            assert is_safe(col) is True, f"{col!r} should be safe"

    def test_rolling_features_marked_safe(self) -> None:
        assert is_safe("Global_active_power_roll_mean_24h") is True
        assert is_safe("Global_active_power_roll_std_168h") is True

    def test_unregistered_column_returns_none(self) -> None:
        """Unknown columns return None (caller treats as unsafe in strict mode)."""
        assert is_safe("nonexistent_feature_xyz") is None


# ─────────────────────────────────────────────────────────────────────
#  Pipeline behavior in strict mode
# ─────────────────────────────────────────────────────────────────────

class TestStrictMode:
    """In strict mode the pipeline silently drops unsafe inputs."""

    def test_pipeline_drops_all_unsafe_input_columns(
        self,
        synthetic_hourly_df: pd.DataFrame,
        default_config: ForecastConfig,
    ) -> None:
        """The 6 contemporaneous columns must not appear in the output X."""
        fm = build_features(synthetic_hourly_df, config=default_config)
        for col in TestRegistry.UNSAFE_RAW_COLUMNS:
            assert col not in fm.X.columns, (
                f"{col!r} leaked into the feature matrix — strict mode failed"
            )
        assert fm.n_unsafe_dropped == len(TestRegistry.UNSAFE_RAW_COLUMNS)

    def test_pipeline_output_contains_only_safe_columns(
        self,
        synthetic_hourly_df: pd.DataFrame,
        default_config: ForecastConfig,
    ) -> None:
        """Every column in the output X must be is_safe()=True."""
        fm = build_features(synthetic_hourly_df, config=default_config)
        for col in fm.X.columns:
            assert is_safe(col) is True, (
                f"{col!r} in output is not registered as safe — leakage policy violated"
            )

    def test_pipeline_includes_expected_safe_features(
        self,
        synthetic_hourly_df: pd.DataFrame,
        default_config: ForecastConfig,
    ) -> None:
        """A sample of expected safe features must be present."""
        fm = build_features(synthetic_hourly_df, config=default_config)
        expected = {
            "hour",
            "day_of_week",
            "is_weekend",
            "hour_sin",
            "Global_active_power_lag_1h",
            "Global_active_power_lag_24h",
            "Global_active_power_roll_mean_24h",
            "is_french_holiday",
        }
        missing = expected - set(fm.X.columns)
        assert not missing, f"Expected features missing: {sorted(missing)}"


# ─────────────────────────────────────────────────────────────────────
#  Pipeline behavior in non-strict mode
# ─────────────────────────────────────────────────────────────────────

class TestNonStrictMode:
    """Non-strict mode is the explicit opt-out — used for ablations only."""

    def test_non_strict_still_drops_unsafe_inputs(
        self,
        synthetic_hourly_df: pd.DataFrame,
        default_config: ForecastConfig,
    ) -> None:
        """Even in non-strict mode, unsafe input columns are dropped by default.

        Non-strict mode only affects what happens when *engineered* features
        produce something unsafe — it does NOT re-enable the original leakage.
        """
        non_strict = replace(
            default_config,
            features=replace(default_config.features, strict_leakage_safe=False),
        )
        fm = build_features(synthetic_hourly_df, config=non_strict)
        for col in TestRegistry.UNSAFE_RAW_COLUMNS:
            assert col not in fm.X.columns


# ─────────────────────────────────────────────────────────────────────
#  Sanity guard: real-data MAE plausibility
# ─────────────────────────────────────────────────────────────────────

class TestRealDataPlausibility:
    """Tests using the real UCI dataset. Skip if data not downloaded."""

    def test_lag1_baseline_mae_is_not_suspiciously_low(
        self,
        real_raw_path: "Path",
        default_config: ForecastConfig,
    ) -> None:
        """A lag-1 naive baseline on the safe feature matrix should have MAE
        within a plausible range for this dataset.

        A leakage-free model on hourly residential consumption should sit in
        the 0.1 – 1.0 kW range. A value far below 0.05 indicates that a
        same-timestamp component of the target has crept back into the matrix.
        """
        from energy_forecasting.data import load_raw
        from energy_forecasting.preprocessing import preprocess

        df = load_raw(real_raw_path)
        processed, _ = preprocess(df, config=default_config)
        fm = build_features(processed, config=default_config)

        # Lag-1 naive: predict y[t] = y[t-1]. Use the lag_1h feature column directly.
        lag_col = f"{default_config.target.column}_lag_1h"
        assert lag_col in fm.X.columns

        y_true = fm.y.to_numpy()
        y_pred = fm.X[lag_col].to_numpy()
        mae = float(np.mean(np.abs(y_true - y_pred)))

        assert 0.05 < mae < 1.5, (
            f"Lag-1 baseline MAE = {mae:.4f} is outside the plausible range "
            f"[0.05, 1.5] for hourly household power. "
            f"Very low value (< 0.05) suggests leakage has crept back in."
        )
