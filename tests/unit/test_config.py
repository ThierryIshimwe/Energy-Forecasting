"""Unit tests for ForecastConfig loading and validation."""

from __future__ import annotations

import pytest

from energy_forecasting.config import (
    DEFAULT_CONFIG_PATH,
    DataConfig,
    EvaluationConfig,
    FeaturesConfig,
    ForecastConfig,
    SplitsConfig,
    TargetConfig,
)
from energy_forecasting.exceptions import ConfigurationError


class TestLoading:
    def test_default_config_loads(self) -> None:
        cfg = ForecastConfig.from_yaml(DEFAULT_CONFIG_PATH)
        assert cfg.target.column == "Global_active_power"
        assert cfg.target.frequency == "1h"
        assert cfg.target.horizon_hours == 24
        assert cfg.features.strict_leakage_safe is True

    def test_content_hash_is_stable(self) -> None:
        cfg1 = ForecastConfig.from_yaml(DEFAULT_CONFIG_PATH)
        cfg2 = ForecastConfig.from_yaml(DEFAULT_CONFIG_PATH)
        assert cfg1.content_hash() == cfg2.content_hash()

    def test_missing_file_raises(self) -> None:
        with pytest.raises(ConfigurationError, match="not found"):
            ForecastConfig.from_yaml("/nonexistent/path/config.yaml")


class TestValidation:
    """Each sub-config must reject invalid values at construction."""

    def test_invalid_frequency(self) -> None:
        with pytest.raises(ConfigurationError, match="frequency"):
            TargetConfig(column="x", frequency="13s", horizon_hours=24, aggregation="mean")

    def test_invalid_aggregation(self) -> None:
        with pytest.raises(ConfigurationError, match="aggregation"):
            TargetConfig(column="x", frequency="1h", horizon_hours=24, aggregation="bad")

    def test_negative_horizon(self) -> None:
        with pytest.raises(ConfigurationError, match="horizon"):
            TargetConfig(column="x", frequency="1h", horizon_hours=0, aggregation="mean")

    def test_invalid_url(self) -> None:
        with pytest.raises(ConfigurationError, match="http"):
            DataConfig(source_url="not-a-url", raw_filename="x.txt")

    def test_invalid_primary_metric(self) -> None:
        with pytest.raises(ConfigurationError, match="primary_metric"):
            EvaluationConfig(
                primary_metric="UNKNOWN",
                guardrail_metrics=("RMSE",),
                prediction_interval=0.95,
                segment_dimensions=("hour",),
            )

    def test_prediction_interval_out_of_range(self) -> None:
        with pytest.raises(ConfigurationError, match="prediction_interval"):
            EvaluationConfig(
                primary_metric="MAE",
                guardrail_metrics=(),
                prediction_interval=1.5,
                segment_dimensions=(),
            )

    def test_too_few_folds(self) -> None:
        with pytest.raises(ConfigurationError, match="n_folds"):
            SplitsConfig(n_folds=1, train_window_days=10, valid_window_days=10, gap_days=0)

    def test_negative_lag(self) -> None:
        with pytest.raises(ConfigurationError, match="lag_hours"):
            FeaturesConfig(
                lag_hours=(1, -2, 24),
                rolling_window_hours=(24,),
                enable_cyclical_encoding=True,
                enable_calendar=True,
                enable_french_holidays=False,
                strict_leakage_safe=True,
            )


class TestImmutability:
    def test_top_level_is_frozen(self) -> None:
        cfg = ForecastConfig.from_yaml(DEFAULT_CONFIG_PATH)
        with pytest.raises(Exception):  # FrozenInstanceError / AttributeError
            cfg.target = None  # type: ignore[misc]

    def test_subconfig_is_frozen(self) -> None:
        cfg = ForecastConfig.from_yaml(DEFAULT_CONFIG_PATH)
        with pytest.raises(Exception):
            cfg.target.column = "oops"  # type: ignore[misc]
