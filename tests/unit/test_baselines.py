"""Unit tests for NaiveLagBaseline and the BaseForecaster contract."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from energy_forecasting.exceptions import ModelError
from energy_forecasting.models import BaseForecaster, NaiveLagBaseline


@pytest.fixture
def sample_xy() -> tuple[pd.DataFrame, pd.Series]:
    """Tiny feature matrix with a lag column the baseline can use."""
    idx = pd.date_range("2010-01-01", periods=10, freq="1h")
    X = pd.DataFrame(
        {
            "Global_active_power_lag_1h": np.arange(10, dtype=float),
            "other_feature": np.zeros(10),
        },
        index=idx,
    )
    y = pd.Series(np.arange(1, 11, dtype=float), index=idx)
    return X, y


class TestNaiveLagBaseline:
    def test_predict_returns_lag_column_values(
        self,
        sample_xy: tuple[pd.DataFrame, pd.Series],
    ) -> None:
        X, y = sample_xy
        model = NaiveLagBaseline("Global_active_power_lag_1h").fit(X, y)
        preds = model.predict(X)
        np.testing.assert_array_equal(preds, X["Global_active_power_lag_1h"].values)
        assert preds.dtype == np.float64

    def test_fit_returns_self(self, sample_xy: tuple[pd.DataFrame, pd.Series]) -> None:
        X, y = sample_xy
        model = NaiveLagBaseline("Global_active_power_lag_1h")
        assert model.fit(X, y) is model

    def test_default_name(self) -> None:
        model = NaiveLagBaseline("Global_active_power_lag_24h")
        assert model.name == "naive_Global_active_power_lag_24h"

    def test_custom_name(self) -> None:
        model = NaiveLagBaseline("Global_active_power_lag_24h", name="daily_naive")
        assert model.name == "daily_naive"

    def test_get_hyperparameters_reports_lag_column(self) -> None:
        model = NaiveLagBaseline("Global_active_power_lag_24h")
        assert model.get_hyperparameters() == {"lag_column": "Global_active_power_lag_24h"}

    def test_missing_column_in_fit_raises(
        self,
        sample_xy: tuple[pd.DataFrame, pd.Series],
    ) -> None:
        X, y = sample_xy
        with pytest.raises(ModelError, match="requires column"):
            NaiveLagBaseline("missing_column").fit(X, y)

    def test_missing_column_in_predict_raises(
        self,
        sample_xy: tuple[pd.DataFrame, pd.Series],
    ) -> None:
        X, y = sample_xy
        model = NaiveLagBaseline("Global_active_power_lag_1h").fit(X, y)
        with pytest.raises(ModelError, match="requires column"):
            model.predict(X.drop(columns=["Global_active_power_lag_1h"]))

    def test_empty_lag_column_name_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            NaiveLagBaseline("")

    def test_satisfies_base_forecaster(self) -> None:
        model = NaiveLagBaseline("Global_active_power_lag_1h")
        assert isinstance(model, BaseForecaster)
