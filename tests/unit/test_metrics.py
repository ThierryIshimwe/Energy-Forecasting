"""Unit tests for the forecasting metric functions.

Two kinds of checks:

1. **Mathematical correctness** — small hand-calculable examples confirm each
   formula yields the expected number.
2. **Library equivalence** — for MAE and RMSE, our custom implementation must
   match scikit-learn's to machine precision. This is the test that earns us
   the right to use the library version going forward without re-deriving.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from sklearn.metrics import mean_absolute_error, mean_squared_error

from energy_forecasting.evaluation import (
    METRIC_FUNCTIONS,
    compute_metrics,
    mae,
    mase,
    rmse,
    smape,
    wape,
)


# ─────────────────────────────────────────────────────────────────────
#  Hand-calculable cases
# ─────────────────────────────────────────────────────────────────────

class TestHandCalculable:
    """Tiny inputs where the answer can be computed by hand."""

    def test_mae_two_points(self) -> None:
        # |10 - 12| + |20 - 17| = 2 + 3 = 5; mean = 2.5
        assert mae([10, 20], [12, 17]) == pytest.approx(2.5)

    def test_rmse_two_points(self) -> None:
        # ((10-12)^2 + (20-17)^2) / 2 = (4 + 9) / 2 = 6.5; sqrt ≈ 2.5495
        assert rmse([10, 20], [12, 17]) == pytest.approx(math.sqrt(6.5))

    def test_wape_one_point(self) -> None:
        # |y - yhat| / |y| = |10 - 12| / 10 = 0.2
        assert wape([10], [12]) == pytest.approx(0.2)

    def test_wape_with_negative_residual(self) -> None:
        # Total error = 2 + 3 = 5; total |y| = 10 + 20 = 30; WAPE = 5/30
        assert wape([10, 20], [12, 17]) == pytest.approx(5 / 30)

    def test_smape_identical_zero(self) -> None:
        # Both arrays all zero — sMAPE limit case is 0
        assert smape([0, 0, 0], [0, 0, 0]) == pytest.approx(0.0)

    def test_smape_known_value(self) -> None:
        # |10 - 8| / ((10 + 8)/2) = 2/9 ≈ 0.2222
        assert smape([10], [8]) == pytest.approx(2 / 9)

    def test_mase_against_naive_train(self) -> None:
        # train = [1, 2, 3, 4, 5]; naive errors (period=1) = [1,1,1,1] → MAE = 1.0
        # forecast MAE = MAE([10, 20], [12, 17]) = 2.5
        # MASE = 2.5 / 1.0 = 2.5
        assert mase([10, 20], [12, 17], y_train=[1, 2, 3, 4, 5], seasonal_period=1) == pytest.approx(2.5)


# ─────────────────────────────────────────────────────────────────────
#  Library equivalence — MAE, RMSE vs scikit-learn
# ─────────────────────────────────────────────────────────────────────

class TestLibraryEquivalence:
    """Our custom MAE and RMSE must match sklearn to machine precision."""

    @pytest.fixture
    def random_pair(self) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed=42)
        y_true = rng.uniform(0.0, 5.0, size=1000)
        y_pred = y_true + rng.normal(0.0, 0.3, size=1000)
        return y_true, y_pred

    def test_mae_matches_sklearn(self, random_pair: tuple[np.ndarray, np.ndarray]) -> None:
        y_true, y_pred = random_pair
        assert mae(y_true, y_pred) == pytest.approx(mean_absolute_error(y_true, y_pred))

    def test_rmse_matches_sklearn(self, random_pair: tuple[np.ndarray, np.ndarray]) -> None:
        y_true, y_pred = random_pair
        sklearn_rmse = math.sqrt(mean_squared_error(y_true, y_pred))
        assert rmse(y_true, y_pred) == pytest.approx(sklearn_rmse)


# ─────────────────────────────────────────────────────────────────────
#  Validation and edge cases
# ─────────────────────────────────────────────────────────────────────

class TestValidation:
    def test_shape_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="same shape"):
            mae([1, 2, 3], [1, 2])

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="not be empty"):
            mae([], [])

    def test_wape_all_zero_truth_raises(self) -> None:
        with pytest.raises(ValueError, match="undefined"):
            wape([0, 0, 0], [1, 1, 1])

    def test_mase_train_too_short(self) -> None:
        with pytest.raises(ValueError, match="cannot support"):
            mase([1, 2], [1, 2], y_train=[1, 2], seasonal_period=24)

    def test_mase_constant_train_raises(self) -> None:
        with pytest.raises(ValueError, match="naive MAE is zero"):
            mase([1, 2], [1, 2], y_train=[3, 3, 3, 3, 3], seasonal_period=1)


# ─────────────────────────────────────────────────────────────────────
#  Bundled API
# ─────────────────────────────────────────────────────────────────────

class TestComputeMetrics:
    def test_returns_four_metrics_without_train(self) -> None:
        m = compute_metrics([1, 2, 3, 4], [1.1, 1.9, 3.2, 3.9])
        assert set(m.keys()) == {"MAE", "RMSE", "WAPE", "sMAPE"}

    def test_includes_mase_when_train_provided(self) -> None:
        m = compute_metrics(
            [1, 2, 3, 4], [1.1, 1.9, 3.2, 3.9],
            y_train=list(range(50)), seasonal_period=1,
        )
        assert "MASE" in m
        assert m["MASE"] > 0

    def test_metric_functions_dict_complete(self) -> None:
        assert set(METRIC_FUNCTIONS.keys()) == {"MAE", "RMSE", "WAPE", "sMAPE"}
