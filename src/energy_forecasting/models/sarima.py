"""SARIMA forecaster — the classical statistical model required by the brief.

Built on ``statsmodels.tsa.statespace.SARIMAX``. Two prediction modes:

- :meth:`predict` — multi-step out-of-sample forecast (the natural ARIMA usage;
  errors compound over the horizon).
- :meth:`predict_rolling_one_step` — extends the model with actual values
  step by step, returning 1-step-ahead predictions over the validation window.
  This is what makes the comparison against ``naive_lag_1h`` apples-to-apples.

Defaults to ``order=(1,1,1)`` and ``seasonal_order=(1,1,1,24)``, motivated by
the EDA findings:
- d=1 from §6 (stationary after first-differencing)
- s=24 from §7 (daily seasonal component dominant in STL)
- P=D=Q=1 as a conservative starting point

For Phase 4A we fit these fixed orders; auto-selection via ``pmdarima.auto_arima``
is in scope for Phase 4F when we move to per-model hyperparameter search.
"""

from __future__ import annotations

import warnings
from typing import Sequence

import numpy as np
import numpy.typing as npt
import pandas as pd

from energy_forecasting.exceptions import ModelError
from energy_forecasting.models.base import BaseForecaster
from energy_forecasting.utils import get_logger

FloatArray = npt.NDArray[np.float64]

_logger = get_logger(__name__)


class SARIMAForecaster(BaseForecaster):
    """SARIMA via statsmodels SARIMAX.

    Args:
        order: Non-seasonal (p, d, q). Default (1, 1, 1).
        seasonal_order: Seasonal (P, D, Q, s). Default (1, 1, 1, 24) — daily.
        exog_columns: Optional list of column names in X to use as exogenous
            regressors (SARIMAX). Useful for calendar features like
            ``hour_sin``, ``hour_cos``, ``is_french_holiday``. Empty by default
            (pure SARIMA).
        enforce_stationarity: Pass-through to SARIMAX. Default True.
        enforce_invertibility: Pass-through to SARIMAX. Default True.
        name: Override the auto-generated model name.
    """

    def __init__(
        self,
        order: tuple[int, int, int] = (1, 1, 1),
        seasonal_order: tuple[int, int, int, int] = (1, 1, 1, 24),
        *,
        exog_columns: Sequence[str] = (),
        enforce_stationarity: bool = True,
        enforce_invertibility: bool = True,
        name: str | None = None,
    ) -> None:
        self.order = order
        self.seasonal_order = seasonal_order
        self.exog_columns: tuple[str, ...] = tuple(exog_columns)
        self.enforce_stationarity = enforce_stationarity
        self.enforce_invertibility = enforce_invertibility
        exog_tag = f"+exog{len(self.exog_columns)}" if self.exog_columns else ""
        self.name: str = name or f"sarima{order}x{seasonal_order}{exog_tag}"
        # Set after fit():
        self._results = None
        self._train_end: pd.Timestamp | None = None

    # ── BaseForecaster contract ─────────────────────────────────

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "SARIMAForecaster":
        """Fit SARIMAX on ``(y, exog)`` where exog is ``X[exog_columns]`` if any."""
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        if len(y) != len(X):
            raise ModelError(f"X ({len(X)}) and y ({len(y)}) must have matching length")
        exog_train = self._select_exog(X)

        _logger.info(
            f"Fitting {self.name} on {len(y):,} hourly observations "
            f"(exog={'yes' if exog_train is not None else 'no'})"
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # SARIMAX prints many convergence warnings
            model = SARIMAX(
                y,
                exog=exog_train,
                order=self.order,
                seasonal_order=self.seasonal_order,
                enforce_stationarity=self.enforce_stationarity,
                enforce_invertibility=self.enforce_invertibility,
            )
            self._results = model.fit(disp=False)
        self._train_end = pd.Timestamp(y.index[-1])
        _logger.info(f"{self.name} fit converged: log-likelihood={self._results.llf:.1f}")
        return self

    def predict(self, X: pd.DataFrame) -> FloatArray:
        """Multi-step out-of-sample forecast for the timestamps in ``X.index``.

        Each prediction uses the model's own previous predictions for lagged
        values — errors compound over the horizon. For a per-timestep
        1-step-ahead evaluation, use :meth:`predict_rolling_one_step` instead.
        """
        self._require_fitted()
        exog = self._select_exog(X)
        forecast = self._results.get_forecast(steps=len(X), exog=exog)
        return np.asarray(forecast.predicted_mean, dtype=np.float64)

    # ── Rolling 1-step-ahead (the protocol used in the leaderboard) ─

    def predict_rolling_one_step(
        self,
        X_valid: pd.DataFrame,
        y_valid: pd.Series,
    ) -> FloatArray:
        """Return 1-step-ahead predictions over the validation window.

        At each timestep ``t`` in ``X_valid.index``, the prediction for ``y[t]``
        uses the model state filtered up to actual ``y[t-1]`` — exactly the
        information lag-1 baseline has access to. Implementation uses
        ``SARIMAXResults.apply`` to extend the model with actuals.
        """
        self._require_fitted()
        if len(X_valid) != len(y_valid):
            raise ModelError(
                f"X_valid ({len(X_valid)}) and y_valid ({len(y_valid)}) must match"
            )

        exog_valid = self._select_exog(X_valid)

        # Apply (refilter) the existing fitted model to the extended endog.
        # This does NOT re-estimate parameters — it just runs the Kalman filter
        # over the new data using the trained coefficients. Cheap.
        extended = self._results.apply(
            endog=y_valid,
            exog=exog_valid,
            refit=False,
        )
        # `dynamic=False` ⇒ each prediction uses actual past values, not
        # predicted past values. Exactly the 1-step-ahead protocol.
        preds = extended.predict(
            start=X_valid.index[0],
            end=X_valid.index[-1],
            dynamic=False,
        )
        return np.asarray(preds, dtype=np.float64)

    # ── Introspection ────────────────────────────────────────────

    def get_hyperparameters(self) -> dict[str, object]:
        return {
            "order": self.order,
            "seasonal_order": self.seasonal_order,
            "exog_columns": list(self.exog_columns),
        }

    # ── Internals ────────────────────────────────────────────────

    def _select_exog(self, X: pd.DataFrame) -> pd.DataFrame | None:
        if not self.exog_columns:
            return None
        missing = set(self.exog_columns) - set(X.columns)
        if missing:
            raise ModelError(f"SARIMA exog columns missing from X: {sorted(missing)}")
        return X[list(self.exog_columns)]

    def _require_fitted(self) -> None:
        if self._results is None:
            raise ModelError(f"{self.name}.predict() called before fit()")

