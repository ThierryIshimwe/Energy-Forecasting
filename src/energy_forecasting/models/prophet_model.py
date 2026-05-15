"""Prophet forecaster — additive trend + seasonality + holidays decomposition.

Beyond a forecaster, Prophet is the *interpretability* model: its output
naturally splits into trend, daily/weekly/yearly seasonality, and a
holiday-effect component. We pass French national holidays (from the
``is_french_holiday`` feature) as the holiday calendar, so the trained model
can be inspected to say things like "Bastille Day adds X kW of expected
consumption". This is the kind of explanation the technical report's
discussion section will draw on.

Design:
- Trend: piecewise-linear (Prophet default).
- Daily seasonality: enabled.
- Weekly seasonality: enabled.
- Yearly seasonality: enabled (we have ~4 years of data).
- Custom holidays: French national holidays extracted from the
  ``is_french_holiday`` column when present in X.
- No lag-feature regressors in v1. Prophet is meant to compete on
  decomposition, not lag-feature usage. (We can add them later if Phase 4F
  tuning argues for it.)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from energy_forecasting.exceptions import ModelError
from energy_forecasting.models.base import BaseForecaster
from energy_forecasting.utils import get_logger

FloatArray = npt.NDArray[np.float64]

_logger = get_logger(__name__)


class ProphetForecaster(BaseForecaster):
    """Prophet forecaster with French holidays."""

    def __init__(
        self,
        *,
        daily_seasonality: bool = True,
        weekly_seasonality: bool = True,
        yearly_seasonality: bool = True,
        changepoint_prior_scale: float = 0.05,  # Prophet default
        seasonality_prior_scale: float = 10.0,  # Prophet default
        holidays_prior_scale: float = 10.0,
        use_holidays: bool = True,
        name: str | None = None,
    ) -> None:
        self.daily_seasonality = daily_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.yearly_seasonality = yearly_seasonality
        self.changepoint_prior_scale = changepoint_prior_scale
        self.seasonality_prior_scale = seasonality_prior_scale
        self.holidays_prior_scale = holidays_prior_scale
        self.use_holidays = use_holidays
        self.name: str = name or "prophet_daily+weekly+yearly+fr_hol"
        self._model: Any = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ProphetForecaster":
        from prophet import Prophet

        if len(X) != len(y):
            raise ModelError(f"X ({len(X)}) and y ({len(y)}) must have matching length")

        df_train = pd.DataFrame({"ds": y.index, "y": y.to_numpy()})
        holidays_df: pd.DataFrame | None = None
        if self.use_holidays and "is_french_holiday" in X.columns:
            holiday_dates = sorted({
                pd.Timestamp(ts).normalize()
                for ts, flag in zip(X.index, X["is_french_holiday"].to_numpy())
                if int(flag) == 1
            })
            if holiday_dates:
                holidays_df = pd.DataFrame({
                    "holiday": "french_national",
                    "ds": holiday_dates,
                })

        _logger.info(
            f"Fitting {self.name} on {len(df_train):,} rows "
            f"(daily={self.daily_seasonality}, weekly={self.weekly_seasonality}, "
            f"yearly={self.yearly_seasonality}, holidays="
            f"{len(holidays_df) if holidays_df is not None else 0})"
        )
        self._model = Prophet(
            daily_seasonality=self.daily_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            yearly_seasonality=self.yearly_seasonality,
            changepoint_prior_scale=self.changepoint_prior_scale,
            seasonality_prior_scale=self.seasonality_prior_scale,
            holidays_prior_scale=self.holidays_prior_scale,
            holidays=holidays_df,
        )
        # Prophet logs a lot of stan output; suppress at warning level.
        import logging
        logging.getLogger("prophet").setLevel(logging.WARNING)
        logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
        self._model.fit(df_train)
        _logger.info(f"{self.name} fit complete")
        return self

    def predict(self, X: pd.DataFrame) -> FloatArray:
        """Return Prophet's yhat for the timestamps in ``X.index``."""
        if self._model is None:
            raise ModelError(f"{self.name}.predict() called before fit()")
        future = pd.DataFrame({"ds": X.index})
        forecast = self._model.predict(future)
        return np.asarray(forecast["yhat"].to_numpy(), dtype=np.float64)

    def predict_components(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return the full Prophet forecast (trend, seasonality, holidays, yhat).

        Used by the modeling notebook's interpretability section.
        """
        if self._model is None:
            raise ModelError("predict_components() called before fit()")
        future = pd.DataFrame({"ds": X.index})
        return self._model.predict(future)

    def get_hyperparameters(self) -> dict[str, object]:
        return {
            "daily_seasonality": self.daily_seasonality,
            "weekly_seasonality": self.weekly_seasonality,
            "yearly_seasonality": self.yearly_seasonality,
            "changepoint_prior_scale": self.changepoint_prior_scale,
            "seasonality_prior_scale": self.seasonality_prior_scale,
            "holidays_prior_scale": self.holidays_prior_scale,
            "use_holidays": self.use_holidays,
        }
