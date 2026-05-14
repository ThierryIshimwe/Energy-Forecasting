"""Forecasting models.

Public API:
    BaseForecaster   — abstract base class every model inherits from
    NaiveLagBaseline — predict y_t = y_{t-lag} from a precomputed lag feature
    SARIMAForecaster — classical SARIMA via statsmodels SARIMAX
"""

from energy_forecasting.models.base import BaseForecaster
from energy_forecasting.models.baselines import NaiveLagBaseline
from energy_forecasting.models.sarima import SARIMAForecaster

__all__ = ["BaseForecaster", "NaiveLagBaseline", "SARIMAForecaster"]
