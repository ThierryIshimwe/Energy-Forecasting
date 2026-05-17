"""Forecasting models.

Public API:
    BaseForecaster    — abstract base class every model inherits from
    NaiveLagBaseline  — predict y_t = y_{t-lag} from a precomputed lag feature
    SARIMAForecaster  — classical SARIMA via statsmodels SARIMAX
    XGBoostForecaster — gradient-boosted trees on the leakage-safe feature matrix
    LSTMForecaster    — univariate LSTM / GRU sequence model (PyTorch)
"""

from energy_forecasting.models.base import BaseForecaster
from energy_forecasting.models.baselines import NaiveLagBaseline
from energy_forecasting.models.lstm_model import LSTMForecaster
from energy_forecasting.models.sarima import SARIMAForecaster
from energy_forecasting.models.xgboost_model import XGBoostForecaster

__all__ = [
    "BaseForecaster",
    "LSTMForecaster",
    "NaiveLagBaseline",
    "SARIMAForecaster",
    "XGBoostForecaster",
]
