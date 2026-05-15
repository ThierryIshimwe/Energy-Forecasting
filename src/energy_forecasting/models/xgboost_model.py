"""XGBoost forecaster — gradient-boosted trees on the leakage-safe feature matrix.

XGBoost is the natural tabular ML baseline: it handles the heterogeneous mix
of calendar, cyclical, lag, and rolling features without preprocessing, learns
nonlinear interactions (e.g. ``hour × is_weekend``) automatically, and is
fast enough for hyperparameter search.

Defaults are conservative starting points (Phase 4F will run Optuna):
- ``n_estimators=500`` with ``learning_rate=0.05`` — gradual learning, enough
  trees to converge but not so many that we overfit on a single training fold.
- ``max_depth=6`` — moderate interaction depth.
- ``subsample=0.8``, ``colsample_bytree=0.8`` — light regularization via
  stochastic gradient boosting.
- ``reg_lambda=1.0`` — default L2.

No early stopping in this implementation: it requires a validation set, but our
BaseForecaster contract only exposes ``fit(X, y)``. Tuned variants in Phase 4F
will use Optuna's inner CV for that purpose.
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


class XGBoostForecaster(BaseForecaster):
    """XGBoost regressor wrapped to satisfy the ``BaseForecaster`` interface."""

    def __init__(
        self,
        *,
        n_estimators: int = 500,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_lambda: float = 1.0,
        reg_alpha: float = 0.0,
        random_state: int = 42,
        n_jobs: int = -1,
        name: str | None = None,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_lambda = reg_lambda
        self.reg_alpha = reg_alpha
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.name: str = name or f"xgb_d{max_depth}_lr{learning_rate}_n{n_estimators}"
        self._model: Any | None = None
        self._feature_names: tuple[str, ...] | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "XGBoostForecaster":
        """Fit XGBoost on ``(X, y)``."""
        from xgboost import XGBRegressor

        if len(X) != len(y):
            raise ModelError(f"X ({len(X)}) and y ({len(y)}) must have matching length")

        _logger.info(
            f"Fitting {self.name} on {len(X):,} rows × {X.shape[1]} features"
        )
        self._feature_names = tuple(X.columns)
        self._model = XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_lambda=self.reg_lambda,
            reg_alpha=self.reg_alpha,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            tree_method="hist",   # fast histogram-based algorithm
            objective="reg:squarederror",
            verbosity=0,
        )
        self._model.fit(X.to_numpy(), y.to_numpy())
        _logger.info(f"{self.name} fit complete")
        return self

    def predict(self, X: pd.DataFrame) -> FloatArray:
        """Return point predictions aligned with ``X.index``."""
        if self._model is None:
            raise ModelError(f"{self.name}.predict() called before fit()")
        if self._feature_names is not None and tuple(X.columns) != self._feature_names:
            raise ModelError(
                f"X has different columns than training. "
                f"Expected {len(self._feature_names)} features, got {X.shape[1]}."
            )
        return np.asarray(self._model.predict(X.to_numpy()), dtype=np.float64)

    def get_hyperparameters(self) -> dict[str, object]:
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "reg_lambda": self.reg_lambda,
            "reg_alpha": self.reg_alpha,
            "random_state": self.random_state,
        }

    def feature_importance(self) -> pd.Series:
        """Return per-feature gain importance, sorted descending."""
        if self._model is None or self._feature_names is None:
            raise ModelError("feature_importance() called before fit()")
        importances = self._model.feature_importances_
        return pd.Series(importances, index=self._feature_names, name="gain").sort_values(
            ascending=False
        )
