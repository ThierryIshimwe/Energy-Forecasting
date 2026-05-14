"""Abstract base class for every forecasting model in the project.

The training loop, evaluator, and inference service all operate on a list of
``BaseForecaster`` instances. Adding a new model is purely additive — subclass,
implement the three methods, append to the model lineup, done.

The class is small *on purpose*. Anything not in this interface is model-
specific and lives in the subclass. We resist adding methods like
``predict_proba`` or ``feature_importance`` to the base because not every
forecaster supports them. Subclasses that do offer them via their own methods.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt
import pandas as pd

FloatArray = npt.NDArray[np.float64]


class BaseForecaster(ABC):
    """Common interface every forecasting model implements.

    Subclasses must set the ``name`` attribute (used for logging and the
    leaderboard) and implement ``fit`` and ``predict``. The default
    ``get_hyperparameters`` returns an empty dict; subclasses with tunable
    parameters override it to return what should be logged with MLflow.

    Subclasses must be importable without their heavy dependencies loaded
    until ``fit`` is called — i.e. ``torch`` / ``prophet`` imports go inside
    the subclass body, not at module top.
    """

    #: Stable identifier for this model instance (e.g. ``"naive_lag_24h"``).
    name: str

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BaseForecaster":
        """Train on ``(X, y)`` and return ``self``.

        Conventions:
        - ``X.index`` and ``y.index`` are assumed identical (caller's job).
        - For models that have no actual training step (e.g. naive lag
          baselines), ``fit`` is still defined for interface symmetry and
          should just return ``self``.
        """

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> FloatArray:
        """Return point predictions, aligned with ``X.index``.

        The returned array has length ``len(X)`` and dtype ``float64``.
        """

    def get_hyperparameters(self) -> dict[str, object]:
        """Return the hyperparameters that define this instance.

        Override in subclasses that have tunable parameters. Used by the
        tracker to log every run's configuration to MLflow.
        """
        return {}

    # ── Convenience defaults that subclasses rarely need to override ──

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        hp = self.get_hyperparameters()
        if hp:
            kv = ", ".join(f"{k}={v!r}" for k, v in hp.items())
            return f"{type(self).__name__}({kv}, name={self.name!r})"
        return f"{type(self).__name__}(name={self.name!r})"
