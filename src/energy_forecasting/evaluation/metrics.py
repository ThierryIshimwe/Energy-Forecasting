r"""Forecasting evaluation metrics.

We implement each metric from scratch following the spirit of the Titanic
SNA notebook (foundational math first, then verify against the library where
a library reference exists). This serves three purposes:

1. **Pedagogical:** every formula is written out explicitly in the docstring
   so a reader can audit what is being computed.
2. **Specification:** the implementation IS the definition. There's no
   ambiguity about which MAPE variant we use (we don't — we use WAPE instead,
   for the reasons stated in §9 of the EDA).
3. **Verification:** MAE and RMSE are checked against scikit-learn in the
   test suite; agreement to machine precision confirms our math is right.

The metric definitions:

* **MAE** — mean absolute error
  $\mathrm{MAE} = \frac{1}{n} \sum_{t=1}^{n} |y_t - \hat y_t|$
* **RMSE** — root mean squared error
  $\mathrm{RMSE} = \sqrt{\frac{1}{n} \sum_{t=1}^{n} (y_t - \hat y_t)^2}$
* **WAPE** — weighted absolute percentage error (robust to near-zero $y$)
  $\mathrm{WAPE} = \frac{\sum |y_t - \hat y_t|}{\sum |y_t|}$
* **MASE** — mean absolute scaled error (Hyndman & Koehler 2006)
  $\mathrm{MASE} = \frac{\mathrm{MAE}(\text{forecast})}{\mathrm{MAE}(\text{seasonal naive on train})}$
* **sMAPE** — symmetric mean absolute percentage error
  $\mathrm{sMAPE} = \frac{1}{n} \sum \frac{|y_t - \hat y_t|}{(|y_t| + |\hat y_t|)/2}$

Why these five and not MAPE: MAPE explodes when actuals are near zero (which
happens nightly in this dataset — see EDA §9). WAPE is the robust analogue;
MASE provides scale-free comparability against a seasonal-naive baseline.
"""

from __future__ import annotations

from typing import Callable, Final

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


def _to_float_array(x: npt.ArrayLike) -> FloatArray:
    """Coerce array-like input to float64 ndarray (preserves NaN)."""
    return np.asarray(x, dtype=np.float64)


def _validate_shapes(y_true: FloatArray, y_pred: FloatArray) -> None:
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"y_true and y_pred must have the same shape; got {y_true.shape} and {y_pred.shape}"
        )
    if y_true.size == 0:
        raise ValueError("y_true and y_pred must not be empty")


def mae(y_true: npt.ArrayLike, y_pred: npt.ArrayLike) -> float:
    r"""Mean absolute error: $\frac{1}{n}\sum |y_t - \hat y_t|$.

    Equivalent to ``sklearn.metrics.mean_absolute_error``; verified in tests.
    """
    yt = _to_float_array(y_true)
    yp = _to_float_array(y_pred)
    _validate_shapes(yt, yp)
    return float(np.mean(np.abs(yt - yp)))


def rmse(y_true: npt.ArrayLike, y_pred: npt.ArrayLike) -> float:
    r"""Root mean squared error: $\sqrt{\frac{1}{n}\sum (y_t - \hat y_t)^2}$.

    Equivalent to ``np.sqrt(sklearn.metrics.mean_squared_error(y_true, y_pred))``;
    verified in tests.
    """
    yt = _to_float_array(y_true)
    yp = _to_float_array(y_pred)
    _validate_shapes(yt, yp)
    return float(np.sqrt(np.mean((yt - yp) ** 2)))


def wape(y_true: npt.ArrayLike, y_pred: npt.ArrayLike) -> float:
    r"""Weighted absolute percentage error: $\frac{\sum |y_t - \hat y_t|}{\sum |y_t|}$.

    Returned as a fraction (multiply by 100 for percentage display). Robust
    to near-zero values where MAPE would explode.

    Raises:
        ValueError: ``sum(|y_true|)`` is zero (cannot weight by zero total).
    """
    yt = _to_float_array(y_true)
    yp = _to_float_array(y_pred)
    _validate_shapes(yt, yp)
    denom = float(np.sum(np.abs(yt)))
    if denom == 0.0:
        raise ValueError("WAPE undefined: sum of |y_true| is zero")
    return float(np.sum(np.abs(yt - yp)) / denom)


def smape(y_true: npt.ArrayLike, y_pred: npt.ArrayLike) -> float:
    r"""Symmetric mean absolute percentage error.

    $\mathrm{sMAPE} = \frac{1}{n} \sum \frac{|y_t - \hat y_t|}{(|y_t| + |\hat y_t|)/2}$

    Returned as a fraction. When both $y$ and $\hat y$ are zero at a point we
    treat the contribution as zero (limit case).
    """
    yt = _to_float_array(y_true)
    yp = _to_float_array(y_pred)
    _validate_shapes(yt, yp)
    denom = (np.abs(yt) + np.abs(yp)) / 2.0
    safe = np.where(denom == 0.0, 1.0, denom)
    contrib = np.where(denom == 0.0, 0.0, np.abs(yt - yp) / safe)
    return float(np.mean(contrib))


def mase(
    y_true: npt.ArrayLike,
    y_pred: npt.ArrayLike,
    *,
    y_train: npt.ArrayLike,
    seasonal_period: int = 1,
) -> float:
    r"""Mean absolute scaled error (Hyndman & Koehler 2006).

    $\mathrm{MASE} = \frac{\mathrm{MAE}(\text{forecast})}{\mathrm{MAE}(\text{naive on training})}$

    The denominator is the in-sample MAE of a seasonal-naive forecast on the
    training data — i.e. $y_t = y_{t-m}$ where $m$ is ``seasonal_period``.
    For ``seasonal_period=1`` this collapses to the random-walk naive.

    MASE < 1 means the model beats the naive baseline; MASE = 1 means it ties.

    Args:
        y_true: Validation truth.
        y_pred: Validation predictions (same shape).
        y_train: Training-set truth, used to compute the denominator. Must be
            longer than ``seasonal_period``.
        seasonal_period: Lag in steps for the seasonal naive (default 1 =
            non-seasonal naive). Use 24 for daily, 168 for weekly with hourly data.

    Raises:
        ValueError: ``y_train`` is too short or its naive forecast has zero MAE.
    """
    yt = _to_float_array(y_true)
    yp = _to_float_array(y_pred)
    _validate_shapes(yt, yp)
    train = _to_float_array(y_train)
    if seasonal_period < 1:
        raise ValueError(f"seasonal_period must be >= 1, got {seasonal_period}")
    if len(train) <= seasonal_period:
        raise ValueError(
            f"y_train of length {len(train)} cannot support seasonal_period={seasonal_period}"
        )
    naive_errors = np.abs(train[seasonal_period:] - train[:-seasonal_period])
    naive_mae = float(np.mean(naive_errors))
    if naive_mae == 0.0:
        raise ValueError("MASE undefined: in-sample naive MAE is zero (constant training series)")
    return float(np.mean(np.abs(yt - yp))) / naive_mae


# Bundled lookup. Functions take just (y_true, y_pred); MASE is excluded from
# this dict because it needs the extra y_train argument and a seasonal period.
METRIC_FUNCTIONS: Final[dict[str, Callable[[npt.ArrayLike, npt.ArrayLike], float]]] = {
    "MAE": mae,
    "RMSE": rmse,
    "WAPE": wape,
    "sMAPE": smape,
}


def compute_metrics(
    y_true: npt.ArrayLike,
    y_pred: npt.ArrayLike,
    *,
    y_train: npt.ArrayLike | None = None,
    seasonal_period: int = 24,
) -> dict[str, float]:
    """Compute MAE, RMSE, WAPE, sMAPE on ``(y_true, y_pred)``.

    If ``y_train`` is provided, also compute MASE with the given
    ``seasonal_period`` (default 24 hours = daily naive).
    """
    out: dict[str, float] = {name: fn(y_true, y_pred) for name, fn in METRIC_FUNCTIONS.items()}
    if y_train is not None:
        out["MASE"] = mase(y_true, y_pred, y_train=y_train, seasonal_period=seasonal_period)
    return out
