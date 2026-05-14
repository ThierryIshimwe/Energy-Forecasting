"""Feature pipeline: composes builders and enforces the leakage policy.

This module is the chokepoint where leakage prevention is mechanically
enforced. It:

1. Extracts the target series and ``y``.
2. Drops any input columns flagged ``leakage_safe=False`` in the registry.
3. Calls each enabled feature builder.
4. Concatenates the resulting columns.
5. Verifies *every* output column is either registered-safe OR (in
   non-strict mode) explicitly opted in.
6. Returns a :class:`FeatureMatrix` containing the safe ``X`` and ``y``.

If any unsafe feature would be emitted in strict mode, :class:`LeakageError`
is raised with a list of offending columns — this is the integration test's
canary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd

from energy_forecasting.exceptions import LeakageError
from energy_forecasting.features.calendar import build_calendar_features
from energy_forecasting.features.cyclical import build_cyclical_features
from energy_forecasting.features.holidays_feature import build_french_holiday_feature
from energy_forecasting.features.lags import build_lag_features
from energy_forecasting.features.registry import FEATURE_REGISTRY, is_safe
from energy_forecasting.features.rolling import build_rolling_features
from energy_forecasting.utils import get_logger

if TYPE_CHECKING:
    from energy_forecasting.config import ForecastConfig

_logger = get_logger(__name__)


@dataclass(frozen=True)
class FeatureMatrix:
    """Typed result of :func:`build_features`.

    Attributes:
        X: Feature matrix. Every column is leakage-safe.
        y: Target series aligned with ``X.index``.
        feature_names: Names of every column in ``X``, in order.
        n_unsafe_dropped: Number of input columns excluded by the policy.
        n_rows_before_dropna: Rows before dropping warm-up NaN rows.
        n_rows_after_dropna: Rows in the final ``X``/``y``.
    """

    X: pd.DataFrame
    y: pd.Series
    feature_names: tuple[str, ...]
    n_unsafe_dropped: int
    n_rows_before_dropna: int
    n_rows_after_dropna: int


def build_features(
    df: pd.DataFrame,
    *,
    config: "ForecastConfig",
) -> FeatureMatrix:
    """Build the model-ready feature matrix from a preprocessed DataFrame.

    Args:
        df: Output of :func:`energy_forecasting.preprocessing.preprocess`.
        config: ForecastConfig instance. Determines which features to build
            and whether strict leakage safety is enforced.

    Returns:
        :class:`FeatureMatrix` ready for splitting and modeling.

    Raises:
        LeakageError: If any feature in the output is not registered-safe and
            ``config.features.strict_leakage_safe`` is True.
    """
    target_col = config.target.column
    if target_col not in df.columns:
        raise LeakageError(
            f"Target column {target_col!r} not in input DataFrame. Columns: {list(df.columns)}"
        )

    # ── 1. Extract target ──────────────────────────────────────
    y = df[target_col].copy()
    y.name = target_col

    # ── 2. Drop unsafe input columns up front (defense in depth) ──
    unsafe_input_cols = [
        c for c in df.columns
        if c != target_col and is_safe(c) is False
    ]
    if unsafe_input_cols:
        _logger.info(
            f"Dropping {len(unsafe_input_cols)} unsafe input columns: {unsafe_input_cols}"
        )

    # Carry forward any safe non-target columns (e.g., outage indicators).
    safe_carry_cols = [
        c for c in df.columns
        if c != target_col and is_safe(c) is True
    ]
    carry_forward = df[safe_carry_cols].copy() if safe_carry_cols else None

    # ── 3. Build engineered features ───────────────────────────
    parts: list[pd.DataFrame] = []
    if carry_forward is not None:
        parts.append(carry_forward)

    fcfg = config.features
    if fcfg.enable_calendar:
        parts.append(build_calendar_features(df))
    if fcfg.enable_cyclical_encoding:
        parts.append(build_cyclical_features(df))
    if fcfg.lag_hours:
        parts.append(
            build_lag_features(y, fcfg.lag_hours, target_name=target_col)
        )
    if fcfg.rolling_window_hours:
        parts.append(
            build_rolling_features(y, fcfg.rolling_window_hours, target_name=target_col)
        )
    if fcfg.enable_french_holidays:
        parts.append(build_french_holiday_feature(df))

    X = pd.concat(parts, axis=1)

    # ── 4. Leakage check on the output ─────────────────────────
    unsafe_in_output: list[str] = []
    unknown_in_output: list[str] = []
    for col in X.columns:
        safety = is_safe(col)
        if safety is False:
            unsafe_in_output.append(col)
        elif safety is None:
            unknown_in_output.append(col)

    if fcfg.strict_leakage_safe and (unsafe_in_output or unknown_in_output):
        problems: list[str] = []
        if unsafe_in_output:
            problems.append(f"explicitly unsafe: {unsafe_in_output}")
        if unknown_in_output:
            problems.append(f"unregistered (treated as unsafe in strict mode): {unknown_in_output}")
        raise LeakageError(
            "Feature pipeline produced leakage-unsafe columns in strict mode.\n"
            "  " + "\n  ".join(problems)
            + "\nEither remove them, register them in features/registry.py, "
            "or set features.strict_leakage_safe=False (and document why)."
        )
    if unsafe_in_output:
        _logger.warning(
            f"Non-strict mode: keeping {len(unsafe_in_output)} unsafe column(s): "
            f"{unsafe_in_output}"
        )

    # ── 5. Drop warm-up NaN rows (created by lags/rolling) ─────
    n_before = len(X)
    # NaN in y means the target itself is missing — also drop.
    combined = pd.concat([y, X], axis=1).dropna()
    y_final = combined.iloc[:, 0]
    X_final = combined.iloc[:, 1:]
    n_after = len(X_final)

    _logger.info(
        f"Feature matrix built: {X_final.shape[1]} features, "
        f"{n_after:,} rows after warm-up dropna (was {n_before:,}, "
        f"retained {n_after / n_before * 100:.2f}%)"
    )

    return FeatureMatrix(
        X=X_final,
        y=y_final,
        feature_names=tuple(X_final.columns),
        n_unsafe_dropped=len(unsafe_input_cols),
        n_rows_before_dropna=n_before,
        n_rows_after_dropna=n_after,
    )
