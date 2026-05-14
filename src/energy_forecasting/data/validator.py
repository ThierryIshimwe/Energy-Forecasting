"""Quality validation for the loaded UCI dataset.

The loader gets bytes into a DataFrame; this module checks whether that
DataFrame is *usable* for forecasting. Failures here are surfaced explicitly —
either by raising :class:`DataValidationError` or by returning a
:class:`ValidationReport` with ``passed=False``.

Checks are organized into four categories:

* **Structural** — index type, sort order, duplicates
* **Schema** — required columns present, dtypes correct
* **Physical plausibility** — power ≥ 0, voltage in realistic European range
* **Sufficiency** — enough rows for the configured rolling-origin CV scheme

The sufficiency check is configuration-aware: pass a :class:`ForecastConfig`
to verify the dataset can support the configured splits.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pandas as pd

from energy_forecasting.exceptions import DataValidationError
from energy_forecasting.utils import get_logger

if TYPE_CHECKING:
    from energy_forecasting.config import ForecastConfig

_logger = get_logger(__name__)


@dataclass(frozen=True)
class ValidationReport:
    """Outcome of :func:`validate_raw`.

    Attributes:
        passed: ``True`` if every check succeeded.
        issues: Human-readable descriptions of every failure (empty when passed).
        n_rows: Total rows in the DataFrame.
        n_missing_rows: Rows with at least one NaN.
        duration_index: ``(start, end)`` if a DatetimeIndex was present.
    """

    passed: bool
    issues: tuple[str, ...] = field(default_factory=tuple)
    n_rows: int = 0
    n_missing_rows: int = 0
    duration_index: tuple[pd.Timestamp, pd.Timestamp] | None = None


# Physical-plausibility guard rails for a single European household.
_REQUIRED_NUMERIC_COLUMNS: frozenset[str] = frozenset(
    {
        "Global_active_power",
        "Global_reactive_power",
        "Voltage",
        "Global_intensity",
        "Sub_metering_1",
        "Sub_metering_2",
        "Sub_metering_3",
    }
)

# Sceaux is on the French grid (nominal 230 V). 100/300 are generous outer bounds.
_VOLTAGE_MIN_V: float = 100.0
_VOLTAGE_MAX_V: float = 300.0
# 50 kW is far above any plausible single-household instantaneous draw.
_POWER_MAX_KW: float = 50.0


def validate_raw(
    df: pd.DataFrame,
    *,
    config: "ForecastConfig | None" = None,
    raise_on_fail: bool = True,
) -> ValidationReport:
    """Run the full quality battery on a loaded raw DataFrame.

    Args:
        df: DataFrame returned by :func:`load_raw`.
        config: When given, additionally check the dataset has enough rows to
            support the configured rolling-origin CV scheme.
        raise_on_fail: If ``True`` (default), raise :class:`DataValidationError`
            on any issue. If ``False``, return a report with ``passed=False``.

    Returns:
        :class:`ValidationReport` with the outcome.

    Raises:
        DataValidationError: When ``raise_on_fail=True`` and any check fails.
    """
    issues: list[str] = []

    # ── Structural ─────────────────────────────────────────────
    if not isinstance(df.index, pd.DatetimeIndex):
        issues.append(f"Index must be DatetimeIndex, got {type(df.index).__name__}")
    else:
        if not df.index.is_monotonic_increasing:
            issues.append("DatetimeIndex must be sorted in ascending order")
        n_dups = int(df.index.duplicated().sum())
        if n_dups > 0:
            issues.append(f"{n_dups:,} duplicate timestamps detected")

    # ── Schema ─────────────────────────────────────────────────
    missing_cols = _REQUIRED_NUMERIC_COLUMNS - set(df.columns)
    if missing_cols:
        issues.append(f"Missing required columns: {sorted(missing_cols)}")
    for col in _REQUIRED_NUMERIC_COLUMNS & set(df.columns):
        if not pd.api.types.is_numeric_dtype(df[col]):
            issues.append(f"Column {col!r} is not numeric (dtype={df[col].dtype})")

    # ── Physical plausibility ──────────────────────────────────
    # Only run these on numeric columns. Non-numeric dtypes are already
    # surfaced by the schema check above; running comparisons on strings
    # here would raise an opaque TypeError.
    if "Global_active_power" in df.columns and pd.api.types.is_numeric_dtype(
        df["Global_active_power"]
    ):
        gap = df["Global_active_power"].dropna()
        if (gap < 0).any():
            issues.append(
                f"Global_active_power has {int((gap < 0).sum()):,} negative values "
                f"(physically impossible)"
            )
        if (gap > _POWER_MAX_KW).any():
            issues.append(
                f"Global_active_power has {int((gap > _POWER_MAX_KW).sum()):,} values "
                f"> {_POWER_MAX_KW} kW (unrealistic for a single household)"
            )
    if "Voltage" in df.columns and pd.api.types.is_numeric_dtype(df["Voltage"]):
        v = df["Voltage"].dropna()
        if not v.empty:
            n_lo = int((v < _VOLTAGE_MIN_V).sum())
            n_hi = int((v > _VOLTAGE_MAX_V).sum())
            if n_lo > 0:
                issues.append(f"Voltage has {n_lo:,} values < {_VOLTAGE_MIN_V} V (unrealistic)")
            if n_hi > 0:
                issues.append(f"Voltage has {n_hi:,} values > {_VOLTAGE_MAX_V} V (unrealistic)")

    # ── Sufficiency (config-aware) ─────────────────────────────
    n_rows = len(df)
    if config is not None and isinstance(df.index, pd.DatetimeIndex) and len(df) > 0:
        # The most demanding fold needs: train_window + n_folds × valid_window days
        # of raw 1-minute data behind it.
        s = config.splits
        min_days = s.train_window_days + s.n_folds * s.valid_window_days
        min_rows = min_days * 24 * 60  # raw frequency is 1 minute
        if n_rows < min_rows:
            issues.append(
                f"Dataset has {n_rows:,} rows but needs >= {min_rows:,} "
                f"for {s.n_folds}-fold CV with {s.train_window_days}d train + "
                f"{s.valid_window_days}d valid windows"
            )

    # ── Build report ───────────────────────────────────────────
    n_missing = int(df.isna().any(axis=1).sum()) if len(df) > 0 else 0
    duration: tuple[pd.Timestamp, pd.Timestamp] | None = None
    if isinstance(df.index, pd.DatetimeIndex) and len(df) > 0:
        duration = (df.index.min(), df.index.max())

    report = ValidationReport(
        passed=len(issues) == 0,
        issues=tuple(issues),
        n_rows=n_rows,
        n_missing_rows=n_missing,
        duration_index=duration,
    )

    if not report.passed:
        msg = f"Data validation failed ({len(issues)} issue(s)):\n" + "\n".join(
            f"  - {i}" for i in issues
        )
        if raise_on_fail:
            raise DataValidationError(msg)
        _logger.warning(msg)
    else:
        _logger.info(
            f"Validation passed: {n_rows:,} rows, {n_missing:,} with missing values, "
            f"range {duration[0]} → {duration[1]}" if duration else f"{n_rows} rows"
        )

    return report
