"""French holiday feature based on the ``holidays`` library.

Why a separate module: the events/ subpackage will eventually own a richer
event catalog (weather, storms, curated incidents). The minimal "is-this-date-
a-jour-férié" feature lives here as a simple binary input for any model.

Leakage discussion: the date is known at forecast creation time, so this
feature is safe even for the prediction timestamp itself.
"""

from __future__ import annotations

import pandas as pd

from energy_forecasting.exceptions import DataValidationError
from energy_forecasting.utils import get_logger

_logger = get_logger(__name__)


def build_french_holiday_feature(df: pd.DataFrame) -> pd.DataFrame:
    """Return ``is_french_holiday`` aligned to ``df.index``.

    Args:
        df: DataFrame with a DatetimeIndex.

    Returns:
        Single-column DataFrame ``is_french_holiday`` (int8).

    Raises:
        DataValidationError: Index is not a DatetimeIndex.
        ImportError: The ``holidays`` package is not installed.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise DataValidationError(
            f"Holiday feature needs DatetimeIndex, got {type(df.index).__name__}"
        )

    try:
        import holidays as _holidays_pkg
    except ImportError as e:
        raise ImportError(
            "The 'holidays' package is required for is_french_holiday. "
            "Install with: pip install holidays"
        ) from e

    years = sorted({int(t.year) for t in df.index})
    fr_holidays = _holidays_pkg.country_holidays("FR", years=years)

    dates_only = df.index.normalize()
    is_holiday = pd.Series(
        [d in fr_holidays for d in dates_only.date],
        index=df.index,
        dtype="int8",
    )
    n_holiday_hours = int(is_holiday.sum())
    _logger.info(
        f"French holidays: {n_holiday_hours:,} of {len(df):,} timestamps "
        f"fall on {len(set(dates_only.date) & set(fr_holidays.keys()))} distinct holiday dates"
    )
    return is_holiday.to_frame("is_french_holiday")
