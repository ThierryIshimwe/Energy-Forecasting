"""Raw data loading for the UCI Household Electric Power Consumption dataset.

The UCI file is a semicolon-delimited text file with separate ``Date`` and
``Time`` columns that need to be joined into a single ``DatetimeIndex``.
Numeric columns use ``'?'`` as the missing-value sentinel.

This module's only job is to parse the bytes into a clean, well-typed
DataFrame. Deeper quality checks (value plausibility, sufficiency for the
configured pipeline) live in :mod:`energy_forecasting.data.validator`.
The two are composed by the ``download_data.py`` script and the EDA notebook.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from energy_forecasting.exceptions import DataValidationError
from energy_forecasting.utils import get_logger

_logger = get_logger(__name__)

# UCI dataset's canonical column names. Used for boundary validation.
EXPECTED_RAW_COLUMNS: frozenset[str] = frozenset(
    {
        "Date",
        "Time",
        "Global_active_power",
        "Global_reactive_power",
        "Voltage",
        "Global_intensity",
        "Sub_metering_1",
        "Sub_metering_2",
        "Sub_metering_3",
    }
)

_NUMERIC_COLUMNS: tuple[str, ...] = (
    "Global_active_power",
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3",
)


def load_raw(file_path: Path | str) -> pd.DataFrame:
    """Load the UCI raw text file into a DateTime-indexed DataFrame.

    Args:
        file_path: Path to the ``data.txt`` file (semicolon-delimited).

    Returns:
        DataFrame with a sorted ``DatetimeIndex`` and seven numeric columns:
        ``Global_active_power``, ``Global_reactive_power``, ``Voltage``,
        ``Global_intensity``, ``Sub_metering_1/2/3``. Missing values (``'?'``
        in the source) are NaN.

    Raises:
        FileNotFoundError: ``file_path`` does not exist.
        DataValidationError: The file could not be parsed, required columns
            are missing, or every timestamp failed to parse.
    """
    p = Path(file_path)
    if not p.is_file():
        raise FileNotFoundError(f"Raw data file not found: {p}")

    _logger.info(f"Loading raw dataset from {p}")

    try:
        df = pd.read_csv(p, sep=";", na_values=["?"], low_memory=False)
    except pd.errors.ParserError as e:
        raise DataValidationError(f"Failed to parse CSV file {p}: {e}") from e
    except UnicodeDecodeError as e:
        raise DataValidationError(f"Encoding error reading {p}: {e}") from e

    # Match columns case-insensitively so minor casing variations don't break ingestion.
    columns_lower = {c.strip().lower(): c for c in df.columns}
    expected_lower = {c.lower() for c in EXPECTED_RAW_COLUMNS}
    missing = expected_lower - set(columns_lower.keys())
    if missing:
        raise DataValidationError(
            f"Missing expected columns: {sorted(missing)}. Found: {sorted(df.columns)}"
        )

    # Combine Date + Time into a single timestamp. UCI uses DD/MM/YYYY (dayfirst).
    date_col = columns_lower["date"]
    time_col = columns_lower["time"]
    df["datetime"] = pd.to_datetime(
        df[date_col].astype(str) + " " + df[time_col].astype(str),
        dayfirst=True,
        errors="coerce",
    )

    n_unparseable = int(df["datetime"].isna().sum())
    if n_unparseable == len(df):
        raise DataValidationError("All timestamps failed to parse — check delimiter and date format")
    if n_unparseable > 0:
        pct = n_unparseable / len(df) * 100
        _logger.warning(
            f"{n_unparseable:,} ({pct:.2f}%) timestamps could not be parsed; rows dropped"
        )
        df = df.dropna(subset=["datetime"])

    df = df.set_index("datetime").sort_index()
    df = df.drop(columns=[date_col, time_col])

    # Defensive numeric coercion. ``na_values=['?']`` should have done this,
    # but malformed cells can still slip through with whitespace etc.
    for col in _NUMERIC_COLUMNS:
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], errors="coerce")

    _logger.info(
        f"Loaded {len(df):,} rows × {df.shape[1]} columns, "
        f"range {df.index.min()} → {df.index.max()}"
    )
    return df
