"""Data layer: raw I/O and quality validation.

Public API:
    load_raw          — parse the UCI text file into a clean DataFrame
    validate_raw      — run the full quality battery on a loaded DataFrame
    ValidationReport  — result type returned by validate_raw

Boundary discipline: this is the only module that knows about file paths,
delimiters, and the UCI dataset's quirks. Everything downstream consumes a
validated DataFrame and trusts its shape.
"""

from energy_forecasting.data.loader import EXPECTED_RAW_COLUMNS, load_raw
from energy_forecasting.data.validator import ValidationReport, validate_raw

__all__ = [
    "EXPECTED_RAW_COLUMNS",
    "ValidationReport",
    "load_raw",
    "validate_raw",
]
