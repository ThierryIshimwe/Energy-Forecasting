"""End-to-end preprocessing orchestrator.

Composes :func:`apply_missing_value_policy` and :func:`resample_to_frequency`
in the canonical order:

    raw (1-min) -> missing-value policy (still 1-min) -> resample to target

It returns both the cleaned DataFrame and a :class:`PreprocessingReport`
capturing what happened — useful for the EDA notebook and for the technical
report's data-quality section.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd

from energy_forecasting.preprocessing.missing import (
    apply_missing_value_policy,
    summarize_missing_gaps,
)
from energy_forecasting.preprocessing.resampling import resample_to_frequency
from energy_forecasting.utils import get_logger

if TYPE_CHECKING:
    from energy_forecasting.config import ForecastConfig

_logger = get_logger(__name__)


@dataclass(frozen=True)
class PreprocessingReport:
    """Summary metadata produced by :func:`preprocess`.

    Attributes:
        input_rows: Rows in the raw input.
        output_rows: Rows in the processed output.
        target_frequency: Pandas offset alias the output was resampled to.
        n_total_gaps: Total consecutive-missing segments in the raw input.
        n_short_gaps_filled: Segments at or below ``short_gap_max_rows``
            that were filled by interpolation.
        n_long_gaps_preserved: Segments above the threshold, kept as NaN.
        longest_gap_rows: Length (in raw rows) of the single longest gap.
        rows_with_residual_nan: Rows in the output still containing NaN
            after the policy (entirely from long outage gaps).
    """

    input_rows: int
    output_rows: int
    target_frequency: str
    n_total_gaps: int
    n_short_gaps_filled: int
    n_long_gaps_preserved: int
    longest_gap_rows: int
    rows_with_residual_nan: int


def preprocess(
    df: pd.DataFrame,
    *,
    config: "ForecastConfig",
) -> tuple[pd.DataFrame, PreprocessingReport]:
    """Run the full preprocessing pipeline using settings from ``config``.

    Args:
        df: Raw DataFrame from :func:`energy_forecasting.data.load_raw`.
        config: Source of truth for ``preprocessing`` and ``target`` settings.

    Returns:
        ``(processed_df, report)`` — the resampled DataFrame and a typed
        summary of what happened.
    """
    input_rows = len(df)

    # ── Diagnostics on the raw input ──────────────────────────
    gap_summary = summarize_missing_gaps(df)
    if gap_summary.empty:
        longest_gap = 0
        n_total_gaps = 0
    else:
        longest_gap = int(gap_summary["length"].max())
        n_total_gaps = len(gap_summary)

    # ── Stage 1: missing-value policy at native frequency ─────
    cleaned = apply_missing_value_policy(
        df,
        short_gap_max_rows=config.preprocessing.short_gap_max_rows,
        interpolation_method=config.preprocessing.interpolation_method,  # type: ignore[arg-type]
    )

    # The policy filled gaps up to short_gap_max_rows; everything longer is preserved.
    # We can reconstruct the counts directly from the original gap summary.
    n_short_gaps_filled = (
        int((gap_summary["length"] <= config.preprocessing.short_gap_max_rows).sum())
        if not gap_summary.empty
        else 0
    )
    n_long_gaps_preserved = n_total_gaps - n_short_gaps_filled

    # ── Stage 2: resample to the configured frequency ─────────
    processed = resample_to_frequency(
        cleaned,
        frequency=config.target.frequency,
        aggregation=config.target.aggregation,  # type: ignore[arg-type]
    )

    measurement_cols = [
        c for c in processed.columns if c not in ("is_outage_gap", "is_originally_missing")
    ]
    rows_with_residual_nan = int(processed[measurement_cols].isna().any(axis=1).sum())

    report = PreprocessingReport(
        input_rows=input_rows,
        output_rows=len(processed),
        target_frequency=config.target.frequency,
        n_total_gaps=n_total_gaps,
        n_short_gaps_filled=n_short_gaps_filled,
        n_long_gaps_preserved=n_long_gaps_preserved,
        longest_gap_rows=longest_gap,
        rows_with_residual_nan=rows_with_residual_nan,
    )

    _logger.info(
        f"Preprocessing complete: {input_rows:,} raw rows -> {len(processed):,} rows "
        f"at {config.target.frequency}, "
        f"{rows_with_residual_nan:,} rows still NaN (long outages)"
    )
    return processed, report
