"""Preprocessing layer: missing-value handling and frequency resampling.

Public API:
    apply_missing_value_policy  — two-stage gap handling (interpolate short, mask long)
    summarize_missing_gaps      — report consecutive-NaN segments
    resample_to_frequency       — downsample numeric and indicator columns
    preprocess                  — config-driven full pipeline
    PreprocessingReport         — typed metadata returned alongside cleaned data

Design rule: this layer never *deletes* rows for outages. It marks them
with indicator columns so downstream feature engineering can decide whether
to mask, model, or learn from them.
"""

from energy_forecasting.preprocessing.missing import (
    apply_missing_value_policy,
    summarize_missing_gaps,
)
from energy_forecasting.preprocessing.pipeline import PreprocessingReport, preprocess
from energy_forecasting.preprocessing.resampling import resample_to_frequency

__all__ = [
    "PreprocessingReport",
    "apply_missing_value_policy",
    "preprocess",
    "resample_to_frequency",
    "summarize_missing_gaps",
]
