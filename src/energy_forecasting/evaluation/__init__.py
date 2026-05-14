"""Evaluation utilities: metrics, segments, statistical tests.

Public API (this submodule is grown incrementally):
    mae, rmse, wape, mase, smape       — five forecasting metrics
    compute_metrics                    — bundle: returns all five at once
    METRIC_FUNCTIONS                   — name → function lookup
"""

from energy_forecasting.evaluation.metrics import (
    METRIC_FUNCTIONS,
    compute_metrics,
    mae,
    mase,
    rmse,
    smape,
    wape,
)

__all__ = [
    "METRIC_FUNCTIONS",
    "compute_metrics",
    "mae",
    "mase",
    "rmse",
    "smape",
    "wape",
]
