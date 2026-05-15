"""Lightweight save/load for fitted forecaster instances.

We use ``joblib`` for everything because it handles every model type our
``BaseForecaster`` subclasses produce on CPU:

- ``XGBoostForecaster`` — joblib's native target.
- ``SARIMAForecaster`` — statsmodels results are picklable.
- ``LSTMForecaster`` (PyTorch) — CPU tensors pickle cleanly.
- ``ProphetForecaster`` — the cmdstanpy backend leaves a serializable state.

Compression (``compress=3``) keeps the on-disk footprint small without the
slow gzip levels. The dispatch is intentionally trivial — we don't want
clever per-model serialization until a real failure mode forces it.

This module is imported by training scripts and (later) by the Streamlit
app loader. It is *not* a model registry or version manager; that's a
Phase 5 concern (MLflow / model cards).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import joblib

from energy_forecasting.utils.logging import get_logger

if TYPE_CHECKING:  # Avoid circular import: models -> utils -> models.
    from energy_forecasting.models.base import BaseForecaster

_logger = get_logger(__name__)


def save_model(model: "BaseForecaster", path: Path | str) -> Path:
    """Persist ``model`` to ``path`` using joblib.

    The destination directory is created if missing. Returns the resolved
    path of the written file.
    """
    p = Path(path).resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, p, compress=3)
    size_kb = p.stat().st_size / 1024
    _logger.info(f"Saved {getattr(model, 'name', type(model).__name__)} to {p} ({size_kb:.1f} KB)")
    return p


def load_model(path: Path | str) -> "BaseForecaster":
    """Load a forecaster previously saved with :func:`save_model`.

    Raises:
        FileNotFoundError: ``path`` does not exist.
    """
    p = Path(path).resolve()
    if not p.is_file():
        raise FileNotFoundError(f"Model file not found: {p}")
    model = joblib.load(p)
    _logger.info(f"Loaded {getattr(model, 'name', type(model).__name__)} from {p}")
    return model
