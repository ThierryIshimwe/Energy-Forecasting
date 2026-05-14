"""Shared pytest fixtures and configuration."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from energy_forecasting.config import DEFAULT_CONFIG_PATH, ForecastConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "data.txt"


@pytest.fixture(scope="session")
def default_config() -> ForecastConfig:
    """The pinned default ForecastConfig used by every test that needs one."""
    return ForecastConfig.from_yaml(DEFAULT_CONFIG_PATH)


@pytest.fixture(scope="session")
def synthetic_hourly_df() -> pd.DataFrame:
    """A small, deterministic hourly DataFrame matching the post-resample schema.

    Covers one full year so weekly + yearly seasonality features can warm up
    correctly. The target is generated with a known daily pattern + noise so
    tests can sanity-check feature builders without needing the real dataset.
    """
    rng = np.random.default_rng(seed=42)
    idx = pd.date_range("2008-01-01", "2008-12-31 23:00", freq="1h")
    # Daily sinusoid + small noise, scaled to realistic household kW values.
    hour = idx.hour.to_numpy()
    base = 1.0 + 0.8 * np.sin(2 * np.pi * (hour - 6) / 24)
    target = base + rng.normal(0.0, 0.1, size=len(idx))
    target = np.clip(target, 0.05, None)

    df = pd.DataFrame(
        {
            "Global_active_power": target.astype("float64"),
            # Include the unsafe columns to verify they get dropped by the registry.
            "Global_reactive_power": rng.uniform(0.0, 0.3, len(idx)).astype("float64"),
            "Voltage": rng.uniform(232.0, 240.0, len(idx)).astype("float64"),
            "Global_intensity": (target * 4.5).astype("float64"),
            "Sub_metering_1": rng.uniform(0.0, 3.0, len(idx)).astype("float64"),
            "Sub_metering_2": rng.uniform(0.0, 3.0, len(idx)).astype("float64"),
            "Sub_metering_3": rng.uniform(0.0, 15.0, len(idx)).astype("float64"),
            "is_outage_gap": np.zeros(len(idx), dtype="int8"),
            "is_originally_missing": np.zeros(len(idx), dtype="int8"),
        },
        index=idx,
    )
    return df


@pytest.fixture(scope="session")
def real_raw_path() -> Path:
    """Path to the real UCI dataset. Tests using this should skip if absent."""
    if not RAW_DATA_PATH.is_file():
        pytest.skip(
            f"Real UCI dataset not present at {RAW_DATA_PATH}. "
            "Run `python scripts/download_data.py` first."
        )
    return RAW_DATA_PATH
