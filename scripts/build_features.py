"""Rebuild data/processed/features.parquet from raw with the corrected
past-only imputation. Used after the missing.py fix that replaced bilateral
interpolation with ffill.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from energy_forecasting.config import DEFAULT_CONFIG_PATH, ForecastConfig
from energy_forecasting.data import load_raw
from energy_forecasting.features import build_features
from energy_forecasting.preprocessing import (
    apply_missing_value_policy,
    resample_to_frequency,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "processed" / "features.parquet"


def main() -> None:
    cfg = ForecastConfig.from_yaml(DEFAULT_CONFIG_PATH)

    df_raw = load_raw(ROOT / "data" / "raw" / "data.txt")
    print(f"Raw rows: {len(df_raw):,}")
    print(f"Rows with NaN: {df_raw.isna().any(axis=1).sum():,}")

    cleaned = apply_missing_value_policy(
        df_raw, short_gap_max_rows=3, interpolation_method="time"
    )
    print(f"After past-only ffill: {cleaned.isna().any(axis=1).sum():,} NaN rows remain "
          f"(only the long-gap segments, which stay NaN by policy)")

    hourly = resample_to_frequency(cleaned, frequency=cfg.target.frequency,
                                    aggregation="mean")
    print(f"Hourly rows: {len(hourly):,}")

    fm = build_features(hourly, config=cfg)
    out_df = pd.concat([fm.X, fm.y.rename(cfg.target.column)], axis=1)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(OUT, compression="snappy")
    print(f"Wrote {OUT}")
    print(f"Shape: {out_df.shape}")


if __name__ == "__main__":
    main()
