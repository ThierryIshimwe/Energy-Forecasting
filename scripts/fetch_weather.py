"""Fetch hourly weather for Paris-Montsouris (5 km from Sceaux) and save to disk.

Variables retained: temp (C), rhum (%), wspd (km/h), plus temp_lag_24h.
Dew point and pressure were tested but dropped — they introduced redundancy
with temp+rhum and slightly degraded XGBoost performance (cross-fold MAE
0.3339 with 6 features vs 0.3302 with 4 features).

Output is saved as a parquet for fast reload.

Source: meteostat (NOAA ISD + Météo-France).
Station: Paris-Montsouris (id=07156), 48.82N 2.34E, 5.2 km from Sceaux (48.78N 2.29E).
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
from meteostat import Hourly

ROOT = Path(__file__).resolve().parents[1]
OUT_PARQUET = ROOT / "data" / "external" / "paris_montsouris_weather.parquet"
OUT_CSV = ROOT / "data" / "external" / "paris_montsouris_weather.csv"

STATION_ID = "07156"  # Paris-Montsouris
STATION_NAME = "Paris-Montsouris"
START = datetime(2006, 12, 16, 17)
END = datetime(2010, 11, 26, 22)


def main() -> None:
    print(f"Fetching hourly weather from {STATION_NAME} (id={STATION_ID})")
    print(f"Range: {START} to {END}")
    weather = Hourly(STATION_ID, START, END).fetch()

    # Keep the three variables that survived feature-variant testing
    keep = ["temp", "rhum", "wspd"]
    weather = weather[keep].copy()
    weather.columns = ["temp_c", "rhum_pct", "wspd_kmh"]
    weather.index.name = "datetime"

    # Add lag-24h temperature (strictly leakage-safe — yesterday's observed value)
    weather["temp_lag_24h_c"] = weather["temp_c"].shift(24)

    print(f"\nShape: {weather.shape}")
    print(f"Date range: {weather.index.min()} -> {weather.index.max()}")
    print("\nCoverage (non-null counts):")
    print(weather.notna().sum().to_string())
    print("\nSummary:")
    print(weather.describe().round(2).to_string())

    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    weather.to_parquet(OUT_PARQUET, compression="snappy")
    weather.to_csv(OUT_CSV)
    print(f"\nSaved {OUT_PARQUET}")
    print(f"Saved {OUT_CSV}")


if __name__ == "__main__":
    main()
