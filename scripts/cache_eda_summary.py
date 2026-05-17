"""Cache the EDA summary numbers and the three seasonality plot series.

The deliverable notebook (00_main_deliverable.ipynb) needs to display EDA
overview numbers and seasonality plots without requiring the evaluator to
re-download the 127 MB UCI source. This script runs the raw-data computations
once and writes small tracked CSVs / JSON that the notebook can load directly.

Outputs (all tracked in git):
    reports/results/eda_summary.json
    reports/results/eda_seasonality_weekly.csv
    reports/results/eda_seasonality_june2008_daily.csv
    reports/results/eda_seasonality_nov2007_hourly.csv
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from energy_forecasting.config import DEFAULT_CONFIG_PATH, ForecastConfig
from energy_forecasting.data import load_raw
from energy_forecasting.preprocessing import summarize_missing_gaps

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "results"


def main() -> None:
    cfg = ForecastConfig.from_yaml(DEFAULT_CONFIG_PATH)
    target = cfg.target.column

    df_raw = load_raw(ROOT / "data" / "raw" / "data.txt")
    n_total = len(df_raw)
    n_missing = int(df_raw.isna().any(axis=1).sum())
    gaps = summarize_missing_gaps(df_raw)
    longest = gaps.loc[gaps["length"].idxmax()]

    summary = {
        "n_total_rows": n_total,
        "n_missing_rows": n_missing,
        "missing_pct": round(n_missing / n_total * 100, 4),
        "n_gap_segments": int(len(gaps)),
        "n_short_gaps": int((gaps["length"] <= 3).sum()),
        "n_long_gaps": int((gaps["length"] > 3).sum()),
        "longest_gap_minutes": int(longest["length"]),
        "longest_gap_start": str(longest["start"]),
        "longest_gap_end": str(longest["end"]),
    }
    (OUT / "eda_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Wrote {OUT / 'eda_summary.json'}")
    print(json.dumps(summary, indent=2))

    s = df_raw[target]
    weekly = s.resample("W").mean().rename("value").to_frame()
    weekly.index.name = "datetime"
    weekly.to_csv(OUT / "eda_seasonality_weekly.csv")
    print(f"Wrote {OUT / 'eda_seasonality_weekly.csv'}  ({len(weekly)} rows)")

    june = s.loc["2008-06-01":"2008-06-30"].resample("D").mean().rename("value").to_frame()
    june.index.name = "datetime"
    june.to_csv(OUT / "eda_seasonality_june2008_daily.csv")
    print(f"Wrote {OUT / 'eda_seasonality_june2008_daily.csv'}  ({len(june)} rows)")

    nov = s.loc["2007-11-05":"2007-11-11"].resample("h").mean().rename("value").to_frame()
    nov.index.name = "datetime"
    nov.to_csv(OUT / "eda_seasonality_nov2007_hourly.csv")
    print(f"Wrote {OUT / 'eda_seasonality_nov2007_hourly.csv'}  ({len(nov)} rows)")


if __name__ == "__main__":
    main()
