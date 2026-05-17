"""Rebuild reports/results/leaderboard.csv from the per-model fold-result CSVs.

Reads the per-fold metric files written by each `run_*.py` script and aggregates
them into one row per model with mean ± std across the 6 rolling-origin folds.
The deliverable notebook's leaderboard cell loads this output.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "reports" / "results"

FOLD_FILES = [
    "baseline_fold_results.csv",
    "xgboost_fold_results.csv",
    "sarima_fold_results.csv",
    "lstm_fold_results.csv",
    "gru_fold_results.csv",
]

METRIC_COLS = ["MAE", "RMSE", "WAPE", "sMAPE", "MASE"]


def main() -> None:
    frames = []
    for f in FOLD_FILES:
        p = R / f
        if p.exists():
            frames.append(pd.read_csv(p))
            print(f"  Loaded {p.name}: {len(frames[-1])} rows")
        else:
            print(f"  Missing {p.name}")
    combined = pd.concat(frames, ignore_index=True)
    agg = combined.groupby("model")[METRIC_COLS].agg(["mean", "std"]).round(4)
    agg.columns = ["_".join(c) for c in agg.columns]
    agg = agg.sort_values("MAE_mean").reset_index()
    out = R / "leaderboard.csv"
    agg.to_csv(out, index=False)
    print(f"\nWrote {out}")
    print(agg.to_string(index=False))


if __name__ == "__main__":
    main()
