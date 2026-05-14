"""Developer verification: load the real feature matrix, build the rolling-origin
split plan, save it to disk, and produce a timeline visualization.

Run via:  .venv/Scripts/python.exe scripts/_dev_visualize_splits.py

Outputs:
  data/processed/split_plan.csv          — the canonical split plan
  reports/figures/split_plan_timeline.png — Gantt-style visualization of the 6 folds

Idempotent: re-running overwrites both outputs.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd

from energy_forecasting.config import DEFAULT_CONFIG_PATH, ForecastConfig
from energy_forecasting.splits import make_rolling_origin_splits, splits_to_dataframe

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    cfg = ForecastConfig.from_yaml(DEFAULT_CONFIG_PATH)

    # 1. Load the real feature matrix (just the index — we don't need values for splits).
    features_path = ROOT / "data" / "processed" / "features.parquet"
    df = pd.read_parquet(features_path, columns=[cfg.target.column])
    print(f"Loaded {len(df):,} rows  ({df.index.min()} -> {df.index.max()})\n")

    # 2. Build folds.
    folds = make_rolling_origin_splits(df.index, config=cfg)
    plan = splits_to_dataframe(folds, index=df.index)
    print("Split plan:")
    print(plan.to_string(index=False))

    # 3. Persist.
    out_csv = ROOT / "data" / "processed" / "split_plan.csv"
    plan.to_csv(out_csv, index=False)
    print(f"\nSaved {out_csv}")

    # 4. Visualize: one row per fold, train bar + validation bar across calendar time.
    fig, ax = plt.subplots(figsize=(13, 5), constrained_layout=True)

    train_color = "#4C78A8"
    valid_color = "#F58518"

    for i, f in enumerate(folds):
        y = len(folds) - i  # plot fold 1 at top
        train_w = (f.train_end - f.train_start).days + 1
        valid_w = (f.valid_end - f.valid_start).days + 1
        ax.barh(y, train_w, left=f.train_start, height=0.6, color=train_color, edgecolor="black", linewidth=0.4)
        ax.barh(y, valid_w, left=f.valid_start, height=0.6, color=valid_color, edgecolor="black", linewidth=0.4)
        ax.text(
            f.train_start + (f.train_end - f.train_start) / 2,
            y,
            f"train {train_w}d",
            ha="center", va="center", fontsize=8, color="white", fontweight="bold",
        )
        ax.text(
            f.valid_start + (f.valid_end - f.valid_start) / 2,
            y,
            f"valid {valid_w}d",
            ha="center", va="center", fontsize=8, color="black", fontweight="bold",
        )

    ax.set_yticks(range(1, len(folds) + 1))
    ax.set_yticklabels([f"fold {len(folds) - i}" for i in range(len(folds))])
    ax.set_xlabel("Calendar time")
    ax.set_title(
        f"Rolling-origin split plan  —  {cfg.splits.n_folds} folds × "
        f"({cfg.splits.train_window_days}d train + {cfg.splits.valid_window_days}d valid"
        + (f", {cfg.splits.gap_days}d gap" if cfg.splits.gap_days else "")
        + ")"
    )
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
    ax.grid(True, axis="x", alpha=0.3)
    ax.set_xlim(df.index.min() - pd.Timedelta(days=15), df.index.max() + pd.Timedelta(days=15))

    legend_handles = [
        mpatches.Patch(color=train_color, label="Training window"),
        mpatches.Patch(color=valid_color, label="Validation window"),
    ]
    ax.legend(handles=legend_handles, loc="lower right")

    out_png = ROOT / "reports" / "figures" / "split_plan_timeline.png"
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=140)
    print(f"Saved {out_png}")


if __name__ == "__main__":
    main()
