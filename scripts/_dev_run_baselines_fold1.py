"""Developer-only: run all three naive baselines on fold 1 of the real data
and print a small leaderboard. Sanity check before the real model lineup is
attempted in Phase 4.

Run via:  .venv/Scripts/python.exe scripts/_dev_run_baselines_fold1.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from energy_forecasting.config import DEFAULT_CONFIG_PATH, ForecastConfig
from energy_forecasting.evaluation import compute_metrics
from energy_forecasting.models import NaiveLagBaseline
from energy_forecasting.splits import make_rolling_origin_splits

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    cfg = ForecastConfig.from_yaml(DEFAULT_CONFIG_PATH)
    target = cfg.target.column

    features_path = ROOT / "data" / "processed" / "features.parquet"
    df = pd.read_parquet(features_path)
    y_all = df[target]
    X_all = df.drop(columns=[target])
    print(f"Loaded features  : {X_all.shape}")
    print(f"Loaded target    : {y_all.shape}")

    folds = make_rolling_origin_splits(X_all.index, config=cfg)

    # Run on each fold individually + aggregate. For now show fold 1 in detail,
    # then a summary line per fold.
    fold1 = folds[0]
    print(f"\nFold 1: train {fold1.train_start} -> {fold1.train_end}")
    print(f"        valid {fold1.valid_start} -> {fold1.valid_end}")

    X_train = X_all.loc[fold1.train_start:fold1.train_end]
    y_train = y_all.loc[fold1.train_start:fold1.train_end]
    X_valid = X_all.loc[fold1.valid_start:fold1.valid_end]
    y_valid = y_all.loc[fold1.valid_start:fold1.valid_end]
    print(f"        train rows={len(X_train):,}, valid rows={len(X_valid):,}")

    baselines = [
        NaiveLagBaseline(f"{target}_lag_1h",   name="naive_lag_1h"),
        NaiveLagBaseline(f"{target}_lag_24h",  name="naive_lag_24h_daily"),
        NaiveLagBaseline(f"{target}_lag_168h", name="naive_lag_168h_weekly"),
    ]

    rows_fold1 = []
    for m in baselines:
        m.fit(X_train, y_train)
        y_pred = m.predict(X_valid)
        metrics = compute_metrics(
            y_valid.to_numpy(),
            y_pred,
            y_train=y_train.to_numpy(),
            seasonal_period=24,
        )
        rows_fold1.append({"model": m.name, **{k: round(v, 4) for k, v in metrics.items()}})

    leaderboard = pd.DataFrame(rows_fold1).sort_values("MAE")
    print("\nFold-1 baseline leaderboard (sorted by MAE):")
    print(leaderboard.to_string(index=False))

    # Aggregate across all 6 folds — show mean ± std per baseline
    print("\n--- All 6 folds: mean +/- std of each metric per baseline ---")
    all_rows: list[dict[str, object]] = []
    for f in folds:
        Xt = X_all.loc[f.train_start:f.train_end]
        yt = y_all.loc[f.train_start:f.train_end]
        Xv = X_all.loc[f.valid_start:f.valid_end]
        yv = y_all.loc[f.valid_start:f.valid_end]
        for m in baselines:
            m.fit(Xt, yt)
            y_pred = m.predict(Xv)
            mets = compute_metrics(yv.to_numpy(), y_pred,
                                   y_train=yt.to_numpy(), seasonal_period=24)
            all_rows.append({"fold": f.fold_id, "model": m.name, **mets,
                             "valid_rows": len(yv)})

    by_fold = pd.DataFrame(all_rows)
    summary = by_fold.groupby("model").agg(
        MAE_mean=("MAE", "mean"), MAE_std=("MAE", "std"),
        RMSE_mean=("RMSE", "mean"), RMSE_std=("RMSE", "std"),
        WAPE_mean=("WAPE", "mean"),
        MASE_mean=("MASE", "mean"),
    ).round(4).sort_values("MAE_mean")
    print(summary.to_string())

    print("\n--- Per-fold MAE for each baseline (so we can see Aug-outage effect) ---")
    pivot = by_fold.pivot(index="model", columns="fold", values="MAE").round(3)
    pivot.columns = [f"fold{c}" for c in pivot.columns]
    pivot["valid_rows_avg"] = by_fold.groupby("model")["valid_rows"].mean().astype(int)
    print(pivot.to_string())


if __name__ == "__main__":
    main()
