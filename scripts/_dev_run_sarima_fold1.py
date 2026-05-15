"""Developer-only: fit SARIMA on real fold 1, observe MAE vs baselines.

Run before integrating SARIMA into 03_modeling.ipynb — confirms fit time and
forecast quality are in the expected regime.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from energy_forecasting.config import DEFAULT_CONFIG_PATH, ForecastConfig
from energy_forecasting.evaluation import compute_metrics
from energy_forecasting.models import NaiveLagBaseline, SARIMAForecaster
from energy_forecasting.splits import make_rolling_origin_splits

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    cfg = ForecastConfig.from_yaml(DEFAULT_CONFIG_PATH)
    target = cfg.target.column

    df = pd.read_parquet(ROOT / "data" / "processed" / "features.parquet")
    y = df[target]
    X = df.drop(columns=[target])
    folds = make_rolling_origin_splits(X.index, config=cfg)
    f1 = folds[0]
    print(f"Fold 1: train {f1.train_start} -> {f1.train_end}  ({(f1.train_end - f1.train_start).days+1}d)")
    print(f"        valid {f1.valid_start} -> {f1.valid_end}  ({(f1.valid_end - f1.valid_start).days+1}d)")

    X_train = X.loc[f1.train_start:f1.train_end]
    y_train = y.loc[f1.train_start:f1.train_end]
    X_valid = X.loc[f1.valid_start:f1.valid_end]
    y_valid = y.loc[f1.valid_start:f1.valid_end]
    print(f"        train rows={len(X_train):,}  valid rows={len(X_valid):,}\n")

    # Fit + rolling 1-step predict
    model = SARIMAForecaster(order=(1, 1, 1), seasonal_order=(1, 1, 1, 24))
    t0 = time.time()
    model.fit(X_train, y_train)
    fit_seconds = time.time() - t0
    print(f"SARIMA fit time: {fit_seconds:.1f}s\n")

    t0 = time.time()
    preds_roll = model.predict_rolling_one_step(X_valid, y_valid)
    roll_seconds = time.time() - t0
    print(f"Rolling 1-step predict time: {roll_seconds:.1f}s")

    metrics_sarima = compute_metrics(
        y_valid.to_numpy(),
        preds_roll,
        y_train=y_train.to_numpy(),
        seasonal_period=24,
    )

    # Naive baseline for direct comparison
    naive = NaiveLagBaseline(f"{target}_lag_1h", name="naive_lag_1h").fit(X_train, y_train)
    metrics_naive = compute_metrics(
        y_valid.to_numpy(),
        naive.predict(X_valid),
        y_train=y_train.to_numpy(),
        seasonal_period=24,
    )

    print("\n=== Fold 1 comparison ===")
    print(f"{'metric':<8} {'SARIMA':>10} {'naive_lag_1h':>14} {'delta':>10}")
    for name in ("MAE", "RMSE", "WAPE", "sMAPE", "MASE"):
        s = metrics_sarima[name]
        n = metrics_naive[name]
        d = s - n
        print(f"{name:<8} {s:>10.4f} {n:>14.4f} {d:>+10.4f}")


if __name__ == "__main__":
    main()
