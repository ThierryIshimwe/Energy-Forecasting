"""Developer-only: fit LSTM (cell_type='lstm') on real fold 1, observe MAE.

Run via:  .venv/Scripts/python.exe scripts/_dev_run_lstm_fold1.py
Expected wall-clock: 1-3 min on CPU with default hyperparameters.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from energy_forecasting.config import DEFAULT_CONFIG_PATH, ForecastConfig
from energy_forecasting.evaluation import compute_metrics
from energy_forecasting.models import LSTMForecaster, NaiveLagBaseline
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

    Xt = X.loc[f1.train_start:f1.train_end]
    yt = y.loc[f1.train_start:f1.train_end]
    Xv = X.loc[f1.valid_start:f1.valid_end]
    yv = y.loc[f1.valid_start:f1.valid_end]
    print(f"Fold 1: train rows={len(Xt):,}  valid rows={len(Xv):,}\n")

    model = LSTMForecaster(cell_type="lstm")  # defaults: lookback=168, hidden=64, epochs=25
    t0 = time.time()
    model.fit(Xt, yt)
    fit_s = time.time() - t0

    t0 = time.time()
    preds = model.predict_rolling_one_step(Xv, yv)
    predict_s = time.time() - t0

    metrics_lstm = compute_metrics(yv.to_numpy(), preds,
                                   y_train=yt.to_numpy(), seasonal_period=24)
    print(f"\nLSTM fit time     : {fit_s:.1f}s")
    print(f"LSTM predict time : {predict_s:.1f}s\n")

    # naive_lag_1h floor
    naive = NaiveLagBaseline(f"{target}_lag_1h", name="naive_lag_1h").fit(Xt, yt)
    metrics_naive = compute_metrics(yv.to_numpy(), naive.predict(Xv),
                                    y_train=yt.to_numpy(), seasonal_period=24)

    print("=== Fold 1 comparison ===")
    print(f"{'metric':<8} {'LSTM':>10} {'naive_lag_1h':>14} {'delta':>10}")
    for name in ("MAE", "RMSE", "WAPE", "sMAPE", "MASE"):
        s = metrics_lstm[name]
        n = metrics_naive[name]
        print(f"{name:<8} {s:>10.4f} {n:>14.4f} {s - n:>+10.4f}")


if __name__ == "__main__":
    main()
