"""Developer-only: fit XGBoost on real fold 1, observe MAE vs naive_lag_1h."""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from energy_forecasting.config import DEFAULT_CONFIG_PATH, ForecastConfig
from energy_forecasting.evaluation import compute_metrics
from energy_forecasting.models import NaiveLagBaseline, XGBoostForecaster
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

    model = XGBoostForecaster()
    t0 = time.time()
    model.fit(Xt, yt)
    fit_s = time.time() - t0
    preds = model.predict(Xv)
    metrics_xgb = compute_metrics(yv.to_numpy(), preds,
                                  y_train=yt.to_numpy(), seasonal_period=24)
    print(f"XGBoost fit time: {fit_s:.1f}s\n")

    naive = NaiveLagBaseline(f"{target}_lag_1h", name="naive_lag_1h").fit(Xt, yt)
    metrics_naive = compute_metrics(yv.to_numpy(), naive.predict(Xv),
                                    y_train=yt.to_numpy(), seasonal_period=24)

    print("=== Fold 1 comparison ===")
    print(f"{'metric':<8} {'XGBoost':>10} {'naive_lag_1h':>14} {'delta':>10}")
    for name in ("MAE", "RMSE", "WAPE", "sMAPE", "MASE"):
        s = metrics_xgb[name]
        n = metrics_naive[name]
        print(f"{name:<8} {s:>10.4f} {n:>14.4f} {s - n:>+10.4f}")-=-0-9
    print("\nTop-10 feature importances (gain):")
    for name, gain in model.feature_importance().head(10).items():
        print(f"  {name:<40s}  {gain:.4f}")


if __name__ == "__main__":
    main()
