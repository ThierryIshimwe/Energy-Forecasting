"""Train XGBoost on the feature matrix augmented with 4 weather features.

Compares to the baseline XGBoost run (no weather). Saves:
    reports/results/xgboost_with_weather_fold_results.csv
    reports/results/xgboost_with_weather_predictions_fold{1..6}.csv
    reports/results/xgboost_with_weather_feature_importance.csv
    models/xgboost_with_weather_fold_{1..6}.joblib
    data/processed/features_with_weather.parquet  (the augmented matrix)
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from energy_forecasting.config import DEFAULT_CONFIG_PATH, ForecastConfig
from energy_forecasting.evaluation import compute_metrics
from energy_forecasting.models import XGBoostForecaster
from energy_forecasting.splits import make_rolling_origin_splits
from energy_forecasting.utils import save_model

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    cfg = ForecastConfig.from_yaml(DEFAULT_CONFIG_PATH)
    target = cfg.target.column

    # ── Load baseline features + weather; align by inner join ────────
    features = pd.read_parquet(ROOT / "data" / "processed" / "features.parquet")
    weather = pd.read_parquet(ROOT / "data" / "external" / "paris_montsouris_weather.parquet")
    print(f"Base features : {features.shape}")
    print(f"Weather       : {weather.shape}")

    combined = features.join(weather, how="inner")
    print(f"After join    : {combined.shape}")
    print(f"Date range    : {combined.index.min()} -> {combined.index.max()}")
    print(f"Weather cols added: {list(weather.columns)}")
    print()

    # Save the augmented feature matrix for downstream reuse
    out_features = ROOT / "data" / "processed" / "features_with_weather.parquet"
    combined.to_parquet(out_features, compression="snappy")
    print(f"Saved {out_features}")
    print()

    y = combined[target]
    X = combined.drop(columns=[target])
    folds = make_rolling_origin_splits(X.index, config=cfg)
    out_dir = ROOT / "reports" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    models_dir = ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    importances: list[pd.Series] = []
    t0_total = time.time()

    for fold in folds:
        Xt = X.loc[fold.train_start:fold.train_end].dropna()
        yt = y.loc[Xt.index]
        Xv = X.loc[fold.valid_start:fold.valid_end].dropna()
        yv = y.loc[Xv.index]

        model = XGBoostForecaster(name="xgb_with_weather")
        t0 = time.time()
        model.fit(Xt, yt)
        fit_s = time.time() - t0
        preds = model.predict(Xv)
        m = compute_metrics(yv.to_numpy(), preds,
                            y_train=yt.to_numpy(), seasonal_period=24)
        row = {"fold": fold.fold_id, "model": model.name,
               "valid_rows": len(yv), "fit_seconds": round(fit_s, 1),
               **{k: round(v, 4) for k, v in m.items()}}
        all_rows.append(row)
        importances.append(model.feature_importance())
        print(f"Fold {fold.fold_id}: train={len(Xt):,} valid={len(Xv):,} | "
              f"fit {fit_s:.1f}s | MAE={m['MAE']:.4f} RMSE={m['RMSE']:.4f} MASE={m['MASE']:.4f}")

        pd.DataFrame({
            "timestamp": Xv.index,
            "y_true": yv.to_numpy(),
            "y_pred": preds,
        }).to_csv(out_dir / f"xgboost_with_weather_predictions_fold{fold.fold_id}.csv", index=False)
        save_model(model, models_dir / f"xgboost_with_weather_fold_{fold.fold_id}.joblib")

    total = time.time() - t0_total
    results = pd.DataFrame(all_rows)
    results.to_csv(out_dir / "xgboost_with_weather_fold_results.csv", index=False)

    importance_mean = (
        pd.concat(importances, axis=1).mean(axis=1).sort_values(ascending=False)
    )
    importance_mean.name = "gain_mean"
    importance_mean.to_csv(out_dir / "xgboost_with_weather_feature_importance.csv")

    print(f"\nAll folds done in {total / 60:.1f} min")
    print("\n=== XGBoost with weather (cross-fold) ===")
    print(results[["fold", "valid_rows", "fit_seconds", "MAE", "RMSE", "WAPE", "MASE"]].to_string(index=False))
    print(f"\nMean MAE  : {results['MAE'].mean():.4f} +/- {results['MAE'].std():.4f}")
    print(f"Mean RMSE : {results['RMSE'].mean():.4f}")
    print(f"Mean MASE : {results['MASE'].mean():.4f}")

    # Compare to baseline XGBoost
    baseline = pd.read_csv(out_dir / "xgboost_fold_results.csv")
    print(f"\n=== Comparison to baseline XGBoost (no weather) ===")
    print(f"Baseline MAE mean : {baseline['MAE'].mean():.4f}")
    print(f"Weather  MAE mean : {results['MAE'].mean():.4f}")
    delta = results['MAE'].mean() - baseline['MAE'].mean()
    pct = delta / baseline['MAE'].mean() * 100
    print(f"Delta             : {delta:+.4f}  ({pct:+.2f}%)")

    print("\nTop 5 weather-feature gains:")
    weather_cols = ["temp_c", "rhum_pct", "wspd_kmh", "temp_lag_24h_c"]
    weather_importance = importance_mean[importance_mean.index.isin(weather_cols)]
    for name, gain in weather_importance.items():
        print(f"  {name:<22s}  {gain:.4f}")


if __name__ == "__main__":
    main()
