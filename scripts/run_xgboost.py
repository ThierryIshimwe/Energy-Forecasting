"""Developer-only: fit XGBoost on every fold and save per-fold results.

Expected wall-clock: ~2 min total (~20s per fold). Outputs:
    reports/results/xgboost_fold_results.csv
    reports/results/xgboost_predictions_fold1.csv
    reports/results/xgboost_feature_importance.csv
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

    df = pd.read_parquet(ROOT / "data" / "processed" / "features.parquet")
    y = df[target]
    X = df.drop(columns=[target])
    folds = make_rolling_origin_splits(X.index, config=cfg)

    out_dir = ROOT / "reports" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    importances_per_fold: list[pd.Series] = []

    total_start = time.time()
    for fold in folds:
        Xt = X.loc[fold.train_start:fold.train_end]
        yt = y.loc[fold.train_start:fold.train_end]
        Xv = X.loc[fold.valid_start:fold.valid_end]
        yv = y.loc[fold.valid_start:fold.valid_end]

        model = XGBoostForecaster(name="xgb_default")
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
        importances_per_fold.append(model.feature_importance())
        print(f"Fold {fold.fold_id}: fit {fit_s:.1f}s | "
              f"MAE={m['MAE']:.4f} RMSE={m['RMSE']:.4f} MASE={m['MASE']:.4f}")

        # Save per-fold predictions (all 6 folds, not just fold 1)
        pd.DataFrame({
            "timestamp": Xv.index,
            "y_true": yv.to_numpy(),
            "y_pred": preds,
        }).to_csv(out_dir / f"xgboost_predictions_fold{fold.fold_id}.csv", index=False)

        # Save trained model artifact for future inference / ensembling
        save_model(model, ROOT / "models" / f"xgboost_default_fold_{fold.fold_id}.joblib")

    total = time.time() - total_start
    results = pd.DataFrame(all_rows)
    results.to_csv(out_dir / "xgboost_fold_results.csv", index=False)

    # Aggregate feature importance: mean across folds
    importance_mean = pd.concat(importances_per_fold, axis=1).mean(axis=1).sort_values(
        ascending=False
    )
    importance_mean.name = "gain_mean"
    importance_mean.to_csv(out_dir / "xgboost_feature_importance.csv")

    print(f"\nAll folds done in {total / 60:.1f} min")
    print("\n=== Per-fold summary ===")
    print(results[["fold", "valid_rows", "fit_seconds", "MAE", "RMSE", "WAPE", "MASE"]].to_string(index=False))
    print(f"\nMean MAE across folds : {results['MAE'].mean():.4f}")
    print(f"Std  MAE across folds : {results['MAE'].std():.4f}")
    print(f"Mean RMSE across folds: {results['RMSE'].mean():.4f}")
    print(f"Mean MASE             : {results['MASE'].mean():.4f}")


if __name__ == "__main__":
    main()
