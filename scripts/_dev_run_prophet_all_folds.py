"""Developer-only: fit Prophet on every fold and save per-fold results.

Expected wall-clock: ~3-5 min total. Outputs:

    reports/results/prophet_fold_results.csv
    reports/results/prophet_predictions_fold{1..6}.csv
    models/prophet_default_fold_{1..6}.joblib
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from energy_forecasting.config import DEFAULT_CONFIG_PATH, ForecastConfig
from energy_forecasting.evaluation import compute_metrics
from energy_forecasting.models import ProphetForecaster
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
    models_dir = ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    start_time = time.time()
    for fold in folds:
        Xt = X.loc[fold.train_start:fold.train_end]
        yt = y.loc[fold.train_start:fold.train_end]
        Xv = X.loc[fold.valid_start:fold.valid_end]
        yv = y.loc[fold.valid_start:fold.valid_end]

        model = ProphetForecaster(name="prophet_default")
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
        print(f"Fold {fold.fold_id}: fit {fit_s:.1f}s | "
              f"MAE={m['MAE']:.4f} RMSE={m['RMSE']:.4f} MASE={m['MASE']:.4f}")

        pd.DataFrame({
            "timestamp": Xv.index,
            "y_true": yv.to_numpy(),
            "y_pred": preds,
        }).to_csv(out_dir / f"prophet_predictions_fold{fold.fold_id}.csv", index=False)

        save_model(model, models_dir / f"prophet_default_fold_{fold.fold_id}.joblib")

    total = time.time() - start_time
    results = pd.DataFrame(all_rows)
    results.to_csv(out_dir / "prophet_fold_results.csv", index=False)
    print(f"\nALL FOLDS DONE in {total / 60:.1f} min")
    print(results.to_string(index=False))
    print(f"\nMean MAE  : {results['MAE'].mean():.4f}")
    print(f"Std  MAE  : {results['MAE'].std():.4f}")


if __name__ == "__main__":
    main()
