"""Developer-only: fit SARIMA on every fold and save per-fold results.

Expected wall-clock: ~70-90 min total (~10-15 min per fold without CPU contention).
Outputs:

    reports/results/sarima_fold_results.csv
    reports/results/sarima_predictions_fold{1..6}.csv
    models/sarima_default_fold_{1..6}.joblib
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from energy_forecasting.config import DEFAULT_CONFIG_PATH, ForecastConfig
from energy_forecasting.evaluation import compute_metrics
from energy_forecasting.models import SARIMAForecaster
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
    start_time = time.time()
    for fold in folds:
        Xt = X.loc[fold.train_start:fold.train_end]
        yt = y.loc[fold.train_start:fold.train_end]
        Xv = X.loc[fold.valid_start:fold.valid_end]
        yv = y.loc[fold.valid_start:fold.valid_end]
        print(f"\n=== Fold {fold.fold_id}: train={len(Xt):,}  valid={len(Xv):,} ===")

        model = SARIMAForecaster(order=(1, 1, 1), seasonal_order=(1, 1, 1, 24),
                                  name="sarima_111_111_24")
        t0 = time.time()
        model.fit(Xt, yt)
        fit_s = time.time() - t0
        preds = model.predict_rolling_one_step(Xv, yv)
        m = compute_metrics(yv.to_numpy(), preds,
                            y_train=yt.to_numpy(), seasonal_period=24)
        row = {"fold": fold.fold_id, "model": model.name,
               "valid_rows": len(yv), "fit_seconds": round(fit_s, 1),
               **{k: round(v, 4) for k, v in m.items()}}
        all_rows.append(row)
        print(f"    fit {fit_s:.1f}s | MAE={m['MAE']:.4f} | RMSE={m['RMSE']:.4f}")

        # Save per-fold predictions (all 6 folds, not just fold 1)
        pd.DataFrame({
            "timestamp": Xv.index,
            "y_true": yv.to_numpy(),
            "y_pred": preds,
        }).to_csv(out_dir / f"sarima_predictions_fold{fold.fold_id}.csv", index=False)

        # Strip the cached Kalman state before saving. statsmodels caches every
        # filtered/smoothed state and covariance from training inside the result
        # object, which blows the pickled size from ~50 KB to ~1.6 GB per fold.
        # The cached state is not needed for future .predict() / .forecast() on
        # new data — those use the fitted params + model spec, both of which
        # remove_data() preserves.
        model._results.remove_data()
        save_model(model, ROOT / "models" / f"sarima_default_fold_{fold.fold_id}.joblib")

    total = time.time() - start_time
    results = pd.DataFrame(all_rows)
    results.to_csv(out_dir / "sarima_fold_results.csv", index=False)
    print(f"\n=== ALL FOLDS DONE in {total / 60:.1f} min ===")
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
