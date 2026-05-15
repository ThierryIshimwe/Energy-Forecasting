"""Developer-only: fit SARIMA on every fold and save per-fold results.

Expected wall-clock: ~60-70 min total (~11 min per fold). Run in background
while other models are being built. Outputs:

    reports/results/sarima_fold_results.csv
    reports/results/sarima_predictions_fold1.csv  (for the AvP plot later)
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

        # Save fold-1 predictions for the AvP plot later
        if fold.fold_id == 1:
            preds_df = pd.DataFrame({
                "timestamp": Xv.index,
                "y_true": yv.to_numpy(),
                "y_pred": preds,
            })
            preds_df.to_csv(out_dir / "sarima_predictions_fold1.csv", index=False)

    total = time.time() - start_time
    results = pd.DataFrame(all_rows)
    results.to_csv(out_dir / "sarima_fold_results.csv", index=False)
    print(f"\n=== ALL FOLDS DONE in {total / 60:.1f} min ===")
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
