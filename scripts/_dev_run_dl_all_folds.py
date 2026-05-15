"""Developer-only: fit LSTM and GRU on every fold and save artifacts.

Expected wall-clock: ~45 min total (~4 min per fit × 6 folds × 2 cell types).
Outputs (for each cell type):
    reports/results/<cell>_fold_results.csv
    reports/results/<cell>_predictions_fold{N}.csv  for N = 1..6
    models/<cell>_default_fold_{N}.joblib            for N = 1..6
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from energy_forecasting.config import DEFAULT_CONFIG_PATH, ForecastConfig
from energy_forecasting.evaluation import compute_metrics
from energy_forecasting.models import LSTMForecaster
from energy_forecasting.splits import make_rolling_origin_splits
from energy_forecasting.utils import save_model

ROOT = Path(__file__).resolve().parents[1]


def run_cell_type(
    cell_type: str,
    folds,
    X: pd.DataFrame,
    y: pd.Series,
    out_dir: Path,
    models_dir: Path,
) -> pd.DataFrame:
    """Train one DL variant across all folds, save everything, return results table."""
    rows: list[dict] = []
    for fold in folds:
        Xt = X.loc[fold.train_start:fold.train_end]
        yt = y.loc[fold.train_start:fold.train_end]
        Xv = X.loc[fold.valid_start:fold.valid_end]
        yv = y.loc[fold.valid_start:fold.valid_end]

        model = LSTMForecaster(cell_type=cell_type, name=f"{cell_type}_default")
        t0 = time.time()
        model.fit(Xt, yt)
        fit_s = time.time() - t0
        preds = model.predict_rolling_one_step(Xv, yv)
        m = compute_metrics(yv.to_numpy(), preds,
                            y_train=yt.to_numpy(), seasonal_period=24)
        rows.append({
            "fold": fold.fold_id,
            "model": model.name,
            "valid_rows": len(yv),
            "fit_seconds": round(fit_s, 1),
            **{k: round(v, 4) for k, v in m.items()},
        })
        print(f"  {cell_type} fold {fold.fold_id}: fit {fit_s:.0f}s | "
              f"MAE={m['MAE']:.4f}  RMSE={m['RMSE']:.4f}  MASE={m['MASE']:.4f}")

        pd.DataFrame({
            "timestamp": Xv.index,
            "y_true": yv.to_numpy(),
            "y_pred": preds,
        }).to_csv(out_dir / f"{cell_type}_predictions_fold{fold.fold_id}.csv", index=False)
        save_model(model, models_dir / f"{cell_type}_default_fold_{fold.fold_id}.joblib")

    return pd.DataFrame(rows)


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

    overall_start = time.time()

    print("=== LSTM all folds ===")
    lstm_results = run_cell_type("lstm", folds, X, y, out_dir, models_dir)
    lstm_results.to_csv(out_dir / "lstm_fold_results.csv", index=False)

    print("\n=== GRU all folds ===")
    gru_results = run_cell_type("gru", folds, X, y, out_dir, models_dir)
    gru_results.to_csv(out_dir / "gru_fold_results.csv", index=False)

    total_min = (time.time() - overall_start) / 60
    print(f"\nALL DL DONE in {total_min:.1f} min")
    print("\n=== LSTM summary ===")
    print(lstm_results.to_string(index=False))
    print(f"\nLSTM Mean MAE  : {lstm_results['MAE'].mean():.4f} +/- {lstm_results['MAE'].std():.4f}")
    print("\n=== GRU summary ===")
    print(gru_results.to_string(index=False))
    print(f"\nGRU  Mean MAE  : {gru_results['MAE'].mean():.4f} +/- {gru_results['MAE'].std():.4f}")


if __name__ == "__main__":
    main()
