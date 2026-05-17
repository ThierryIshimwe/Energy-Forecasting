"""Developer-only: stacking ensemble over the saved per-fold predictions.

How it works
------------
For every (fold, timestamp), each base model produced a prediction during
its rolling-origin evaluation — those predictions are saved at
``reports/results/<model>_predictions_fold{1..6}.csv``. Concatenated, they
form a (n_rows × n_models) matrix of *out-of-fold* predictions where each
row's predictions were made by models that did NOT train on that row.

We fit a small Ridge meta-learner that maps (per-row base predictions) ->
(true target value). The meta-learner is trained on folds 1..5 and
evaluated on fold 6 (held out). Predictions are clipped at 0 because the
target is non-negative.

Outputs:
    reports/results/stacking_predictions_fold6.csv
    reports/results/stacking_summary.csv  (includes the meta-learner coefs)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from energy_forecasting.evaluation import compute_metrics

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "reports" / "results"


def load_predictions(model_slug: str) -> pd.DataFrame:
    """Load per-fold predictions for one model, return a long DataFrame."""
    frames = []
    for fold_id in range(1, 7):
        p = RESULTS_DIR / f"{model_slug}_predictions_fold{fold_id}.csv"
        if not p.exists():
            print(f"  missing: {p.name} — skipping")
            return pd.DataFrame()
        df = pd.read_csv(p, parse_dates=["timestamp"])
        df["fold"] = fold_id
        df["model"] = model_slug
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    base_models = ["xgboost", "sarima", "lstm", "gru"]
    print("Loading base-model per-fold predictions:")
    long_frames = []
    for slug in base_models:
        df = load_predictions(slug)
        if df.empty:
            print(f"  {slug:<10} (missing folds) — excluded from ensemble")
            continue
        long_frames.append(df)
        print(f"  {slug:<10} loaded ({len(df):,} rows)")

    if not long_frames:
        print("\nNo base-model predictions found; exiting.")
        return

    long_df = pd.concat(long_frames, ignore_index=True)

    # Pivot to (timestamp, fold) × model
    wide = long_df.pivot_table(
        index=["timestamp", "fold"],
        columns="model",
        values="y_pred",
    ).reset_index()
    wide = wide.merge(
        long_df[["timestamp", "fold", "y_true"]].drop_duplicates(),
        on=["timestamp", "fold"],
    )
    model_cols = [c for c in wide.columns if c not in ("timestamp", "fold", "y_true")]
    wide = wide.dropna(subset=model_cols + ["y_true"]).reset_index(drop=True)

    print(f"\nStacking matrix: {len(wide):,} rows × {len(model_cols)} base models "
          f"({', '.join(model_cols)})")

    # Train on folds 1-5, hold out fold 6
    train_mask = wide["fold"] < 6
    test_mask = wide["fold"] == 6
    print(f"Train (folds 1-5): {int(train_mask.sum()):,} rows")
    print(f"Test  (fold   6 ): {int(test_mask.sum()):,} rows\n")

    X_train = wide.loc[train_mask, model_cols].to_numpy()
    y_train = wide.loc[train_mask, "y_true"].to_numpy()
    X_test = wide.loc[test_mask, model_cols].to_numpy()
    y_test = wide.loc[test_mask, "y_true"].to_numpy()

    meta = Ridge(alpha=1.0, positive=False, fit_intercept=True, random_state=42)
    meta.fit(X_train, y_train)
    stack_preds = np.clip(meta.predict(X_test), a_min=0, a_max=None)

    print("Ridge meta-learner coefficients:")
    for name, w in zip(model_cols, meta.coef_):
        print(f"  {name:<10} weight = {w:+.4f}")
    print(f"  intercept           = {meta.intercept_:+.4f}")

    stack_metrics = compute_metrics(y_test, stack_preds,
                                    y_train=y_train, seasonal_period=24)

    print("\n=== Fold-6 leaderboard ===")
    print(f"{'model':<14} {'MAE':>8} {'RMSE':>8} {'MASE':>8}")
    print("-" * 42)
    print(f"{'STACKING':<14} {stack_metrics['MAE']:>8.4f} {stack_metrics['RMSE']:>8.4f} {stack_metrics['MASE']:>8.4f}")
    for col in model_cols:
        single_preds = wide.loc[test_mask, col].to_numpy()
        m = compute_metrics(y_test, single_preds, y_train=y_train, seasonal_period=24)
        print(f"{col:<14} {m['MAE']:>8.4f} {m['RMSE']:>8.4f} {m['MASE']:>8.4f}")

    # Persist
    pd.DataFrame({
        "timestamp": wide.loc[test_mask, "timestamp"].values,
        "y_true": y_test,
        "y_pred": stack_preds,
    }).to_csv(RESULTS_DIR / "stacking_predictions_fold6.csv", index=False)

    summary = pd.DataFrame({
        "base_model": model_cols + ["intercept"],
        "coefficient": list(meta.coef_) + [meta.intercept_],
    })
    summary.to_csv(RESULTS_DIR / "stacking_summary.csv", index=False)

    summary_metrics = pd.DataFrame([{
        "model": "stacking_ridge",
        "fold": 6,
        "valid_rows": int(test_mask.sum()),
        **{k: round(v, 4) for k, v in stack_metrics.items()},
    }])
    summary_metrics.to_csv(RESULTS_DIR / "stacking_fold_results.csv", index=False)
    print("\nSaved stacking_predictions_fold6.csv, stacking_summary.csv, stacking_fold_results.csv")


if __name__ == "__main__":
    main()
