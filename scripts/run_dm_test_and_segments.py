"""Developer-only: Diebold-Mariano significance test + per-segment analysis.

Both consume the saved per-fold prediction CSVs in reports/results/. No model
retraining. Outputs:

    reports/results/dm_test_matrix.csv     (pairwise DM p-values)
    reports/results/segment_mae_by_hour.csv
    reports/results/segment_mae_by_dow.csv
    reports/results/segment_mae_by_month.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "reports" / "results"


def load_predictions_long(model_slug: str) -> pd.DataFrame:
    frames = []
    for fold_id in range(1, 7):
        p = RESULTS_DIR / f"{model_slug}_predictions_fold{fold_id}.csv"
        if not p.exists():
            return pd.DataFrame()
        df = pd.read_csv(p, parse_dates=["timestamp"])
        df["fold"] = fold_id
        df["model"] = model_slug
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def diebold_mariano(e1: np.ndarray, e2: np.ndarray, h: int = 1) -> tuple[float, float]:
    """One-sided DM test on absolute-error loss.

    Returns (DM statistic, two-sided p-value). Negative DM means model 1
    has *lower* mean absolute error than model 2 (model 1 better).

    Implementation: Harvey-Leybourne-Newbold (1997) small-sample correction,
    Newey-West HAC variance with lag h-1 = 0 for 1-step-ahead forecasts.
    """
    d = np.abs(e1) - np.abs(e2)
    n = len(d)
    mean_d = float(np.mean(d))
    # Newey-West autocovariance up to lag h-1
    gamma0 = float(np.var(d, ddof=0))
    var_d_hac = gamma0  # h=1 implies lag-0 only
    if var_d_hac <= 0:
        return 0.0, 1.0
    dm = mean_d / np.sqrt(var_d_hac / n)
    # Harvey-Leybourne-Newbold small-sample correction
    correction = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    dm_hln = dm * correction
    # Two-sided p-value using t-distribution (HLN's recommendation)
    p_value = 2 * (1 - stats.t.cdf(abs(dm_hln), df=n - 1))
    return float(dm_hln), float(p_value)


def main() -> None:
    models = ["xgboost", "sarima", "lstm", "gru"]
    long_frames = {m: load_predictions_long(m) for m in models}
    long_frames = {m: df for m, df in long_frames.items() if not df.empty}
    print(f"Loaded predictions for: {list(long_frames.keys())}")

    # ── Diebold-Mariano pairwise matrix ────────────────────────
    print("\n=== Diebold-Mariano test (1-step-ahead, HLN-corrected) ===\n")
    aligned = {}
    for m, df in long_frames.items():
        df_sorted = df.sort_values(["fold", "timestamp"]).reset_index(drop=True)
        aligned[m] = df_sorted

    common_index = aligned[models[0]][["fold", "timestamp", "y_true"]]
    for m in models[1:]:
        if m not in aligned:
            continue
        common_index = common_index.merge(
            aligned[m][["fold", "timestamp"]], on=["fold", "timestamp"]
        )

    errors = {}
    for m, df in aligned.items():
        df_m = df.merge(common_index[["fold", "timestamp"]], on=["fold", "timestamp"])
        errors[m] = (df_m["y_true"] - df_m["y_pred"]).to_numpy()
    n_obs = len(next(iter(errors.values())))
    print(f"Comparing on {n_obs:,} aligned observations across all folds.\n")

    model_list = list(errors.keys())
    dm_matrix = pd.DataFrame(index=model_list, columns=model_list, dtype=object)
    for i, m1 in enumerate(model_list):
        for j, m2 in enumerate(model_list):
            if i == j:
                dm_matrix.loc[m1, m2] = "—"
            else:
                dm, p = diebold_mariano(errors[m1], errors[m2], h=1)
                marker = " *" if p < 0.05 else "  "
                dm_matrix.loc[m1, m2] = f"DM={dm:+.2f}, p={p:.3f}{marker}"
    print(dm_matrix.to_string())
    print("\nNegative DM => row model has lower MAE than column model.")
    print("`*` marks pairs significantly different at p<0.05.\n")

    # Compact numeric matrix for the report (just p-values)
    pmat = pd.DataFrame(index=model_list, columns=model_list, dtype=float)
    for m1 in model_list:
        for m2 in model_list:
            if m1 == m2:
                pmat.loc[m1, m2] = np.nan
            else:
                _, p = diebold_mariano(errors[m1], errors[m2], h=1)
                pmat.loc[m1, m2] = round(p, 4)
    pmat.to_csv(RESULTS_DIR / "dm_test_pvalues.csv")
    print(f"Saved {RESULTS_DIR / 'dm_test_pvalues.csv'}")

    # ── Per-segment MAE ────────────────────────────────────────
    print("\n=== Per-segment MAE by model ===\n")
    full_long = pd.concat(long_frames.values(), ignore_index=True)
    full_long["abs_err"] = (full_long["y_true"] - full_long["y_pred"]).abs()
    full_long["hour"] = full_long["timestamp"].dt.hour
    full_long["dow"] = full_long["timestamp"].dt.dayofweek
    full_long["month"] = full_long["timestamp"].dt.month

    by_hour = full_long.groupby(["model", "hour"])["abs_err"].mean().unstack("hour").round(4)
    by_dow = full_long.groupby(["model", "dow"])["abs_err"].mean().unstack("dow").round(4)
    by_month = full_long.groupby(["model", "month"])["abs_err"].mean().unstack("month").round(4)

    by_hour.to_csv(RESULTS_DIR / "segment_mae_by_hour.csv")
    by_dow.to_csv(RESULTS_DIR / "segment_mae_by_dow.csv")
    by_month.to_csv(RESULTS_DIR / "segment_mae_by_month.csv")

    print("By hour-of-day (MAE):")
    print(by_hour.to_string())
    print("\nBy day-of-week (MAE; 0=Mon, 6=Sun):")
    print(by_dow.to_string())
    print("\nBy month (MAE):")
    print(by_month.to_string())

    print(f"\nSaved: segment_mae_by_hour.csv, segment_mae_by_dow.csv, segment_mae_by_month.csv")


if __name__ == "__main__":
    main()
