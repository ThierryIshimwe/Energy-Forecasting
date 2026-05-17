"""Run the three naive baselines across all 6 rolling-origin folds and save
the per-fold results to reports/results/baseline_fold_results.csv. This is
the file the deliverable notebook's leaderboard cell loads.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from energy_forecasting.config import DEFAULT_CONFIG_PATH, ForecastConfig
from energy_forecasting.evaluation import compute_metrics
from energy_forecasting.models import NaiveLagBaseline
from energy_forecasting.splits import make_rolling_origin_splits

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "results" / "baseline_fold_results.csv"


def main() -> None:
    cfg = ForecastConfig.from_yaml(DEFAULT_CONFIG_PATH)
    target = cfg.target.column

    df = pd.read_parquet(ROOT / "data" / "processed" / "features.parquet")
    y_all = df[target]
    X_all = df.drop(columns=[target])

    folds = make_rolling_origin_splits(X_all.index, config=cfg)
    baselines = [
        NaiveLagBaseline(f"{target}_lag_1h",   name="naive_lag_1h"),
        NaiveLagBaseline(f"{target}_lag_24h",  name="naive_lag_24h_daily"),
        NaiveLagBaseline(f"{target}_lag_168h", name="naive_lag_168h_weekly"),
    ]

    rows: list[dict] = []
    for f in folds:
        Xt = X_all.loc[f.train_start:f.train_end]
        yt = y_all.loc[f.train_start:f.train_end]
        Xv = X_all.loc[f.valid_start:f.valid_end].dropna()
        yv = y_all.loc[Xv.index]
        for m in baselines:
            m.fit(Xt, yt)
            y_pred = m.predict(Xv)
            metrics = compute_metrics(yv.to_numpy(), y_pred,
                                       y_train=yt.to_numpy(), seasonal_period=24)
            rows.append({
                "fold": f.fold_id,
                "model": m.name,
                "valid_rows": len(yv),
                **{k: round(v, 4) for k, v in metrics.items()},
            })

    out = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"Wrote {OUT}")
    summary = out.groupby("model").agg(MAE_mean=("MAE", "mean"),
                                        MAE_std=("MAE", "std")).round(4).sort_values("MAE_mean")
    print(summary)


if __name__ == "__main__":
    main()
