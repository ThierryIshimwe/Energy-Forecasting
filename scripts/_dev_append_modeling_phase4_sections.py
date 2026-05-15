"""Developer-only: append §8 (SARIMA) and §9 (XGBoost) to the modeling notebook.

The XGBoost section reads its real fold results from
``reports/results/xgboost_fold_results.csv`` and the feature-importance file
``reports/results/xgboost_feature_importance.csv``.

The SARIMA section currently holds the fold-1 result hard-coded plus a TODO
note: the full per-fold results land at ``reports/results/sarima_fold_results.csv``
when the background run finishes. Once that file exists, rerunning this script
(or just re-executing the notebook) picks up the full data.
"""

from __future__ import annotations

import json
from pathlib import Path

NB_PATH = Path(__file__).resolve().parents[1] / "notebooks" / "03_modeling.ipynb"


def _md(cid: str, *lines: str) -> dict:
    return {"cell_type": "markdown", "id": cid, "metadata": {}, "source": list(lines)}


def _code(cid: str, *lines: str) -> dict:
    return {
        "cell_type": "code",
        "id": cid,
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": list(lines),
    }


NEW_CELLS: list[dict] = [
    # ── §8 SARIMA ──────────────────────────────────────────────────
    _md(
        "15-sarima-header",
        "## 8 · SARIMA — Classical Statistical Model\n",
        "\n",
        "[`SARIMAForecaster`](../src/energy_forecasting/models/sarima.py) wraps "
        "`statsmodels.tsa.statespace.SARIMAX`. The configured order is "
        "`(1,1,1)x(1,1,1,24)` — first-difference (from §6 stationarity tests), "
        "seasonal period 24 (from §7 STL), conservative `P=D=Q=1` as a starting point.\n",
        "\n",
        "We evaluate using the rolling 1-step-ahead protocol "
        "([`predict_rolling_one_step`](../src/energy_forecasting/models/sarima.py)) — "
        "at each validation hour the SARIMA filter is extended with the actual past value, "
        "matching the information set the naïve baselines have access to."
    ),
    _code(
        "16-sarima-results-code",
        "# =============================================================\n",
        "# 8.1  SARIMA RESULTS\n",
        "# =============================================================\n",
        "import json\n",
        "from pathlib import Path\n",
        "\n",
        "sarima_results_path = Path('..') / 'reports' / 'results' / 'sarima_fold_results.csv'\n",
        "\n",
        "if sarima_results_path.exists():\n",
        "    sarima_results = pd.read_csv(sarima_results_path)\n",
        "    print('Loaded SARIMA results from disk:')\n",
        "    display(sarima_results)\n",
        "    print()\n",
        "    summary = sarima_results[['MAE','RMSE','WAPE','sMAPE','MASE']].agg(['mean','std']).round(4)\n",
        "    print('Aggregate across folds:')\n",
        "    display(summary)\n",
        "else:\n",
        "    # Fallback: hard-coded fold-1 number from the standalone run while the\n",
        "    # full-folds background job finishes.\n",
        "    print('SARIMA full-folds run still in progress — showing fold-1 result only.')\n",
        "    sarima_fold1 = pd.DataFrame([{\n",
        "        'fold': 1, 'model': 'sarima_(1,1,1)x(1,1,1,24)', 'valid_rows': 720,\n",
        "        'fit_seconds': 651.2,\n",
        "        'MAE': 0.3675, 'RMSE': 0.5053, 'WAPE': 0.3805,\n",
        "        'sMAPE': 0.4052, 'MASE': 0.6198,\n",
        "    }])\n",
        "    display(sarima_fold1)\n",
        "    print('\\nFold-1 SARIMA vs naive_lag_1h (same fold):')\n",
        "    print('  MAE  :  0.3675 vs 0.4014  (delta -0.034 / -8.4%)')\n",
        "    print('  RMSE :  0.5053 vs 0.5864  (delta -0.081)')\n",
        "    print('  MASE :  0.6198 vs 0.6770  (delta -0.057)')\n",
        "    print('\\nWhen sarima_fold_results.csv arrives, this cell auto-updates.')",
    ),
    _md(
        "17-sarima-interp",
        "**Interpretation (fold 1 standalone run).** SARIMA beats `naive_lag_1h` "
        "by **−8.4% MAE** (0.3675 vs 0.4014). The MASE of 0.62 confirms it: less "
        "than 1 means it beats the in-sample daily-naïve baseline on its own ground. "
        "This is the first model in the lineup doing *genuine forecasting* rather "
        "than just persistence.\n",
        "\n",
        "**The cost:** fitting `(1,1,1)×(1,1,1,24)` on 8,607 hourly observations took "
        "**651 seconds (~11 minutes)** on a single fold. The full-leaderboard run "
        "across 6 folds is in progress and will take ~70 min wall-clock total. The "
        "1-step rolling prediction itself is fast (<1 s for 720 hours) — the cost "
        "is concentrated in the MLE fit.\n",
        "\n",
        "**Implication:** SARIMA is competitive but expensive. For Phase 4F "
        "hyperparameter tuning we'll consider reducing seasonal order (e.g., "
        "`(0,1,1,24)`) to keep the search budget tractable."
    ),
    # ── §9 XGBoost ─────────────────────────────────────────────────
    _md(
        "18-xgb-header",
        "## 9 · XGBoost — Gradient-Boosted Trees on Engineered Features\n",
        "\n",
        "[`XGBoostForecaster`](../src/energy_forecasting/models/xgboost_model.py) "
        "consumes the 31-feature leakage-safe matrix directly. Defaults: "
        "`max_depth=6, learning_rate=0.05, n_estimators=500`, light stochastic "
        "regularization (`subsample=0.8, colsample_bytree=0.8`). No early stopping "
        "in this evaluation; Phase 4F will add Optuna tuning with inner CV.\n",
        "\n",
        "Unlike SARIMA, the per-row prediction is naturally 1-step-ahead — the "
        "feature matrix already contains the relevant lags built with `.shift()` "
        "by the feature pipeline, so XGBoost's `predict(X)` is the correct protocol."
    ),
    _code(
        "19-xgb-results-code",
        "# =============================================================\n",
        "# 9.1  XGBOOST FOLD-BY-FOLD RESULTS\n",
        "# =============================================================\n",
        "xgb_results = pd.read_csv(Path('..') / 'reports' / 'results' / 'xgboost_fold_results.csv')\n",
        "display(xgb_results)\n",
        "\n",
        "print('\\nCross-fold aggregate:')\n",
        "summary = xgb_results[['MAE','RMSE','WAPE','sMAPE','MASE']].agg(['mean','std']).round(4)\n",
        "display(summary)",
    ),
    _code(
        "20-xgb-vs-baselines-code",
        "# =============================================================\n",
        "# 9.2  XGBOOST vs BASELINES — UPDATED LEADERBOARD\n",
        "# =============================================================\n",
        "# Combine baseline + XGBoost results into one leaderboard.\n",
        "baseline_results = pd.read_csv(Path('..') / 'reports' / 'results' / 'baseline_fold_results.csv')\n",
        "combined = pd.concat([\n",
        "    baseline_results,\n",
        "    xgb_results.rename(columns={'model': 'model'}),\n",
        "], ignore_index=True)\n",
        "\n",
        "metric_cols = ['MAE','RMSE','WAPE','sMAPE','MASE']\n",
        "leaderboard = (\n",
        "    combined.groupby('model')[metric_cols]\n",
        "    .agg(['mean','std']).round(4)\n",
        "    .sort_values(('MAE','mean'))\n",
        ")\n",
        "leaderboard.columns = ['_'.join(c) for c in leaderboard.columns]\n",
        "display(leaderboard)\n",
        "\n",
        "# Persist for the report\n",
        "leaderboard.to_csv(Path('..') / 'reports' / 'results' / 'leaderboard_phase4_partial.csv')\n",
        "print('Saved leaderboard_phase4_partial.csv (will be extended as SARIMA / LSTM / Prophet results arrive).')",
    ),
    _code(
        "21-xgb-feature-importance",
        "# =============================================================\n",
        "# 9.3  FEATURE IMPORTANCE — WHICH FEATURES CARRY THE SIGNAL?\n",
        "# =============================================================\n",
        "importance = pd.read_csv(Path('..') / 'reports' / 'results' / 'xgboost_feature_importance.csv',\n",
        "                        index_col=0)\n",
        "importance.columns = ['gain_mean']\n",
        "top15 = importance.head(15)\n",
        "\n",
        "fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)\n",
        "ax.barh(top15.index[::-1], top15['gain_mean'][::-1], color='#1f77b4', edgecolor='black', linewidth=0.4)\n",
        "ax.set_title('XGBoost — top 15 features by mean gain across folds')\n",
        "ax.set_xlabel('Mean gain')\n",
        "ax.grid(True, axis='x', alpha=0.3)\n",
        "fig.savefig(Path('..') / 'reports' / 'figures' / 'xgboost_feature_importance.png',\n",
        "            dpi=140, bbox_inches='tight')\n",
        "plt.show()\n",
        "\n",
        "print('Top 5 contributors:')\n",
        "for name, gain in top15.head(5)['gain_mean'].items():\n",
        "    print(f'  {name:<40s}  {gain:.4f}')",
    ),
    _code(
        "22-xgb-avp",
        "# =============================================================\n",
        "# 9.4  ACTUAL vs PREDICTED — XGBOOST on fold 1\n",
        "# =============================================================\n",
        "xgb_preds = pd.read_csv(\n",
        "    Path('..') / 'reports' / 'results' / 'xgboost_predictions_fold1.csv',\n",
        "    parse_dates=['timestamp']\n",
        ").set_index('timestamp')\n",
        "\n",
        "# Same 7-day window as the baseline AvP plot for direct comparison\n",
        "window = xgb_preds.iloc[:24 * 7]\n",
        "resid = window['y_true'] - window['y_pred']\n",
        "\n",
        "fig, axes = plt.subplots(2, 1, figsize=(13, 8), constrained_layout=True, sharex=True)\n",
        "axes[0].plot(window.index, window['y_true'], color='black', linewidth=1.6, label='Actual')\n",
        "axes[0].plot(window.index, window['y_pred'], color='#d62728', linewidth=1, alpha=0.85, label='XGBoost')\n",
        "axes[0].set_title('Actual vs Predicted — XGBoost, fold 1, first 7 days')\n",
        "axes[0].set_ylabel('kW')\n",
        "axes[0].legend(loc='upper right'); axes[0].grid(True, alpha=0.3)\n",
        "\n",
        "axes[1].plot(window.index, resid, color='#d62728', linewidth=1, alpha=0.85)\n",
        "axes[1].axhline(0, color='black', linewidth=0.6)\n",
        "axes[1].set_title('Residuals (y_true − y_pred)'); axes[1].set_ylabel('kW')\n",
        "axes[1].grid(True, alpha=0.3)\n",
        "fig.savefig(Path('..') / 'reports' / 'figures' / 'xgboost_actual_vs_predicted_fold1.png',\n",
        "            dpi=140, bbox_inches='tight')\n",
        "plt.show()\n",
        "\n",
        "print(f'Window MAE  : {resid.abs().mean():.4f}')\n",
        "print(f'Window RMSE : {(resid ** 2).mean() ** 0.5:.4f}')",
    ),
    _md(
        "23-xgb-interp",
        "**Interpretation.** XGBoost is the current leader on every metric:\n",
        "\n",
        "- **Cross-fold MAE ≈ 0.336** (mean of 6 folds), down from 0.380 for `naive_lag_1h` "
        "(−12%) and 0.368 for SARIMA on fold 1 (−9%).\n",
        "- **MASE ≈ 0.57** — comfortably beats the in-sample daily-naïve baseline. \n",
        "- **Per-fold pattern matches the EDA structural finding**: folds 5–6 "
        "(Oct–Nov heating-ramp) are hardest (MAE ≈ 0.40), and fold 3 (August "
        "outage period, lower mean consumption + fewer rows) is easiest (MAE ≈ 0.21). "
        "*Both signals were predicted in EDA §10 — the model didn't change which "
        "months are hard, it just lowered the error within each*.\n",
        "\n",
        "**Why XGBoost wins.** The feature-importance plot tells the story: ~25% of "
        "total gain comes from `Global_active_power_lag_1h` alone, then short-term "
        "rolling stats and cyclical hour features. Decision trees can split on "
        "`hour × is_weekend` interactions that linear SARIMA cannot represent "
        "without explicit interaction terms. The `is_french_holiday` feature shows "
        "up in the top 10 — the calendar regressors are pulling weight.\n",
        "\n",
        "**Residual plot reading.** Most residuals sit within ±0.5 kW. The spikes "
        "align with consumption peaks (evening 18–21h) — the model under-predicts "
        "the highest peaks slightly, a typical right-tail behaviour for "
        "MSE-trained learners on a right-skewed target (EDA §9). For Phase 4F we "
        "might try log-transforming the target before fitting to address this."
    ),
]


def main() -> None:
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
    existing = {c.get("id") for c in nb["cells"]}
    added = 0
    for cell in NEW_CELLS:
        if cell["id"] in existing:
            continue
        nb["cells"].append(cell)
        added += 1
    NB_PATH.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(f"Appended {added} cells. Notebook now has {len(nb['cells'])} cells total.")


if __name__ == "__main__":
    main()
