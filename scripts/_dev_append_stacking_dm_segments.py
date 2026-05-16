"""Developer-only: append §12 (stacking) + §13 (DM + per-segment) to the
modeling notebook. All cells read from CSVs already on disk; no retraining."""

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
    # ── §12 STACKING ENSEMBLE ──────────────────────────────────────
    _md(
        "31-stacking-header",
        "## 12 · Stacking Ensemble (Bonus)\n",
        "\n",
        "**Bonus content beyond the brief.** A ridge meta-learner combines the per-fold predictions from the four base models (XGBoost, SARIMA, LSTM, GRU). The hypothesis: if base-model errors decorrelate, weighted combination should beat any single model. The output is a number — either ensembling helps (+%) or it doesn't (≈ best single model).\n",
        "\n",
        "**Methodology.** Folds 1–5 train the ridge meta-learner on (per-row base predictions) → (true target). Fold 6 holds out and is the test set. Predictions are clipped at 0 (target is non-negative). The ridge `alpha=1.0` is a conservative default; tuning could shift weights but unlikely to flip the conclusion. Outputs persisted at [`reports/results/stacking_*.csv`](../reports/results/)."
    ),
    _code(
        "32-stacking-results",
        "# =============================================================\n",
        "# 12.1  STACKING ENSEMBLE — FOLD 6 LEADERBOARD\n",
        "# =============================================================\n",
        "stack_path = Path('..') / 'reports' / 'results' / 'stacking_summary.csv'\n",
        "stack_fold = Path('..') / 'reports' / 'results' / 'stacking_fold_results.csv'\n",
        "if stack_path.exists():\n",
        "    weights = pd.read_csv(stack_path)\n",
        "    fold6 = pd.read_csv(stack_fold)\n",
        "    print('Ridge meta-learner weights:')\n",
        "    display(weights)\n",
        "    print('\\nFold-6 metrics for the stacked predictor:')\n",
        "    display(fold6)\n",
        "else:\n",
        "    print('Run scripts/_dev_run_stacking_ensemble.py to generate.')",
    ),
    _md(
        "33-stacking-interp",
        "**Interpretation.** On the held-out fold 6:\n",
        "\n",
        "| model | MAE |\n",
        "|---|---|\n",
        "| LSTM | 0.3735 |\n",
        "| **STACKING (ridge)** | **0.3745** |\n",
        "| GRU | 0.3747 |\n",
        "| XGBoost | 0.4020 |\n",
        "| SARIMA | 0.4156 |\n",
        "\n",
        "The stacked ensemble **essentially ties LSTM** on the holdout fold, and is **0.027 MAE better than XGBoost alone on fold 6** (–7%). But: cross-fold XGBoost mean MAE is 0.336, so on the *easier* folds XGBoost beats the ensemble. The ensemble's value is concentrated on the hardest folds where individual models disagree more.\n",
        "\n",
        "**Bonus value-add for the report.** This is an honest negative-ish result: model errors are correlated enough that simple linear stacking does not break the 3% MAE cluster open. It tells us where the next gain *cannot* come from (more model variety) and where it *could* come from (better features, target transforms, multivariate sequence input for LSTM/GRU, weather data).\n",
        "\n",
        "**Limitation.** The ensemble is fit on a single train/test split (folds 1–5 / fold 6). A leave-one-fold-out cross-validated stacker would give a less biased estimate of the ensemble's typical performance. Recorded as future work."
    ),
    # ── §13 DM + per-segment ─────────────────────────────────────
    _md(
        "34-dm-header",
        "## 13 · Statistical Significance & Per-Segment Performance\n",
        "\n",
        "Two analyses the brief doesn't explicitly require but ENEL engineers will ask for. Both go beyond aggregate MAE to ask: *(a) is the leaderboard ordering statistically meaningful, or just noise?* and *(b) where do models actually win or lose — peak hours, weekends, summer/winter?*\n",
        "\n",
        "Both consume the saved per-fold prediction CSVs ([`reports/results/<model>_predictions_fold*.csv`](../reports/results/)). No model retraining."
    ),
    _code(
        "35-dm-code",
        "# =============================================================\n",
        "# 13.1  DIEBOLD-MARIANO PAIRWISE SIGNIFICANCE TEST\n",
        "# =============================================================\n",
        "# Newey-West HAC variance (lag 0 for 1-step-ahead) + Harvey-Leybourne-\n",
        "# Newbold small-sample correction. Negative DM = row model has lower\n",
        "# absolute error than column model. p < 0.05 = significant at 5%.\n",
        "dm_p = pd.read_csv(Path('..') / 'reports' / 'results' / 'dm_test_pvalues.csv', index_col=0)\n",
        "print('Diebold-Mariano two-sided p-values (n=4,114 paired observations):')\n",
        "display(dm_p.round(4))\n",
        "\n",
        "print('\\nReading: a value of 0.017 in the (xgboost, sarima) cell means we reject')\n",
        "print('the null \"XGBoost and SARIMA have equal MAE\" at the 5%% level — i.e. XGBoost')\n",
        "print('is significantly better than SARIMA on this dataset, not just numerically better.')",
    ),
    _md(
        "36-dm-interp",
        "**Interpretation of the DM matrix.** Only **XGBoost vs SARIMA** is significant at α = 0.05 (p = 0.017). Every other pair has p > 0.05.\n",
        "\n",
        "**What this means:** XGBoost's lead is *statistically real* against SARIMA but *not* statistically distinguishable from LSTM (p = 0.10) or GRU (p = 0.35). The 3% MAE cluster of XGBoost / GRU / LSTM is essentially a tie. **The choice between these three architectures is a soft preference, not a strict ranking.**\n",
        "\n",
        "Practical implications:\n",
        "- For the report's *\"which model wins\"* question, XGBoost is the most defensible pick (lowest MAE, significantly beats one competitor, no other competitor significantly beats it).\n",
        "- For a Phase-4F hyperparameter-tuning effort, *all three* of XGBoost / GRU / LSTM are credible targets — XGBoost isn't a foregone conclusion.\n",
        "- For SARIMA, the test confirms it is genuinely behind the ML/DL models on this dataset. Its value going forward is in *interpretability* (predict_components-style decomposition), not raw accuracy."
    ),
    _code(
        "37-segment-code",
        "# =============================================================\n",
        "# 13.2  PER-SEGMENT MAE (hour-of-day, day-of-week, month)\n",
        "# =============================================================\n",
        "by_hour = pd.read_csv(Path('..') / 'reports' / 'results' / 'segment_mae_by_hour.csv', index_col=0)\n",
        "by_dow = pd.read_csv(Path('..') / 'reports' / 'results' / 'segment_mae_by_dow.csv', index_col=0)\n",
        "by_month = pd.read_csv(Path('..') / 'reports' / 'results' / 'segment_mae_by_month.csv', index_col=0)\n",
        "\n",
        "fig, axes = plt.subplots(3, 1, figsize=(13, 12), constrained_layout=True)\n",
        "by_hour.T.plot(ax=axes[0], marker='o', linewidth=1.5)\n",
        "axes[0].set_title('MAE by hour of day — peak hours (18-21) are 2-3x harder than overnight (3-6)')\n",
        "axes[0].set_xlabel('Hour'); axes[0].set_ylabel('MAE (kW)')\n",
        "axes[0].set_xticks(range(0, 24)); axes[0].grid(True, alpha=0.3); axes[0].legend(title='Model', loc='upper left')\n",
        "\n",
        "by_dow.T.plot(ax=axes[1], marker='o', linewidth=1.5)\n",
        "axes[1].set_title('MAE by day of week (0 = Monday, 6 = Sunday) — weekends slightly harder')\n",
        "axes[1].set_xlabel('Day of week'); axes[1].set_ylabel('MAE (kW)')\n",
        "axes[1].set_xticks(range(7))\n",
        "axes[1].set_xticklabels(['Mon','Tue','Wed','Thu','Fri','Sat','Sun'])\n",
        "axes[1].grid(True, alpha=0.3); axes[1].legend(title='Model', loc='upper left')\n",
        "\n",
        "by_month.T.plot(ax=axes[2], marker='o', linewidth=1.5)\n",
        "axes[2].set_title('MAE by validation month — August (vacation) easiest, October (heating ramp) hardest')\n",
        "axes[2].set_xlabel('Month (validation period only)'); axes[2].set_ylabel('MAE (kW)')\n",
        "axes[2].grid(True, alpha=0.3); axes[2].legend(title='Model', loc='upper left')\n",
        "\n",
        "fig.savefig(Path('..') / 'reports' / 'figures' / 'segment_mae_by_dimension.png', dpi=140, bbox_inches='tight')\n",
        "plt.show()\n",
        "\n",
        "print('Saved reports/figures/segment_mae_by_dimension.png')",
    ),
    _md(
        "38-segment-interp",
        "**Interpretation of segment behaviour.**\n",
        "\n",
        "**By hour of day.** Every model has the same shape: easy at 3–6 AM (MAE ≈ 0.17), harder at peak hours 18–21 (MAE ≈ 0.45). The 2.5× peak/trough difference is the dominant pattern. **XGBoost is best at quiet hours; GRU/LSTM are slightly better at peak hours.** For an energy operator who cares most about peak-hour accuracy, GRU may be the better deployment choice — even though XGBoost wins the overall leaderboard. This is the kind of nuance that distinguishes \"highest average MAE\" from \"best for operations\".\n",
        "\n",
        "**By day of week.** Weekday-vs-weekend gap is modest (~5% MAE difference). Saturday/Sunday slightly harder, consistent with the EDA finding that weekend consumption patterns are wider-spread.\n",
        "\n",
        "**By month.** Strong pattern: August is the easiest validation month (MAE ≈ 0.22) because the household is *empty* during the French summer holidays — consumption drops to a near-flat low level. October is the hardest (MAE ≈ 0.44), because heating is being switched back on and consumption becomes more volatile. This connects directly to the EDA §10 segment-profile finding: the months with high *variance* in consumption are the months hard to forecast, not necessarily the months with high *mean*.\n",
        "\n",
        "**Implication for the report.** A single \"best model\" claim is over-simple. The correct framing is:\n",
        "- **XGBoost** is the best overall single model and significantly beats SARIMA.\n",
        "- **GRU** is competitive on every metric and is the better choice if peak-hour accuracy is the operational priority.\n",
        "- **All four models share the same structural failure modes** (peak hours, October) — improving these would require better features (weather, occupancy signal), not a new architecture."
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
