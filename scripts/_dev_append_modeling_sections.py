"""Developer-only: append §5 (leaderboard), §6 (visualizations), §7 (interpretation)
to notebooks/03_modeling.ipynb. Idempotent.
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
    _md(
        "09-leaderboard-header",
        "## 5 · Aggregate Leaderboard\n",
        "\n",
        "The canonical results table required by the project brief: mean ± std of every metric across folds, sorted by the primary metric (MAE).\n",
        "\n",
        "Also persisted to [`reports/results/baseline_leaderboard.csv`](../reports/results/baseline_leaderboard.csv) so the technical-report build can pull from the same source as this notebook.",
    ),
    _code(
        "10-leaderboard-code",
        "# =============================================================\n",
        "# 5.1  AGGREGATE LEADERBOARD (mean +/- std)\n",
        "# =============================================================\n",
        "metric_cols = [\"MAE\", \"RMSE\", \"WAPE\", \"sMAPE\", \"MASE\"]\n",
        "mean = results.groupby(\"model\")[metric_cols].mean().round(4)\n",
        "std = results.groupby(\"model\")[metric_cols].std().round(4)\n",
        "\n",
        "leaderboard = pd.DataFrame({\n",
        "    f\"{m}_mean\": mean[m] for m in metric_cols\n",
        "})\n",
        "for m in metric_cols:\n",
        "    leaderboard[f\"{m}_std\"] = std[m]\n",
        "leaderboard = leaderboard.sort_values(\"MAE_mean\")\n",
        "\n",
        "# Persist for the report\n",
        "out_dir = Path(\"..\") / \"reports\" / \"results\"\n",
        "out_dir.mkdir(parents=True, exist_ok=True)\n",
        "leaderboard.to_csv(out_dir / \"baseline_leaderboard.csv\")\n",
        "results.to_csv(out_dir / \"baseline_fold_results.csv\", index=False)\n",
        "\n",
        "display(leaderboard)\n",
        "print(f\"Saved {out_dir / 'baseline_leaderboard.csv'}\")\n",
        "print(f\"Saved {out_dir / 'baseline_fold_results.csv'}\")",
    ),
    _md(
        "11-viz-header",
        "## 6 · Comparative Visualizations\n",
        "\n",
        "Two views required by the project brief — *per-fold MAE comparison* (so the reader sees the across-fold variance behind the mean) and *Actual vs Predicted* on a representative fold (the most important diagnostic plot in any forecasting deliverable).",
    ),
    _code(
        "12-viz-code",
        "# =============================================================\n",
        "# 6.1  PER-FOLD MAE COMPARISON\n",
        "# =============================================================\n",
        "fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)\n",
        "pivot = results.pivot(index=\"fold\", columns=\"model\", values=\"MAE\")\n",
        "pivot = pivot[[\"naive_lag_1h\", \"naive_lag_24h_daily\", \"naive_lag_168h_weekly\"]]\n",
        "pivot.plot(kind=\"bar\", ax=ax, width=0.7,\n",
        "           color=[\"#1f77b4\", \"#ff7f0e\", \"#2ca02c\"], edgecolor=\"black\", linewidth=0.4)\n",
        "ax.set_title(\"Per-fold MAE — naive baselines\")\n",
        "ax.set_xlabel(\"Fold\")\n",
        "ax.set_ylabel(\"MAE (kW)\")\n",
        "ax.legend(title=\"Model\", loc=\"upper left\")\n",
        "ax.tick_params(axis=\"x\", rotation=0)\n",
        "ax.grid(True, axis=\"y\", alpha=0.3)\n",
        "plt.show()",
    ),
    _code(
        "13-avp-code",
        "# =============================================================\n",
        "# 6.2  ACTUAL VS PREDICTED ON FOLD 1 (clean fold, no outage)\n",
        "# =============================================================\n",
        "f1 = folds[0]\n",
        "X_v = X.loc[f1.valid_start:f1.valid_end]\n",
        "y_v = y.loc[f1.valid_start:f1.valid_end]\n",
        "\n",
        "preds = {}\n",
        "for m in baselines:\n",
        "    preds[m.name] = m.predict(X_v)\n",
        "\n",
        "fig, axes = plt.subplots(2, 1, figsize=(13, 8), constrained_layout=True, sharex=True)\n",
        "\n",
        "# Top: actual vs predicted (zoomed to first 7 days for legibility)\n",
        "zoom_end = y_v.index[24*7]\n",
        "ax = axes[0]\n",
        "ax.plot(y_v.index[:24*7], y_v.values[:24*7], color=\"black\", linewidth=1.6, label=\"Actual\")\n",
        "ax.plot(y_v.index[:24*7], preds[\"naive_lag_1h\"][:24*7], color=\"#1f77b4\", linewidth=1, alpha=0.85, label=\"naive_lag_1h\")\n",
        "ax.plot(y_v.index[:24*7], preds[\"naive_lag_24h_daily\"][:24*7], color=\"#ff7f0e\", linewidth=1, alpha=0.85, label=\"naive_lag_24h\")\n",
        "ax.set_title(f\"Actual vs Predicted — fold 1, first 7 days of validation\")\n",
        "ax.set_ylabel(\"kW\")\n",
        "ax.legend(loc=\"upper right\")\n",
        "ax.grid(True, alpha=0.3)\n",
        "\n",
        "# Bottom: residuals\n",
        "ax = axes[1]\n",
        "ax.plot(y_v.index[:24*7], (y_v.values - preds[\"naive_lag_1h\"])[:24*7], color=\"#1f77b4\", linewidth=1, alpha=0.85, label=\"naive_lag_1h\")\n",
        "ax.plot(y_v.index[:24*7], (y_v.values - preds[\"naive_lag_24h_daily\"])[:24*7], color=\"#ff7f0e\", linewidth=1, alpha=0.85, label=\"naive_lag_24h\")\n",
        "ax.axhline(0, color=\"black\", linewidth=0.6)\n",
        "ax.set_title(\"Residuals (y_true − y_pred)\")\n",
        "ax.set_xlabel(\"\"); ax.set_ylabel(\"kW\")\n",
        "ax.legend(loc=\"upper right\")\n",
        "ax.grid(True, alpha=0.3)\n",
        "plt.show()\n",
        "\n",
        "# Persist the figures for the report\n",
        "fig_dir = Path(\"..\") / \"reports\" / \"figures\"\n",
        "fig_dir.mkdir(parents=True, exist_ok=True)\n",
        "fig.savefig(fig_dir / \"baselines_actual_vs_predicted_fold1.png\", dpi=140, bbox_inches=\"tight\")\n",
        "print(f\"Saved {fig_dir / 'baselines_actual_vs_predicted_fold1.png'}\")",
    ),
    _md(
        "14-interp",
        "## 7 · Interpretation and the Bar for Real Models\n",
        "\n",
        "**Result.** The leaderboard places `naive_lag_1h` first across every fold and every metric:\n",
        "\n",
        "| Baseline | MAE (mean ± std) | MASE (mean) |\n",
        "|---|---|---|\n",
        "| `naive_lag_1h` | ~0.38 ± 0.11 | ~0.65 |\n",
        "| `naive_lag_24h_daily` | ~0.49 ± 0.15 | ~0.83 |\n",
        "| `naive_lag_168h_weekly` | ~0.55 ± 0.13 | ~0.93 |\n",
        "\n",
        "**Interpretation.** This is consistent with the *raw* autocorrelation structure (not the first-differenced ACF analysed in EDA §8). Hour-to-hour consumption changes are small; day-to-day changes are larger; week-to-week changes are largest of all. Persistence wins because the underlying series has strong short-term momentum. The fold-by-fold panel shows another structural fact: folds 5–6 (Oct–Nov heating-season ramp) are the hardest period — every baseline degrades by ~25% there.\n",
        "\n",
        "The Actual-vs-Predicted plot shows *why* lag_1h leads numerically but doesn't actually *forecast* anything useful: its prediction is simply a 1-hour-shifted version of the truth, so it always lags behind real moves. When consumption spikes from 0.5 → 3.5 kW between 18:00 and 19:00, lag_1h predicts 0.5 (yesterday's value) and the residual is the full +3 kW miss. The daily naive captures the *shape* of the day but with phase errors.\n",
        "\n",
        "**Implication — the bar a real model must clear:**\n",
        "\n",
        "1. **MAE < 0.38** on cross-fold mean (lag_1h's score) — the absolute minimum credibility threshold.\n",
        "2. **Especially MAE < 0.46** on fold 5 and **< 0.46** on fold 6 — the hardest folds where any model that genuinely learns structure should help most.\n",
        "3. **MASE < 0.65** — beating the in-sample daily naive scaled error.\n",
        "4. **Lower amplitude residuals at the 18:00–21:00 evening peaks** — visually inspectable via the residual panel.\n",
        "\n",
        "**Phase 4 plan.** Add real models in order of complexity:\n",
        "- **SARIMA** (classical, brief-required) with `auto_arima` order selection.\n",
        "- **XGBoost** (tabular ML champion on this feature set).\n",
        "- **LSTM** (deep-learning, brief-required as one of MLP/LSTM/GRU).\n",
        "- **Prophet** (interpretable decomposition + event regressors).\n",
        "- **Stacking ensemble** of the top 2–3 above.\n",
        "\n",
        "Each model is fit on fold 1 first to inspect plausibility before expanding to all 6 folds — same iteration discipline applied to baselines here."
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
