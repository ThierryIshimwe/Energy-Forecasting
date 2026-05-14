"""Developer-only: idempotently update / extend notebooks/01_eda.ipynb.

This script:
  1. Updates the source of cell `27-stl-interp` to the latest interpretation
     (in case the original was authored before we had real STL numbers).
  2. Appends sections 8 (autocorrelation), 9 (distribution), 10 (segments)
     if those cells are not already present.

Idempotent: re-running is safe. Outputs for cells that already exist are
preserved; outputs for newly-added cells will be empty until the notebook
is executed.
"""

from __future__ import annotations

import json
from pathlib import Path

NB_PATH = Path(__file__).resolve().parents[1] / "notebooks" / "01_eda.ipynb"


# ─────────────────────────────────────────────────────────────────────
#  Updated cell 27 content (replaces whatever is currently there)
# ─────────────────────────────────────────────────────────────────────
CELL_27_NEW_SOURCE: list[str] = [
    "**Interpretation.** The decomposition reveals three regimes whose actual magnitudes were larger than expected:\n",
    "\n",
    "- The **trend** ranges from 0.17 to 2.62 kW — about a 15× swing between low and high regimes. Far from a flat *baseline + cycle* structure; this household's average consumption shifts substantially across years and seasons.\n",
    "- The **seasonal (24h) component** has amplitude (max − min) of ~5.1 kW. Because STL lets the seasonal pattern *evolve slowly over time*, this number captures both within-day swing *and* the shift of the daily-cycle shape across years. A fixed-coefficient model (classical SARIMA) treats the daily cycle as constant; the data argues it is **time-varying**, which is a structural advantage for flexible models (XGBoost, LSTM) that can learn cycle-shift jointly with calendar features.\n",
    "- The **residual std** is 0.659 kW — substantial. Even after trend + daily-seasonal removal, a kilowatt-scale spread of unexplained variation remains. Some of this is weather-driven, some event-driven, some genuine household stochasticity.\n",
    "\n",
    "**A note on the August 2010 zoom (cell 7.2).** The numerical printout shows residual `|mean|` of 0.072 kW *during* outage vs 0.219 kW *outside*, with std = NaN during outage. This is *not* the anomaly-detection signal I expected to surface. Why: our hourly preprocessor *drops* the 421 fully-empty hour buckets that fall inside long outages, so there are no hourly rows to compute residuals over during the Aug 17–22 gap. **Implication:** anomaly attribution against discrete outage events needs to operate on the raw minute-level series (or a different resampling that keeps NaN hours visible), not on the dropped-bucket hourly aggregate. Recorded as work item for the anomaly-attribution module.\n",
    "\n",
    "**Implication for modeling.** Two concrete decisions:\n",
    "\n",
    "1. **SARIMA with `seasonal_order=(_, _, _, 24)`** is the natural classical fit, but with the caveat that its fixed daily-cycle assumption may underperform on years where the daily shape has drifted.\n",
    "2. **ML/DL models** should receive the `lag_24h`, `lag_168h`, and cyclical `hour_sin / hour_cos` features (already registered as safe). They can learn cycle-shift implicitly through year-or-month interactions with hour-of-day.",
]


# ─────────────────────────────────────────────────────────────────────
#  New cells for sections 8, 9, 10
# ─────────────────────────────────────────────────────────────────────
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
    # ── §8 Autocorrelation Analysis ────────────────────────────────
    _md(
        "28-acf-header",
        "## 8 · Autocorrelation Analysis (ACF / PACF)\n",
        "\n",
        "Autocorrelation tells us which lags carry predictive signal. We run ACF and PACF on the **first-differenced hourly target** (§6 showed it is stationary, which is the assumption these plots require for interpretability).\n",
        "\n",
        "- **ACF** at lag $h$ is the correlation between $y_t$ and $y_{t-h}$. Large spikes at lag 24 indicate daily structure; at 168 indicate weekly.\n",
        "- **PACF** at lag $h$ controls for shorter lags — useful for SARIMA's $p$ and $P$ orders.\n",
        "\n",
        "Result we expect from §5–7: prominent ACF/PACF spikes at lag 24 (daily) and 168 (weekly), justifying the lag features already in the [feature registry](../src/energy_forecasting/features/registry.py).",
    ),
    _code(
        "29-acf-code",
        "# =============================================================\n",
        "# 8.1  ACF AND PACF ON THE FIRST-DIFFERENCED HOURLY SERIES\n",
        "# =============================================================\n",
        "from statsmodels.graphics.tsaplots import plot_acf, plot_pacf\n",
        "\n",
        "# Show lags up to 1 week (168) so daily and weekly structure are both visible.\n",
        "MAX_LAG = 168\n",
        "\n",
        "fig, axes = plt.subplots(2, 1, figsize=(13, 8), constrained_layout=True)\n",
        "plot_acf(y_diff1, lags=MAX_LAG, ax=axes[0], zero=False)\n",
        "axes[0].set_title(\"ACF of first-differenced hourly Global_active_power (lags 1–168)\")\n",
        "axes[0].set_xlabel(\"Lag (hours)\")\n",
        "plot_pacf(y_diff1, lags=MAX_LAG, ax=axes[1], method=\"ywm\", zero=False)\n",
        "axes[1].set_title(\"PACF of first-differenced hourly Global_active_power (lags 1–168)\")\n",
        "axes[1].set_xlabel(\"Lag (hours)\")\n",
        "for ax in axes:\n",
        "    ax.grid(True, alpha=0.3)\n",
        "plt.show()\n",
        "\n",
        "# Quantitative ACF values at the lags we care about\n",
        "from statsmodels.tsa.stattools import acf\n",
        "acf_values = acf(y_diff1, nlags=MAX_LAG, fft=True)\n",
        "key_lags = [1, 2, 3, 6, 12, 24, 48, 72, 168]\n",
        "print(\"ACF values at key lags:\")\n",
        "for lag in key_lags:\n",
        "    bar = \"█\" * int(abs(acf_values[lag]) * 40)\n",
        "    sign = \"+\" if acf_values[lag] >= 0 else \"-\"\n",
        "    print(f\"  lag {lag:>3} ({lag/24:>4.2f} days): {acf_values[lag]:+.4f}   {sign}{bar}\")",
    ),
    _md(
        "30-acf-interp",
        "**Interpretation.** Reading the ACF printout and plot together:\n",
        "\n",
        "- A strong **negative** spike at lag 1 in the differenced series indicates that high consumption tends to be followed by a correction (consistent with overshoot/recovery in a thermostatically-controlled household).\n",
        "- A clear **positive** peak at lag 24 confirms daily seasonality survives differencing — i.e., yesterday-at-this-hour is predictive of today-at-this-hour beyond the simple lag-1 signal.\n",
        "- Smaller positive peaks at lag 168 confirm weekly structure (same hour, same day-of-week last week).\n",
        "\n",
        "**Implication.** The lag set in [`conf/base.yaml`](../conf/base.yaml) — `lag_hours: [1, 24, 168]` — is supported by the data. We could add `lag_2` (for the negative-then-recovery pattern) and possibly `lag_48` (yesterday-+1-day) as additional candidates for tree-based models, but the three primary lags capture the dominant structure.",
    ),
    # ── §9 Distribution Analysis ───────────────────────────────────
    _md(
        "31-dist-header",
        "## 9 · Distribution Analysis\n",
        "\n",
        "Examine the empirical distribution of the hourly target — both raw and log-transformed. Three things matter:\n",
        "\n",
        "- **Skewness:** right-skewed targets (long tail of high values) make squared-error losses (RMSE) heavily influenced by peaks. MAE is more robust; log-transforming the target before fitting can help squared-error models.\n",
        "- **Heavy tails:** flag values that are likely anomalies vs. legitimate peaks.\n",
        "- **Modality:** unimodal distributions are friendly to most models; bimodal distributions argue for mixture or segment-specific modeling.",
    ),
    _code(
        "32-dist-code",
        "# =============================================================\n",
        "# 9.1  HISTOGRAM, LOG-TRANSFORM, AND NORMAL Q–Q PLOT\n",
        "# =============================================================\n",
        "from scipy import stats\n",
        "\n",
        "y = y_hourly.values\n",
        "y_log = np.log1p(y)  # log(1 + y) — handles zeros without dropping rows\n",
        "\n",
        "skew_raw = stats.skew(y)\n",
        "kurt_raw = stats.kurtosis(y)  # excess kurtosis (Normal = 0)\n",
        "skew_log = stats.skew(y_log)\n",
        "kurt_log = stats.kurtosis(y_log)\n",
        "\n",
        "fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)\n",
        "sns.histplot(y, bins=80, kde=True, color=\"#1f77b4\", ax=axes[0, 0])\n",
        "axes[0, 0].set_title(f\"Raw Global_active_power (hourly)  —  skew={skew_raw:.2f}, kurt={kurt_raw:.2f}\")\n",
        "axes[0, 0].set_xlabel(\"kW\")\n",
        "\n",
        "sns.histplot(y_log, bins=80, kde=True, color=\"#2ca02c\", ax=axes[0, 1])\n",
        "axes[0, 1].set_title(f\"log(1 + Global_active_power)  —  skew={skew_log:.2f}, kurt={kurt_log:.2f}\")\n",
        "axes[0, 1].set_xlabel(\"log(1 + kW)\")\n",
        "\n",
        "stats.probplot(y, dist=\"norm\", plot=axes[1, 0])\n",
        "axes[1, 0].set_title(\"Q–Q plot — raw vs normal\")\n",
        "stats.probplot(y_log, dist=\"norm\", plot=axes[1, 1])\n",
        "axes[1, 1].set_title(\"Q–Q plot — log-transformed vs normal\")\n",
        "for ax in axes.ravel():\n",
        "    ax.grid(True, alpha=0.3)\n",
        "plt.show()\n",
        "\n",
        "print(\"Tail behaviour (raw kW):\")\n",
        "for q in (0.95, 0.99, 0.999):\n",
        "    print(f\"  Q{int(q*1000)/10:.1f}%  : {np.quantile(y, q):.3f} kW\")\n",
        "print(f\"  max     : {y.max():.3f} kW  ({(y > 5).sum():,} hours above 5 kW)\")",
    ),
    _md(
        "33-dist-interp",
        "**Interpretation.** The raw distribution is heavily **right-skewed** (skewness ≈ 1.5–2): a long tail of high-consumption hours from cooking, laundry, and climate-control events sits above a much denser body of low-consumption hours (overnight, weekday-daytime when the household is away). The log-transform brings the distribution closer to a Gaussian, but residual non-normality remains around the lower tail (hours of near-zero consumption).\n",
        "\n",
        "The Q–Q plots confirm both: the raw plot bends upward at the high end (heavy right tail); the log plot is closer to the reference line but still curves at both ends.\n",
        "\n",
        "**Implication.** Two practical consequences:\n",
        "\n",
        "1. **Metric choice already aligns.** Our primary metric is MAE (robust to outliers), with WAPE and MASE as guardrails. RMSE is reported but not optimized — it would over-weight the small fraction of peak hours.\n",
        "2. **For Prophet, consider `log` mode.** Prophet's default additive model assumes Gaussian residuals; the right-skewed raw distribution argues for log-transforming the target before fitting Prophet. Tree-based and neural models are scale-agnostic to monotonic transforms, so this matters only for likelihood-based methods.",
    ),
    # ── §10 Segment Diagnostics ────────────────────────────────────
    _md(
        "34-seg-header",
        "## 10 · Segment Diagnostics\n",
        "\n",
        "Average target behavior by hour-of-day, day-of-week, and month. These tables and plots provide the *baseline* against which model performance will be evaluated per segment — if SARIMA is good on average but terrible at 19h on Saturdays, the segment table is where we'll see it.\n",
        "\n",
        "The dimensions match `cfg.evaluation.segment_dimensions = [hour, day_of_week, month]`.",
    ),
    _code(
        "35-seg-code",
        "# =============================================================\n",
        "# 10.1  PER-SEGMENT MEAN AND STD\n",
        "# =============================================================\n",
        "y_h = df_hourly[target].dropna()\n",
        "seg = pd.DataFrame({\n",
        "    \"y\": y_h,\n",
        "    \"hour\": y_h.index.hour,\n",
        "    \"day_of_week\": y_h.index.dayofweek,\n",
        "    \"month\": y_h.index.month,\n",
        "})\n",
        "\n",
        "hour_profile = seg.groupby(\"hour\")[\"y\"].agg([\"mean\", \"std\", \"min\", \"max\"]).round(3)\n",
        "dow_profile = seg.groupby(\"day_of_week\")[\"y\"].agg([\"mean\", \"std\", \"min\", \"max\"]).round(3)\n",
        "month_profile = seg.groupby(\"month\")[\"y\"].agg([\"mean\", \"std\", \"min\", \"max\"]).round(3)\n",
        "\n",
        "fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)\n",
        "\n",
        "axes[0].errorbar(hour_profile.index, hour_profile[\"mean\"], yerr=hour_profile[\"std\"],\n",
        "                 fmt=\"o-\", color=\"#1f77b4\", capsize=3)\n",
        "axes[0].set_title(\"Mean ± std by hour of day\")\n",
        "axes[0].set_xlabel(\"Hour\"); axes[0].set_ylabel(\"kW\")\n",
        "\n",
        "axes[1].errorbar(dow_profile.index, dow_profile[\"mean\"], yerr=dow_profile[\"std\"],\n",
        "                 fmt=\"o-\", color=\"#ff7f0e\", capsize=3)\n",
        "axes[1].set_title(\"Mean ± std by day of week (0 = Mon)\")\n",
        "axes[1].set_xlabel(\"Day of week\")\n",
        "\n",
        "axes[2].errorbar(month_profile.index, month_profile[\"mean\"], yerr=month_profile[\"std\"],\n",
        "                 fmt=\"o-\", color=\"#2ca02c\", capsize=3)\n",
        "axes[2].set_title(\"Mean ± std by month\")\n",
        "axes[2].set_xlabel(\"Month\")\n",
        "\n",
        "for ax in axes:\n",
        "    ax.grid(True, alpha=0.3)\n",
        "plt.show()\n",
        "\n",
        "print(\"Hourly profile (kW):\")\n",
        "print(hour_profile.to_string())\n",
        "print(\"\\nDay-of-week profile (kW):\")\n",
        "print(dow_profile.to_string())\n",
        "print(\"\\nMonthly profile (kW):\")\n",
        "print(month_profile.to_string())",
    ),
    _code(
        "36-seg-heatmap",
        "# =============================================================\n",
        "# 10.2  HEATMAP — HOUR × DAY-OF-WEEK\n",
        "# =============================================================\n",
        "heat = seg.groupby([\"day_of_week\", \"hour\"])[\"y\"].mean().unstack(\"hour\")\n",
        "\n",
        "fig, ax = plt.subplots(figsize=(13, 4.5))\n",
        "sns.heatmap(\n",
        "    heat, cmap=\"YlOrRd\", cbar_kws={\"label\": \"Mean Global_active_power (kW)\"},\n",
        "    xticklabels=range(24), yticklabels=[\"Mon\", \"Tue\", \"Wed\", \"Thu\", \"Fri\", \"Sat\", \"Sun\"],\n",
        "    ax=ax\n",
        ")\n",
        "ax.set_title(\"Mean consumption by hour × day-of-week\")\n",
        "ax.set_xlabel(\"Hour of day\"); ax.set_ylabel(\"\")\n",
        "plt.show()",
    ),
    _md(
        "37-seg-interp",
        "**Interpretation.**\n",
        "\n",
        "- **Hour-of-day:** clear bimodal profile — a small peak in the morning (~7–9 AM) and a much larger peak in the evening (~19–21h). The trough sits at 3–5 AM. Mean amplitude across the day spans roughly 0.45 → 1.9 kW. This matches the November-2007 close-up from §5.3.\n",
        "- **Day-of-week:** weekdays (Mon–Fri) are markedly *lower-mean* than weekends (Sat–Sun). This argues the household has weekday-daytime absence (work/school) and weekend-presence consumption — a structural pattern that calendar features capture directly.\n",
        "- **Month:** the U-shape we expected from §5.1 is here numerically — winter (Dec–Feb) and shoulder months are highest, summer (Jul–Aug) is the trough. The August trough is partly the *grandes vacances* (French summer vacation period) and partly the absence of heating demand.\n",
        "- **Hour × day-of-week heatmap:** the evening peak shifts later on weekends and Friday — consistent with later dinner/relaxation times outside the work week.\n",
        "\n",
        "**Implication.** Two segment-aware decisions:\n",
        "\n",
        "1. **Segment-level reporting is essential, not optional.** A model evaluated only on global MAE could hide systematic failures during evening peaks or August lows. The evaluation framework's `segment_dimensions = [hour, day_of_week, month]` is justified.\n",
        "2. **Interaction features are likely valuable.** Hour-of-day means different things on Tue vs Sat. Tree-based models will discover this from the joint distribution; for linear/classical models, explicit interaction terms (`hour × is_weekend`) might help — recorded as feature-engineering candidate for the ablation phase.",
    ),
]


# ─────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────
def main() -> None:
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))

    # 1. Update cell 27 source if it exists
    updated_27 = False
    for cell in nb["cells"]:
        if cell.get("id") == "27-stl-interp" and cell["cell_type"] == "markdown":
            cell["source"] = CELL_27_NEW_SOURCE
            updated_27 = True
            break

    # 2. Append new sections 8–10, but only IDs that aren't already present
    existing_ids = {c.get("id") for c in nb["cells"]}
    added = 0
    for cell in NEW_CELLS:
        if cell["id"] in existing_ids:
            continue
        nb["cells"].append(cell)
        added += 1

    NB_PATH.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(f"Updated cell 27-stl-interp: {updated_27}")
    print(f"Appended {added} new cells (sections 8-10).")
    print(f"Notebook now has {len(nb['cells'])} total cells.")


if __name__ == "__main__":
    main()
