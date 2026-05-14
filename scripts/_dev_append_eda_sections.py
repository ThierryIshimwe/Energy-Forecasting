"""Developer-only: append sections 5–7 to notebooks/01_eda.ipynb.

This is *not* part of the production pipeline. It exists because notebook
JSON is awkward to author by hand, and bash here-docs fight with the
apostrophes that appear in print strings. By keeping this script in the
repo (rather than running an opaque inline script), the operation that
modified the notebook stays auditable in git history.

Safe to delete after the notebook stabilizes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

NB_PATH = Path(__file__).resolve().parents[1] / "notebooks" / "01_eda.ipynb"

CellType = Literal["markdown", "code"]


def md(cell_id: str, *lines: str) -> dict:
    """Build a markdown cell with one source line per arg."""
    return {
        "cell_type": "markdown",
        "id": cell_id,
        "metadata": {},
        "source": list(lines),
    }


def code(cell_id: str, *lines: str) -> dict:
    """Build a code cell with one source line per arg."""
    return {
        "cell_type": "code",
        "id": cell_id,
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": list(lines),
    }


# ─────────────────────────────────────────────────────────────────────
#  Sections 5–7
# ─────────────────────────────────────────────────────────────────────

NEW_CELLS: list[dict] = [
    md(
        "17-viz-header",
        "## 5 · Multi-Scale Time Series Visualization\n",
        "\n",
        "Three nested views of `Global_active_power`:\n",
        "\n",
        "- **5.1 Full timeline** (weekly mean) — the big picture over ~47 months, with visible gaps where long outages live.\n",
        "- **5.2 One-month window** (June 2008 — picked because it sits *outside* any of the top-17 longest outages identified in §4.3).\n",
        "- **5.3 One-week window** (5–11 November 2007 — a clean week deep in the dataset, showing daily and weekday/weekend structure).\n",
        "\n",
        "The window choices are deliberate: showing close-ups during the August 2010 outage would conflate consumption patterns with data acquisition failure.",
    ),
    code(
        "18-viz-code",
        "# =============================================================\n",
        "# 5.1–5.3  MULTI-SCALE VISUALIZATION\n",
        "# =============================================================\n",
        "target = cfg.target.column\n",
        "series_min = df_raw[target]\n",
        "\n",
        "full_weekly  = series_min.resample(\"W\").mean()\n",
        "month_window = series_min.loc[\"2008-06-01\":\"2008-06-30\"].resample(\"D\").mean()\n",
        "week_window  = series_min.loc[\"2007-11-05\":\"2007-11-11\"].resample(\"h\").mean()\n",
        "\n",
        "fig, axes = plt.subplots(3, 1, figsize=(13, 11), constrained_layout=True)\n",
        "axes[0].plot(full_weekly.index, full_weekly.values, color=\"#1f77b4\", linewidth=1.4)\n",
        "axes[0].set_title(\"5.1  Full timeline — weekly mean of Global_active_power\")\n",
        "axes[0].set_ylabel(\"kW\")\n",
        "axes[1].plot(month_window.index, month_window.values, color=\"#ff7f0e\", linewidth=1.8, marker=\"o\", markersize=4)\n",
        "axes[1].set_title(\"5.2  June 2008 — daily mean (clean month, no long outage)\")\n",
        "axes[1].set_ylabel(\"kW\")\n",
        "axes[2].plot(week_window.index, week_window.values, color=\"#2ca02c\", linewidth=1.4)\n",
        "axes[2].set_title(\"5.3  5–11 Nov 2007 — hourly mean (one week)\")\n",
        "axes[2].set_ylabel(\"kW\")\n",
        "for ax in axes:\n",
        "    ax.grid(True, alpha=0.3)\n",
        "    ax.tick_params(axis=\"x\", rotation=20)\n",
        "plt.show()\n",
        "\n",
        "print(\"Weekly summary statistics for 5–11 Nov 2007:\")\n",
        "print(f\"  Mean    : {week_window.mean():.3f} kW\")\n",
        "print(f\"  Peak    : {week_window.max():.3f} kW at {week_window.idxmax()}\")\n",
        "print(f\"  Trough  : {week_window.min():.3f} kW at {week_window.idxmin()}\")",
    ),
    md(
        "19-viz-interp",
        "**Interpretation.** The full timeline shows a clear *annual* cycle: consumption rises in winter (Dec–Feb, heating-dominated months for a Paris household) and falls in summer (Jun–Aug, low heating demand and the August vacation effect). Trend is roughly flat across years — this household's *average* consumption is stable; the variation is seasonal.\n",
        "\n",
        "The June 2008 daily view shows a sub-monthly rhythm: some days double-peak (around the mean), others fall to half of it. Without weather data we can't fully explain individual days, but the variability is real.\n",
        "\n",
        "The November 2007 weekly close-up is the clearest evidence of *daily* and *weekly* seasonality together: every day has a morning ramp around 6–8 AM and an evening peak around 19–21h; weekend days (last two) show a different shape — flatter mornings, slightly later evening peak. This is the structural signal SARIMA was designed to model.\n",
        "\n",
        "**Implication.** Three seasonality scales (annual, weekly, daily) are present and visible. The modeling lineup must reflect this: lag features at 1h, 24h, and 168h (1 week) are non-negotiable; SARIMA's seasonal period should be 24 (hourly seasonality dominates the day-ahead horizon).",
    ),
    md(
        "20-stat-header",
        "## 6 · Stationarity Analysis\n",
        "\n",
        "A series $y_t$ is **(weakly) stationary** if its mean and variance are constant in time and its autocovariance $\\mathrm{Cov}(y_t, y_{t+h})$ depends only on lag $h$, not on $t$. ARIMA-family models assume stationarity after differencing.\n",
        "\n",
        "Two complementary tests:\n",
        "\n",
        "- **Augmented Dickey–Fuller (ADF)** — null hypothesis: the series has a unit root (non-stationary). Small $p$ → reject non-stationarity.\n",
        "- **KPSS** — null hypothesis: the series is stationary around a level. Small $p$ → reject stationarity.\n",
        "\n",
        "Using both reduces false confidence:\n",
        "\n",
        "| ADF rejects $H_0$? | KPSS rejects $H_0$? | Verdict |\n",
        "|:--:|:--:|:--|\n",
        "| ✅ | ❌ | Likely stationary |\n",
        "| ❌ | ✅ | Likely non-stationary |\n",
        "| ✅ | ✅ | Mixed — seasonal structure |\n",
        "| ❌ | ❌ | Mixed — inspect visually |\n",
        "\n",
        "We test (a) the cleaned hourly target series produced by [`preprocess`](../src/energy_forecasting/preprocessing/pipeline.py), and (b) the first-differenced series.",
    ),
    code(
        "21-stat-code",
        "# =============================================================\n",
        "# 6.1  PREPROCESS TO HOURLY, RUN ADF + KPSS\n",
        "# =============================================================\n",
        "from energy_forecasting.preprocessing import preprocess\n",
        "from statsmodels.tsa.stattools import adfuller, kpss\n",
        "import warnings\n",
        "\n",
        "df_hourly, prep_report = preprocess(df_raw, config=cfg)\n",
        "y_hourly = df_hourly[target].dropna()\n",
        "y_diff1 = y_hourly.diff().dropna()\n",
        "\n",
        "print(f\"Hourly series length     : {len(y_hourly):,}\")\n",
        "print(f\"First-diff series length : {len(y_diff1):,}\")\n",
        "print(f\"Long outages preserved   : {prep_report.n_long_gaps_preserved} (in raw 1-min data)\\n\")\n",
        "\n",
        "def run_stationarity(ts, label):\n",
        "    with warnings.catch_warnings():\n",
        "        warnings.simplefilter(\"ignore\")\n",
        "        adf_stat, adf_p, _, _, adf_crit, _ = adfuller(ts, maxlag=40, autolag=\"AIC\")\n",
        "        kpss_stat, kpss_p, _, kpss_crit = kpss(ts, regression=\"c\", nlags=\"auto\")\n",
        "    adf_stationary = adf_p < 0.05\n",
        "    kpss_stationary = kpss_p > 0.05\n",
        "    if adf_stationary and kpss_stationary:\n",
        "        verdict = \"Likely STATIONARY\"\n",
        "    elif not adf_stationary and not kpss_stationary:\n",
        "        verdict = \"Likely NON-STATIONARY\"\n",
        "    else:\n",
        "        verdict = \"MIXED evidence (seasonal structure likely)\"\n",
        "    print(f\"=== {label} ===\")\n",
        "    print(f\"  ADF  statistic = {adf_stat:>9.3f}   p = {adf_p:.4f}   critical 5% = {adf_crit['5%']:.3f}\")\n",
        "    print(f\"  KPSS statistic = {kpss_stat:>9.3f}   p = {kpss_p:.4f}   critical 5% = {kpss_crit['5%']:.3f}\")\n",
        "    print(f\"  Verdict: {verdict}\\n\")\n",
        "\n",
        "run_stationarity(y_hourly, \"Hourly Global_active_power (original)\")\n",
        "run_stationarity(y_diff1, \"Hourly Global_active_power (first-differenced)\")",
    ),
    code(
        "22-stat-rolling",
        "# =============================================================\n",
        "# 6.2  ROLLING-WINDOW VISUAL CROSS-CHECK\n",
        "# =============================================================\n",
        "WIN = 24 * 7  # one week\n",
        "\n",
        "fig, axes = plt.subplots(2, 1, figsize=(13, 8), constrained_layout=True)\n",
        "for ax, series, title in [\n",
        "    (axes[0], y_hourly, \"Original hourly series — 1-week rolling mean & std\"),\n",
        "    (axes[1], y_diff1, \"First-differenced — 1-week rolling mean & std\"),\n",
        "]:\n",
        "    rmean = series.rolling(WIN, min_periods=WIN // 2).mean()\n",
        "    rstd = series.rolling(WIN, min_periods=WIN // 2).std()\n",
        "    ax.plot(series.index, series.values, alpha=0.35, linewidth=0.6, label=\"series\", color=\"#6c757d\")\n",
        "    ax.plot(rmean.index, rmean.values, linewidth=1.8, label=\"rolling mean\", color=\"#1f77b4\")\n",
        "    ax.plot(rstd.index, rstd.values, linewidth=1.8, label=\"rolling std\", color=\"#d62728\")\n",
        "    ax.set_title(title); ax.set_ylabel(\"kW\")\n",
        "    ax.legend(loc=\"upper right\"); ax.grid(True, alpha=0.3)\n",
        "plt.show()",
    ),
    md(
        "23-stat-interp",
        "**Interpretation.** The ADF–KPSS verdict on the original series is *mixed* — ADF rejects the unit-root null (the series isn't a random walk) but KPSS also rejects stationarity (the mean varies seasonally). This is the textbook signal of a **seasonally structured** series: not unit-rooted, but not flat-mean either. The rolling-mean overlay confirms it visually — seasonal swings shift the mean across years.\n",
        "\n",
        "After first differencing, both tests agree on stationarity: the differenced series is centred around zero with constant variance.\n",
        "\n",
        "**Implication.** SARIMA's non-seasonal differencing order $d$ should be **1**. The seasonal differencing order $D$ depends on whether annual or weekly cycles dominate at the 24-hour-ahead horizon — Box–Jenkins selection in the modeling notebook will pick it. For ML/DL models, lag features carry equivalent information; stationarity is not strictly required.",
    ),
    md(
        "24-stl-header",
        "## 7 · Seasonal Decomposition\n",
        "\n",
        "Additive decomposition into three components:\n",
        "\n",
        "$$y_t = T_t + S_t + R_t$$\n",
        "\n",
        "where $T_t$ is the slowly-varying trend, $S_t$ is the seasonal component (period = 24 hours = daily cycle), and $R_t$ is the residual. **STL** (Seasonal-Trend decomposition via LOESS, Cleveland et al. 1990) handles outliers better than classical decomposition and lets the seasonal component vary slowly through time.\n",
        "\n",
        "We then **zoom on the August 2010 outage** to look at residual behavior: a sustained $|R_t|$ excursion around a known outage is exactly the signal the anomaly-attribution module will later use.",
    ),
    code(
        "25-stl-code",
        "# =============================================================\n",
        "# 7.1  STL DECOMPOSITION (DAILY PERIOD)\n",
        "# =============================================================\n",
        "from statsmodels.tsa.seasonal import STL\n",
        "\n",
        "# STL needs a contiguous series. The hourly target has long-outage NaNs;\n",
        "# fill them with the daily-cycle mean for the decomposition only\n",
        "# (exploratory — never used as model input).\n",
        "y_for_stl = df_hourly[target].copy()\n",
        "y_for_stl = y_for_stl.fillna(y_for_stl.groupby(y_for_stl.index.hour).transform(\"mean\"))\n",
        "\n",
        "stl_result = STL(y_for_stl, period=24, robust=True).fit()\n",
        "\n",
        "fig, axes = plt.subplots(4, 1, figsize=(13, 11), constrained_layout=True, sharex=True)\n",
        "axes[0].plot(y_for_stl.index, y_for_stl.values, color=\"#1f77b4\", linewidth=0.6)\n",
        "axes[0].set_title(\"Observed (hourly, NaN-imputed for STL)\"); axes[0].set_ylabel(\"kW\")\n",
        "axes[1].plot(stl_result.trend.index, stl_result.trend.values, color=\"#2ca02c\", linewidth=1.2)\n",
        "axes[1].set_title(\"Trend\"); axes[1].set_ylabel(\"kW\")\n",
        "axes[2].plot(stl_result.seasonal.index, stl_result.seasonal.values, color=\"#ff7f0e\", linewidth=0.6)\n",
        "axes[2].set_title(\"Seasonal (period = 24h)\"); axes[2].set_ylabel(\"kW\")\n",
        "axes[3].plot(stl_result.resid.index, stl_result.resid.values, color=\"#d62728\", linewidth=0.4)\n",
        "axes[3].set_title(\"Residual\"); axes[3].set_ylabel(\"kW\")\n",
        "for ax in axes:\n",
        "    ax.grid(True, alpha=0.3)\n",
        "plt.show()\n",
        "\n",
        "print(f\"Seasonal amplitude (max - min) : {stl_result.seasonal.max() - stl_result.seasonal.min():.3f} kW\")\n",
        "print(f\"Trend range                    : {stl_result.trend.min():.3f} – {stl_result.trend.max():.3f} kW\")\n",
        "print(f\"Residual std                   : {stl_result.resid.std():.3f} kW\")",
    ),
    code(
        "26-stl-august",
        "# =============================================================\n",
        "# 7.2  ZOOM ON THE AUGUST 2010 OUTAGE — RESIDUAL BEHAVIOR\n",
        "# =============================================================\n",
        "zoom_start, zoom_end = \"2010-08-10\", \"2010-09-05\"\n",
        "obs   = y_for_stl.loc[zoom_start:zoom_end]\n",
        "trend = stl_result.trend.loc[zoom_start:zoom_end]\n",
        "resid = stl_result.resid.loc[zoom_start:zoom_end]\n",
        "\n",
        "outage_a = pd.Timestamp(\"2010-08-17 21:02:00\")\n",
        "outage_b = pd.Timestamp(\"2010-08-22 21:27:00\")\n",
        "\n",
        "fig, axes = plt.subplots(2, 1, figsize=(13, 7), constrained_layout=True, sharex=True)\n",
        "axes[0].plot(obs.index, obs.values, color=\"#1f77b4\", linewidth=1, label=\"observed (imputed)\")\n",
        "axes[0].plot(trend.index, trend.values, color=\"#2ca02c\", linewidth=1.6, label=\"trend\")\n",
        "axes[0].axvspan(outage_a, outage_b, alpha=0.18, color=\"red\", label=\"Aug 17–22 outage\")\n",
        "axes[0].set_title(\"Observed vs trend, mid-Aug to early-Sep 2010\"); axes[0].set_ylabel(\"kW\"); axes[0].legend()\n",
        "axes[1].plot(resid.index, resid.values, color=\"#d62728\", linewidth=0.7)\n",
        "axes[1].axhline(0, color=\"black\", linewidth=0.6)\n",
        "axes[1].axvspan(outage_a, outage_b, alpha=0.18, color=\"red\")\n",
        "axes[1].set_title(\"STL residual — sustained excursion during outage = anomaly attribution candidate\")\n",
        "axes[1].set_ylabel(\"kW\")\n",
        "for ax in axes:\n",
        "    ax.grid(True, alpha=0.3)\n",
        "plt.show()\n",
        "\n",
        "during_outage = resid.loc[outage_a:outage_b]\n",
        "outside_outage = pd.concat([resid.loc[:outage_a], resid.loc[outage_b:]])\n",
        "print(f\"Residual |mean| during outage   : {during_outage.abs().mean():.3f} kW\")\n",
        "print(f\"Residual |mean| outside outage  : {outside_outage.abs().mean():.3f} kW\")\n",
        "print(f\"Residual std during outage      : {during_outage.std():.3f} kW\")\n",
        "print(f\"Residual std outside outage     : {outside_outage.std():.3f} kW\")",
    ),
    md(
        "27-stl-interp",
        "**Interpretation.** The decomposition reveals three regimes whose actual magnitudes were larger than expected:\n",
        "\n",
        "- The **trend** ranges from 0.17 to 2.62 kW — about a 15× swing between low and high regimes. Far from a flat \"baseline + cycle\" structure; this household's average consumption shifts substantially across years and seasons.\n",
        "- The **seasonal (24h) component** has amplitude (max − min) of ~5.1 kW. Because STL lets the seasonal pattern *evolve slowly over time*, this number captures both within-day swing *and* the shift of the daily-cycle shape across years. A fixed-coefficient model (classical SARIMA) treats the daily cycle as constant; the data argues it is **time-varying**, which is a structural advantage for flexible models (XGBoost, LSTM) that can learn cycle-shift jointly with calendar features.\n",
        "- The **residual std** is 0.659 kW — substantial. Even after trend + daily-seasonal removal, a kilowatt-scale spread of unexplained variation remains. Some of this is weather-driven, some is event-driven, some is genuine household stochasticity.\n",
        "\n",
        "**A note on the August 2010 zoom (cell 7.2).** The numerical comparison in the printout shows residual `|mean|` of 0.072 kW *during* outage vs 0.219 kW *outside*, with std = NaN during outage. This is not the anomaly-detection signal I expected to surface. The reason: our hourly preprocessor already *drops* the 421 fully-empty hour buckets that fall inside long outages, so there are simply no hourly rows to compute residuals over during the Aug 17–22 gap — the few rows present are at the edges where data partially returned. **Implication:** anomaly attribution against discrete outage events needs to operate on the raw minute-level series (or with a different resampling that keeps NaN hours), not on the dropped-bucket hourly aggregate. Recorded as work item for the anomaly-attribution module.\n",
        "\n",
        "**Implication for modeling.** Two concrete decisions:\n",
        "\n",
        "1. **SARIMA with `seasonal_order=(_, _, _, 24)`** is the natural classical fit, but with the caveat that its fixed daily-cycle assumption may underperform on years where the daily shape has drifted.\n",
        "2. **ML/DL models** should receive the `lag_24h`, `lag_168h`, and cyclical `hour_sin/hour_cos` features (already registered as safe). They can learn cycle-shift implicitly through year-or-month interactions with hour-of-day.",
    ),
]


def main() -> None:
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
    existing_ids = {c.get("id") for c in nb["cells"]}
    new_ids = {c["id"] for c in NEW_CELLS}
    if new_ids & existing_ids:
        print(f"Some cell ids already present — refusing to duplicate: {sorted(new_ids & existing_ids)}")
        return
    nb["cells"].extend(NEW_CELLS)
    NB_PATH.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(f"Appended {len(NEW_CELLS)} cells. Notebook now has {len(nb['cells'])} total cells.")


if __name__ == "__main__":
    main()
