"""Developer-only: finalize notebooks/01_eda.ipynb.

Three things:
  1. Update cell `30-acf-interp` to reflect that lag 168 ACF (+0.208) is in
     fact slightly LARGER than lag 24 (+0.196). The original draft underplayed
     the weekly signal.
  2. Append §11 — the EDA → modeling handoff table that codifies every
     decision this notebook produced.
  3. Persist the handoff table to reports/results/eda_handoff.csv so other
     modules can read it.

Idempotent: re-running is safe.
"""

from __future__ import annotations

import json
from pathlib import Path

NB_PATH = Path(__file__).resolve().parents[1] / "notebooks" / "01_eda.ipynb"


CELL_30_NEW_SOURCE: list[str] = [
    "**Interpretation.** Reading the ACF printout and plot together gives a more precise picture than the §5 visual close-up alone:\n",
    "\n",
    "- A **negative** spike at lag 1 (−0.067) and a slightly stronger negative at lag 2 (−0.121) on the *differenced* series indicate that high hourly increments tend to be followed by corrections — consistent with overshoot/recovery in a thermostatically-controlled load.\n",
    "- A clear **positive** peak at lag 24 (+0.196) confirms daily seasonality survives differencing — yesterday-at-this-hour predicts today-at-this-hour beyond simple persistence.\n",
    "- **Lag 168 (+0.208) is the strongest peak across the lag range we tested** — slightly larger than lag 24 — meaning the *weekly* cycle carries more predictive signal at hourly resolution than the daily cycle alone. This was the only ACF outcome that surprised me; I expected daily to dominate.\n",
    "- Lags 48 and 72 (+0.185, +0.178) show the daily pattern persists across multiple days, but with declining strength.\n",
    "\n",
    "**Implication.**\n",
    "\n",
    "1. The lag set in [`conf/base.yaml`](../conf/base.yaml) — `lag_hours: [1, 24, 168]` — is supported, and the **lag_168 feature deserves first-class attention** during feature-importance analysis, not just lag_24.\n",
    "2. For SARIMA, the seasonal order should likely be tested with both `m=24` (daily) and `m=168` (weekly) as candidate periods. pmdarima's `auto_arima` will pick; we'll let it.\n",
    "3. We could add `lag_2h` (to capture the negative-then-recovery signature) as a feature candidate for tree-based models. Recorded in the modeling handoff.",
]


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


HANDOFF_CELLS: list[dict] = [
    _md(
        "38-handoff-header",
        "## 11 · EDA → Modeling Handoff\n",
        "\n",
        "Every decision this notebook produced, in one auditable table. The same fields are also persisted to [`reports/results/eda_handoff.csv`](../reports/results/eda_handoff.csv) so the modeling notebook reads them rather than re-deriving them — preventing accidental drift between EDA conclusions and modeling assumptions.\n",
        "\n",
        "If a decision changes here, it changes everywhere downstream. If a downstream notebook contradicts a row here, that contradiction is a bug.",
    ),
    _code(
        "39-handoff-code",
        "# =============================================================\n",
        "# 11.1  HANDOFF TABLE — EVERY DECISION FROM THIS NOTEBOOK\n",
        "# =============================================================\n",
        "import os\n",
        "\n",
        "handoff = pd.DataFrame([\n",
        "    {\n",
        "        \"area\": \"Target & frequency\",\n",
        "        \"decision\": f\"{cfg.target.column} at {cfg.target.frequency} ({cfg.target.aggregation})\",\n",
        "        \"evidence\": \"§5 — daily and weekly cycles visible at hourly resolution\",\n",
        "    },\n",
        "    {\n",
        "        \"area\": \"Forecast horizon\",\n",
        "        \"decision\": f\"{cfg.target.horizon_hours} hours ahead (day-ahead)\",\n",
        "        \"evidence\": \"§5.3 — daily cycle structure clearly visible at this horizon\",\n",
        "    },\n",
        "    {\n",
        "        \"area\": \"Missing-value policy\",\n",
        "        \"decision\": f\"Interpolate gaps ≤ {cfg.preprocessing.short_gap_max_rows} rows; preserve longer gaps as NaN with outage indicator\",\n",
        "        \"evidence\": \"§4.3 — 54 short / 17 long bimodal split; longest gap = 7,226 rows\",\n",
        "    },\n",
        "    {\n",
        "        \"area\": \"Stationarity (for ARIMA family)\",\n",
        "        \"decision\": \"d = 1 (first difference); seasonal D selected by auto_arima\",\n",
        "        \"evidence\": \"§6 — ADF rejects unit root, KPSS rejects stationarity in original; both confirm stationary after differencing\",\n",
        "    },\n",
        "    {\n",
        "        \"area\": \"Seasonality regime\",\n",
        "        \"decision\": \"Multiplicative time-varying — STL with period=24 captures the dominant daily cycle\",\n",
        "        \"evidence\": \"§7 — seasonal amplitude 5.1 kW; visible shape drift across years\",\n",
        "    },\n",
        "    {\n",
        "        \"area\": \"Lag features\",\n",
        "        \"decision\": f\"{list(cfg.features.lag_hours)} hours; lag_168 carries the largest ACF (+0.21)\",\n",
        "        \"evidence\": \"§8 — ACF spikes at lag 24/48/72/168\",\n",
        "    },\n",
        "    {\n",
        "        \"area\": \"Rolling features\",\n",
        "        \"decision\": f\"Past-only rolling mean/std/min/max over windows {list(cfg.features.rolling_window_hours)}\",\n",
        "        \"evidence\": \"§7 — residual std of 0.66 kW argues for short-term variability features\",\n",
        "    },\n",
        "    {\n",
        "        \"area\": \"Calendar features\",\n",
        "        \"decision\": \"hour, day_of_week, month, is_weekend + sin/cos cyclical encodings\",\n",
        "        \"evidence\": \"§10 — weekday vs weekend gap of +19%; monthly U-shape with August trough\",\n",
        "    },\n",
        "    {\n",
        "        \"area\": \"External regressors\",\n",
        "        \"decision\": \"French national holidays (enabled). Weather: deferred to events module (Phase 4).\",\n",
        "        \"evidence\": \"§10 — annual cycle suggests seasonal regressors will add value; brief specifies weather as future direction\",\n",
        "    },\n",
        "    {\n",
        "        \"area\": \"Leakage policy\",\n",
        "        \"decision\": f\"Strict mode = {cfg.features.strict_leakage_safe}. Reject contemporaneous raw features.\",\n",
        "        \"evidence\": \"§3 — Global_intensity max × Voltage median ≈ Global_active_power max (P = V·I exact); registry test enforces\",\n",
        "    },\n",
        "    {\n",
        "        \"area\": \"Validation strategy\",\n",
        "        \"decision\": f\"Rolling-origin, {cfg.splits.n_folds} folds × {cfg.splits.valid_window_days}d valid windows on {cfg.splits.train_window_days}d train\",\n",
        "        \"evidence\": \"cfg.splits — chosen for stable mean±std estimates without prohibitive cost\",\n",
        "    },\n",
        "    {\n",
        "        \"area\": \"Primary metric\",\n",
        "        \"decision\": f\"{cfg.evaluation.primary_metric} (guardrails: {', '.join(cfg.evaluation.guardrail_metrics)})\",\n",
        "        \"evidence\": \"§9 — right-skewed distribution argues for MAE over RMSE as primary; WAPE/MASE for cross-series comparability\",\n",
        "    },\n",
        "    {\n",
        "        \"area\": \"Target transform (Prophet)\",\n",
        "        \"decision\": \"Consider log(1+y) for Prophet; tree/neural models unchanged\",\n",
        "        \"evidence\": \"§9 — skewness ≈ 1.5; Q-Q plot bends sharply at high tail\",\n",
        "    },\n",
        "    {\n",
        "        \"area\": \"Segment reporting\",\n",
        "        \"decision\": f\"Mandatory per-segment metrics on {', '.join(cfg.evaluation.segment_dimensions)}\",\n",
        "        \"evidence\": \"§10 — heatmap shows hour × dow interaction; global metric would hide failures at evening peaks\",\n",
        "    },\n",
        "    {\n",
        "        \"area\": \"Anomaly attribution note\",\n",
        "        \"decision\": \"Operate on raw 1-min series; hourly aggregate drops empty buckets and hides outage signals\",\n",
        "        \"evidence\": \"§7.2 — STL residual during Aug 17–22 outage is artefactual (rows dropped, not flagged)\",\n",
        "    },\n",
        "])\n",
        "\n",
        "out_path = Path('..') / 'reports' / 'results' / 'eda_handoff.csv'\n",
        "out_path.parent.mkdir(parents=True, exist_ok=True)\n",
        "handoff.to_csv(out_path, index=False)\n",
        "print(f'Persisted {len(handoff)} handoff rows to {out_path.resolve()}')\n",
        "handoff",
    ),
    _md(
        "40-handoff-close",
        "---\n",
        "\n",
        "## Closing reflection\n",
        "\n",
        "Three things this EDA changed in my expectation of the modeling phase:\n",
        "\n",
        "1. **The weekly cycle matters more than I thought.** Lag 168 ACF (+0.208) edged out lag 24 (+0.196). Models that treat weekly seasonality as secondary will leave accuracy on the table.\n",
        "2. **The daily-cycle shape drifts across years.** STL's seasonal amplitude of 5.1 kW reflects both within-day swing and cross-year shift. Fixed-coefficient SARIMA may underperform on the most recent fold if the cycle has drifted; flexible models that learn calendar × time-of-day interactions should benefit.\n",
        "3. **August is the structural anomaly month.** A 60% drop from January to August (1.46 → 0.57 kW mean) — the *grandes vacances* effect plus no heating demand. Any model that treats months symmetrically (no calendar features) will systematically over-forecast Augusts.\n",
        "\n",
        "The full handoff table above lives at [`reports/results/eda_handoff.csv`](../reports/results/eda_handoff.csv) for the modeling notebook to read.",
    ),
]


def main() -> None:
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))

    # 1. Update cell 30 (ACF interpretation)
    updated_30 = False
    for cell in nb["cells"]:
        if cell.get("id") == "30-acf-interp" and cell["cell_type"] == "markdown":
            cell["source"] = CELL_30_NEW_SOURCE
            updated_30 = True
            break

    # 2. Append handoff cells if missing
    existing_ids = {c.get("id") for c in nb["cells"]}
    added = 0
    for cell in HANDOFF_CELLS:
        if cell["id"] in existing_ids:
            continue
        nb["cells"].append(cell)
        added += 1

    NB_PATH.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(f"Updated cell 30-acf-interp: {updated_30}")
    print(f"Appended {added} handoff cells.")
    print(f"Notebook now has {len(nb['cells'])} total cells.")


if __name__ == "__main__":
    main()
