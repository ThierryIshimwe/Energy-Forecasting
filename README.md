# Energy Forecasting — Household Electricity Consumption

A reproducible, leakage-safe time-series forecasting pipeline for hourly household electricity consumption. Four model families (SARIMA, XGBoost, LSTM, GRU) compared on a uniform rolling-origin protocol, with statistical-significance testing, per-segment analysis, a stacking ensemble, weather augmentation, and a Streamlit dashboard.

> **Course:** Artificial Intelligence Techniques — LUISS Guido Carli
> **Industry partner:** Enel Global ICT
> **Group members:** Thierry Ishimwe, Linda Carla Zorzoli, Mariavittoria Giurato, Cesar Dushimimana
> **Brief:** [`reports/Project_Work_Luiss_AI_Techniques_200326.pdf`](reports/Project_Work_Luiss_AI_Techniques_200326.pdf)

---

## Run the deliverable notebook (4 commands)

The deliverable notebook ships with outputs pre-rendered and loads tracked artifacts only — no raw-data download required for the evaluator view.

```bash
git clone https://github.com/ThierryIshimwe/Energy-Forecasting.git
cd Energy-Forecasting
python -m venv .venv && .venv\Scripts\activate          # Windows
pip install -r requirements.txt && pip install -e .
```

Then open [`notebooks/00_main_deliverable.ipynb`](notebooks/00_main_deliverable.ipynb) in Jupyter Lab or VS Code and *Run All*. To execute from the command line:

```bash
pip install -r requirements-dev.txt
python -m nbconvert --to notebook --execute notebooks/00_main_deliverable.ipynb \
    --output 00_main_deliverable.ipynb --ExecutePreprocessor.timeout=600
```

---

## Submission artifacts

| Artifact | Location | What it contains |
|---|---|---|
| Deliverable notebook | [`notebooks/00_main_deliverable.ipynb`](notebooks/00_main_deliverable.ipynb) | End-to-end story: problem, EDA, features, models, leaderboard, DM significance, per-segment, additional work, discussion, reproducibility |
| 5-page technical report | [`reports/output/technical_report.pdf`](reports/output/technical_report.pdf) | Brief-required PDF |
| 1–2 slide executive summary | [`reports/output/executive_summary.pptx`](reports/output/executive_summary.pptx) | Brief-required PPT |
| Executive-summary PDF (preview) | [`reports/output/executive_summary.pdf`](reports/output/executive_summary.pdf) | Landscape PDF of the same content, viewable without PowerPoint |

---

## Headline results

Cross-fold mean MAE (lower is better, ± 1σ across 6 rolling-origin folds):

| Model | MAE | RMSE | MASE |
|---|---:|---:|---:|
| **XGBoost + weather** | **0.334 ± 0.07** | 0.46 | 0.55 |
| XGBoost (baseline) | 0.336 ± 0.07 | 0.47 | 0.57 |
| GRU | 0.339 ± 0.09 | 0.48 | 0.58 |
| LSTM | 0.341 ± 0.07 | 0.47 | 0.58 |
| SARIMA(1,1,1)(1,1,1,24) | 0.345 ± 0.08 | 0.49 | 0.59 |
| Naive lag-1h | 0.380 ± 0.11 | 0.56 | 0.65 |

**Statistical significance** (Diebold–Mariano, α = 0.05): only XGBoost vs SARIMA is significant (p = 0.022). The top three (XGBoost, GRU, LSTM) are a statistical tie.

---

## Repository structure

```
energy-forecasting/
├── conf/base.yaml                    # Single source of truth for the pipeline
├── data/
│   ├── raw/                          # UCI source (gitignored; download via scripts/download_data.py)
│   ├── processed/features.parquet    # Engineered feature matrix (tracked, ~1.4 MB)
│   └── external/                     # Paris-Montsouris weather parquet (tracked)
├── notebooks/
│   ├── 00_main_deliverable.ipynb     # ⭐ Submission deliverable
│   ├── 01_eda.ipynb                  # EDA process notebook
│   ├── 02_feature_engineering.ipynb  # Feature build process notebook
│   └── 03_modeling.ipynb             # Model training process notebook
├── src/energy_forecasting/           # Installable package
│   ├── config.py                     # ForecastConfig (loaded from conf/base.yaml)
│   ├── data/                         # UCI loader + integrity checks
│   ├── preprocessing/                # Missing-value policy (past-only fill), resampling
│   ├── features/                     # Leakage-safe feature registry + builders
│   ├── splits/                       # Rolling-origin CV
│   ├── models/                       # SARIMA, XGBoost, LSTM, GRU, naive baselines
│   ├── evaluation/                   # MAE/RMSE/WAPE/sMAPE/MASE + DM test + segments
│   └── utils/                        # Logging, seeds, atomic I/O
├── app/                              # Streamlit dashboard (loads saved .joblib models)
├── scripts/                          # CLI entry points + reproducible build scripts
├── tests/                            # Unit + integration + leakage tests (64+ tests)
├── reports/
│   ├── output/                       # PDF report + PPTX deck (submission artifacts)
│   ├── figures/                      # Leaderboard, per-segment, AvP plots
│   └── results/                      # Per-fold prediction + metric CSVs (all tracked)
├── pyproject.toml
├── requirements.txt                  # Runtime deps (pinned)
├── requirements-dev.txt              # Dev tools + report builders (pinned)
└── README.md                         # this file
```

---

## How leakage is prevented

The pipeline is leakage-safe by construction. Three guarantees:

1. **Same-timestamp components of the target are excluded.** `Voltage`, `Global_intensity`, `Sub_metering_1/2/3`, and `Global_reactive_power` are registered in [`features/registry.py`](src/energy_forecasting/features/registry.py) with `leakage_safe=False`. The feature pipeline refuses to emit them when `strict_leakage_safe: true` (the default). Verified by [`tests/integration/test_leakage.py`](tests/integration/test_leakage.py).

2. **Lag and rolling features use only past observations.** Lags use `Series.shift(N)` with N ≥ 1; rolling features use `Series.shift(1).rolling(W)` so the window at time *t* covers `[t-W, t-1]`. Verified by [`tests/unit/test_baselines.py`](tests/unit/test_baselines.py) and the feature-builder tests.

3. **Missing-value imputation is past-only.** Short gaps (≤3 minutes) are filled with `ffill()` from the last past observation, never with a bilateral blend that would pull from future timestamps. Long gaps remain NaN. Verified by [`tests/unit/test_missing_value_policy.py`](tests/unit/test_missing_value_policy.py).

---

## Pipeline overview

```
Raw UCI (1-min) ─► validate ─► past-only impute ─► resample to hourly
                                                          ↓
                                       Leakage-safe feature pipeline
                                                          ↓
                              31 engineered features (lags, rolling stats,
                                 calendar, cyclical, holiday, outage flags)
                                                          ↓
                           Rolling-origin CV (6 folds × 30-day validation)
                                                          ↓
       ┌─ naive baselines ─ SARIMA ─ XGBoost ─ LSTM ─ GRU ─ Stacking ─┐
       └────────────── + Weather-augmented XGBoost ────────────────────┘
                                                          ↓
                MAE / RMSE / WAPE / sMAPE / MASE  +  Diebold-Mariano
                                                          ↓
                         Per-segment analysis  +  Streamlit dashboard
```

---

## Regenerating artifacts from raw

Only needed to reproduce the artifacts themselves; not needed to read the deliverable.

```bash
python scripts/download_data.py                           # ~20 MB, SHA-256 verified
python scripts/build_features.py                   # features.parquet
.venv/Scripts/python.exe scripts/run_naive_baselines.py   # ~30 sec
.venv/Scripts/python.exe scripts/run_xgboost.py    # ~1 min
.venv/Scripts/python.exe scripts/run_sarima.py     # ~14 min
.venv/Scripts/python.exe scripts/run_lstm_and_gru.py         # ~45 min (LSTM + GRU)
.venv/Scripts/python.exe scripts/run_xgboost_with_weather.py     # ~1 min
.venv/Scripts/python.exe scripts/run_stacking.py    # ~30 sec
.venv/Scripts/python.exe scripts/run_dm_test_and_segments.py      # ~5 sec
.venv/Scripts/python.exe scripts/build_technical_report.py        # PDF report
.venv/Scripts/python.exe scripts/build_executive_summary.py       # PPTX deck
```

Total wall-clock: ~90 minutes (DL dominates).

---

## Reproducibility guarantees

- Python pinned to **3.11** ([`.python-version`](.python-version)).
- All dependencies version-pinned in [`requirements.txt`](requirements.txt) (runtime) and [`requirements-dev.txt`](requirements-dev.txt) (dev tools + report builders).
- Global RNG seed = **42** ([`conf/base.yaml`](conf/base.yaml)).
- Source dataset SHA-256 verified at download ([`scripts/download_data.py`](scripts/download_data.py)).
- Every artifact carries the content hash of [`conf/base.yaml`](conf/base.yaml).
- **64+ tests** (`pytest tests/`), including:
  - [`tests/integration/test_leakage.py`](tests/integration/test_leakage.py) — registry contract
  - [`tests/unit/test_missing_value_policy.py`](tests/unit/test_missing_value_policy.py) — past-only fill
  - [`tests/unit/test_splits.py`](tests/unit/test_splits.py) — rolling-origin invariants
  - [`tests/unit/test_metrics.py`](tests/unit/test_metrics.py) — metric correctness

---

## License

[MIT](LICENSE) — for educational and research purposes (LUISS × Enel Project Work).
