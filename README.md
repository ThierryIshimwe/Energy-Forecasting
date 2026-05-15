# Energy Forecasting — Household Electricity Consumption

A reproducible, leakage-safe time-series forecasting pipeline for household electric power consumption, comparing classical statistical, gradient-boosting, and deep-learning approaches, with anomaly-attribution against real-world events.

> **Course:** AI Techniques — LUISS Guido Carli (Department of AI, Data and Decision Sciences)
> **Industry partner:** Enel Global ICT
> **Brief:** [`reports/Project_Work_Luiss_AI_Techniques_200326.pdf`](reports/Project_Work_Luiss_AI_Techniques_200326.pdf)
> **Latest progress + Phase-4 leaderboard:** see [`reports/HANDOFF_REPORT.md`](reports/HANDOFF_REPORT.md) — the auditable status doc that we update along the way.

---

## Overview

We forecast hourly active power consumption for a single household in **Sceaux (Paris area), France**, using the UCI *Individual Household Electric Power Consumption* dataset:

- **2,075,259** minute-level measurements
- **December 2006 → November 2010** (~47 months)
- **7 measured variables** (active/reactive power, voltage, intensity, 3 sub-metering circuits)
- **Forecast target:** `Global_active_power` (kW), resampled to hourly mean
- **Forecast horizon:** 24 hours (day-ahead)
- **Validation:** rolling-origin cross-validation, 6 folds × 30-day windows

The pipeline covers the full forecasting workflow — from data integrity verification through EDA, leakage-safe feature engineering, multi-model comparison with hyperparameter tuning, statistical significance testing, and anomaly attribution against curated real-world events (French holidays, school vacations, storms).

## Key Findings

> *Populated after the model training and evaluation phases. Placeholders below show the structure.*

| Metric | Value |
|--------|-------|
| Dataset | UCI Household Power, 47 months |
| Forecast target | Hourly `Global_active_power` (kW) |
| Models compared | **9** (3 baselines + ARIMA + SARIMA + XGBoost + LSTM + Prophet + Stacking) |
| Best single model | **TBD** |
| Best ensemble | **TBD** |
| Best MAE (validation) | **TBD** |
| Best RMSE (validation) | **TBD** |
| Anomalies detected | **TBD** |
| Anomalies attributed to known events | **TBD** |
| Most impactful event | **TBD** |

## Pipeline Sections

| # | Stage | Notebook / Script | Highlights |
|---|-------|-------------------|------------|
| 0 | Setup & data | `scripts/download_data.py` | UCI fetch + SHA-256 integrity check |
| 1 | EDA | `notebooks/01_eda.ipynb` | Seasonality, stationarity (ADF/KPSS), missingness, segment profiling |
| 2 | Feature engineering | `notebooks/02_feature_engineering.ipynb` | **Leakage-safe** lags, rolling stats, calendar, cyclical encodings |
| 3 | Modeling | `notebooks/03_modeling.ipynb` | 9 models, Optuna tuning, rolling-origin CV |
| 4 | Results analysis | `notebooks/04_results_analysis.ipynb` | Leaderboard, segment performance, Diebold-Mariano significance |
| 5 | Anomaly attribution | `src/energy_forecasting/attribution/` | Residual-based detection + event-catalog matching |
| 6 | Reporting | `reports/source/technical_report.qmd` | Quarto → PDF (5-page report) + Pandoc → PowerPoint slides |
| 7 | App | `app/app.py` | Streamlit dashboard with real model inference |

## Architecture

The project follows a layered design with a **single source of truth** for configuration ([`conf/base.yaml`](conf/base.yaml)). Leakage prevention is mechanical, not documentary — the feature registry refuses to emit contemporaneous-variable features when `strict_leakage_safe: true`.

```
Raw data (UCI) → Validator → Preprocessor → Feature factory → Splitter
                                                                  ↓
   ┌──────────── 9 models (baselines, ARIMA, SARIMA, XGBoost, LSTM, Prophet) ───────────┐
   └──────────────────────────────────────────────────────────────────────────────────┘
                                                                  ↓
                                          Optuna tuner → Evaluator → MLflow tracker
                                                                  ↓
                            ┌── Anomaly attribution ──┐   ┌── Reports (Quarto/LaTeX) ──┐
                            │  + event catalog (FR    │   │  + Streamlit app           │
                            │   holidays, weather)    │   └────────────────────────────┘
                            └─────────────────────────┘
```

Full design rationale in [`docs/architecture.md`](docs/architecture.md).

## Repository Structure

```
energy-forecasting/
├── conf/                          # Configuration (single source of truth)
│   ├── base.yaml                  # Default pipeline config
│   ├── models/                    # Per-model hyperparameter search spaces
│   └── experiments/               # Named ablation overrides
├── data/
│   ├── raw/                       # Immutable inputs (gitignored, hash-verified)
│   ├── interim/                   # Cleaned, not-yet-featured
│   ├── processed/                 # Final feature matrices
│   ├── external/                  # French holidays, weather, events
│   └── samples/                   # Small fixtures (committed) for tests
├── docs/                          # Architecture, decisions, results
├── notebooks/                     # Story-driven exploration
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_modeling.ipynb
│   ├── 04_results_analysis.ipynb
│   └── _legacy/                   # Archived original team-member work
├── src/energy_forecasting/        # Installable package
│   ├── config.py                  # ForecastConfig (SSoT)
│   ├── data/                      # Loader, validator, checksums
│   ├── preprocessing/             # Missing handling, resampling
│   ├── features/                  # Leakage-safe feature registry
│   ├── splits/                    # Rolling-origin CV
│   ├── models/                    # BaseForecaster + 9 implementations
│   ├── tuning/                    # Optuna integration
│   ├── evaluation/                # Metrics, significance tests, segments
│   ├── events/                    # French holidays, school cal, weather
│   ├── attribution/               # Anomaly detection + event matching
│   ├── tracking/                  # MLflow logger, model cards
│   ├── inference/                 # Production prediction service
│   └── utils/                     # Logging, seeds, atomic I/O
├── app/                           # Streamlit dashboard
├── scripts/                       # CLI entry points
├── tests/                         # Pytest suite (unit + integration + leakage)
├── reports/
│   ├── source/                    # Quarto sources (.qmd) for the 5-page report
│   ├── output/                    # Rendered PDF + slides
│   ├── figures/                   # Auto-generated charts
│   └── results/                   # Reproducible leaderboard + per-fold CSVs
├── pyproject.toml                 # Package metadata + tool configs
├── requirements.txt               # Pinned runtime dependencies
├── requirements-dev.txt           # Dev tools (lint, test, type-check)
├── Makefile                       # One-command operations
└── .pre-commit-config.yaml        # Format/lint/strip hooks
```

## Getting Started

### Prerequisites

- **Python 3.11** (see [`.python-version`](.python-version))
- **Git** for cloning
- **Quarto** (only needed for PDF report generation): [quarto.org](https://quarto.org/)

### Installation

```bash
# Clone the repository
git clone https://github.com/ThierryIshimwe/energy-forecasting.git
cd energy-forecasting

# Create and activate a virtual environment (recommended)
python -m venv .venv
.venv\Scripts\activate           # Windows
# source .venv/bin/activate      # macOS/Linux

# Install everything (package + dev tools + pre-commit hooks)
make setup
```

### Running the Pipeline

```bash
make data         # Download UCI dataset + verify SHA-256 (~130 MB)
make eda          # Execute the EDA notebook
make features     # Build the leakage-safe feature matrix
make tune         # Hyperparameter optimization (slow — runs Optuna)
make train        # Train all 9 models with best hyperparameters
make evaluate     # Generate leaderboard + per-fold + segment results
make report       # Compile the technical report PDF + slides
make app          # Launch the Streamlit dashboard
```

Run `make help` for the full list of targets.

### Reproducibility

All artifacts (data, models, metrics, predictions) carry the SHA-256 of the [`base.yaml`](conf/base.yaml) that produced them. A clean clone reproducing this pipeline should yield byte-identical results given:

- Same Python version (3.11)
- Same pinned dependencies (`requirements.txt`)
- Same RNG seed (default `42`, in `reproducibility.seed`)
- Same source dataset hash (verified at download time)

## Design Principles

This project commits to seven principles, documented in detail in [`docs/architecture.md`](docs/architecture.md):

1. **Leakage safety by construction.** Features carry a `leakage_safe` flag in code, enforced by the registry.
2. **Single source of truth.** All configuration in [`conf/base.yaml`](conf/base.yaml). No drift across notebooks.
3. **Designed experiments.** Each model tests a specific hypothesis (linear vs. nonlinear, tabular vs. sequential, etc.).
4. **Honest by default.** No synthetic data or hardcoded metrics. Every number is computed or labeled as example.
5. **Reproducible from a cold clone.** `make all` regenerates everything.
6. **Predictable code.** Aggressive consistency. Type annotations everywhere. One way to do each thing.
7. **Separation of concerns.** Data, features, models, evaluation, app — each layer has a stable contract.

## Dependencies

Highlights of the runtime stack (see [`requirements.txt`](requirements.txt) for the full annotated list with version pins):

- **Data & I/O:** `pandas`, `numpy`, `pyarrow`, `pyyaml`
- **Statistical models:** `statsmodels`, `pmdarima`, `scipy`
- **Machine learning:** `scikit-learn`, `xgboost`
- **Deep learning:** `torch` (LSTM)
- **Modern forecasting:** `prophet`
- **Hyperparameter tuning:** `optuna`
- **Experiment tracking:** `mlflow`
- **Visualization:** `matplotlib`, `seaborn`, `plotly`
- **Calendars:** `holidays` (French national)
- **App:** `streamlit`
- **Logging:** `loguru`

Development extras (in [`requirements-dev.txt`](requirements-dev.txt)): `pytest`, `black`, `isort`, `flake8`, `mypy`, `pre-commit`, `nbstripout`.

## License

[MIT](LICENSE) — for educational and research purposes (LUISS Guido Carli × Enel Project Work).
