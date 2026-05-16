# Handoff Report — Energy Forecasting Project

**From:** Thierry Ishimwe
**To:** Project team
**Date:** May 2026 (last updated mid-month, see change log below)
**Repo state at handoff:** see `git log` for commit history; the canonical config hash is in [`conf/base.yaml`](../conf/base.yaml) and the package version is `0.1.0`.

### Change log

- **v0.4 (current)** — SARIMA persistence gap closed: retrained all 6 folds with `remove_data()` applied before save (14 min wall-clock, deterministic — same metrics as v0.3). All 6 SARIMA prediction CSVs and 6 SARIMA `.joblib` files now exist (~215-256 MB each, local only; `.gitignore` excludes them as with every other model family). **Every model in the lineup is now end-to-end persistable.**
- **v0.3** — SARIMA all 6 folds done (MAE 0.345); LSTM + GRU all 6 folds done (MAE 0.343 / 0.339); model artifact persistence added (`utils/model_io.py`); 18 trained models + 18 prediction CSVs saved to `models/` and `reports/results/`; SARIMA AvP plot added to notebook §8.2. **Brief's "≥1 classical + ≥1 DL" requirement now satisfied with one classical (SARIMA) + two DL variants (LSTM, GRU).** Only Prophet, stacking ensemble, app rewrite, and reports remain.
- **v0.2** — XGBoost trained on all 6 folds (cross-fold MAE 0.336, current leader). SARIMA fold-1 done (MAE 0.368); remaining 5 folds running in background. LSTM and Prophet classes written but not yet trained.
- **v0.1 (initial)** — Phases 1–3 complete: foundation, EDA, feature engineering, baselines. First version of this report.

---

## 1. Executive Summary

I took the team's existing work (initial EDA notebook, feature engineering, modeling notebook, basic `src/` modules, Streamlit app skeleton) and rebuilt it into an installable, tested, leakage-safe Python package with reproducible notebooks. The architecture, EDA, feature engineering, baseline modeling, and the entire brief-required model lineup are complete.

**Brief compliance.** The brief asks for "at least two of: (a) classical statistical models (ARIMA/SARIMA), (b) deep learning models (MLP/LSTM/GRU)" evaluated with RMSE and MAE. This is **fully satisfied**: SARIMA(1,1,1)×(1,1,1,24) + LSTM + GRU are all trained, evaluated on 6-fold rolling-origin CV, with the canonical results table required by the report saved at [`reports/results/leaderboard_phase4.csv`](results/leaderboard_phase4.csv).

**Final leaderboard (cross-fold mean MAE, all 6 folds, leakage-free):**

| Rank | Model | MAE | MASE | Family |
|---:|---|---|---|---|
| 1 | XGBoost | **0.336** | 0.572 | Gradient-boosted trees |
| 2 | GRU | 0.339 | 0.578 | Deep learning |
| 3 | LSTM | 0.343 | 0.584 | Deep learning |
| 4 | SARIMA | 0.345 | 0.588 | Classical statistical |
| 5 | naïve_lag_1h | 0.380 | 0.647 | Persistence baseline |

The four real models cluster within a **3% MAE band**. None dominates; XGBoost edges out by 1% MAE over GRU, which itself beats LSTM consistently across all 6 folds.

**Still pending** (bonus / report-phase / quality-polish work):

- Prophet (class written, not yet trained — fast: ~3-5 min when run).
- Stacking ensemble (trains in ~30 s from the saved per-fold predictions).
- AvP + residual plots for SARIMA folds 2-6, LSTM, and GRU (XGBoost has them; SARIMA fold-1 has them; the rest need parity).
- Per-segment analysis (MAE by hour-of-day × model, by day-of-week, by month) in the modeling notebook.
- Diebold-Mariano significance test between the top models (the 4 real models cluster in a 3% MAE band — a DM test would tell us whether XGBoost is *significantly* better than GRU).
- 5-page technical-report PDF + 1-2 slide executive-summary PPT.
- Unified deliverable notebook (single end-to-end story for submission).
- Streamlit app rewrite (current is archived at `app/app_legacy.py.bak`; uses synthetic data).

**Most important finding from auditing the original work:** the headline result of **MAE = 0.0136** in the original `modeling.ipynb` was caused by **target leakage**. The Random Forest used `Voltage`, `Global_intensity`, and `Sub_metering_1/2/3` as features — but these are contemporaneous physical measurements that are mathematically related to the target (`P = V × I` from Ohm's law; sub-meters are *components* of `Global_active_power`). The model wasn't forecasting — it was reconstructing the target from its own components. After fixing this, the realistic baseline MAE is **~0.38 kW** (1-step-ahead naïve). Every model trained in the rebuild has to beat that to deserve a place in the report, and all four do.

---

## 2. What Was Preserved From The Original Work

The original EDA work was genuinely good and most of its analytical conclusions hold. Specifically:

| Original artifact | Status | Notes |
|---|---|---|
| `notebooks/EDA.ipynb` (Section 1 — data overview, missingness deep-dive) | ✅ Conclusions preserved and verified | Numbers I recompute match yours exactly: 2,075,259 rows, 25,979 missing (1.25%), 71 gaps, longest = 7,226 rows |
| `notebooks/EDA.ipynb` (Section 2 — Senior workflow with forecast contract, leakage register, segment diagnostics) | ✅ Concept preserved and *enforced in code* | Your leakage register was documentary; in the rebuild it's a real registry that mechanically rejects unsafe features |
| `src/data_loader.py` (semicolon-delimited CSV parser, datetime join) | ✅ Logic preserved | Rewritten with type hints, validation, error handling |
| `src/preprocessing.py` (two-stage missing policy: interpolate short, mask long, indicator columns) | ✅ Logic preserved | Same policy, same default thresholds; rewritten with type hints |
| Rolling-origin cross-validation | ✅ Concept preserved | Expanded from 4 to 6 folds |
| `reports/eda_feature_handoff.csv` | ✅ Format preserved | Regenerated with 15 decisions traceable to specific EDA sections |

The legacy files are archived for reference at [`notebooks/_legacy/`](../notebooks/_legacy/).

---

## 3. What Was Rebuilt, And Why

### 3.1 The leakage problem (critical)

**Original:** `notebooks/feature_engineering.ipynb` and `notebooks/modeling.ipynb` produced a feature matrix containing `Voltage`, `Global_intensity`, `Global_reactive_power`, `Sub_metering_1`, `Sub_metering_2`, `Sub_metering_3` — all measured at the *same timestamp* as the target.

**Why this is wrong:** at forecast-creation time, none of these values are known. The model in production would never have them. Using them in training means evaluating on a problem that doesn't exist operationally. The MAE = 0.0136 was an artefact of the model essentially computing `P = V × I` with the answer present in the inputs.

**Fix (rebuild):** introduced [`src/energy_forecasting/features/registry.py`](../src/energy_forecasting/features/registry.py) — a code-level registry that classifies every column as `leakage_safe=True/False` with a written rationale. The feature pipeline in [`features/pipeline.py`](../src/energy_forecasting/features/pipeline.py) refuses to emit any unsafe feature when `strict_leakage_safe=True` in [`conf/base.yaml`](../conf/base.yaml). This is enforced by [`tests/integration/test_leakage.py`](../tests/integration/test_leakage.py) — 15 tests that run on every commit and fail if leakage reappears.

### 3.2 The infrastructure was incomplete

**Original `src/`:** `train.py` and `evaluate.py` were stub files. `train.py` referenced `from models.sarima import SARIMAModel` and `from models.xgboost import XGBoostModel` — but no `models/` directory existed. `features.py` was just a docstring. The training pipeline could not run end-to-end without crashing.

**Original Streamlit app:** beautiful layout, sidebar, sections — but every forecast was synthetic (`np.linspace(last*0.95, last*1.05, ...)`), every metric was hardcoded ("RMSE 0.245"), and anomaly counts were literal string templates ("3 anomalies detected"). It looked finished but didn't run any real model.

**Original `requirements.txt`:** 10 lines, unpinned. Missing `pmdarima`, `prophet`, `optuna`, `mlflow`, `torch`, `pyarrow`, `loguru`, `holidays`, etc. — most of the libraries needed for the brief's deliverables.

**Original `README.md`:** 17 lines, mostly bullet points.

### 3.3 No single source of truth

**Original:** target column, frequency, horizon, fold parameters were redefined in three different notebooks with slight drift (one had `frequency='H'`, another worked at 1-minute resolution).

**Fix:** [`src/energy_forecasting/config.py`](../src/energy_forecasting/config.py) defines a `ForecastConfig` dataclass loaded once from [`conf/base.yaml`](../conf/base.yaml). Every notebook, script, and module reads from it. A content-hash on the config is embedded in every artifact for traceability.

---

## 4. Current Repository State (Phase by Phase)

### ✅ Phase 1 — Foundation, data, preprocessing, features (complete)

- **Installable package** at [`src/energy_forecasting/`](../src/energy_forecasting/) with `pyproject.toml`, editable install via `pip install -e .`
- **Config object** with validation and content-hashing
- **Annotated, pinned `requirements.txt`** (sectioned with comments explaining where each package is used)
- **Annotated, pinned `requirements-dev.txt`** with lint/test/notebook tools
- **Makefile** with one-command operations (`make data`, `make eda`, `make train`, etc.)
- **`.pre-commit-config.yaml`** with black, isort, flake8, mypy, nbstripout
- **Data loader** with schema validation and physical-plausibility checks
- **Download script** ([`scripts/download_data.py`](../scripts/download_data.py)) fetches UCI dataset, verifies SHA-256
- **Preprocessing pipeline** — two-stage missing-value policy + frequency resampling
- **Feature registry** with leakage-safe enforcement (29 entries)
- **Feature builders** — calendar (7), cyclical (6), lag (3), rolling (12), holidays (1), indicators (2)

### ✅ Phase 2 — EDA, feature engineering, and modeling notebooks (complete)

- [`notebooks/01_eda.ipynb`](../notebooks/01_eda.ipynb) — **41 cells**, fully executed. Sections: data overview, audit, missingness deep-dive, multi-scale time-series viz, ADF/KPSS stationarity, STL decomposition, ACF/PACF, distribution analysis, segment diagnostics, EDA→modeling handoff.
- [`notebooks/02_feature_engineering.ipynb`](../notebooks/02_feature_engineering.ipynb) — **19 cells**, fully executed. Loads raw, preprocesses, builds the 31-feature leakage-safe matrix, demonstrates the leakage policy firing live, persists the artefact.
- [`notebooks/03_modeling.ipynb`](../notebooks/03_modeling.ipynb) — **15 cells**, fully executed. Loads features.parquet (with config-hash check), builds the 6-fold rolling-origin split plan, fits 3 naïve baselines across all folds, produces the canonical leaderboard, generates Actual-vs-Predicted and residual plots.

### ✅ Phase 3 — Splits, metrics, baseline models (complete)

- **Rolling-origin splitter** ([`src/energy_forecasting/splits/`](../src/energy_forecasting/splits/)) — 10 unit tests, verified on real data with timeline visualization saved at [`reports/figures/split_plan_timeline.png`](figures/split_plan_timeline.png)
- **Metrics module** ([`src/energy_forecasting/evaluation/metrics.py`](../src/energy_forecasting/evaluation/metrics.py)) — MAE, RMSE, WAPE, MASE, sMAPE with custom implementations verified against sklearn (MAE/RMSE) and hand-calculable test cases. 17 unit tests pass.
- **BaseForecaster ABC** ([`src/energy_forecasting/models/base.py`](../src/energy_forecasting/models/base.py)) — interface every model satisfies
- **NaiveLagBaseline** ([`src/energy_forecasting/models/baselines.py`](../src/energy_forecasting/models/baselines.py)) — three instances (lag-1h, lag-24h, lag-168h)
- **Baseline performance leaderboard:** [`reports/results/baseline_leaderboard.csv`](results/baseline_leaderboard.csv) and per-fold detail at [`baseline_fold_results.csv`](results/baseline_fold_results.csv)

### ✅ Phase 4 — Real model lineup (brief-required portion complete)

- **SARIMA** ([`src/energy_forecasting/models/sarima.py`](../src/energy_forecasting/models/sarima.py)) — class complete, `order=(1,1,1)×(1,1,1,24)`, supports both multi-step `predict` and 1-step-ahead `predict_rolling_one_step`.
  - ✅ **All 6 folds done** — total wall-clock 160 min. Cross-fold mean MAE = **0.345 ± 0.079**, MASE = 0.588. Beats every naïve baseline; ranks **4th on the leaderboard** behind XGBoost (0.336), GRU (0.339), and LSTM (0.343). Per-fold table at [`reports/results/sarima_fold_results.csv`](results/sarima_fold_results.csv).
  - ✅ Notebook section §8 auto-loads the per-fold CSV; the cell shows the full 6-fold table and the cross-fold mean/std.
  - ⚠️ **Persistence gap (deliberate, see §4.5 below):** only the metrics CSV and fold-1 predictions are saved on disk; trained model objects and predictions for folds 2-6 were not captured. Retraining costs ~160 min, so this is accepted unless stacking ensemble specifically requires SARIMA OOF predictions.
- **XGBoost** ([`src/energy_forecasting/models/xgboost_model.py`](../src/energy_forecasting/models/xgboost_model.py)) — class + tests complete. Defaults: `max_depth=6, learning_rate=0.05, n_estimators=500`.
  - ✅ **All 6 folds done.** Mean MAE = **0.3361** ± 0.067 (current leader). Per-fold results at [`reports/results/xgboost_fold_results.csv`](results/xgboost_fold_results.csv).
  - ✅ **Feature importance** computed and saved to [`reports/results/xgboost_feature_importance.csv`](results/xgboost_feature_importance.csv) and visualized at [`reports/figures/xgboost_feature_importance.png`](figures/xgboost_feature_importance.png). Top contributors: `lag_1h` (25%), `roll_max_3h`, `hour_cos`, `month_cos`, `hour`, `lag_168h`, `is_french_holiday`.
  - ✅ Notebook section §9 added with leaderboard, feature importance plot, and Actual-vs-Predicted plot.
- **LSTM** ([`src/energy_forecasting/models/lstm_model.py`](../src/energy_forecasting/models/lstm_model.py)) — class supports both LSTM and GRU via `cell_type` parameter. Univariate sequence model (lookback=168h, hidden=64, 1 layer, 25 epochs, batch=64, Adam). Two-mode prediction API (`predict` recursive, `predict_rolling_one_step` for the leaderboard).
  - ✅ **All 6 folds done** (run time ~25 min). Cross-fold mean MAE = **0.343 ± 0.069**, MASE = 0.584. Per-fold results at [`reports/results/lstm_fold_results.csv`](results/lstm_fold_results.csv).
  - ✅ 6 trained model artifacts saved at [`models/lstm_default_fold_<N>.joblib`](../models/) (~65 KB each, loadable with `utils.load_model`).
  - ✅ Per-fold predictions saved at [`reports/results/lstm_predictions_fold<N>.csv`](results/) for all 6 folds.
  - ✅ Notebook section §10 with full table + cross-fold aggregate.
- **GRU** — same class with `cell_type='gru'`. Tests whether the simpler architecture without an output gate is sufficient for this dataset.
  - ✅ **All 6 folds done** (run time ~45 min, slower than LSTM due to CPU contention). Cross-fold mean MAE = **0.339 ± 0.086** — **beats LSTM consistently** across folds (1-2% MAE lower).
  - ✅ 6 trained model artifacts saved at [`models/gru_default_fold_<N>.joblib`](../models/) (~49 KB each).
  - ✅ Per-fold predictions saved at [`reports/results/gru_predictions_fold<N>.csv`](results/).
  - ✅ Notebook section §11 with full table + head-to-head LSTM-vs-GRU finding + the canonical 7-model leaderboard for the report.
  - **Headline finding worth surfacing in the report:** *for this dataset, at hourly resolution with a 1-week lookback, GRU's simpler architecture sufficed — the extra LSTM gate did not help.*
- **Prophet** ([`src/energy_forecasting/models/prophet_model.py`](../src/energy_forecasting/models/prophet_model.py)) — class written, registered, import-verified. Daily + weekly + yearly seasonality, French national holidays pulled from the `is_french_holiday` feature. Provides `predict_components()` for the interpretable trend / seasonality / holiday breakdown. **Not yet trained — explicitly framed as bonus content beyond the brief's minimum requirements.** Notebook §12 not yet added.
- **Stacking ensemble** — **not started.** Will combine the top 2-4 base models via a simple linear (ridge) meta-learner trained on the per-fold predictions we already saved to disk. No retraining of base models needed. Notebook §13 not yet added.

### 4.5 Model artifact persistence — what's saved and what isn't

The persistence helper is at [`src/energy_forecasting/utils/model_io.py`](../src/energy_forecasting/utils/model_io.py) — two functions, `save_model` / `load_model`, both joblib-based, both round-trip-verified for LSTM/GRU/XGBoost.

| Model | Per-fold metrics | Fold-1 predictions | Folds 2-6 predictions | Trained model files |
|---|---|---|---|---|
| 3× naïve baselines | ✅ | ✅ | ❌ | n/a (no training, predictions can be recomputed trivially) |
| **XGBoost** | ✅ | ✅ | ✅ | ✅ (6 × ~680 KB) |
| **SARIMA** | ✅ | ✅ | ✅ | ✅ (6 × ~215-256 MB; cached state stripped via `remove_data()`) |
| **LSTM** | ✅ | ✅ | ✅ | ✅ (6 × ~65 KB) |
| **GRU** | ✅ | ✅ | ✅ | ✅ (6 × ~49 KB) |
| Prophet | — | — | — | — (not trained yet) |

**Total persisted artifacts right now:**
- 24 trained model files in [`models/`](../models/) (6 each of XGBoost, SARIMA, LSTM, GRU)
- 24 per-fold prediction CSVs in [`reports/results/`](results/)
- 5 fold-results CSVs (1 per model family): `baseline_fold_results.csv`, `xgboost_fold_results.csv`, `sarima_fold_results.csv`, `lstm_fold_results.csv`, `gru_fold_results.csv`
- The canonical leaderboard at [`reports/results/leaderboard_phase4.csv`](results/leaderboard_phase4.csv)

**SARIMA file size note:** statsmodels' `SARIMAXResults` object remains larger than the other model families (~215 MB per fold vs <1 MB for the rest) even after `remove_data()` is applied — the Kalman filter and smoother result objects keep per-timestep state that statsmodels' public API cannot fully strip without breaking `.predict()` on new data. All `models/*.joblib` files are git-ignored regardless, so this only affects local disk (~1.4 GB total for SARIMA), not the repo size.

**Loading a trained model from disk** (one-liner that powers stacking + the future app rewrite):
```python
from energy_forecasting.utils import load_model
xgb_fold1 = load_model("models/xgboost_default_fold_1.joblib")
y_pred = xgb_fold1.predict(X_valid)
```

**Loading any saved model on the colleague's machine** (one-liner that powers the stacking ensemble + app rewrite):
```python
from energy_forecasting.utils import load_model
m = load_model("models/sarima_default_fold_6.joblib")  # or xgboost / lstm / gru
y_pred = m.predict(X_valid) if hasattr(m, "predict_rolling_one_step") else m.predict(X_valid)
```

**Note for the colleague:** because `models/*.joblib` is `.gitignored`, the trained models do not arrive in their clone. They have two options to obtain them:
1. **Re-run the training scripts** — all four scripts at [`scripts/_dev_run_*_all_folds.py`](../scripts/) are deterministic given the same seed. Total wall-clock to regenerate all 24 model files: XGBoost ~1 min + SARIMA ~14 min + LSTM ~25 min + GRU ~45 min ≈ ~85 minutes.
2. **Use only the per-fold prediction CSVs** (which ARE committed to git in `reports/results/`). These are sufficient for building the stacking ensemble and authoring the report; only the Streamlit app rewrite genuinely needs the model files.

### ❌ Not started

- **Streamlit app rewrite** — current `app/app.py` is archived at `app/app_legacy.py.bak` (it used synthetic data). Needs a clean rewrite that loads real trained models from disk.
- **Technical report PDF** (max 5 pages) — required by brief. Quarto template structure is planned in `reports/source/` but not authored.
- **Executive summary** (1–2 slide PPT) — required by brief
- **Hyperparameter tuning** with Optuna
- **MLflow experiment tracking** integration
- **Anomaly attribution module** with event catalog (French holidays already implemented as a feature; weather and curated events deferred)

---

## 5. Honest Performance Numbers (Leakage-Free)

All numbers below are computed on the 6-fold rolling-origin split using the clean feature matrix. The 1-step-ahead protocol is used uniformly: at each validation hour, the prediction uses actual information up to the previous hour.

### Current leaderboard (sorted by MAE)

| Rank | Model | MAE mean | MAE std | RMSE mean | MASE mean | Folds tested |
|---|---|---|---|---|---|---|
| 1 | **`xgb_default`** | **0.336** | 0.067 | 0.469 | 0.572 | 6/6 |
| 2 | `gru_default` | 0.339 | 0.086 | 0.475 | 0.578 | 6/6 |
| 3 | `lstm_default` | 0.343 | 0.069 | 0.480 | 0.584 | 6/6 |
| 4 | `sarima_(1,1,1)x(1,1,1,24)` | 0.345 | 0.079 | 0.490 | 0.588 | 6/6 |
| 5 | `naive_lag_1h` (persistence) | 0.380 | 0.107 | 0.563 | 0.647 | 6/6 |
| 6 | `naive_lag_24h_daily` | 0.487 | 0.153 | 0.715 | 0.828 | 6/6 |
| 7 | `naive_lag_168h_weekly` | 0.546 | 0.131 | 0.777 | 0.930 | 6/6 |

**Reading the leaderboard:** the four real models cluster within a **3% MAE band**. XGBoost edges out GRU by ~1%, GRU consistently beats LSTM, and SARIMA is just behind LSTM. The architecture-family ordering (tree-based > sequence-DL > classical-statistical) holds, but the gap between any two adjacent models is small enough that **a stacking ensemble could move the needle** — which is exactly why §13 of the modeling notebook is on the to-do list.

The canonical CSV behind this table is at [`reports/results/leaderboard_phase4.csv`](results/leaderboard_phase4.csv). The technical-report PDF will pull from there directly.

### Per-fold detail (XGBoost) — the per-fold spread matters

| Fold | Period | MAE | RMSE | Valid rows |
|---|---|---|---|---|
| 1 | Jun 2010 | 0.333 | 0.470 | 720 |
| 2 | Jul 2010 | 0.334 | 0.450 | 720 |
| 3 | Aug 2010 | 0.214 | 0.312 | 601 (outage) |
| 4 | Sep 2010 | 0.337 | 0.459 | 654 |
| 5 | Oct 2010 | 0.396 | 0.553 | 699 |
| 6 | Nov 2010 | 0.402 | 0.571 | 720 |

**Reading the per-fold table:** the same structural pattern the EDA predicted holds — folds 5–6 (heating-ramp) are hardest (MAE ≈ 0.40), fold 3 (August outage period with lower mean consumption and fewer rows) is easiest (MAE ≈ 0.21). XGBoost lowered the error inside every fold but didn't change which months are hard. That's a good sign — the model is learning real structure, not gaming an easy regime.

### Bar for any future model (e.g. Prophet, stacking ensemble, retuned variants)

To take the lead, a new model must clear cross-fold MAE < **0.336** (XGBoost). To matter for the report at all, it has to clear at least the lag-1 naïve (0.380). The four current real models all sit between 0.336 and 0.345 — a tight band — so a stacking ensemble has a credible shot at sub-0.336 if the errors decorrelate.

### Compare against the original

The original `model_leaderboard.csv` reported Random Forest MAE = 0.0136. That number is invalid (target leakage from contemporaneous physical features). The honest realistic regime for any sensible model on this dataset is the **0.33–0.40 kW MAE range** — exactly where the new leaderboard sits. The factor between the original bogus number and the realistic regime is ~25× — a useful calibration for anyone evaluating future modeling work on this dataset.

---

## 6. Important Per-Fold Detail (Worth Reading)

Validation fold 3 (covering August 2010) has only **601** valid rows instead of the nominal 720. This is because the hourly resampler dropped 119 empty hour buckets corresponding to the longest outage in the dataset (Aug 17–22, 2010 = 7,226 missing minutes). Folds 4 and 5 are also short by 66 and 21 rows respectively.

**Implication:** per-fold metrics are not directly comparable in variance — folds with fewer rows have noisier MAE estimates. The aggregate leaderboard reports mean and std across folds, but **per-fold tables must always be shown**, not just the mean, otherwise the August-outage fold gets masked. This is visible in [`reports/results/baseline_fold_results.csv`](results/baseline_fold_results.csv).

---

## 7. Repository Layout (For Orientation)

```
energy-forecasting/
├── conf/base.yaml                          # Single source of truth for ALL parameters
├── data/
│   ├── raw/data.txt                        # UCI dataset (2,075,259 rows)
│   └── processed/
│       ├── features.parquet                # 34,000 × 32 leakage-safe matrix
│       ├── features.metadata.json          # Config-hash audit trail
│       └── split_plan.csv                  # 6-fold rolling-origin plan
├── notebooks/
│   ├── 01_eda.ipynb                        # 41 cells, executed
│   ├── 02_feature_engineering.ipynb        # 19 cells, executed
│   ├── 03_modeling.ipynb                   # 15 cells, executed (baselines only)
│   └── _legacy/                            # Original team work, archived
├── src/energy_forecasting/                 # Installable package
│   ├── config.py                           # ForecastConfig dataclass
│   ├── data/                               # Loader + validator
│   ├── preprocessing/                      # Missing-value + resampling
│   ├── features/                           # Registry + builders + pipeline
│   ├── splits/                             # Rolling-origin
│   ├── evaluation/                         # Metrics module
│   ├── models/                             # BaseForecaster + Naive + SARIMA
│   └── utils/                              # Logging, seeds, atomic I/O
├── tests/                                  # 64 tests total, 28 in CI-critical paths
│   ├── unit/                               # config, splits, metrics, baselines
│   └── integration/test_leakage.py         # The leakage guard (15 tests)
├── scripts/
│   ├── download_data.py                    # UCI fetcher with hash verification
│   └── _dev_*.py                           # Developer scripts for notebook appending
├── reports/
│   ├── results/                            # Persisted leaderboards & fold results
│   ├── figures/                            # Auto-generated plots
│   ├── HANDOFF_REPORT.md                   # This document
│   └── Project_Work_Luiss_AI_Techniques_200326.pdf   # Original brief
├── pyproject.toml                          # Package metadata + tool configs
├── requirements.txt                        # Pinned runtime deps, annotated
├── requirements-dev.txt                    # Pinned dev deps
├── Makefile                                # `make data | features | train | report`
├── README.md
└── .pre-commit-config.yaml
```

---

## 8. How To Pick Up The Work

### 8.1 Environment setup (one-time)

```bash
# Clone (already done if you're reading this in the repo)
cd energy-forecasting

# Create the virtual environment (Python 3.11)
python -m venv .venv
.venv\Scripts\activate                      # Windows
# source .venv/bin/activate                 # macOS/Linux

# Install everything
pip install -r requirements-dev.txt
pip install -e .

# Verify it works
pytest                                       # should show ~64 passing tests
python scripts/download_data.py              # downloads UCI dataset (~20 MB)

# Quick sanity check that the trained models load cleanly:
python -c "from energy_forecasting.utils import load_model; \
  m = load_model('models/xgboost_default_fold_1.joblib'); \
  print(m.name, m.get_hyperparameters())"
```

### 8.2 Reproduce what's there

```bash
.venv/Scripts/python.exe -m nbconvert --execute --inplace --to notebook \
    notebooks/01_eda.ipynb \
    notebooks/02_feature_engineering.ipynb \
    notebooks/03_modeling.ipynb
```

After this, open each in VS Code → right-click tab → *Reopen Editor With...* → *Jupyter Notebook* to see executed cells.

### 8.3 Next concrete tasks (in priority order)

The brief-required modeling is **done**. What's left is bonus content + the report-phase deliverables. In execution order:

1. **Train Prophet on all 6 folds (~3-5 min wall-clock).** Class is at [`src/energy_forecasting/models/prophet_model.py`](../src/energy_forecasting/models/prophet_model.py). Write a small all-folds script similar to [`scripts/_dev_run_dl_all_folds.py`](../scripts/_dev_run_dl_all_folds.py) — instantiate `ProphetForecaster()`, fit, predict, save model + per-fold predictions using `utils.save_model`. Add notebook §12 with the same auto-load CSV pattern §10/§11 use. Value-add framing for the section header: *"Prophet is the only model in the lineup that produces an interpretable forecast-component decomposition (trend + seasonality + holiday effect). Use `predict_components()` to extract the per-component contribution for the report's discussion."*

2. **Add stacking ensemble (~30 s, no retraining of bases).** All 18 per-fold prediction CSVs are already on disk. Build a small script that: loads each model's per-fold predictions, concatenates them as a (n_rows, n_models) matrix, fits a `Ridge(alpha=1.0)` meta-learner using out-of-fold predictions, and reports the resulting MAE. Add notebook §13 with the headline answer: *"Does combining errors decorrelate enough to help? Δ MAE = X%."*

3. **Update modeling notebook §9.3 / §10 / §11** with per-segment analysis (MAE by hour-of-day, day-of-week, month) for the top models. Uses the saved predictions + the EDA's `segment_dimensions` from `cfg.evaluation`.

4. **Optional — retrofit SARIMA persistence (~160 min wall-clock).** Only if you want SARIMA in the stacking ensemble or the Streamlit app. Otherwise, leave it as documented in §4.5.

5. **Author the 5-page technical report PDF.** Suggested structure: Introduction (¼ page) → EDA Summary (¾ page, distilled from notebook 01) → Methodology (1 page, leakage policy + rolling-origin CV + 1-step-ahead protocol) → Results (1½ pages, the §29 leaderboard + Actual-vs-Predicted plots for the top 2 models from `reports/figures/`) → Discussion (¾ page, GRU-vs-LSTM finding + tight-cluster finding + bar against the leaked-feature 0.0136 fiction) → Conclusion (¼ page). Use Quarto or LaTeX. All numbers should `pd.read_csv` from `reports/results/` to stay sync'd with the notebook.

6. **Generate the 1–2 slide executive summary PPT** from the same data sources. Slide 1 = the leaderboard. Slide 2 = the leakage-correction story (the 25× MAE delta between original bogus 0.0136 and honest 0.336).

7. **Rewrite [`app/app.py`](../app/app.py)** (only if time permits — not in the brief's hard requirements). The synthetic-data placeholder is at [`app/app_legacy.py.bak`](../app/app_legacy.py.bak). Rebuild with `utils.load_model("models/xgboost_default_fold_6.joblib")` (most recent training window) and real inference.

### 8.4 Useful shortcuts

- All configuration is in [`conf/base.yaml`](../conf/base.yaml). To change horizon, frequency, fold count, lag set, etc., edit there — every notebook and module picks it up.
- To run only the leakage tests: `pytest -m leakage -v`
- The dev scripts in [`scripts/_dev_*.py`](../scripts/) are one-time utilities for notebook authoring; they can be deleted once the work stabilizes.
- The `features.metadata.json` sidecar stores the config hash that produced `features.parquet`. The modeling notebook asserts this matches the current config; if you change config and forget to rebuild features, modeling will fail loudly rather than silently train on stale features.

---

## 9. Key Design Decisions Worth Knowing About

1. **Leakage prevention by construction, not documentation.** Code rejects unsafe features. The original team's "leakage register" was a CSV that nobody enforced.
2. **Single source of truth.** One config object. No drift across notebooks.
3. **Honest by default.** Every number in the notebooks is computed from the loaded data. No synthetic forecasts, no hardcoded metrics. The old app showed `"RMSE 0.245"` as a literal string — that is no longer possible because the new architecture wires the leaderboard CSV directly to display widgets.
4. **Reproducible from cold clone.** Python pinned at 3.11, requirements pinned, RNG seeds set globally, data hash verified on download, artefacts carry config-hash provenance.
5. **Designed experiments, not model zoos.** Each model in the lineup tests a specific hypothesis (linear vs nonlinear, tabular vs sequential, interpretable vs black-box). The point isn't to try ten things — it's to learn something from each.
6. **1-step-ahead protocol for evaluation.** All models predict `y[t]` using actual information up to `t-1`. This matches the implicit protocol of the naïve baselines. The brief's `horizon_hours=24` refers to validation-window structure, not per-prediction horizon. We can switch to 24-step-ahead later if needed; it requires a separate feature pipeline that filters lags < 24h.

---

## 10. Acknowledgements

The original EDA work (especially the missingness deep-dive in Section 1 and the senior workflow framing in Section 2) was thoughtful and the conclusions hold up. The rebuild preserved those conclusions and built them into enforceable code rather than documentation. The leakage issue caught in the original `feature_engineering.ipynb` is the kind of subtle bug that even careful work can produce — flagging it now means we don't ship a report with an invalid headline number.

Questions about anything in this report or the codebase: see commit history, file docstrings, and the test suite. Every architectural decision has a written rationale somewhere in the repo.

