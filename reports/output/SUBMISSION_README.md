# Submission — Energy Forecasting Deliverable

**Course:** Artificial Intelligence Techniques — LUISS Guido Carli
**Industry partner:** Enel Global ICT
**Group members:** Thierry Ishimwe, Linda Carla Zorzoli, Mariavittoria Giurato, Cesar Dushimimana

This zip is a **self-contained, runnable** bundle of the deliverable notebook with everything it needs to execute end-to-end. The notebook ships with all outputs pre-rendered, so you can read it without running anything; the package + artifacts are included so you can re-execute it if you want to verify reproducibility.

The 5-page technical report and the 1–2 slide executive summary are submitted as **separate uploads** alongside this zip.

## Contents

```
.
├── 00_main_deliverable.ipynb              ← the deliverable notebook (pre-executed)
├── requirements.txt                       ← pinned runtime dependencies
├── pyproject.toml                         ← package metadata
├── README.md                              ← project-level README
├── SUBMISSION_README.md                   ← this file
├── conf/base.yaml                         ← single-source-of-truth config
├── src/energy_forecasting/                ← installable Python package
├── data/
│   ├── processed/features.parquet         ← 31 leakage-safe features × 34,000 hourly rows
│   └── external/paris_montsouris_weather.parquet  ← weather data for §10.3
└── reports/
    ├── results/                           ← per-fold predictions + metric CSVs
    └── figures/                           ← leaderboard, per-segment, AvP plots
```

## How to read it

Open `00_main_deliverable.ipynb` in Jupyter Lab, VS Code, or any notebook viewer. Every cell already has its output rendered — tables, plots, and prints are all visible without re-executing.

## How to re-execute it (optional, for reproducibility)

```bash
# 1. Create a virtual environment
python -m venv .venv
.venv\Scripts\activate           # Windows
# source .venv/bin/activate      # macOS / Linux

# 2. Install runtime deps + the local package
pip install -r requirements.txt
pip install -e .

# 3. Install nbconvert and re-execute the notebook
pip install nbconvert
python -m nbconvert --to notebook --execute 00_main_deliverable.ipynb \
    --output 00_main_deliverable.ipynb --ExecutePreprocessor.timeout=600
```

The notebook does **not** download anything from the network — it reads only the parquet and CSV files bundled here.

## Notebook structure

Section headers are tagged so the evaluator can immediately see what's brief-required vs added-value beyond minimum:

- **§1 Problem & Dataset** *(Brief)*
- **§2 Key Findings — Executive Summary** *(Brief)*
- **§3 External Context** *(Additional value: grounds findings in documented French / EU events)*
- **§4 EDA Highlights** *(Brief)*
- **§5 Feature Engineering** *(Brief)*
- **§6 Modeling Protocol & Lineup** *(Brief — ≥1 classical + ≥1 DL)*
- **§7 Leaderboard** *(Brief — MAE / RMSE table)*
- **§8 Statistical Significance — Diebold–Mariano** *(Additional value: tests whether leaderboard ordering is statistically real)*
- **§9 Per-Segment Performance** *(Additional value: surfaces operationally important structure)*
- **§10 Additional Modeling Work** *(Additional)*
  - §10.1 XGBoost — 4th model family beyond brief minimum
  - §10.2 Stacking Ensemble — tests base-model error decorrelation
  - §10.3 Weather Augmentation — exogenous regressors give −0.77 % MAE on XGBoost
- **§11 Streamlit Dashboard** *(Additional value: interactive forecast inspection)*
- **§12 Discussion**
- **§13 Conclusion + Future Work**
- **§14 Reproducibility Appendix**

## Headline results

Cross-fold mean MAE (lower is better, ± 1σ across 6 rolling-origin folds):

| Model | MAE |
|---|---:|
| **XGBoost + weather** | **0.334 ± 0.07** |
| XGBoost (baseline) | 0.336 ± 0.07 |
| GRU | 0.339 ± 0.09 |
| LSTM | 0.341 ± 0.07 |
| SARIMA(1,1,1)(1,1,1,24) | 0.345 ± 0.08 |
| Naive lag-1h | 0.380 ± 0.11 |

**Statistical significance** (Diebold–Mariano, α = 0.05): only XGBoost vs SARIMA is significant (p = 0.022). The top three are a statistical tie.

## Project repository

The full repository (including unit tests, model-training scripts, and the Streamlit dashboard) lives at:
https://github.com/ThierryIshimwe/Energy-Forecasting
