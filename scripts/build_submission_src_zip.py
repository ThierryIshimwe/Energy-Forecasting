"""Build the prof-compliant src.zip submission bundle.

Flat layout (numbered scripts visible at top, supporting package one folder down).
"""

from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_ZIP = ROOT / "reports" / "output" / "src.zip"
STAGING = ROOT / "_src_staging"

# Scripts to copy + renumber. Order = chronological run order.
SCRIPTS_RENAME = [
    ("build_features.py", "01_build_features.py"),
    ("fetch_weather.py", "02_fetch_weather.py"),
    ("cache_eda_summary.py", "03_cache_eda_summary.py"),
    ("run_naive_baselines.py", "04_run_naive_baselines.py"),
    ("run_xgboost.py", "05_run_xgboost.py"),
    ("run_sarima.py", "06_run_sarima.py"),
    ("run_lstm_and_gru.py", "07_run_lstm_and_gru.py"),
    ("run_xgboost_with_weather.py", "08_run_xgboost_with_weather.py"),
    ("run_stacking.py", "09_run_stacking.py"),
    ("run_dm_test_and_segments.py", "10_run_dm_test_and_segments.py"),
    ("aggregate_leaderboard.py", "11_aggregate_leaderboard.py"),
    ("build_technical_report.py", "12_build_technical_report.py"),
    ("build_executive_summary.py", "13_build_presentation.py"),
]

# Path rewrites applied to copied scripts (their ROOT was 'scripts/..' = repo root;
# in the flat zip they are AT repo root of the bundle, so ROOT = parent, not parents[1]).
SCRIPT_REWRITES = [
    ("parents[1]", "parent"),
    ('"data" / "processed"', '"data"'),
    ("'data' / 'processed'", "'data'"),
    ('"data" / "external"', '"data"'),
    ("'data' / 'external'", "'data'"),
    ('"reports" / "results"', '"results"'),
    ("'reports' / 'results'", "'results'"),
    ('"reports" / "figures"', '"figures"'),
    ("'reports' / 'figures'", "'figures'"),
    ('"reports" / "output"', '"."'),
    ("'reports' / 'output'", "'.'"),
    ('"conf" / "base.yaml"', '"conf.yaml"'),
    ("'conf' / 'base.yaml'", "'conf.yaml'"),
]

# Path rewrites applied to the notebook (its working dir is the bundle root).
NOTEBOOK_REWRITES = [
    ("Path('..') / 'reports' / 'results'", "Path('.') / 'results'"),
    ("Path('..') / 'reports' / 'figures'", "Path('.') / 'figures'"),
    ("Path('..') / 'reports'", "Path('.')"),
    ("Path('..') / 'data' / 'processed' / 'features.parquet'",
     "Path('.') / 'data' / 'features.parquet'"),
    ("Path('..') / 'data' / 'processed'", "Path('.') / 'data'"),
    ("Path('..') / 'data' / 'external'", "Path('.') / 'data'"),
    # sys.path bootstrap: package is at ./energy_forecasting/, so add . to sys.path
    ("_SRC = Path('..').resolve() / 'src'", "_SRC = Path('.').resolve()"),
]

# config.py's DEFAULT_CONFIG_PATH assumes src/energy_forecasting/config.py
# and base.yaml at <root>/conf/. In the flat zip, the package is at
# <root>/energy_forecasting/config.py and the config is at <root>/conf.yaml.
CONFIG_REWRITE = (
    'DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "conf" / "base.yaml"',
    'DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "conf.yaml"',
)


def patch_script(text: str) -> str:
    for old, new in SCRIPT_REWRITES:
        text = text.replace(old, new)
    return text


def patch_notebook_cell(text: str) -> str:
    for old, new in NOTEBOOK_REWRITES:
        text = text.replace(old, new)
    return text


def build() -> None:
    if STAGING.exists():
        shutil.rmtree(STAGING)
    STAGING.mkdir(parents=True)
    for sub in ("data", "results", "figures"):
        (STAGING / sub).mkdir()

    # 1. Notebook (with path rewrites)
    nb_path = ROOT / "notebooks" / "00_main_deliverable.ipynb"
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    for c in nb["cells"]:
        if c.get("cell_type") != "code":
            continue
        src_lines = c.get("source", [])
        joined = "".join(src_lines) if isinstance(src_lines, list) else src_lines
        new = patch_notebook_cell(joined)
        if new != joined:
            c["source"] = new.splitlines(keepends=True)
    (STAGING / "main.ipynb").write_text(
        json.dumps(nb, indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    # 2. Top-level files
    shutil.copy(ROOT / "requirements.txt", STAGING / "requirements.txt")
    shutil.copy(ROOT / "conf" / "base.yaml", STAGING / "conf.yaml")

    # 3. Renumbered scripts
    for old, new in SCRIPTS_RENAME:
        sp = ROOT / "scripts" / old
        if not sp.exists():
            print(f"  WARNING: missing {old}")
            continue
        (STAGING / new).write_text(patch_script(sp.read_text(encoding="utf-8")),
                                    encoding="utf-8")

    # 4. Package (one folder; can't flatten without breaking imports)
    pkg_src = ROOT / "src" / "energy_forecasting"
    pkg_dst = STAGING / "energy_forecasting"
    shutil.copytree(pkg_src, pkg_dst, ignore=shutil.ignore_patterns("__pycache__"))
    # Patch DEFAULT_CONFIG_PATH inside the bundled package
    cfg_py = pkg_dst / "config.py"
    txt = cfg_py.read_text(encoding="utf-8")
    if CONFIG_REWRITE[0] in txt:
        cfg_py.write_text(txt.replace(*CONFIG_REWRITE), encoding="utf-8")
    else:
        print("  WARNING: DEFAULT_CONFIG_PATH line not found in bundled config.py")

    # 5. Data / results / figures
    shutil.copy(ROOT / "data" / "processed" / "features.parquet",
                STAGING / "data" / "features.parquet")
    shutil.copy(ROOT / "data" / "external" / "paris_montsouris_weather.parquet",
                STAGING / "data" / "paris_montsouris_weather.parquet")
    for f in (ROOT / "reports" / "results").glob("*"):
        if f.suffix in (".csv", ".json"):
            shutil.copy(f, STAGING / "results" / f.name)
    for f in (ROOT / "reports" / "figures").glob("*.png"):
        shutil.copy(f, STAGING / "figures" / f.name)

    # 6. README inside src
    readme = """# Energy Forecasting — Source Code

Self-contained bundle for the LUISS × Enel project: hourly household electricity
forecasting with four model families on a leakage-safe pipeline.

## Python and library versions

- Python **3.11**
- All package versions are pinned in [`requirements.txt`](requirements.txt).

## How to read the results (fast)

The notebook ships with every output pre-rendered, so no retraining is needed
to read the deliverable:

```bash
pip install -r requirements.txt
```

Then open [`main.ipynb`](main.ipynb) in Jupyter Lab or VS Code and *Run All*
(~10 seconds; only loads tracked artifacts from `data/`, `results/`,
`figures/`).

## How to retrain everything from scratch (~90 minutes total)

Each script is self-contained and run in the order indicated by its
numeric prefix:

| # | Script | What it does | Wall-clock |
|---|---|---|---|
| 01 | `01_build_features.py` | UCI raw → `data/features.parquet` (31 leakage-safe features) | ~30 s |
| 02 | `02_fetch_weather.py` | Paris–Montsouris weather → `data/paris_montsouris_weather.parquet` | ~10 s |
| 03 | `03_cache_eda_summary.py` | EDA summary JSON + seasonality plot series → `results/eda_*` | ~20 s |
| 04 | `04_run_naive_baselines.py` | 3 persistence baselines, 6 folds each | ~30 s |
| 05 | `05_run_xgboost.py` | XGBoost, 6 folds | ~1 min |
| 06 | `06_run_sarima.py` | SARIMA(1,1,1)(1,1,1,24), 6 folds | ~14 min |
| 07 | `07_run_lstm_and_gru.py` | LSTM + GRU, 6 folds each | ~45 min |
| 08 | `08_run_xgboost_with_weather.py` | XGBoost on features + weather, 6 folds | ~1 min |
| 09 | `09_run_stacking.py` | Ridge meta-learner over the 4 base models | ~30 s |
| 10 | `10_run_dm_test_and_segments.py` | Diebold–Mariano + per-segment MAE | ~5 s |
| 11 | `11_aggregate_leaderboard.py` | Aggregate per-fold CSVs → `results/leaderboard.csv` | ~2 s |
| 12 | `12_build_technical_report.py` | Regenerate technical report PDF | ~5 s |
| 13 | `13_build_presentation.py` | Regenerate the presentation deck | ~3 s |

Outputs land in `results/` (CSV / JSON) and `figures/` (PNG).

## Layout

```
.
├── README.md                    ← this file
├── requirements.txt             ← pinned Python deps
├── conf.yaml                    ← single source of truth (target, frequency, fold config)
├── main.ipynb                   ← deliverable notebook (entry point)
├── 01_*.py … 13_*.py            ← numbered, role-named scripts in chronological run order
├── energy_forecasting/          ← supporting Python package imported by the scripts
├── data/                        ← features.parquet + paris_montsouris_weather.parquet
├── results/                     ← per-fold CSVs + leaderboard.csv + DM matrix + EDA JSON
└── figures/                     ← leaderboard, per-segment, AvP plots
```
"""
    (STAGING / "README.md").write_text(readme, encoding="utf-8")

    # 7. Zip everything under a top-level "src/" arcname
    if OUT_ZIP.exists():
        OUT_ZIP.unlink()
    with zipfile.ZipFile(OUT_ZIP, "w", zipfile.ZIP_DEFLATED) as z:
        for p in STAGING.rglob("*"):
            if p.is_file():
                arc = Path("src") / p.relative_to(STAGING)
                z.write(p, arcname=str(arc))
    shutil.rmtree(STAGING)
    size_kb = OUT_ZIP.stat().st_size / 1024
    print(f"Wrote {OUT_ZIP}  ({size_kb:.0f} KB)")


if __name__ == "__main__":
    build()
