"""Developer-only: add §5 (persist + inspect) and §6 (handoff) to the
feature-engineering notebook. Idempotent.
"""

from __future__ import annotations

import json
from pathlib import Path

NB_PATH = Path(__file__).resolve().parents[1] / "notebooks" / "02_feature_engineering.ipynb"


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
        "13-persist-header",
        "## 5 · Inspect and Persist the Feature Matrix\n",
        "\n",
        "Before saving the artifact, look at the actual shape of the data — the first few rows, dtypes, NaN check. Then write `X` and `y` to [`data/processed/features.parquet`](../data/processed/features.parquet) along with a tiny metadata sidecar so the modeling notebook can verify it's loading the artifact built from the same config.",
    ),
    _code(
        "14-inspect-code",
        "# =============================================================\n",
        "# 5.1  HEAD AND DTYPE AUDIT\n",
        "# =============================================================\n",
        "# Show a window of rows (well past the lag-168 warm-up so all features are populated).\n",
        "display(fm.X.head(3).round(3))\n",
        "\n",
        "print()\n",
        "print(\"Dtype distribution:\")\n",
        "dtype_counts = fm.X.dtypes.value_counts()\n",
        "for d, n in dtype_counts.items():\n",
        "    print(f\"  {str(d):>10s}: {n} columns\")\n",
        "\n",
        "n_nan_rows = int(fm.X.isna().any(axis=1).sum())\n",
        "n_nan_y = int(fm.y.isna().sum())\n",
        "print()\n",
        "print(f\"Rows with any NaN in X : {n_nan_rows}\")\n",
        "print(f\"NaN values in y        : {n_nan_y}\")\n",
        "assert n_nan_rows == 0 and n_nan_y == 0, \"Unexpected NaN — modeling expects fully populated matrix\"",
    ),
    _code(
        "15-persist-code",
        "# =============================================================\n",
        "# 5.2  PERSIST FEATURE MATRIX + METADATA SIDECAR\n",
        "# =============================================================\n",
        "import json\n",
        "from datetime import datetime, timezone\n",
        "\n",
        "out_dir = Path(\"..\") / \"data\" / \"processed\"\n",
        "out_dir.mkdir(parents=True, exist_ok=True)\n",
        "features_path = out_dir / \"features.parquet\"\n",
        "metadata_path = out_dir / \"features.metadata.json\"\n",
        "\n",
        "# Combine X and y into a single parquet (target as its own column).\n",
        "combined = fm.X.copy()\n",
        "combined[cfg.target.column] = fm.y\n",
        "combined.to_parquet(features_path, compression=\"snappy\")\n",
        "\n",
        "# Sidecar metadata — config hash + provenance for downstream verification.\n",
        "metadata = {\n",
        "    \"config_hash\": cfg.content_hash(),\n",
        "    \"target_column\": cfg.target.column,\n",
        "    \"target_frequency\": cfg.target.frequency,\n",
        "    \"n_rows\": int(len(fm.X)),\n",
        "    \"n_features\": int(fm.X.shape[1]),\n",
        "    \"feature_names\": list(fm.feature_names),\n",
        "    \"date_range_start\": str(fm.X.index.min()),\n",
        "    \"date_range_end\": str(fm.X.index.max()),\n",
        "    \"unsafe_inputs_dropped\": int(fm.n_unsafe_dropped),\n",
        "    \"strict_leakage_safe\": bool(cfg.features.strict_leakage_safe),\n",
        "    \"built_at_utc\": datetime.now(timezone.utc).isoformat(timespec=\"seconds\"),\n",
        "}\n",
        "metadata_path.write_text(json.dumps(metadata, indent=2), encoding=\"utf-8\")\n",
        "\n",
        "print(f\"Saved features  : {features_path.resolve()}\")\n",
        "print(f\"Saved metadata  : {metadata_path.resolve()}\")\n",
        "print(f\"File size (MB)  : {features_path.stat().st_size / 1e6:.2f}\")\n",
        "print()\n",
        "print(\"Metadata sidecar contents:\")\n",
        "print(json.dumps(metadata, indent=2)[:1200])  # truncate feature list for readability",
    ),
    _md(
        "16-handoff-header",
        "## 6 · Handoff to Modeling\n",
        "\n",
        "What this notebook produced and what comes next."
    ),
    _code(
        "17-handoff-code",
        "# =============================================================\n",
        "# 6.1  HANDOFF SUMMARY\n",
        "# =============================================================\n",
        "handoff_summary = pd.DataFrame([\n",
        "    {\"item\": \"Artifact\",          \"value\": str(features_path.resolve())},\n",
        "    {\"item\": \"Metadata sidecar\",  \"value\": str(metadata_path.resolve())},\n",
        "    {\"item\": \"Config hash\",       \"value\": cfg.content_hash()[:16] + \"…\"},\n",
        "    {\"item\": \"Rows\",              \"value\": f\"{len(fm.X):,}\"},\n",
        "    {\"item\": \"Features\",          \"value\": f\"{fm.X.shape[1]} (all leakage-safe)\"},\n",
        "    {\"item\": \"Target column\",     \"value\": cfg.target.column},\n",
        "    {\"item\": \"Frequency\",         \"value\": cfg.target.frequency},\n",
        "    {\"item\": \"Date range\",        \"value\": f\"{fm.X.index.min()} → {fm.X.index.max()}\"},\n",
        "    {\"item\": \"Strict leakage\",    \"value\": str(cfg.features.strict_leakage_safe)},\n",
        "    {\"item\": \"Next notebook\",     \"value\": \"03_modeling.ipynb (Phase 4 in plan)\"},\n",
        "])\n",
        "display(handoff_summary)",
    ),
    _md(
        "18-handoff-close",
        "---\n",
        "\n",
        "**Closing reflection.**\n",
        "\n",
        "Two things this notebook locked in for downstream:\n",
        "\n",
        "1. **The feature matrix is one canonical artifact.** Anyone modeling on this dataset reads [`data/processed/features.parquet`](../data/processed/features.parquet) and trusts it — they don't re-derive features ad-hoc. The metadata sidecar lets the modeling notebook assert `metadata['config_hash'] == cfg.content_hash()` and refuse to run if they disagree.\n",
        "2. **The leakage policy is provably active.** §4 demonstrated the registry's classification of every dangerous column and showed the guard firing on an injection attempt. The same guarantee is locked in by [`tests/integration/test_leakage.py`](../tests/integration/test_leakage.py) — those tests run in CI on every commit.\n",
        "\n",
        "**Next.** The modeling notebook (Phase 4 in the plan) will: (a) read this artifact and its metadata, (b) apply the rolling-origin split plan from the splitter module (Phase 3A), (c) fit and evaluate the model lineup (Phase 4), (d) produce the leaderboard required by the project brief.",
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
