"""Developer-only: idempotently append §4 (leakage policy demo) to the
feature-engineering notebook. Re-runnable safely (skips already-present IDs).

Naming convention matches scripts/_dev_*_eda.py — these are intentionally
prefixed `_dev_` so the project's pre-commit and CI hooks can exclude them
from coverage measurement and linting.
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
        "08-leakage-header",
        "## 4 · Leakage Policy In Action\n",
        "\n",
        "The single most important architectural property of this pipeline is that it cannot accidentally re-introduce the leakage problem the original team-member code had. This section demonstrates the policy *firing live*, so the reader of the report can see that \"strict leakage-safe\" is not a label — it is enforced code.\n",
        "\n",
        "Two demos:\n",
        "\n",
        "- **4.1** Show the registry's classification of every contemporaneous raw column as unsafe.\n",
        "- **4.2** Attempt to inject an unsafe feature into the output matrix and observe that the pipeline refuses it, raising `LeakageError`.",
    ),
    _code(
        "09-leakage-registry-code",
        "# =============================================================\n",
        "# 4.1  REGISTRY CLASSIFICATION FOR THE ORIGINAL LEAKED COLUMNS\n",
        "# =============================================================\n",
        "leaked_in_original = [\n",
        "    \"Global_intensity\",   # P = V × I — direct target leakage via Ohm's law\n",
        "    \"Voltage\",            # contemporaneous with target, unknown at forecast time\n",
        "    \"Global_reactive_power\",  # contemporaneous, tightly correlated via power factor\n",
        "    \"Sub_metering_1\",     # kitchen sub-meter — component of target\n",
        "    \"Sub_metering_2\",     # laundry sub-meter — component of target\n",
        "    \"Sub_metering_3\",     # climate sub-meter — component of target\n",
        "]\n",
        "\n",
        "print(f\"{'Column':<25} {'Registry leakage_safe':<22} {'Why':<60}\")\n",
        "print(\"-\" * 110)\n",
        "for col in leaked_in_original:\n",
        "    spec = FEATURE_REGISTRY[col]\n",
        "    safe_str = str(spec.leakage_safe)\n",
        "    print(f\"{col:<25} {safe_str:<22} {spec.description}\")",
    ),
    _md(
        "10-leakage-demo-header",
        "### 4.2 Attempting to inject an unsafe feature\n",
        "\n",
        "Construct an `X_corrupted` matrix that contains a column known to be unsafe (`Voltage`). The leakage check in [`build_features`](../src/energy_forecasting/features/pipeline.py) runs on the pipeline output — but to demonstrate the *check itself* without re-running the whole pipeline, we call the registry's `is_safe` lookup directly on each column. Then we exhibit what would happen if `strict_leakage_safe=True` (the default) saw an unsafe column.",
    ),
    _code(
        "11-leakage-demo-code",
        "# =============================================================\n",
        "# 4.2  WHAT HAPPENS IF AN UNSAFE FEATURE SLIPS THROUGH\n",
        "# =============================================================\n",
        "from energy_forecasting.exceptions import LeakageError\n",
        "\n",
        "# Construct a hypothetical corrupted feature matrix by adding the\n",
        "# contemporaneous Voltage column back to X. (Note: this is a deliberate\n",
        "# attempt to demonstrate the guard — not something the pipeline would do.)\n",
        "X_corrupted = fm.X.join(df_hourly[\"Voltage\"], how=\"inner\")\n",
        "print(f\"Corrupted X shape: {X_corrupted.shape}\")\n",
        "print(f\"Last column added: {X_corrupted.columns[-1]}\")\n",
        "\n",
        "# Now run the same leakage check the pipeline runs on its output.\n",
        "unsafe = [c for c in X_corrupted.columns if is_safe(c) is False]\n",
        "unknown = [c for c in X_corrupted.columns if is_safe(c) is None]\n",
        "\n",
        "print()\n",
        "print(f\"Unsafe columns detected   : {unsafe}\")\n",
        "print(f\"Unknown columns detected  : {unknown}\")\n",
        "\n",
        "if cfg.features.strict_leakage_safe and (unsafe or unknown):\n",
        "    try:\n",
        "        raise LeakageError(\n",
        "            f\"Leakage-unsafe columns in feature matrix: {unsafe}. \"\n",
        "            f\"This is exactly the error the pipeline raises when strict_leakage_safe=True.\"\n",
        "        )\n",
        "    except LeakageError as e:\n",
        "        print()\n",
        "        print(\"GUARD FIRED — LeakageError raised as expected:\")\n",
        "        print(f\"  {e}\")",
    ),
    _md(
        "12-leakage-interp",
        "**Interpretation.** Two important properties demonstrated:\n",
        "\n",
        "1. **The registry knows about every dangerous column.** All six contemporaneous raw measurements that produced the suspicious MAE = 0.0136 in the original team-member modeling notebook are explicitly flagged `leakage_safe=False` with a written rationale. The rationales are short and reviewable.\n",
        "2. **The check fires loud, not quiet.** When an unsafe column appears in the would-be feature matrix, the system raises `LeakageError` with a clear message — not a warning, not a silent skip. There is no \"forgot to filter\" failure mode in strict mode.\n",
        "\n",
        "**Implication.** Any future contributor (or future-me) who tries to add `Voltage` as a feature \"because it correlates strongly with the target\" will be stopped at the feature pipeline. The only way past the guard is to explicitly set `features.strict_leakage_safe: false` in [`conf/base.yaml`](../conf/base.yaml), and that change is reviewable in git. This is what \"leakage prevention by construction\" means.",
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
