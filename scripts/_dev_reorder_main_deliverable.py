"""Move §11 External Context to §3 (right after the Executive Summary) and
renumber every downstream section + cross-reference."""

from __future__ import annotations

import json
from pathlib import Path

NB_PATH = Path(__file__).resolve().parents[1] / "notebooks" / "00_main_deliverable.ipynb"


def _src(cell: dict) -> str:
    s = cell.get("source", [])
    return "".join(s) if isinstance(s, list) else s


def _set_src(cell: dict, new: str) -> None:
    cell["source"] = new.splitlines(keepends=True)


def _replace_all(cell: dict, mapping: list[tuple[str, str]]) -> None:
    s = _src(cell)
    for old, new in mapping:
        s = s.replace(old, new)
    _set_src(cell, s)


def main() -> None:
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
    cells = nb["cells"]

    # ── 1. Locate the External Context cell (currently §11). ─────────
    ec_idx = next(
        (
            i for i, c in enumerate(cells)
            if c.get("cell_type") == "markdown"
            and "External Context" in _src(c)
            and "## 11 ·" in _src(c)
        ),
        None,
    )
    if ec_idx is None:
        raise SystemExit("External Context cell not found")

    ec_cell = cells.pop(ec_idx)

    # ── 2. Update External-Context content for the new §3 position. ──
    _replace_all(
        ec_cell,
        [
            ("## 11 · External Context", "## 3 · External Context"),
            # Forward-reference renumbering (these point to sections that come AFTER §3):
            ("§3, §6, and §8", "§4, §7, and §9"),
            ("the per-segment analysis (§8) flags",
             "the per-segment analysis in §9 will flag"),
            ("The per-segment analysis (§8)",
             "The per-segment analysis (§9)"),
        ],
    )

    # ── 3. Insert after the Executive Summary (cell id 03-findings). ──
    findings_idx = next(i for i, c in enumerate(cells) if c.get("id") == "03-findings")
    cells.insert(findings_idx + 1, ec_cell)

    # ── 4. Section-header renumbering for the cells that shifted. ────
    HEADERS = [
        ("## 3 · EDA",            "## 4 · EDA"),
        ("## 4 · Feature",        "## 5 · Feature"),
        ("## 5 · Modeling",       "## 6 · Modeling"),
        ("## 6 · Leaderboard",    "## 7 · Leaderboard"),
        ("## 7 · Statistical",    "## 8 · Statistical"),
        ("## 8 · Per-Segment",    "## 9 · Per-Segment"),
        ("## 9 · Additional",     "## 10 · Additional"),
        ("## 10 · Discussion",    "## 11 · Discussion"),
        # §12 Conclusion and §13 Reproducibility are already correctly numbered.
    ]

    SUBSECTIONS = [
        ("### 9.1 XGBoost",  "### 10.1 XGBoost"),
        ("### 9.2 Stacking", "### 10.2 Stacking"),
        ("### 9.3 Prophet",  "### 10.3 Prophet"),
    ]

    # ── 5. Cross-reference updates inside prose. ────────────────────
    CROSS_REFS = [
        # In §8 DM interpretation (was §7) — references peak hours which moved §8→§9:
        ("see §8", "see §9"),
        # In §11 Discussion (was §10) — multiple refs to renumbered sections:
        ("(Diebold-Mariano, §7)", "(Diebold-Mariano, §8)"),
        ("(§9.2)", "(§10.2)"),
        ("analysis (§8) shows", "analysis (§9) shows"),
        ("Per-segment analysis (§8)", "Per-segment analysis (§9)"),
        # In §12 Conclusion + Future Work — references External Context which moved §11→§3:
        ("real-world context (§11)", "real-world context (§3)"),
        ("Verify the §11 ", "Verify the §3 "),
    ]

    for cell in cells:
        if cell.get("cell_type") != "markdown":
            continue
        _replace_all(cell, HEADERS + SUBSECTIONS + CROSS_REFS)

    NB_PATH.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(f"Reordered: {NB_PATH.name}")
    print(f"  - External Context moved to §3 (right after Executive Summary)")
    print(f"  - 8 section headers renumbered; 3 subsections renumbered")
    print(f"  - 7 cross-reference patterns updated")


if __name__ == "__main__":
    main()
