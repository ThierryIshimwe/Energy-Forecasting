"""Feature registry: the single source of truth for which features are leakage-safe.

Each feature (or feature family) is registered with metadata. The pipeline
consults the registry on every column it would emit and refuses to ship
unsafe features in strict mode.

What "leakage-safe" means here:

    A feature is *leakage-safe* if its value is computable from information
    available at the forecast creation timestamp — i.e., before the target
    value for that timestamp would be known.

Examples of safe features:
    - Calendar features (hour, day of week) — known from the timestamp itself
    - Lag features (target shifted by ≥1 period) — observed in the past
    - Rolling statistics (mean of past N values, shifted by 1) — past-only

Examples of UNSAFE features (and why):
    - ``Global_intensity`` — the current/amperage at the SAME timestamp as
      the target. Power = Voltage × Intensity, so this gives the model the
      answer directly. Also: in production we wouldn't *know* future
      intensity any more than we'd know future power.
    - ``Sub_metering_1/2/3`` — sub-circuit consumption that SUMS into the
      global active power target. Using these is target leakage by
      construction.
    - ``Voltage`` — physically correlated with active power through the
      load's power factor. Not as direct as intensity, but still
      contemporaneous and unavailable at forecast time.

The original team-member implementation included all three families above as
features, resulting in MAE of 0.0136 — an artifact of leakage, not a real
forecasting result. The registry below makes that bug mechanically impossible.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

FeatureCategory = Literal[
    "raw",  # source column from the UCI dataset
    "calendar",
    "lag",
    "rolling",
    "cyclical",
    "external",  # holidays, weather, events
    "indicator",  # outage flags, etc.
]


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    """Metadata describing one feature or one family of features.

    Attributes:
        name_pattern: Either an exact column name (``"hour"``) or a prefix
            ending in ``"*"`` (``"Global_active_power_lag_*"``). The registry
            uses prefix matching for the wildcard form.
        category: Coarse classification (see :data:`FeatureCategory`).
        leakage_safe: ``True`` iff this feature is safe to use as a model input.
        description: One-line explanation suitable for the model card.
    """

    name_pattern: str
    category: FeatureCategory
    leakage_safe: bool
    description: str


# ─────────────────────────────────────────────────────────────────────
#  THE REGISTRY
#
#  Edit this dict to add/change a feature's leakage classification.
#  The pipeline reads from here on every run — there is no other location.
# ─────────────────────────────────────────────────────────────────────

FEATURE_REGISTRY: dict[str, FeatureSpec] = {
    # ── Target column (always present, never used as its own feature) ─
    "Global_active_power": FeatureSpec(
        name_pattern="Global_active_power",
        category="raw",
        leakage_safe=True,
        description="The forecast target. Safe to use only as past lags/rolling stats.",
    ),

    # ── Contemporaneous raw columns — UNSAFE. ─────────────────────────
    # These three are why the original modeling notebook's MAE was suspect.
    # They are physically/structurally related to the target at the SAME
    # timestamp, which is exactly the definition of target leakage.
    "Global_intensity": FeatureSpec(
        name_pattern="Global_intensity",
        category="raw",
        leakage_safe=False,
        description=(
            "Current intensity at the same timestamp as target. P = V × I gives "
            "the model the answer directly."
        ),
    ),
    "Voltage": FeatureSpec(
        name_pattern="Voltage",
        category="raw",
        leakage_safe=False,
        description=(
            "Same-timestamp voltage. Physically related to active power through "
            "the load's power factor; unknown at forecast time."
        ),
    ),
    "Global_reactive_power": FeatureSpec(
        name_pattern="Global_reactive_power",
        category="raw",
        leakage_safe=False,
        description=(
            "Same-timestamp reactive power. Tightly correlated with active power "
            "via the household's overall power factor."
        ),
    ),
    "Sub_metering_1": FeatureSpec(
        name_pattern="Sub_metering_1",
        category="raw",
        leakage_safe=False,
        description="Kitchen sub-meter at same timestamp. Component of Global_active_power.",
    ),
    "Sub_metering_2": FeatureSpec(
        name_pattern="Sub_metering_2",
        category="raw",
        leakage_safe=False,
        description="Laundry sub-meter at same timestamp. Component of Global_active_power.",
    ),
    "Sub_metering_3": FeatureSpec(
        name_pattern="Sub_metering_3",
        category="raw",
        leakage_safe=False,
        description="Climate sub-meter at same timestamp. Component of Global_active_power.",
    ),

    # ── Calendar features — safe (derivable from the timestamp alone) ─
    "hour": FeatureSpec("hour", "calendar", True, "Hour of day, 0–23."),
    "day_of_week": FeatureSpec("day_of_week", "calendar", True, "Day of week, 0 (Mon) – 6 (Sun)."),
    "day_of_month": FeatureSpec("day_of_month", "calendar", True, "Day of month, 1–31."),
    "day_of_year": FeatureSpec("day_of_year", "calendar", True, "Day of year, 1–366."),
    "month": FeatureSpec("month", "calendar", True, "Month, 1–12."),
    "quarter": FeatureSpec("quarter", "calendar", True, "Quarter, 1–4."),
    "is_weekend": FeatureSpec("is_weekend", "calendar", True, "1 if Saturday or Sunday, else 0."),

    # ── Cyclical encodings — safe (deterministic from timestamp) ─────
    "hour_sin": FeatureSpec("hour_sin", "cyclical", True, "sin(2π × hour / 24)."),
    "hour_cos": FeatureSpec("hour_cos", "cyclical", True, "cos(2π × hour / 24)."),
    "day_of_week_sin": FeatureSpec(
        "day_of_week_sin", "cyclical", True, "sin(2π × day_of_week / 7)."
    ),
    "day_of_week_cos": FeatureSpec(
        "day_of_week_cos", "cyclical", True, "cos(2π × day_of_week / 7)."
    ),
    "month_sin": FeatureSpec("month_sin", "cyclical", True, "sin(2π × month / 12)."),
    "month_cos": FeatureSpec("month_cos", "cyclical", True, "cos(2π × month / 12)."),

    # ── Lag features (target shifted N hours into the past) — safe ───
    "Global_active_power_lag_*": FeatureSpec(
        "Global_active_power_lag_*",
        "lag",
        True,
        "Target value N hours before the prediction timestamp.",
    ),

    # ── Rolling statistics over past values — safe (shift(1) applied) ─
    "Global_active_power_roll_mean_*": FeatureSpec(
        "Global_active_power_roll_mean_*",
        "rolling",
        True,
        "Mean of the past N hours of target, computed with shift(1) to avoid leakage.",
    ),
    "Global_active_power_roll_std_*": FeatureSpec(
        "Global_active_power_roll_std_*",
        "rolling",
        True,
        "Std of the past N hours of target, computed with shift(1) to avoid leakage.",
    ),
    "Global_active_power_roll_min_*": FeatureSpec(
        "Global_active_power_roll_min_*",
        "rolling",
        True,
        "Min of the past N hours of target, computed with shift(1) to avoid leakage.",
    ),
    "Global_active_power_roll_max_*": FeatureSpec(
        "Global_active_power_roll_max_*",
        "rolling",
        True,
        "Max of the past N hours of target, computed with shift(1) to avoid leakage.",
    ),

    # ── External: French holidays — safe (known from calendar) ───────
    "is_french_holiday": FeatureSpec(
        "is_french_holiday",
        "external",
        True,
        "1 if the date is a French national holiday (jour férié).",
    ),
    "is_french_school_vacation": FeatureSpec(
        "is_french_school_vacation",
        "external",
        True,
        "1 if the date is in Zone C (Paris area) school vacation period.",
    ),

    # ── Outage indicators — safe (knowable at prediction time) ────────
    "is_outage_gap": FeatureSpec(
        "is_outage_gap",
        "indicator",
        True,
        "1 if this hour contained any outage minute in the raw data.",
    ),
    "is_originally_missing": FeatureSpec(
        "is_originally_missing",
        "indicator",
        True,
        "1 if this hour contained any raw row with missing values.",
    ),
}


def is_safe(column_name: str) -> bool | None:
    """Look up whether a column is leakage-safe.

    Matches exact names first, then prefix wildcards from the registry.

    Returns:
        ``True`` or ``False`` if the column is registered.
        ``None`` if the column name is unknown — callers should treat unknown
        columns as unsafe by default in strict mode.
    """
    if column_name in FEATURE_REGISTRY:
        return FEATURE_REGISTRY[column_name].leakage_safe
    for pattern, spec in FEATURE_REGISTRY.items():
        if pattern.endswith("*") and column_name.startswith(pattern[:-1]):
            return spec.leakage_safe
    return None
