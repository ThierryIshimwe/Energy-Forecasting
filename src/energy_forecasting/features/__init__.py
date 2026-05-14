"""Feature engineering with mechanical leakage prevention.

This subpackage implements the project's leakage policy as *code*, not as
documentation. Every feature is registered with a ``leakage_safe`` flag and
the :func:`build_features` pipeline refuses to emit unsafe features when
``config.features.strict_leakage_safe = True`` (the default).

Public API:
    FeatureSpec         — metadata about one feature
    FEATURE_REGISTRY    — module-level dict of name → FeatureSpec
    is_safe             — look up whether a column name is leakage-safe
    build_features      — config-driven feature pipeline
    FeatureMatrix       — typed result returned by build_features
"""

from energy_forecasting.features.pipeline import FeatureMatrix, build_features
from energy_forecasting.features.registry import FEATURE_REGISTRY, FeatureSpec, is_safe

__all__ = [
    "FEATURE_REGISTRY",
    "FeatureMatrix",
    "FeatureSpec",
    "build_features",
    "is_safe",
]
