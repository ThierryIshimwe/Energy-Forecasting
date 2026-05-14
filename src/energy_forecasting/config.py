"""ForecastConfig — the single source of truth for the entire pipeline.

Every module that needs to know about *anything* configurable (target column,
forecast horizon, validation strategy, hyperparameter budgets, RNG seed)
imports this object. There is no other authoritative location.

Design rules:

1. **Frozen and slotted.** Configs are immutable after construction; mutation
   is a programming error. ``slots=True`` also catches typos like
   ``cfg.frequecy`` at runtime instead of silently creating a new attribute.

2. **Validated on construction.** Invalid values raise :class:`ConfigurationError`
   immediately at load time, with a clear message. Garbage data never reaches
   downstream modules.

3. **Hashable.** ``content_hash()`` returns a stable digest of all values, used
   to tag artifacts so any output can be traced back to the exact config that
   produced it.

4. **YAML-loadable.** ``ForecastConfig.from_yaml(path)`` is the canonical entry
   point. Direct construction is supported for tests.

The configuration is partitioned into named sub-sections (``data``, ``target``,
``features``, ...) rather than a flat namespace, so unrelated changes do not
collide in code review.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from energy_forecasting.exceptions import ConfigurationError

# ── Allowed value sets (used at validation time) ─────────────────────
_VALID_FREQUENCIES: frozenset[str] = frozenset({"1min", "5min", "15min", "30min", "1h", "1D"})
_VALID_AGGREGATIONS: frozenset[str] = frozenset({"mean", "sum", "median", "first", "last"})
_VALID_INTERPOLATIONS: frozenset[str] = frozenset({"time", "linear"})
_VALID_METRICS: frozenset[str] = frozenset({"MAE", "RMSE", "MAPE", "WAPE", "MASE", "sMAPE"})
_VALID_SEGMENTS: frozenset[str] = frozenset({"hour", "day_of_week", "month", "is_weekend"})


# ─────────────────────────────────────────────────────────────────────
#  Sub-section dataclasses
# ─────────────────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class DataConfig:
    """Where raw data comes from and how its integrity is verified."""

    source_url: str
    raw_filename: str
    expected_sha256: str | None = None  # None = trust-on-first-use; populated after first download

    def __post_init__(self) -> None:
        if not self.source_url.startswith(("http://", "https://")):
            raise ConfigurationError(
                f"data.source_url must be an http(s) URL, got: {self.source_url!r}"
            )
        if not self.raw_filename:
            raise ConfigurationError("data.raw_filename must be non-empty")


@dataclass(frozen=True, slots=True)
class TargetConfig:
    """The forecasting contract: what we predict, at what frequency, how far ahead."""

    column: str
    frequency: str
    horizon_hours: int
    aggregation: str

    def __post_init__(self) -> None:
        if not self.column:
            raise ConfigurationError("target.column must be non-empty")
        if self.frequency not in _VALID_FREQUENCIES:
            raise ConfigurationError(
                f"target.frequency {self.frequency!r} not in {sorted(_VALID_FREQUENCIES)}"
            )
        if self.aggregation not in _VALID_AGGREGATIONS:
            raise ConfigurationError(
                f"target.aggregation {self.aggregation!r} not in {sorted(_VALID_AGGREGATIONS)}"
            )
        if self.horizon_hours < 1:
            raise ConfigurationError("target.horizon_hours must be >= 1")


@dataclass(frozen=True, slots=True)
class PreprocessingConfig:
    """How missing values and outages are handled."""

    short_gap_max_rows: int
    interpolation_method: str

    def __post_init__(self) -> None:
        if self.short_gap_max_rows < 1:
            raise ConfigurationError("preprocessing.short_gap_max_rows must be >= 1")
        if self.interpolation_method not in _VALID_INTERPOLATIONS:
            raise ConfigurationError(
                f"preprocessing.interpolation_method {self.interpolation_method!r} "
                f"not in {sorted(_VALID_INTERPOLATIONS)}"
            )


@dataclass(frozen=True, slots=True)
class FeaturesConfig:
    """Which engineered features are built, and the leakage policy."""

    lag_hours: tuple[int, ...]
    rolling_window_hours: tuple[int, ...]
    enable_cyclical_encoding: bool
    enable_calendar: bool
    enable_french_holidays: bool
    strict_leakage_safe: bool

    def __post_init__(self) -> None:
        if any(h < 1 for h in self.lag_hours):
            raise ConfigurationError("features.lag_hours must contain only positive integers")
        if any(w < 1 for w in self.rolling_window_hours):
            raise ConfigurationError(
                "features.rolling_window_hours must contain only positive integers"
            )
        if not self.strict_leakage_safe:
            # Not an error, but worth flagging in the config-load logs.
            # We don't raise here because explicit opt-out is sometimes wanted
            # (e.g., for an ablation that demonstrates leakage). The feature
            # pipeline still requires this flag to be False before emitting
            # unsafe features.
            pass


@dataclass(frozen=True, slots=True)
class SplitsConfig:
    """Rolling-origin temporal cross-validation parameters."""

    n_folds: int
    train_window_days: int
    valid_window_days: int
    gap_days: int

    def __post_init__(self) -> None:
        if self.n_folds < 2:
            raise ConfigurationError("splits.n_folds must be >= 2")
        if self.train_window_days < 1:
            raise ConfigurationError("splits.train_window_days must be >= 1")
        if self.valid_window_days < 1:
            raise ConfigurationError("splits.valid_window_days must be >= 1")
        if self.gap_days < 0:
            raise ConfigurationError("splits.gap_days must be >= 0")


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    """Metrics, prediction intervals, and segmentation for evaluation."""

    primary_metric: str
    guardrail_metrics: tuple[str, ...]
    prediction_interval: float
    segment_dimensions: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.primary_metric not in _VALID_METRICS:
            raise ConfigurationError(
                f"evaluation.primary_metric {self.primary_metric!r} "
                f"not in {sorted(_VALID_METRICS)}"
            )
        for m in self.guardrail_metrics:
            if m not in _VALID_METRICS:
                raise ConfigurationError(
                    f"evaluation.guardrail_metrics contains invalid metric {m!r}"
                )
        if not 0.0 < self.prediction_interval < 1.0:
            raise ConfigurationError(
                f"evaluation.prediction_interval must be in (0, 1), got {self.prediction_interval}"
            )
        for s in self.segment_dimensions:
            if s not in _VALID_SEGMENTS:
                raise ConfigurationError(
                    f"evaluation.segment_dimensions contains unknown segment {s!r}"
                )


@dataclass(frozen=True, slots=True)
class TuningConfig:
    """Hyperparameter optimization budgets and strategy."""

    n_trials: int
    timeout_minutes_per_model: int | None
    sampler: str

    def __post_init__(self) -> None:
        if self.n_trials < 1:
            raise ConfigurationError("tuning.n_trials must be >= 1")
        if self.timeout_minutes_per_model is not None and self.timeout_minutes_per_model < 1:
            raise ConfigurationError(
                "tuning.timeout_minutes_per_model must be >= 1 (or null for no timeout)"
            )


@dataclass(frozen=True, slots=True)
class ReproConfig:
    """Reproducibility knobs: RNG seed and deterministic flags."""

    seed: int
    deterministic_torch: bool


# ─────────────────────────────────────────────────────────────────────
#  Top-level ForecastConfig
# ─────────────────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class ForecastConfig:
    """The complete pipeline configuration. One instance per run."""

    data: DataConfig
    target: TargetConfig
    preprocessing: PreprocessingConfig
    features: FeaturesConfig
    splits: SplitsConfig
    evaluation: EvaluationConfig
    tuning: TuningConfig
    reproducibility: ReproConfig

    # ── Construction ────────────────────────────────────────────────
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ForecastConfig":
        """Build from a nested dictionary (typically loaded from YAML)."""
        try:
            return cls(
                data=DataConfig(**data["data"]),
                target=TargetConfig(**data["target"]),
                preprocessing=PreprocessingConfig(**data["preprocessing"]),
                features=FeaturesConfig(
                    lag_hours=tuple(data["features"]["lag_hours"]),
                    rolling_window_hours=tuple(data["features"]["rolling_window_hours"]),
                    enable_cyclical_encoding=data["features"]["enable_cyclical_encoding"],
                    enable_calendar=data["features"]["enable_calendar"],
                    enable_french_holidays=data["features"]["enable_french_holidays"],
                    strict_leakage_safe=data["features"]["strict_leakage_safe"],
                ),
                splits=SplitsConfig(**data["splits"]),
                evaluation=EvaluationConfig(
                    primary_metric=data["evaluation"]["primary_metric"],
                    guardrail_metrics=tuple(data["evaluation"]["guardrail_metrics"]),
                    prediction_interval=data["evaluation"]["prediction_interval"],
                    segment_dimensions=tuple(data["evaluation"]["segment_dimensions"]),
                ),
                tuning=TuningConfig(**data["tuning"]),
                reproducibility=ReproConfig(**data["reproducibility"]),
            )
        except KeyError as e:
            raise ConfigurationError(f"Missing required config section/key: {e}") from e
        except TypeError as e:
            # Surfaces unexpected/missing keys in any sub-dataclass.
            raise ConfigurationError(f"Invalid config field: {e}") from e

    @classmethod
    def from_yaml(cls, path: Path | str) -> "ForecastConfig":
        """Load and validate config from a YAML file."""
        p = Path(path)
        if not p.is_file():
            raise ConfigurationError(f"Config file not found: {p}")
        with p.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        if not isinstance(data, dict):
            raise ConfigurationError(f"Config file must contain a mapping at top level: {p}")
        return cls.from_dict(data)

    # ── Serialization ───────────────────────────────────────────────
    def to_dict(self) -> dict[str, Any]:
        """Return the config as a plain (JSON-serializable) dictionary."""
        return asdict(self)

    def content_hash(self) -> str:
        """SHA-256 of the canonical JSON representation. Stable across runs.

        Use this to tag artifacts (trained models, evaluation outputs) so any
        result can be traced back to the exact config that produced it.
        """
        canonical = json.dumps(self.to_dict(), sort_keys=True, default=str)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ─────────────────────────────────────────────────────────────────────
#  Conventional default location
# ─────────────────────────────────────────────────────────────────────

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "conf" / "base.yaml"
"""Resolved path to the project's default ``conf/base.yaml``.

Notebooks and scripts that don't want to think about config paths can simply
call ``ForecastConfig.from_yaml(DEFAULT_CONFIG_PATH)``.
"""
