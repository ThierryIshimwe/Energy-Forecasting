"""Custom exception hierarchy for the energy-forecasting package.

Design rule: every place we can fail predictably raises one of these — never
a bare ``Exception`` or ``ValueError`` with a vague message. Callers (notebooks,
the app, scripts) can then ``except`` precisely on the failure mode they want
to handle.

Hierarchy:

    EnergyForecastingError
    ├── ConfigurationError      # invalid or inconsistent ForecastConfig
    ├── DataValidationError     # raw data fails schema or quality checks
    ├── LeakageError            # feature pipeline would emit unsafe features
    ├── ModelError              # training or prediction failed
    │   └── HyperparameterError # tuning could not find a valid configuration
    └── ArtifactNotFoundError   # expected model / dataset / split missing
"""

from __future__ import annotations


class EnergyForecastingError(Exception):
    """Base class for all errors raised by this package."""


class ConfigurationError(EnergyForecastingError):
    """The ForecastConfig is internally inconsistent or contains invalid values."""


class DataValidationError(EnergyForecastingError):
    """Raw data does not satisfy the contract required by downstream stages."""


class LeakageError(EnergyForecastingError):
    """A feature pipeline tried to emit a feature flagged as not leakage-safe.

    Raised by the feature registry when ``strict_leakage_safe=True`` and an
    unsafe feature appears in the build list. This error exists to make
    leakage a *mechanical impossibility* rather than a documentation footnote.
    """


class ModelError(EnergyForecastingError):
    """Model training or prediction failed for reasons specific to the model."""


class HyperparameterError(ModelError):
    """Hyperparameter optimization could not find any valid configuration."""


class ArtifactNotFoundError(EnergyForecastingError):
    """An expected artifact (trained model, processed dataset, split plan) is missing."""
