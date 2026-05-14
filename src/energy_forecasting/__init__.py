"""Energy Forecasting — household electricity consumption modeling.

A reproducible, leakage-safe forecasting pipeline built for the Enel × LUISS
Project Work. The public API is intentionally small: most callers should reach
into the relevant submodule (data, features, models, evaluation) directly.
"""

from energy_forecasting._version import __version__

__all__ = ["__version__"]
