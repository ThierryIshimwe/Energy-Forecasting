"""Logging configuration built on loguru.

Why loguru over stdlib logging: zero-boilerplate setup, structured output by
default, easy file rotation, and a single ``logger`` object that doesn't need
per-module configuration. Calls in package code use ``get_logger(__name__)``
to keep the same module-tagged style the stdlib offers.

Default output is human-readable on stderr at INFO level. Scripts that want
structured logs (JSON for ingestion) can call ``configure_logging`` with
``json=True``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal

from loguru import logger as _root_logger

LogLevel = Literal["TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"]

_DEFAULT_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
    "<level>{message}</level>"
)


def configure_logging(
    level: LogLevel = "INFO",
    *,
    log_file: Path | None = None,
    json: bool = False,
) -> None:
    """Configure the root loguru logger for the current process.

    Idempotent: re-calling replaces the previous configuration.

    Args:
        level: Minimum level to emit (records below this are dropped).
        log_file: If given, also write to this file with automatic rotation
            at 10 MB and 7-day retention. Parent directory must exist.
        json: When True, emit serialized JSON records instead of human-readable
            text. Useful for piping logs to ingestion tools.
    """
    _root_logger.remove()
    _root_logger.add(
        sys.stderr,
        level=level,
        format=_DEFAULT_FORMAT if not json else "{message}",
        serialize=json,
        backtrace=True,
        diagnose=False,  # diagnose=True leaks values in tracebacks — avoid in production
    )
    if log_file is not None:
        _root_logger.add(
            log_file,
            level=level,
            rotation="10 MB",
            retention="7 days",
            compression="gz",
            serialize=json,
            backtrace=True,
            diagnose=False,
        )


def get_logger(name: str) -> "type[_root_logger]":
    """Return a bound logger tagged with ``name`` (typically ``__name__``)."""
    return _root_logger.bind(name=name)  # type: ignore[return-value]


# Configure with sensible defaults on import. Scripts can override.
configure_logging(level="INFO")
