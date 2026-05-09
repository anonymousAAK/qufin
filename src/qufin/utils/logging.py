"""Logging configuration for qufin."""

from __future__ import annotations

import logging

_LOG_FORMAT = "%(asctime)s [%(name)s] %(levelname)s: %(message)s"


def get_logger(name: str, level: str = "WARNING") -> logging.Logger:
    """Get a configured logger for qufin modules."""
    logger = logging.getLogger(f"qufin.{name}")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_LOG_FORMAT))
        logger.addHandler(handler)
    logger.setLevel(getattr(logging, level.upper(), logging.WARNING))
    return logger
