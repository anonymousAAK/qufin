"""Backend abstraction layer for quantum framework dispatch."""

from __future__ import annotations

from qufin.backends.base import Backend
from qufin.backends.mock import MockBackend

__all__ = ["Backend", "MockBackend"]
