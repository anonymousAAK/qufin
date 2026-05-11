"""Data ingestion: equities, macro, synthetic generators, universes, caching."""

from __future__ import annotations

from qufin.data.synthetic import gbm_paths, heston_paths, merton_jump_paths

__all__ = ["gbm_paths", "heston_paths", "merton_jump_paths"]


def get_fred_provider(api_key: str | None = None):
    """Lazy import for FREDProvider (requires fredapi)."""
    from qufin.data.macro import FREDProvider
    return FREDProvider(api_key=api_key)
