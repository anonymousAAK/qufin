"""Data ingestion: equities, macro, synthetic, caching, streaming, warehouse."""

from __future__ import annotations

from qufin.data.synthetic import gbm_paths, heston_paths, merton_jump_paths

__all__ = ["gbm_paths", "heston_paths", "merton_jump_paths"]


def get_fred_provider(api_key: str | None = None):
    """Lazy import for FREDProvider (requires fredapi)."""
    from qufin.data.macro import FREDProvider
    return FREDProvider(api_key=api_key)


def get_refinitiv_source(app_key: str = "", **kwargs):
    """Lazy import for RefinitivDataSource (requires eikon)."""
    from qufin.data.refinitiv import RefinitivConfig, RefinitivDataSource
    config = RefinitivConfig(app_key=app_key, **kwargs)
    return RefinitivDataSource(config=config)


def get_bloomberg_source(config=None, **kwargs):
    """Lazy import for BloombergDataSource (requires blpapi + Terminal license)."""
    from qufin.data.bloomberg import BloombergDataSource
    return BloombergDataSource(config=config, **kwargs)
