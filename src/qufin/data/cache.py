"""Parquet caching for downloaded data.

Caches data to ~/.cache/qufin/ to avoid repeated API calls.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from qufin.utils.settings import get_settings


def _cache_dir() -> Path:
    d = get_settings().cache_dir
    d.mkdir(parents=True, exist_ok=True)
    return d


def _cache_key(prefix: str, *args: object) -> str:
    raw = f"{prefix}:" + ":".join(str(a) for a in args)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def get_cached(prefix: str, *args: object) -> pd.DataFrame | None:
    """Try to load a cached DataFrame."""
    key = _cache_key(prefix, *args)
    path = _cache_dir() / f"{prefix}_{key}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return None


def put_cache(df: pd.DataFrame, prefix: str, *args: object) -> Path:
    """Save a DataFrame to the parquet cache. Returns the file path."""
    key = _cache_key(prefix, *args)
    path = _cache_dir() / f"{prefix}_{key}.parquet"
    df.to_parquet(path)
    return path


def clear_cache(prefix: str | None = None) -> int:
    """Clear cached files. If prefix given, only clear matching files.

    Returns the number of files deleted.
    """
    cache = _cache_dir()
    count = 0
    pattern = f"{prefix}_*.parquet" if prefix else "*.parquet"
    for f in cache.glob(pattern):
        f.unlink()
        count += 1
    return count
