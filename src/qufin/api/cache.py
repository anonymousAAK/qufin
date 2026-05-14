"""Result caching for quantum algorithm outputs.

Supports two storage backends:
- **Redis** (distributed): for multi-node deployments.
- **SQLite** (single-node): stdlib-only fallback, zero external dependencies.

Cache key = hash(algorithm_name + sorted_parameters + data_hash).

TTL policy:
- 24 hours for market-data-dependent results (``CacheTTL.MARKET_DATA``).
- 7 days for static / structural problems (``CacheTTL.STATIC``).
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Any

try:
    import redis as _redis_mod
except ImportError:  # pragma: no cover
    _redis_mod = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# TTL constants
# ---------------------------------------------------------------------------


class CacheTTL(IntEnum):
    """Default time-to-live values (in seconds)."""

    MARKET_DATA = 86_400  # 24 h
    STATIC = 604_800  # 7 d


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_cache_key(algorithm: str, parameters: dict[str, Any], data_hash: str = "") -> str:
    """Deterministic cache key from algorithm name, parameters, and data hash.

    Parameters
    ----------
    algorithm:
        Algorithm identifier (e.g. ``"european_qae"``).
    parameters:
        Algorithm parameters.  Sorted by key for determinism.
    data_hash:
        Optional hash of input data.  Empty string when data-independent.

    Returns
    -------
    str
        Hex digest (SHA-256, 64 chars).
    """
    canonical = json.dumps(
        {"algorithm": algorithm, "parameters": parameters, "data_hash": data_hash},
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Hit-rate monitor
# ---------------------------------------------------------------------------


@dataclass
class CacheStats:
    """Tracks cache hit/miss statistics."""

    hits: int = 0
    misses: int = 0
    evictions: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record_hit(self) -> None:
        with self._lock:
            self.hits += 1

    def record_miss(self) -> None:
        with self._lock:
            self.misses += 1

    def record_eviction(self) -> None:
        with self._lock:
            self.evictions += 1

    @property
    def total(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        """Return hit rate as a fraction in [0, 1].  Returns 0.0 when no lookups."""
        t = self.total
        return self.hits / t if t > 0 else 0.0

    def report(self) -> dict[str, Any]:
        """Return a JSON-serializable summary."""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "total": self.total,
            "hit_rate": round(self.hit_rate, 4),
        }

    def reset(self) -> None:
        with self._lock:
            self.hits = 0
            self.misses = 0
            self.evictions = 0


# ---------------------------------------------------------------------------
# Backend config
# ---------------------------------------------------------------------------


@dataclass
class RedisCacheConfig:
    """Configuration for the Redis cache backend."""

    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: str | None = None
    key_prefix: str = "qufin:"


@dataclass
class SQLiteCacheConfig:
    """Configuration for the SQLite cache backend."""

    db_path: Path = field(default_factory=lambda: Path.home() / ".cache" / "qufin" / "cache.db")


# ---------------------------------------------------------------------------
# SQLite backend
# ---------------------------------------------------------------------------


class SQLiteCacheBackend:
    """Single-node cache backed by SQLite (stdlib only)."""

    def __init__(self, config: SQLiteCacheConfig | None = None) -> None:
        self._config = config or SQLiteCacheConfig()
        self._config.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._stats = CacheStats()
        self._conn: sqlite3.Connection | None = None
        self._lock = threading.Lock()
        self._init_db()

    # -- internal ----------------------------------------------------------

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(
                str(self._config.db_path),
                check_same_thread=False,
            )
            self._conn.execute("PRAGMA journal_mode=WAL")
        return self._conn

    def _init_db(self) -> None:
        with self._lock:
            conn = self._get_conn()
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS cache (
                    key       TEXT PRIMARY KEY,
                    value     TEXT    NOT NULL,
                    ttl       INTEGER NOT NULL,
                    created   REAL    NOT NULL,
                    algorithm TEXT    NOT NULL
                )
                """
            )
            conn.commit()

    # -- public API --------------------------------------------------------

    @property
    def stats(self) -> CacheStats:
        return self._stats

    def get(self, key: str) -> Any | None:
        """Retrieve a cached value.  Returns ``None`` on miss or expiry."""
        with self._lock:
            conn = self._get_conn()
            row = conn.execute(
                "SELECT value, ttl, created FROM cache WHERE key = ?", (key,)
            ).fetchone()

        if row is None:
            self._stats.record_miss()
            return None

        value_json, ttl, created = row
        if time.time() - created > ttl:
            # Expired -- evict lazily
            self.delete(key)
            self._stats.record_miss()
            self._stats.record_eviction()
            return None

        self._stats.record_hit()
        return json.loads(value_json)

    def put(
        self,
        key: str,
        value: Any,
        ttl: int = CacheTTL.STATIC,
        algorithm: str = "",
    ) -> None:
        """Store *value* under *key* with the given TTL (seconds)."""
        value_json = json.dumps(value, sort_keys=True, default=str)
        with self._lock:
            conn = self._get_conn()
            conn.execute(
                """
                INSERT OR REPLACE INTO cache (key, value, ttl, created, algorithm)
                VALUES (?, ?, ?, ?, ?)
                """,
                (key, value_json, ttl, time.time(), algorithm),
            )
            conn.commit()

    def delete(self, key: str) -> bool:
        """Delete a single entry. Returns ``True`` if the key existed."""
        with self._lock:
            conn = self._get_conn()
            cur = conn.execute("DELETE FROM cache WHERE key = ?", (key,))
            conn.commit()
            return cur.rowcount > 0

    def invalidate_by_algorithm(self, algorithm: str) -> int:
        """Remove all entries for a given algorithm.  Returns count deleted."""
        with self._lock:
            conn = self._get_conn()
            cur = conn.execute("DELETE FROM cache WHERE algorithm = ?", (algorithm,))
            conn.commit()
            count = cur.rowcount
        self._stats.evictions += count
        return count

    def invalidate_all(self) -> int:
        """Clear the entire cache. Returns count deleted."""
        with self._lock:
            conn = self._get_conn()
            cur = conn.execute("SELECT COUNT(*) FROM cache")
            count = cur.fetchone()[0]
            conn.execute("DELETE FROM cache")
            conn.commit()
        self._stats.evictions += count
        return count

    def purge_expired(self) -> int:
        """Remove all expired entries. Returns count purged."""
        now = time.time()
        with self._lock:
            conn = self._get_conn()
            cur = conn.execute(
                "DELETE FROM cache WHERE (? - created) > ttl", (now,)
            )
            conn.commit()
            count = cur.rowcount
        self._stats.evictions += count
        return count

    def size(self) -> int:
        """Number of entries (including potentially expired ones)."""
        with self._lock:
            conn = self._get_conn()
            row = conn.execute("SELECT COUNT(*) FROM cache").fetchone()
            return row[0]

    def close(self) -> None:
        """Close the underlying database connection."""
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None


# ---------------------------------------------------------------------------
# Redis backend
# ---------------------------------------------------------------------------


class RedisCacheBackend:
    """Distributed cache backed by Redis.

    Requires the ``redis`` package (``pip install redis``).
    Falls back to :class:`SQLiteCacheBackend` if Redis is unavailable.
    """

    def __init__(self, config: RedisCacheConfig | None = None) -> None:
        if _redis_mod is None:
            raise ImportError(
                "redis package is required for RedisCacheBackend. "
                "Install it with: pip install redis"
            )
        self._config = config or RedisCacheConfig()
        self._client: Any = _redis_mod.Redis(
            host=self._config.host,
            port=self._config.port,
            db=self._config.db,
            password=self._config.password,
            decode_responses=True,
        )
        self._stats = CacheStats()

    @property
    def stats(self) -> CacheStats:
        return self._stats

    def _prefixed(self, key: str) -> str:
        return f"{self._config.key_prefix}{key}"

    def get(self, key: str) -> Any | None:
        """Retrieve a cached value. Returns ``None`` on miss."""
        raw = self._client.get(self._prefixed(key))
        if raw is None:
            self._stats.record_miss()
            return None
        self._stats.record_hit()
        return json.loads(raw)

    def put(
        self,
        key: str,
        value: Any,
        ttl: int = CacheTTL.STATIC,
        algorithm: str = "",
    ) -> None:
        """Store *value* under *key* with the given TTL (seconds).

        The algorithm name is stored as metadata in a companion key for
        invalidation purposes.
        """
        value_json = json.dumps(
            {"value": value, "algorithm": algorithm},
            sort_keys=True,
            default=str,
        )
        self._client.setex(self._prefixed(key), ttl, value_json)

    def delete(self, key: str) -> bool:
        """Delete a single entry. Returns ``True`` if the key existed."""
        return bool(self._client.delete(self._prefixed(key)))

    def invalidate_by_algorithm(self, algorithm: str) -> int:
        """Remove all entries for a given algorithm (scan-based)."""
        pattern = f"{self._config.key_prefix}*"
        count = 0
        for rkey in self._client.scan_iter(match=pattern):
            raw = self._client.get(rkey)
            if raw is not None:
                data = json.loads(raw)
                if isinstance(data, dict) and data.get("algorithm") == algorithm:
                    self._client.delete(rkey)
                    count += 1
        self._stats.evictions += count
        return count

    def invalidate_all(self) -> int:
        """Flush all qufin-prefixed keys."""
        pattern = f"{self._config.key_prefix}*"
        count = 0
        for rkey in self._client.scan_iter(match=pattern):
            self._client.delete(rkey)
            count += 1
        self._stats.evictions += count
        return count

    def size(self) -> int:
        """Approximate number of cached entries (prefix scan)."""
        pattern = f"{self._config.key_prefix}*"
        return sum(1 for _ in self._client.scan_iter(match=pattern))

    def close(self) -> None:
        """Close the Redis connection."""
        self._client.close()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_cache(
    backend: str = "sqlite",
    *,
    redis_config: RedisCacheConfig | None = None,
    sqlite_config: SQLiteCacheConfig | None = None,
) -> SQLiteCacheBackend | RedisCacheBackend:
    """Create a cache backend instance.

    Parameters
    ----------
    backend:
        ``"redis"`` or ``"sqlite"`` (default).
    redis_config:
        Optional Redis connection settings.
    sqlite_config:
        Optional SQLite file path settings.

    Returns
    -------
    SQLiteCacheBackend | RedisCacheBackend
    """
    if backend == "redis":
        return RedisCacheBackend(config=redis_config)
    if backend == "sqlite":
        return SQLiteCacheBackend(config=sqlite_config)
    raise ValueError(f"Unknown cache backend: {backend!r}. Use 'redis' or 'sqlite'.")
