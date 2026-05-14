"""Tests for qufin.api.cache — result caching with SQLite and Redis backends."""

from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from qufin.api.cache import (
    CacheStats,
    CacheTTL,
    RedisCacheBackend,
    RedisCacheConfig,
    SQLiteCacheBackend,
    SQLiteCacheConfig,
    create_cache,
    make_cache_key,
)

# ---------------------------------------------------------------------------
# make_cache_key
# ---------------------------------------------------------------------------


class TestMakeCacheKey:
    def test_deterministic(self) -> None:
        k1 = make_cache_key("algo", {"a": 1, "b": 2}, "datahash")
        k2 = make_cache_key("algo", {"a": 1, "b": 2}, "datahash")
        assert k1 == k2

    def test_order_independent(self) -> None:
        k1 = make_cache_key("algo", {"b": 2, "a": 1})
        k2 = make_cache_key("algo", {"a": 1, "b": 2})
        assert k1 == k2

    def test_different_algorithm_different_key(self) -> None:
        k1 = make_cache_key("algo1", {"x": 1})
        k2 = make_cache_key("algo2", {"x": 1})
        assert k1 != k2

    def test_different_data_hash_different_key(self) -> None:
        k1 = make_cache_key("algo", {"x": 1}, "hash_a")
        k2 = make_cache_key("algo", {"x": 1}, "hash_b")
        assert k1 != k2

    def test_key_is_hex_sha256(self) -> None:
        key = make_cache_key("test", {})
        assert len(key) == 64
        assert all(c in "0123456789abcdef" for c in key)


# ---------------------------------------------------------------------------
# CacheTTL
# ---------------------------------------------------------------------------


class TestCacheTTL:
    def test_market_data_24h(self) -> None:
        assert CacheTTL.MARKET_DATA == 86_400

    def test_static_7d(self) -> None:
        assert CacheTTL.STATIC == 604_800


# ---------------------------------------------------------------------------
# CacheStats
# ---------------------------------------------------------------------------


class TestCacheStats:
    def test_initial_state(self) -> None:
        stats = CacheStats()
        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.evictions == 0
        assert stats.total == 0
        assert stats.hit_rate == 0.0

    def test_record_hit(self) -> None:
        stats = CacheStats()
        stats.record_hit()
        stats.record_hit()
        assert stats.hits == 2
        assert stats.total == 2
        assert stats.hit_rate == 1.0

    def test_record_miss(self) -> None:
        stats = CacheStats()
        stats.record_miss()
        assert stats.misses == 1
        assert stats.hit_rate == 0.0

    def test_hit_rate_mixed(self) -> None:
        stats = CacheStats()
        stats.record_hit()
        stats.record_miss()
        stats.record_hit()
        stats.record_miss()
        assert stats.hit_rate == pytest.approx(0.5)

    def test_report(self) -> None:
        stats = CacheStats()
        stats.record_hit()
        stats.record_miss()
        stats.record_eviction()
        rpt = stats.report()
        assert rpt["hits"] == 1
        assert rpt["misses"] == 1
        assert rpt["evictions"] == 1
        assert rpt["total"] == 2
        assert rpt["hit_rate"] == 0.5

    def test_reset(self) -> None:
        stats = CacheStats()
        stats.record_hit()
        stats.record_miss()
        stats.record_eviction()
        stats.reset()
        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.evictions == 0


# ---------------------------------------------------------------------------
# SQLiteCacheBackend
# ---------------------------------------------------------------------------


class TestSQLiteCacheBackend:
    @pytest.fixture()
    def cache(self, tmp_path: Path) -> SQLiteCacheBackend:
        cfg = SQLiteCacheConfig(db_path=tmp_path / "test_cache.db")
        return SQLiteCacheBackend(config=cfg)

    def test_put_and_get(self, cache: SQLiteCacheBackend) -> None:
        cache.put("k1", {"result": 42}, algorithm="test_algo")
        val = cache.get("k1")
        assert val == {"result": 42}

    def test_get_miss(self, cache: SQLiteCacheBackend) -> None:
        val = cache.get("nonexistent")
        assert val is None
        assert cache.stats.misses == 1

    def test_hit_increments_stats(self, cache: SQLiteCacheBackend) -> None:
        cache.put("k1", "hello")
        cache.get("k1")
        assert cache.stats.hits == 1

    def test_ttl_expiration(self, cache: SQLiteCacheBackend) -> None:
        cache.put("k1", "value", ttl=1)
        time.sleep(1.1)
        val = cache.get("k1")
        assert val is None
        assert cache.stats.evictions == 1

    def test_delete(self, cache: SQLiteCacheBackend) -> None:
        cache.put("k1", "value")
        assert cache.delete("k1") is True
        assert cache.get("k1") is None

    def test_delete_nonexistent(self, cache: SQLiteCacheBackend) -> None:
        assert cache.delete("nope") is False

    def test_invalidate_by_algorithm(self, cache: SQLiteCacheBackend) -> None:
        cache.put("a1", 1, algorithm="algo_a")
        cache.put("a2", 2, algorithm="algo_a")
        cache.put("b1", 3, algorithm="algo_b")
        count = cache.invalidate_by_algorithm("algo_a")
        assert count == 2
        assert cache.get("a1") is None
        assert cache.get("b1") == 3

    def test_invalidate_all(self, cache: SQLiteCacheBackend) -> None:
        cache.put("k1", 1)
        cache.put("k2", 2)
        count = cache.invalidate_all()
        assert count == 2
        assert cache.size() == 0

    def test_purge_expired(self, cache: SQLiteCacheBackend) -> None:
        cache.put("short", "val", ttl=1)
        cache.put("long", "val", ttl=3600)
        time.sleep(1.1)
        purged = cache.purge_expired()
        assert purged == 1
        assert cache.size() == 1

    def test_size(self, cache: SQLiteCacheBackend) -> None:
        assert cache.size() == 0
        cache.put("a", 1)
        cache.put("b", 2)
        assert cache.size() == 2

    def test_overwrite_existing_key(self, cache: SQLiteCacheBackend) -> None:
        cache.put("k1", "old")
        cache.put("k1", "new")
        assert cache.get("k1") == "new"
        assert cache.size() == 1

    def test_close_and_reopen(self, tmp_path: Path) -> None:
        cfg = SQLiteCacheConfig(db_path=tmp_path / "reopen.db")
        c1 = SQLiteCacheBackend(config=cfg)
        c1.put("persist", {"val": 99})
        c1.close()

        c2 = SQLiteCacheBackend(config=cfg)
        assert c2.get("persist") == {"val": 99}
        c2.close()

    def test_default_ttl_is_static(self, cache: SQLiteCacheBackend) -> None:
        cache.put("k", "v")
        # Should not expire within a second
        val = cache.get("k")
        assert val == "v"

    def test_complex_value(self, cache: SQLiteCacheBackend) -> None:
        value = {"prices": [1.1, 2.2, 3.3], "metadata": {"algo": "qaoa", "depth": 5}}
        cache.put("complex", value)
        assert cache.get("complex") == value


# ---------------------------------------------------------------------------
# RedisCacheBackend (mocked)
# ---------------------------------------------------------------------------


class TestRedisCacheBackend:
    @pytest.fixture()
    def mock_redis_mod(self):
        """Patch the redis module so RedisCacheBackend can be instantiated."""
        mock_client = MagicMock()
        mock_redis_class = MagicMock(return_value=mock_client)

        with patch("qufin.api.cache._redis_mod") as mock_mod:
            mock_mod.Redis = mock_redis_class
            yield mock_mod, mock_client

    def test_put_calls_setex(self, mock_redis_mod) -> None:
        _, client = mock_redis_mod
        cache = RedisCacheBackend()
        cache.put("k1", {"result": 42}, ttl=3600, algorithm="test")
        client.setex.assert_called_once()
        args = client.setex.call_args
        assert args[0][0] == "qufin:k1"
        assert args[0][1] == 3600

    def test_get_hit(self, mock_redis_mod) -> None:
        _, client = mock_redis_mod
        stored = json.dumps({"value": {"x": 1}, "algorithm": "test"})
        client.get.return_value = stored
        cache = RedisCacheBackend()
        val = cache.get("k1")
        assert val == {"value": {"x": 1}, "algorithm": "test"}
        assert cache.stats.hits == 1

    def test_get_miss(self, mock_redis_mod) -> None:
        _, client = mock_redis_mod
        client.get.return_value = None
        cache = RedisCacheBackend()
        val = cache.get("k1")
        assert val is None
        assert cache.stats.misses == 1

    def test_delete(self, mock_redis_mod) -> None:
        _, client = mock_redis_mod
        client.delete.return_value = 1
        cache = RedisCacheBackend()
        assert cache.delete("k1") is True

    def test_import_error_when_no_redis(self) -> None:
        with patch("qufin.api.cache._redis_mod", None), pytest.raises(
            ImportError, match="redis package is required"
        ):
            RedisCacheBackend()

    def test_custom_prefix(self, mock_redis_mod) -> None:
        _, client = mock_redis_mod
        cfg = RedisCacheConfig(key_prefix="myapp:")
        cache = RedisCacheBackend(config=cfg)
        cache.put("k1", "v")
        args = client.setex.call_args
        assert args[0][0] == "myapp:k1"

    def test_close(self, mock_redis_mod) -> None:
        _, client = mock_redis_mod
        cache = RedisCacheBackend()
        cache.close()
        client.close.assert_called_once()


# ---------------------------------------------------------------------------
# create_cache factory
# ---------------------------------------------------------------------------


class TestCreateCache:
    def test_sqlite_default(self, tmp_path: Path) -> None:
        cfg = SQLiteCacheConfig(db_path=tmp_path / "factory.db")
        cache = create_cache("sqlite", sqlite_config=cfg)
        assert isinstance(cache, SQLiteCacheBackend)

    def test_redis_backend(self) -> None:
        with patch("qufin.api.cache._redis_mod") as mock_mod:
            mock_mod.Redis = MagicMock(return_value=MagicMock())
            cache = create_cache("redis")
            assert isinstance(cache, RedisCacheBackend)

    def test_unknown_backend_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown cache backend"):
            create_cache("memcached")
