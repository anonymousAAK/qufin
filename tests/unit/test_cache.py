"""Tests for data caching."""

from __future__ import annotations

import pandas as pd
import pytest

from qufin.data.cache import clear_cache, get_cached, put_cache


class TestCache:
    def test_put_and_get(self, tmp_path: object) -> None:
        import qufin.utils.settings as s
        original = s._settings
        s._settings = s.Settings(cache_dir=tmp_path)  # type: ignore[arg-type]
        try:
            df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
            put_cache(df, "test", "key1")
            result = get_cached("test", "key1")
            assert result is not None
            pd.testing.assert_frame_equal(df, result)
        finally:
            s._settings = original

    def test_get_missing_returns_none(self, tmp_path: object) -> None:
        import qufin.utils.settings as s
        original = s._settings
        s._settings = s.Settings(cache_dir=tmp_path)  # type: ignore[arg-type]
        try:
            assert get_cached("nonexistent", "key") is None
        finally:
            s._settings = original

    def test_clear_cache(self, tmp_path: object) -> None:
        import qufin.utils.settings as s
        original = s._settings
        s._settings = s.Settings(cache_dir=tmp_path)  # type: ignore[arg-type]
        try:
            df = pd.DataFrame({"x": [1]})
            put_cache(df, "test", "a")
            put_cache(df, "test", "b")
            deleted = clear_cache("test")
            assert deleted == 2
            assert get_cached("test", "a") is None
        finally:
            s._settings = original
