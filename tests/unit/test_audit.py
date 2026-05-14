"""Tests for compliance audit trail."""

from __future__ import annotations

import csv
import io
import json
import threading
from pathlib import Path

import pytest

from qufin.compliance.audit import (
    AuditEntry,
    AuditStore,
    QueryFilter,
    compute_input_hash,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def store(tmp_path: Path) -> AuditStore:
    """Provide a fresh SQLite-backed audit store per test."""
    db = tmp_path / "audit.db"
    s = AuditStore.from_sqlite(db)
    yield s
    s.close()


@pytest.fixture()
def sample_entry() -> AuditEntry:
    return AuditEntry(
        user="trader_1",
        algorithm="quantum_var",
        params={"confidence": 0.99, "n_qubits": 6},
        input_hash="sha256:abc123",
        result={"var_95": 0.042},
        duration_ms=1523.7,
    )


def _make_entry(
    user: str = "u",
    algorithm: str = "algo",
    timestamp: str | None = None,
    **kwargs,
) -> AuditEntry:
    kw: dict = {"user": user, "algorithm": algorithm, **kwargs}
    if timestamp is not None:
        kw["timestamp"] = timestamp
    return AuditEntry(**kw)


# ---------------------------------------------------------------------------
# AuditEntry dataclass
# ---------------------------------------------------------------------------


class TestAuditEntry:
    def test_defaults(self) -> None:
        e = AuditEntry(user="u", algorithm="a")
        assert e.user == "u"
        assert e.algorithm == "a"
        assert e.params == {}
        assert e.result == {}
        assert e.duration_ms == 0.0
        assert len(e.entry_id) == 36  # UUID
        assert "T" in e.timestamp  # ISO-8601

    def test_frozen(self) -> None:
        e = AuditEntry(user="u", algorithm="a")
        with pytest.raises(AttributeError):
            e.user = "other"  # type: ignore[misc]

    def test_custom_fields(self, sample_entry: AuditEntry) -> None:
        assert sample_entry.user == "trader_1"
        assert sample_entry.params["confidence"] == 0.99
        assert sample_entry.duration_ms == 1523.7


# ---------------------------------------------------------------------------
# compute_input_hash
# ---------------------------------------------------------------------------


class TestComputeInputHash:
    def test_bytes_input(self) -> None:
        h = compute_input_hash(b"hello")
        assert h.startswith("sha256:")
        assert len(h) == 7 + 64  # prefix + hex digest

    def test_string_input(self) -> None:
        h = compute_input_hash("hello")
        assert h == compute_input_hash(b"hello")

    def test_deterministic(self) -> None:
        assert compute_input_hash("data") == compute_input_hash("data")

    def test_different_data_different_hash(self) -> None:
        assert compute_input_hash("a") != compute_input_hash("b")


# ---------------------------------------------------------------------------
# AuditStore — basic CRUD
# ---------------------------------------------------------------------------


class TestAuditStoreCRUD:
    def test_log_and_get(self, store: AuditStore, sample_entry: AuditEntry) -> None:
        eid = store.log(sample_entry)
        assert eid == sample_entry.entry_id
        fetched = store.get(eid)
        assert fetched is not None
        assert fetched.user == sample_entry.user
        assert fetched.algorithm == sample_entry.algorithm
        assert fetched.params == sample_entry.params
        assert fetched.duration_ms == sample_entry.duration_ms

    def test_get_missing_returns_none(self, store: AuditStore) -> None:
        assert store.get("nonexistent-id") is None

    def test_count(self, store: AuditStore) -> None:
        assert store.count() == 0
        store.log(_make_entry())
        assert store.count() == 1
        store.log(_make_entry())
        assert store.count() == 2

    def test_log_after_close_raises(self, store: AuditStore) -> None:
        store.close()
        with pytest.raises(ValueError, match="closed"):
            store.log(_make_entry())

    def test_context_manager(self, tmp_path: Path) -> None:
        db = tmp_path / "ctx.db"
        with AuditStore.from_sqlite(db) as s:
            s.log(_make_entry())
            assert s.count() == 1
        # After context exit, store is closed.
        assert s._closed

    def test_in_memory_store(self) -> None:
        with AuditStore.from_sqlite(":memory:") as s:
            s.log(_make_entry())
            assert s.count() == 1


# ---------------------------------------------------------------------------
# Querying
# ---------------------------------------------------------------------------


class TestAuditStoreQuery:
    def test_query_all(self, store: AuditStore) -> None:
        for i in range(5):
            store.log(_make_entry(user=f"u{i}"))
        assert len(store.query()) == 5

    def test_query_by_algorithm(self, store: AuditStore) -> None:
        store.log(_make_entry(algorithm="var"))
        store.log(_make_entry(algorithm="cvar"))
        store.log(_make_entry(algorithm="var"))
        results = store.query(algorithm="var")
        assert len(results) == 2
        assert all(r.algorithm == "var" for r in results)

    def test_query_by_user(self, store: AuditStore) -> None:
        store.log(_make_entry(user="alice"))
        store.log(_make_entry(user="bob"))
        results = store.query(user="alice")
        assert len(results) == 1
        assert results[0].user == "alice"

    def test_query_by_date_range(self, store: AuditStore) -> None:
        store.log(_make_entry(timestamp="2026-01-01T00:00:00+00:00"))
        store.log(_make_entry(timestamp="2026-06-15T12:00:00+00:00"))
        store.log(_make_entry(timestamp="2026-12-31T23:59:59+00:00"))

        results = store.query(
            start="2026-03-01T00:00:00+00:00",
            end="2026-09-01T00:00:00+00:00",
        )
        assert len(results) == 1
        assert "06-15" in results[0].timestamp

    def test_query_with_limit(self, store: AuditStore) -> None:
        for _ in range(10):
            store.log(_make_entry())
        results = store.query(limit=3)
        assert len(results) == 3

    def test_query_combined_filters(self, store: AuditStore) -> None:
        store.log(_make_entry(user="alice", algorithm="var",
                              timestamp="2026-05-01T00:00:00+00:00"))
        store.log(_make_entry(user="alice", algorithm="cvar",
                              timestamp="2026-05-01T00:00:00+00:00"))
        store.log(_make_entry(user="bob", algorithm="var",
                              timestamp="2026-05-01T00:00:00+00:00"))
        results = store.query(user="alice", algorithm="var")
        assert len(results) == 1

    def test_query_empty_db(self, store: AuditStore) -> None:
        assert store.query() == []
        assert store.query(algorithm="nope") == []

    def test_query_filter_object(self, store: AuditStore) -> None:
        store.log(_make_entry(algorithm="x"))
        qf = QueryFilter(algorithm="x")
        results = store.query_filter(qf)
        assert len(results) == 1

    def test_query_no_match(self, store: AuditStore) -> None:
        store.log(_make_entry(algorithm="var"))
        assert store.query(algorithm="cvar") == []


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


class TestAuditStoreExport:
    def test_export_csv_string(self, store: AuditStore) -> None:
        store.log(_make_entry(user="alice", algorithm="var"))
        entries = store.query()
        csv_str = store.export_csv(entries)
        reader = csv.DictReader(io.StringIO(csv_str))
        rows = list(reader)
        assert len(rows) == 1
        assert rows[0]["user"] == "alice"
        assert rows[0]["algorithm"] == "var"

    def test_export_csv_file(self, store: AuditStore, tmp_path: Path) -> None:
        store.log(_make_entry())
        entries = store.query()
        outfile = tmp_path / "out.csv"
        store.export_csv(entries, outfile)
        assert outfile.exists()
        content = outfile.read_text(encoding="utf-8")
        assert "entry_id" in content

    def test_export_json_string(self, store: AuditStore) -> None:
        store.log(_make_entry(user="bob", algorithm="cvar"))
        entries = store.query()
        json_str = store.export_json(entries)
        data = json.loads(json_str)
        assert len(data) == 1
        assert data[0]["user"] == "bob"

    def test_export_json_file(self, store: AuditStore, tmp_path: Path) -> None:
        store.log(_make_entry())
        entries = store.query()
        outfile = tmp_path / "out.json"
        store.export_json(entries, outfile)
        assert outfile.exists()
        data = json.loads(outfile.read_text(encoding="utf-8"))
        assert len(data) == 1

    def test_export_empty(self, store: AuditStore) -> None:
        csv_str = store.export_csv([])
        assert "entry_id" in csv_str  # header present
        reader = csv.DictReader(io.StringIO(csv_str))
        assert list(reader) == []

        json_str = store.export_json([])
        assert json.loads(json_str) == []

    def test_export_preserves_params_as_json(self, store: AuditStore) -> None:
        store.log(_make_entry(params={"nested": {"a": 1}}))
        entries = store.query()
        csv_str = store.export_csv(entries)
        reader = csv.DictReader(io.StringIO(csv_str))
        row = next(reader)
        parsed = json.loads(row["params"])
        assert parsed == {"nested": {"a": 1}}


# ---------------------------------------------------------------------------
# Edge cases & concurrency
# ---------------------------------------------------------------------------


class TestAuditStoreEdgeCases:
    def test_duplicate_entry_id_raises(self, store: AuditStore) -> None:
        e1 = _make_entry()
        store.log(e1)
        e2 = AuditEntry(
            user="other",
            algorithm="other",
            entry_id=e1.entry_id,
        )
        with pytest.raises(Exception):  # IntegrityError  # noqa: B017
            store.log(e2)

    def test_concurrent_writes(self, tmp_path: Path) -> None:
        db = tmp_path / "concurrent.db"
        store = AuditStore.from_sqlite(db)
        errors: list[Exception] = []

        def writer(tid: int) -> None:
            try:
                for i in range(20):
                    store.log(_make_entry(user=f"t{tid}", algorithm=f"a{i}"))
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=writer, args=(t,)) for t in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        assert store.count() == 80
        store.close()

    def test_wal_mode_enabled(self, tmp_path: Path) -> None:
        db = tmp_path / "wal.db"
        store = AuditStore.from_sqlite(db)
        cur = store._conn.cursor()
        cur.execute("PRAGMA journal_mode;")
        mode = cur.fetchone()[0]
        cur.close()
        assert mode == "wal"
        store.close()

    def test_persistence_across_reopens(self, tmp_path: Path) -> None:
        db = tmp_path / "persist.db"
        with AuditStore.from_sqlite(db) as s:
            s.log(_make_entry(user="persist_user"))

        with AuditStore.from_sqlite(db) as s:
            assert s.count() == 1
            entries = s.query(user="persist_user")
            assert len(entries) == 1

    def test_large_params_and_result(self, store: AuditStore) -> None:
        big_params = {f"key_{i}": list(range(100)) for i in range(50)}
        big_result = {"data": "x" * 10_000}
        e = _make_entry(params=big_params, result=big_result)
        store.log(e)
        fetched = store.get(e.entry_id)
        assert fetched is not None
        assert fetched.params == big_params
        assert fetched.result == big_result


# ---------------------------------------------------------------------------
# Postgres guard
# ---------------------------------------------------------------------------


class TestPostgresGuard:
    def test_import_error_without_psycopg2(self) -> None:
        """from_postgres raises ImportError when psycopg2 is absent."""
        # We cannot guarantee psycopg2 is missing, but we test the
        # code path by monkeypatching.
        import sys

        saved = sys.modules.get("psycopg2")
        sys.modules["psycopg2"] = None  # type: ignore[assignment]
        try:
            with pytest.raises(ImportError, match="psycopg2"):
                AuditStore.from_postgres("host=localhost dbname=test")
        finally:
            if saved is not None:
                sys.modules["psycopg2"] = saved
            else:
                sys.modules.pop("psycopg2", None)
