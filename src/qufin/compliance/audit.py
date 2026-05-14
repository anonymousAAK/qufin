"""Immutable audit trail for algorithm executions.

Provides append-only logging of every algorithm run with full provenance:
timestamp, user, algorithm name, parameters, input hash, result, and
wall-clock duration.  Default storage is SQLite in WAL mode; optional
Postgres backend via psycopg2.

Typical usage::

    store = AuditStore.from_sqlite("/path/to/audit.db")
    entry = AuditEntry(
        user="quant_desk_1",
        algorithm="quantum_var",
        params={"confidence": 0.99, "n_qubits": 6},
        input_hash="sha256:abc123...",
        result={"var_95": 0.042},
        duration_ms=1523.7,
    )
    store.log(entry)

    # Query and export
    results = store.query(algorithm="quantum_var", start="2026-01-01")
    store.export_csv(results, "audit_q1.csv")
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import sqlite3
import threading
import uuid
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

__all__ = [
    "AuditEntry",
    "AuditStore",
    "QueryFilter",
    "compute_input_hash",
]

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AuditEntry:
    """Single audit log entry.

    Parameters
    ----------
    user : str
        Identifier of the user / service account that triggered the run.
    algorithm : str
        Name of the algorithm executed (e.g. ``"quantum_var"``).
    params : dict[str, Any]
        Algorithm hyper-parameters as a JSON-serialisable dict.
    input_hash : str
        Deterministic hash of the input data (use :func:`compute_input_hash`).
    result : dict[str, Any]
        Algorithm outputs as a JSON-serialisable dict.
    duration_ms : float
        Wall-clock duration in milliseconds.
    entry_id : str
        UUID assigned automatically if not provided.
    timestamp : str
        ISO-8601 UTC timestamp, auto-generated when omitted.
    """

    user: str
    algorithm: str
    params: dict[str, Any] = field(default_factory=dict)
    input_hash: str = ""
    result: dict[str, Any] = field(default_factory=dict)
    duration_ms: float = 0.0
    entry_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


@dataclass
class QueryFilter:
    """Filter criteria for audit log queries.

    All fields are optional; ``None`` means *no filter on that field*.
    """

    start: str | None = None
    end: str | None = None
    algorithm: str | None = None
    user: str | None = None
    limit: int | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def compute_input_hash(data: bytes | str) -> str:
    """Return a ``sha256:<hex>`` hash of *data* for provenance tracking."""
    if isinstance(data, str):
        data = data.encode("utf-8")
    digest = hashlib.sha256(data).hexdigest()
    return f"sha256:{digest}"


def _parse_iso(ts: str) -> datetime:
    """Parse an ISO-8601 string to a timezone-aware datetime."""
    dt = datetime.fromisoformat(ts)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


# ---------------------------------------------------------------------------
# Audit store
# ---------------------------------------------------------------------------


_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS audit_log (
    entry_id    TEXT PRIMARY KEY,
    timestamp   TEXT NOT NULL,
    user        TEXT NOT NULL,
    algorithm   TEXT NOT NULL,
    params      TEXT NOT NULL,
    input_hash  TEXT NOT NULL,
    result      TEXT NOT NULL,
    duration_ms REAL NOT NULL
);
"""

_CREATE_INDEX_TS = (
    "CREATE INDEX IF NOT EXISTS idx_audit_ts ON audit_log (timestamp);"
)
_CREATE_INDEX_ALGO = (
    "CREATE INDEX IF NOT EXISTS idx_audit_algo ON audit_log (algorithm);"
)
_CREATE_INDEX_USER = (
    "CREATE INDEX IF NOT EXISTS idx_audit_user ON audit_log (user);"
)


class AuditStore:
    """Append-only audit log backed by SQLite or Postgres.

    Use the factory class-methods :meth:`from_sqlite` or
    :meth:`from_postgres` to obtain an instance.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, conn: Any, *, backend: str = "sqlite") -> None:
        self._conn = conn
        self._backend = backend
        self._lock = threading.Lock()
        self._closed = False

    @classmethod
    def from_sqlite(cls, path: str | Path = ":memory:") -> AuditStore:
        """Create a store backed by a SQLite database in WAL mode.

        Parameters
        ----------
        path : str | Path
            File-system path for the database, or ``":memory:"`` for an
            in-memory database (useful for tests).
        """
        path_str = str(path)
        conn = sqlite3.connect(path_str, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        conn.execute(_CREATE_TABLE)
        conn.execute(_CREATE_INDEX_TS)
        conn.execute(_CREATE_INDEX_ALGO)
        conn.execute(_CREATE_INDEX_USER)
        conn.commit()
        return cls(conn, backend="sqlite")

    @classmethod
    def from_postgres(cls, dsn: str) -> AuditStore:
        """Create a store backed by a Postgres database.

        Parameters
        ----------
        dsn : str
            Postgres connection string, e.g.
            ``"host=localhost dbname=audit user=app"``.

        Raises
        ------
        ImportError
            If *psycopg2* is not installed.
        """
        try:
            import psycopg2  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "psycopg2 is required for Postgres support. "
                "Install it with: pip install psycopg2-binary"
            ) from exc

        conn = psycopg2.connect(dsn)
        conn.autocommit = False
        cur = conn.cursor()
        # Postgres uses the same DDL (TEXT, REAL are valid types).
        cur.execute(_CREATE_TABLE.replace("TEXT PRIMARY KEY", "TEXT PRIMARY KEY"))
        cur.execute(_CREATE_INDEX_TS)
        cur.execute(_CREATE_INDEX_ALGO)
        cur.execute(_CREATE_INDEX_USER)
        conn.commit()
        cur.close()
        return cls(conn, backend="postgres")

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def log(self, entry: AuditEntry) -> str:
        """Append an audit entry.  Returns the ``entry_id``.

        Raises
        ------
        ValueError
            If the store has been closed.
        """
        if self._closed:
            raise ValueError("AuditStore is closed.")

        sql = (
            "INSERT INTO audit_log "
            "(entry_id, timestamp, user, algorithm, params, "
            "input_hash, result, duration_ms) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
        )
        if self._backend == "postgres":
            sql = sql.replace("?", "%s")

        values = (
            entry.entry_id,
            entry.timestamp,
            entry.user,
            entry.algorithm,
            json.dumps(entry.params, default=str),
            entry.input_hash,
            json.dumps(entry.result, default=str),
            entry.duration_ms,
        )

        with self._lock:
            cur = self._conn.cursor()
            cur.execute(sql, values)
            self._conn.commit()
            cur.close()

        return entry.entry_id

    def get(self, entry_id: str) -> AuditEntry | None:
        """Retrieve a single entry by its ID, or ``None`` if not found."""
        sql = "SELECT * FROM audit_log WHERE entry_id = ?"
        if self._backend == "postgres":
            sql = sql.replace("?", "%s")

        cur = self._conn.cursor()
        cur.execute(sql, (entry_id,))
        row = cur.fetchone()
        cur.close()
        if row is None:
            return None
        return self._row_to_entry(row)

    def query(
        self,
        *,
        start: str | None = None,
        end: str | None = None,
        algorithm: str | None = None,
        user: str | None = None,
        limit: int | None = None,
    ) -> list[AuditEntry]:
        """Query audit entries matching the given filters.

        Parameters
        ----------
        start : str, optional
            ISO-8601 lower bound (inclusive) on timestamp.
        end : str, optional
            ISO-8601 upper bound (inclusive) on timestamp.
        algorithm : str, optional
            Exact algorithm name filter.
        user : str, optional
            Exact user name filter.
        limit : int, optional
            Maximum number of rows to return.

        Returns
        -------
        list[AuditEntry]
            Matching entries ordered by timestamp ascending.
        """
        return self.query_filter(
            QueryFilter(
                start=start,
                end=end,
                algorithm=algorithm,
                user=user,
                limit=limit,
            )
        )

    def query_filter(self, qf: QueryFilter) -> list[AuditEntry]:
        """Query using a :class:`QueryFilter` object."""
        clauses: list[str] = []
        params: list[Any] = []
        ph = "%s" if self._backend == "postgres" else "?"

        if qf.start is not None:
            clauses.append(f"timestamp >= {ph}")
            params.append(qf.start)
        if qf.end is not None:
            clauses.append(f"timestamp <= {ph}")
            params.append(qf.end)
        if qf.algorithm is not None:
            clauses.append(f"algorithm = {ph}")
            params.append(qf.algorithm)
        if qf.user is not None:
            clauses.append(f"user = {ph}")
            params.append(qf.user)

        sql = "SELECT * FROM audit_log"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY timestamp ASC"
        if qf.limit is not None:
            sql += f" LIMIT {int(qf.limit)}"

        cur = self._conn.cursor()
        cur.execute(sql, params)
        rows = cur.fetchall()
        cur.close()
        return [self._row_to_entry(r) for r in rows]

    def count(self) -> int:
        """Return total number of audit entries."""
        cur = self._conn.cursor()
        cur.execute("SELECT COUNT(*) FROM audit_log")
        n = cur.fetchone()[0]
        cur.close()
        return n

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_csv(
        self, entries: Sequence[AuditEntry], path: str | Path | None = None
    ) -> str:
        """Export entries to CSV.

        Parameters
        ----------
        entries : Sequence[AuditEntry]
            Entries to export (typically from :meth:`query`).
        path : str | Path, optional
            If provided, write to this file.  Otherwise return the CSV
            as a string.

        Returns
        -------
        str
            The CSV content.
        """
        fieldnames = [
            "entry_id",
            "timestamp",
            "user",
            "algorithm",
            "params",
            "input_hash",
            "result",
            "duration_ms",
        ]
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=fieldnames)
        writer.writeheader()
        for e in entries:
            row = asdict(e)
            row["params"] = json.dumps(row["params"], default=str)
            row["result"] = json.dumps(row["result"], default=str)
            writer.writerow(row)

        content = buf.getvalue()
        if path is not None:
            Path(path).write_text(content, encoding="utf-8")
        return content

    def export_json(
        self, entries: Sequence[AuditEntry], path: str | Path | None = None
    ) -> str:
        """Export entries to JSON.

        Parameters
        ----------
        entries : Sequence[AuditEntry]
            Entries to export.
        path : str | Path, optional
            If provided, write to this file.  Otherwise return the JSON
            as a string.

        Returns
        -------
        str
            The JSON content.
        """
        data = [asdict(e) for e in entries]
        content = json.dumps(data, indent=2, default=str)
        if path is not None:
            Path(path).write_text(content, encoding="utf-8")
        return content

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the underlying database connection."""
        if not self._closed:
            self._conn.close()
            self._closed = True

    def __enter__(self) -> AuditStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_entry(row: tuple) -> AuditEntry:
        """Convert a database row tuple to an :class:`AuditEntry`."""
        return AuditEntry(
            entry_id=row[0],
            timestamp=row[1],
            user=row[2],
            algorithm=row[3],
            params=json.loads(row[4]),
            input_hash=row[5],
            result=json.loads(row[6]),
            duration_ms=row[7],
        )
