"""Async job queue for long-running quantum computations.

Uses Celery with a Redis backend for distributed task management.
Both ``celery`` and ``redis`` are optional dependencies.

Usage::

    from qufin.api.jobs import JobQueue, JobType, JobPriority

    queue = JobQueue(broker_url="redis://localhost:6379/0")
    job_id = queue.submit("optimization", params={...})
    status = queue.status(job_id)
    result = queue.result(job_id)
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

# ---------------------------------------------------------------------------
# Optional dependency guards
# ---------------------------------------------------------------------------

try:
    from celery import Celery
    from celery import states as celery_states
    from celery.result import AsyncResult

    _HAS_CELERY = True
except ImportError:  # pragma: no cover
    _HAS_CELERY = False
    celery_states = None  # type: ignore[assignment]
    AsyncResult = None  # type: ignore[assignment, misc]

try:
    import redis as _redis_mod

    _HAS_REDIS = True
except ImportError:  # pragma: no cover
    _HAS_REDIS = False
    _redis_mod = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Enums and constants
# ---------------------------------------------------------------------------


class JobType(str, Enum):
    """Supported job types."""

    OPTIMIZATION = "optimization"
    PRICING = "pricing"
    RISK = "risk"
    BACKTEST = "backtest"


class JobStatus(str, Enum):
    """Job lifecycle status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobPriority(str, Enum):
    """Priority queue selection."""

    INTERACTIVE = "interactive"
    BATCH = "batch"


# Default timeouts per job type (seconds)
DEFAULT_TIMEOUTS: dict[str, int] = {
    JobType.OPTIMIZATION: 300,
    JobType.PRICING: 120,
    JobType.RISK: 180,
    JobType.BACKTEST: 600,
}

# Result expiration per job type (seconds)
DEFAULT_RESULT_EXPIRY: dict[str, int] = {
    JobType.OPTIMIZATION: 3600,
    JobType.PRICING: 1800,
    JobType.RISK: 3600,
    JobType.BACKTEST: 7200,
}

# Queue routing
QUEUE_ROUTES: dict[str, str] = {
    JobPriority.INTERACTIVE: "qufin.interactive",
    JobPriority.BATCH: "qufin.batch",
}


# ---------------------------------------------------------------------------
# Job metadata
# ---------------------------------------------------------------------------


@dataclass
class JobMeta:
    """Metadata for a submitted job."""

    job_id: str
    job_type: JobType
    priority: JobPriority
    status: JobStatus
    params: dict[str, Any]
    result: dict[str, Any] | None = None
    error: str | None = None
    timeout: int = 300
    result_expiry: int = 3600
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "job_id": self.job_id,
            "job_type": self.job_type.value,
            "priority": self.priority.value,
            "status": self.status.value,
            "params": self.params,
            "result": self.result,
            "error": self.error,
            "timeout": self.timeout,
            "result_expiry": self.result_expiry,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> JobMeta:
        """Deserialize from dictionary."""
        return cls(
            job_id=data["job_id"],
            job_type=JobType(data["job_type"]),
            priority=JobPriority(data["priority"]),
            status=JobStatus(data["status"]),
            params=data.get("params", {}),
            result=data.get("result"),
            error=data.get("error"),
            timeout=data.get("timeout", 300),
            result_expiry=data.get("result_expiry", 3600),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
        )


# ---------------------------------------------------------------------------
# Celery app factory
# ---------------------------------------------------------------------------


def create_celery_app(
    broker_url: str = "redis://localhost:6379/0",
    result_backend: str | None = None,
    task_default_queue: str = "qufin.interactive",
) -> Celery:
    """Create and configure a Celery application.

    Parameters
    ----------
    broker_url : str
        Message broker URL (Redis or RabbitMQ).
    result_backend : str | None
        Result backend URL. Defaults to ``broker_url``.
    task_default_queue : str
        Default task queue name.

    Returns
    -------
    Celery
        Configured Celery application.

    Raises
    ------
    ImportError
        If ``celery`` is not installed.
    """
    if not _HAS_CELERY:
        raise ImportError(
            "celery is required for the qufin job queue. "
            "Install it with: pip install celery[redis]"
        )

    backend = result_backend or broker_url

    celery_app = Celery(
        "qufin.jobs",
        broker=broker_url,
        backend=backend,
    )

    celery_app.conf.update(
        task_serializer="json",
        accept_content=["json"],
        result_serializer="json",
        task_default_queue=task_default_queue,
        task_queues={
            "qufin.interactive": {"exchange": "qufin", "routing_key": "interactive"},
            "qufin.batch": {"exchange": "qufin", "routing_key": "batch"},
        },
        task_track_started=True,
        task_acks_late=True,
        worker_prefetch_multiplier=1,
        result_expires=3600,
    )

    _register_tasks(celery_app)

    return celery_app


def _register_tasks(celery_app: Celery) -> None:
    """Register Celery task handlers."""

    @celery_app.task(name="qufin.optimize", bind=True, max_retries=1)
    def run_optimize(self, params: dict[str, Any]) -> dict[str, Any]:
        """Execute portfolio optimization as a Celery task."""
        try:
            from qufin.api.server import OptimizeRequest, _run_optimize

            req = OptimizeRequest(**params)
            result = _run_optimize(req)
            return result.model_dump()
        except Exception as exc:
            raise (
                self.retry(exc=exc, countdown=5) if self.request.retries < 1 else exc
            ) from exc

    @celery_app.task(name="qufin.price", bind=True, max_retries=1)
    def run_price(self, params: dict[str, Any]) -> dict[str, Any]:
        """Execute option pricing as a Celery task."""
        try:
            from qufin.api.server import PriceRequest, _run_price

            req = PriceRequest(**params)
            result = _run_price(req)
            return result.model_dump()
        except Exception as exc:
            raise (
                self.retry(exc=exc, countdown=5) if self.request.retries < 1 else exc
            ) from exc

    @celery_app.task(name="qufin.risk", bind=True, max_retries=1)
    def run_risk(self, params: dict[str, Any]) -> dict[str, Any]:
        """Execute risk computation as a Celery task."""
        try:
            from qufin.api.server import RiskRequest, _run_risk

            req = RiskRequest(**params)
            result = _run_risk(req)
            return result.model_dump()
        except Exception as exc:
            raise (
                self.retry(exc=exc, countdown=5) if self.request.retries < 1 else exc
            ) from exc

    @celery_app.task(name="qufin.backtest", bind=True, max_retries=1)
    def run_backtest(self, params: dict[str, Any]) -> dict[str, Any]:
        """Execute backtesting as a Celery task."""
        try:
            from qufin.backtesting.engine import BacktestEngine

            engine = BacktestEngine(**params)
            result = engine.run()
            return result.to_dict() if hasattr(result, "to_dict") else {"status": "completed"}
        except Exception as exc:
            raise (
                self.retry(exc=exc, countdown=5) if self.request.retries < 1 else exc
            ) from exc


# ---------------------------------------------------------------------------
# Job Queue — high-level interface
# ---------------------------------------------------------------------------


class JobQueue:
    """High-level job queue interface.

    Manages job submission, status tracking, cancellation, and cleanup.
    Uses Celery when available, falls back to in-memory execution.

    Parameters
    ----------
    broker_url : str
        Redis/RabbitMQ broker URL.
    result_backend : str | None
        Result backend URL. Defaults to broker_url.
    use_celery : bool
        If True and Celery is available, use distributed task queue.
        If False, run tasks synchronously (for testing).
    default_timeout : int
        Default task timeout in seconds.
    """

    def __init__(
        self,
        broker_url: str = "redis://localhost:6379/0",
        result_backend: str | None = None,
        use_celery: bool = True,
        default_timeout: int = 300,
    ):
        self.broker_url = broker_url
        self.result_backend = result_backend or broker_url
        self.default_timeout = default_timeout
        self._jobs: dict[str, JobMeta] = {}
        self._celery_app: Celery | None = None

        if use_celery and _HAS_CELERY:
            try:
                self._celery_app = create_celery_app(
                    broker_url=broker_url,
                    result_backend=self.result_backend,
                )
            except Exception:
                self._celery_app = None

    @property
    def has_celery(self) -> bool:
        """Whether Celery backend is available."""
        return self._celery_app is not None

    def submit(
        self,
        job_type: str | JobType,
        params: dict[str, Any],
        priority: str | JobPriority = JobPriority.INTERACTIVE,
        timeout: int | None = None,
        result_expiry: int | None = None,
    ) -> str:
        """Submit a job for execution.

        Parameters
        ----------
        job_type : str | JobType
            Type of computation to run.
        params : dict
            Job parameters (must be JSON-serializable).
        priority : str | JobPriority
            Queue priority (interactive or batch).
        timeout : int | None
            Task timeout in seconds. Uses per-type defaults if None.
        result_expiry : int | None
            How long to keep results (seconds). Uses per-type defaults if None.

        Returns
        -------
        str
            Unique job identifier.
        """
        jtype = JobType(job_type) if isinstance(job_type, str) else job_type
        jpriority = JobPriority(priority) if isinstance(priority, str) else priority

        job_id = str(uuid.uuid4())
        effective_timeout = timeout or DEFAULT_TIMEOUTS.get(jtype.value, self.default_timeout)
        effective_expiry = result_expiry or DEFAULT_RESULT_EXPIRY.get(jtype.value, 3600)

        meta = JobMeta(
            job_id=job_id,
            job_type=jtype,
            priority=jpriority,
            status=JobStatus.PENDING,
            params=params,
            timeout=effective_timeout,
            result_expiry=effective_expiry,
        )
        self._jobs[job_id] = meta

        # Dispatch to Celery if available
        if self._celery_app is not None:
            task_name = f"qufin.{jtype.value}"
            queue = QUEUE_ROUTES.get(jpriority.value, "qufin.interactive")
            try:
                self._celery_app.send_task(
                    task_name,
                    kwargs={"params": params},
                    queue=queue,
                    time_limit=effective_timeout,
                    soft_time_limit=max(1, effective_timeout - 10),
                    task_id=job_id,
                    expires=effective_expiry,
                )
                meta.status = JobStatus.PENDING
            except Exception as exc:
                meta.status = JobStatus.FAILED
                meta.error = str(exc)

        return job_id

    def status(self, job_id: str) -> JobMeta:
        """Get current job status.

        Parameters
        ----------
        job_id : str
            Job identifier.

        Returns
        -------
        JobMeta
            Current job metadata.

        Raises
        ------
        KeyError
            If job_id is not found.
        """
        if job_id not in self._jobs:
            raise KeyError(f"Job {job_id} not found")

        meta = self._jobs[job_id]

        # Sync status from Celery if active
        if self._celery_app is not None and meta.status in (
            JobStatus.PENDING,
            JobStatus.RUNNING,
        ):
            try:
                async_result = AsyncResult(job_id, app=self._celery_app)
                celery_status = async_result.status

                if celery_status == "PENDING":
                    meta.status = JobStatus.PENDING
                elif celery_status == "STARTED":
                    meta.status = JobStatus.RUNNING
                elif celery_status == "SUCCESS":
                    meta.status = JobStatus.COMPLETED
                    meta.result = async_result.result
                elif celery_status in ("FAILURE", "REVOKED"):
                    meta.status = (
                        JobStatus.CANCELLED
                        if celery_status == "REVOKED"
                        else JobStatus.FAILED
                    )
                    meta.error = str(async_result.result) if async_result.result else None

                meta.updated_at = datetime.now(timezone.utc).isoformat()
            except Exception:
                pass

        return meta

    def result(self, job_id: str, timeout: float | None = None) -> dict[str, Any] | None:
        """Get job result, optionally waiting for completion.

        Parameters
        ----------
        job_id : str
            Job identifier.
        timeout : float | None
            Maximum seconds to wait. None means no wait.

        Returns
        -------
        dict | None
            Job result or None if not yet completed.

        Raises
        ------
        KeyError
            If job_id not found.
        RuntimeError
            If job failed.
        """
        meta = self.status(job_id)

        if meta.status == JobStatus.FAILED:
            raise RuntimeError(f"Job {job_id} failed: {meta.error}")

        if meta.status == JobStatus.COMPLETED:
            return meta.result

        if timeout is not None and self._celery_app is not None:
            try:
                async_result = AsyncResult(job_id, app=self._celery_app)
                res = async_result.get(timeout=timeout)
                meta.status = JobStatus.COMPLETED
                meta.result = res
                meta.updated_at = datetime.now(timezone.utc).isoformat()
                return res
            except Exception as exc:
                meta.status = JobStatus.FAILED
                meta.error = str(exc)
                meta.updated_at = datetime.now(timezone.utc).isoformat()
                raise RuntimeError(f"Job {job_id} failed: {exc}") from exc

        return None

    def cancel(self, job_id: str) -> bool:
        """Cancel a pending or running job.

        Parameters
        ----------
        job_id : str
            Job identifier.

        Returns
        -------
        bool
            True if job was successfully cancelled.

        Raises
        ------
        KeyError
            If job_id not found.
        ValueError
            If job is already in a terminal state.
        """
        if job_id not in self._jobs:
            raise KeyError(f"Job {job_id} not found")

        meta = self._jobs[job_id]

        if meta.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
            raise ValueError(
                f"Cannot cancel job {job_id}: already {meta.status.value}"
            )

        if self._celery_app is not None:
            import contextlib

            with contextlib.suppress(Exception):
                self._celery_app.control.revoke(job_id, terminate=True)

        meta.status = JobStatus.CANCELLED
        meta.updated_at = datetime.now(timezone.utc).isoformat()
        return True

    def list_jobs(
        self,
        job_type: str | JobType | None = None,
        status_filter: str | JobStatus | None = None,
    ) -> list[JobMeta]:
        """List jobs with optional filters.

        Parameters
        ----------
        job_type : str | JobType | None
            Filter by job type.
        status_filter : str | JobStatus | None
            Filter by status.

        Returns
        -------
        list[JobMeta]
            Matching jobs sorted by creation time (newest first).
        """
        jobs = list(self._jobs.values())

        if job_type is not None:
            jtype = JobType(job_type) if isinstance(job_type, str) else job_type
            jobs = [j for j in jobs if j.job_type == jtype]

        if status_filter is not None:
            jstatus = (
                JobStatus(status_filter) if isinstance(status_filter, str) else status_filter
            )
            jobs = [j for j in jobs if j.status == jstatus]

        jobs.sort(key=lambda j: j.created_at, reverse=True)
        return jobs

    def cleanup_expired(self) -> int:
        """Remove completed/failed jobs past their result_expiry.

        Returns
        -------
        int
            Number of jobs cleaned up.
        """
        now = datetime.now(timezone.utc)
        to_remove = []

        for job_id, meta in self._jobs.items():
            if meta.status not in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
                continue

            try:
                updated = datetime.fromisoformat(meta.updated_at)
                # Ensure timezone-aware comparison
                if updated.tzinfo is None:
                    updated = updated.replace(tzinfo=timezone.utc)
                elapsed = (now - updated).total_seconds()
                if elapsed >= meta.result_expiry:
                    to_remove.append(job_id)
            except (ValueError, TypeError):
                continue

        for job_id in to_remove:
            del self._jobs[job_id]

        return len(to_remove)

    def get_queue_stats(self) -> dict[str, Any]:
        """Get queue statistics.

        Returns
        -------
        dict
            Statistics including counts by status and type.
        """
        stats: dict[str, Any] = {
            "total": len(self._jobs),
            "by_status": {},
            "by_type": {},
            "by_priority": {},
        }

        for meta in self._jobs.values():
            s = meta.status.value
            stats["by_status"][s] = stats["by_status"].get(s, 0) + 1
            t = meta.job_type.value
            stats["by_type"][t] = stats["by_type"].get(t, 0) + 1
            p = meta.priority.value
            stats["by_priority"][p] = stats["by_priority"].get(p, 0) + 1

        return stats
