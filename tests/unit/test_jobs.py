"""Tests for the qufin async job queue.

All Celery/Redis interactions are fully mocked.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from qufin.api.jobs import (
    DEFAULT_RESULT_EXPIRY,
    DEFAULT_TIMEOUTS,
    QUEUE_ROUTES,
    JobMeta,
    JobPriority,
    JobQueue,
    JobStatus,
    JobType,
    create_celery_app,
)

# ---------------------------------------------------------------------------
# JobType / JobStatus / JobPriority enums
# ---------------------------------------------------------------------------


class TestEnums:
    """Tests for job-related enums."""

    def test_job_types(self):
        assert JobType.OPTIMIZATION == "optimization"
        assert JobType.PRICING == "pricing"
        assert JobType.RISK == "risk"
        assert JobType.BACKTEST == "backtest"

    def test_job_statuses(self):
        assert JobStatus.PENDING == "pending"
        assert JobStatus.RUNNING == "running"
        assert JobStatus.COMPLETED == "completed"
        assert JobStatus.FAILED == "failed"
        assert JobStatus.CANCELLED == "cancelled"

    def test_job_priorities(self):
        assert JobPriority.INTERACTIVE == "interactive"
        assert JobPriority.BATCH == "batch"

    def test_default_timeouts(self):
        assert DEFAULT_TIMEOUTS[JobType.OPTIMIZATION] == 300
        assert DEFAULT_TIMEOUTS[JobType.BACKTEST] == 600

    def test_queue_routes(self):
        assert QUEUE_ROUTES[JobPriority.INTERACTIVE] == "qufin.interactive"
        assert QUEUE_ROUTES[JobPriority.BATCH] == "qufin.batch"


# ---------------------------------------------------------------------------
# JobMeta
# ---------------------------------------------------------------------------


class TestJobMeta:
    """Tests for JobMeta dataclass."""

    def test_create(self):
        meta = JobMeta(
            job_id="test-123",
            job_type=JobType.OPTIMIZATION,
            priority=JobPriority.INTERACTIVE,
            status=JobStatus.PENDING,
            params={"tickers": ["AAPL"]},
        )
        assert meta.job_id == "test-123"
        assert meta.job_type == JobType.OPTIMIZATION
        assert meta.status == JobStatus.PENDING

    def test_to_dict(self):
        meta = JobMeta(
            job_id="test-456",
            job_type=JobType.PRICING,
            priority=JobPriority.BATCH,
            status=JobStatus.RUNNING,
            params={"spot": 100},
            timeout=120,
        )
        d = meta.to_dict()
        assert d["job_id"] == "test-456"
        assert d["job_type"] == "pricing"
        assert d["priority"] == "batch"
        assert d["status"] == "running"
        assert d["timeout"] == 120

    def test_from_dict(self):
        data = {
            "job_id": "test-789",
            "job_type": "risk",
            "priority": "interactive",
            "status": "completed",
            "params": {"weights": {}},
            "result": {"var": 0.05},
            "error": None,
            "timeout": 180,
            "result_expiry": 3600,
            "created_at": "2024-01-01T00:00:00+00:00",
            "updated_at": "2024-01-01T00:01:00+00:00",
        }
        meta = JobMeta.from_dict(data)
        assert meta.job_type == JobType.RISK
        assert meta.status == JobStatus.COMPLETED
        assert meta.result == {"var": 0.05}

    def test_roundtrip(self):
        meta = JobMeta(
            job_id="rt-1",
            job_type=JobType.BACKTEST,
            priority=JobPriority.BATCH,
            status=JobStatus.FAILED,
            params={"param": 1},
            error="timeout",
        )
        restored = JobMeta.from_dict(meta.to_dict())
        assert restored.job_id == meta.job_id
        assert restored.job_type == meta.job_type
        assert restored.status == meta.status
        assert restored.error == meta.error


# ---------------------------------------------------------------------------
# JobQueue — no Celery (sync mode)
# ---------------------------------------------------------------------------


class TestJobQueueSync:
    """Tests for JobQueue without Celery (sync/in-memory mode)."""

    def test_create_queue_no_celery(self):
        queue = JobQueue(use_celery=False)
        assert not queue.has_celery

    def test_submit_job(self):
        queue = JobQueue(use_celery=False)
        job_id = queue.submit("optimization", {"tickers": ["AAPL", "MSFT"]})
        assert isinstance(job_id, str)
        assert len(job_id) > 0

    def test_status_pending(self):
        queue = JobQueue(use_celery=False)
        job_id = queue.submit("pricing", {"spot": 100})
        meta = queue.status(job_id)
        assert meta.status == JobStatus.PENDING
        assert meta.job_type == JobType.PRICING

    def test_status_nonexistent(self):
        queue = JobQueue(use_celery=False)
        with pytest.raises(KeyError, match="not found"):
            queue.status("nonexistent-id")

    def test_result_not_completed(self):
        queue = JobQueue(use_celery=False)
        job_id = queue.submit("risk", {"weights": {}})
        result = queue.result(job_id)
        assert result is None

    def test_result_completed(self):
        queue = JobQueue(use_celery=False)
        job_id = queue.submit("optimization", {"tickers": []})
        meta = queue._jobs[job_id]
        meta.status = JobStatus.COMPLETED
        meta.result = {"weights": {"AAPL": 1.0}}
        result = queue.result(job_id)
        assert result == {"weights": {"AAPL": 1.0}}

    def test_result_failed_raises(self):
        queue = JobQueue(use_celery=False)
        job_id = queue.submit("pricing", {})
        meta = queue._jobs[job_id]
        meta.status = JobStatus.FAILED
        meta.error = "something broke"
        with pytest.raises(RuntimeError, match="failed"):
            queue.result(job_id)

    def test_cancel_pending(self):
        queue = JobQueue(use_celery=False)
        job_id = queue.submit("risk", {})
        assert queue.cancel(job_id)
        meta = queue.status(job_id)
        assert meta.status == JobStatus.CANCELLED

    def test_cancel_already_completed(self):
        queue = JobQueue(use_celery=False)
        job_id = queue.submit("optimization", {})
        queue._jobs[job_id].status = JobStatus.COMPLETED
        with pytest.raises(ValueError, match="Cannot cancel"):
            queue.cancel(job_id)

    def test_cancel_nonexistent(self):
        queue = JobQueue(use_celery=False)
        with pytest.raises(KeyError, match="not found"):
            queue.cancel("nonexistent-id")

    def test_list_jobs_empty(self):
        queue = JobQueue(use_celery=False)
        assert queue.list_jobs() == []

    def test_list_jobs_filtered(self):
        queue = JobQueue(use_celery=False)
        queue.submit("optimization", {})
        queue.submit("pricing", {})
        queue.submit("pricing", {})

        opt_jobs = queue.list_jobs(job_type="optimization")
        assert len(opt_jobs) == 1

        price_jobs = queue.list_jobs(job_type="pricing")
        assert len(price_jobs) == 2

    def test_list_jobs_by_status(self):
        queue = JobQueue(use_celery=False)
        id1 = queue.submit("optimization", {})
        queue.submit("pricing", {})
        queue._jobs[id1].status = JobStatus.COMPLETED

        pending = queue.list_jobs(status_filter="pending")
        assert len(pending) == 1
        completed = queue.list_jobs(status_filter="completed")
        assert len(completed) == 1

    def test_submit_with_priority(self):
        queue = JobQueue(use_celery=False)
        job_id = queue.submit("optimization", {}, priority="batch")
        meta = queue.status(job_id)
        assert meta.priority == JobPriority.BATCH

    def test_submit_with_custom_timeout(self):
        queue = JobQueue(use_celery=False)
        job_id = queue.submit("optimization", {}, timeout=999)
        meta = queue.status(job_id)
        assert meta.timeout == 999

    def test_submit_with_custom_expiry(self):
        queue = JobQueue(use_celery=False)
        job_id = queue.submit("pricing", {}, result_expiry=7200)
        meta = queue.status(job_id)
        assert meta.result_expiry == 7200

    def test_default_timeout_per_type(self):
        queue = JobQueue(use_celery=False)
        opt_id = queue.submit("optimization", {})
        bt_id = queue.submit("backtest", {})
        assert queue.status(opt_id).timeout == 300
        assert queue.status(bt_id).timeout == 600

    def test_default_expiry_per_type(self):
        queue = JobQueue(use_celery=False)
        opt_id = queue.submit("optimization", {})
        bt_id = queue.submit("backtest", {})
        assert queue.status(opt_id).result_expiry == DEFAULT_RESULT_EXPIRY["optimization"]
        assert queue.status(bt_id).result_expiry == DEFAULT_RESULT_EXPIRY["backtest"]


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------


class TestCleanup:
    """Tests for result expiration and cleanup."""

    def test_cleanup_removes_expired(self):
        queue = JobQueue(use_celery=False)
        job_id = queue.submit("optimization", {}, result_expiry=1)
        queue._jobs[job_id].status = JobStatus.COMPLETED
        # Set updated_at far enough in the past to exceed the 1-second expiry
        past = (datetime.now(timezone.utc) - timedelta(seconds=60)).isoformat()
        queue._jobs[job_id].updated_at = past

        removed = queue.cleanup_expired()
        assert removed == 1
        assert len(queue._jobs) == 0

    def test_cleanup_keeps_active(self):
        queue = JobQueue(use_celery=False)
        queue.submit("optimization", {})  # PENDING — should not be cleaned
        removed = queue.cleanup_expired()
        assert removed == 0
        assert len(queue._jobs) == 1

    def test_cleanup_keeps_non_expired(self):
        queue = JobQueue(use_celery=False)
        job_id = queue.submit("optimization", {}, result_expiry=99999)
        queue._jobs[job_id].status = JobStatus.COMPLETED
        removed = queue.cleanup_expired()
        assert removed == 0


# ---------------------------------------------------------------------------
# Queue stats
# ---------------------------------------------------------------------------


class TestQueueStats:
    """Tests for get_queue_stats."""

    def test_stats_empty(self):
        queue = JobQueue(use_celery=False)
        stats = queue.get_queue_stats()
        assert stats["total"] == 0
        assert stats["by_status"] == {}

    def test_stats_counts(self):
        queue = JobQueue(use_celery=False)
        id1 = queue.submit("optimization", {}, priority="interactive")
        queue.submit("pricing", {}, priority="batch")
        queue._jobs[id1].status = JobStatus.COMPLETED

        stats = queue.get_queue_stats()
        assert stats["total"] == 2
        assert stats["by_status"]["completed"] == 1
        assert stats["by_status"]["pending"] == 1
        assert stats["by_type"]["optimization"] == 1
        assert stats["by_type"]["pricing"] == 1
        assert stats["by_priority"]["interactive"] == 1
        assert stats["by_priority"]["batch"] == 1


# ---------------------------------------------------------------------------
# Celery integration (mocked)
# ---------------------------------------------------------------------------


class TestCeleryIntegration:
    """Tests for Celery integration with mocked Celery app."""

    @patch("qufin.api.jobs._HAS_CELERY", False)
    def test_create_celery_app_raises_without_celery(self):
        with pytest.raises(ImportError, match="celery is required"):
            create_celery_app()

    @patch("qufin.api.jobs._HAS_CELERY", True)
    @patch("qufin.api.jobs.create_celery_app")
    def test_queue_with_celery(self, mock_create):
        mock_app = MagicMock()
        mock_create.return_value = mock_app

        queue = JobQueue(use_celery=True)
        assert queue.has_celery

    @patch("qufin.api.jobs._HAS_CELERY", True)
    @patch("qufin.api.jobs.create_celery_app")
    def test_submit_dispatches_to_celery(self, mock_create):
        mock_app = MagicMock()
        mock_create.return_value = mock_app

        queue = JobQueue(use_celery=True)
        queue.submit("optimization", {"tickers": ["AAPL"]}, priority="batch")

        mock_app.send_task.assert_called_once()
        call_kwargs = mock_app.send_task.call_args
        assert call_kwargs[0][0] == "qufin.optimization"
        assert call_kwargs[1]["queue"] == "qufin.batch"

    @patch("qufin.api.jobs._HAS_CELERY", True)
    @patch("qufin.api.jobs.create_celery_app")
    def test_cancel_revokes_celery_task(self, mock_create):
        mock_app = MagicMock()
        mock_create.return_value = mock_app

        queue = JobQueue(use_celery=True)
        job_id = queue.submit("pricing", {})
        queue.cancel(job_id)

        mock_app.control.revoke.assert_called_once_with(job_id, terminate=True)

    @patch("qufin.api.jobs._HAS_CELERY", True)
    @patch("qufin.api.jobs.create_celery_app")
    @patch("qufin.api.jobs.AsyncResult")
    def test_status_syncs_from_celery(self, mock_ar_cls, mock_create):
        mock_app = MagicMock()
        mock_create.return_value = mock_app

        mock_ar = MagicMock()
        mock_ar.status = "SUCCESS"
        mock_ar.result = {"weights": {"AAPL": 1.0}}
        mock_ar_cls.return_value = mock_ar

        queue = JobQueue(use_celery=True)
        job_id = queue.submit("optimization", {})
        meta = queue.status(job_id)
        assert meta.status == JobStatus.COMPLETED
        assert meta.result == {"weights": {"AAPL": 1.0}}

    @patch("qufin.api.jobs._HAS_CELERY", True)
    @patch("qufin.api.jobs.create_celery_app")
    @patch("qufin.api.jobs.AsyncResult")
    def test_status_failure_from_celery(self, mock_ar_cls, mock_create):
        mock_app = MagicMock()
        mock_create.return_value = mock_app

        mock_ar = MagicMock()
        mock_ar.status = "FAILURE"
        mock_ar.result = Exception("task error")
        mock_ar_cls.return_value = mock_ar

        queue = JobQueue(use_celery=True)
        job_id = queue.submit("pricing", {})
        meta = queue.status(job_id)
        assert meta.status == JobStatus.FAILED

    @patch("qufin.api.jobs._HAS_CELERY", True)
    @patch("qufin.api.jobs.create_celery_app")
    @patch("qufin.api.jobs.AsyncResult")
    def test_status_revoked_from_celery(self, mock_ar_cls, mock_create):
        mock_app = MagicMock()
        mock_create.return_value = mock_app

        mock_ar = MagicMock()
        mock_ar.status = "REVOKED"
        mock_ar.result = None
        mock_ar_cls.return_value = mock_ar

        queue = JobQueue(use_celery=True)
        job_id = queue.submit("risk", {})
        meta = queue.status(job_id)
        assert meta.status == JobStatus.CANCELLED
