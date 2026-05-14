"""REST API and async job queue for qufin.

Provides a FastAPI-based REST API with endpoints for portfolio optimization,
option pricing, and risk computation, plus a Celery-based async job queue
for long-running quantum computations.

Optional dependencies: ``pip install fastapi uvicorn celery redis``
"""

from __future__ import annotations

__all__ = [
    "JobQueue",
    "JobStatus",
    "create_app",
]


def create_app(**kwargs):
    """Create and return the FastAPI application.

    Raises :class:`ImportError` if ``fastapi`` is not installed.
    """
    from qufin.api.server import create_app as _create_app

    return _create_app(**kwargs)


def get_job_queue(**kwargs):
    """Return a :class:`JobQueue` instance.

    Raises :class:`ImportError` if ``celery`` is not installed.
    """
    from qufin.api.jobs import JobQueue

    return JobQueue(**kwargs)
