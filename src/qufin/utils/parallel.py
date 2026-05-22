"""Parallel and batched circuit execution utilities.

Provides ``parallel_execute`` for concurrent circuit runs via
ThreadPoolExecutor, and ``batch_execute`` for memory-friendly batching.
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from qufin.backends.base import Backend, CircuitResult


def parallel_execute(
    circuits: list[Any],
    backend: Backend,
    shots: int = 1024,
    max_workers: int = 4,
    progress_callback: Callable[[int, int], None] | None = None,
) -> list[CircuitResult]:
    """Execute multiple circuits concurrently.

    Parameters
    ----------
    circuits : list
        Quantum circuits to execute.
    backend : Backend
        qufin backend instance.
    shots : int
        Shots per circuit.
    max_workers : int
        Maximum parallel threads.
    progress_callback : callable, optional
        Called as ``callback(completed, total)`` after each circuit finishes.

    Returns
    -------
    list[CircuitResult]
        Results in the same order as *circuits*.
    """
    if not circuits:
        return []

    total = len(circuits)
    results: list[CircuitResult | None] = [None] * total

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_to_idx = {
            pool.submit(backend.run, circ, shots): idx
            for idx, circ in enumerate(circuits)
        }
        for done_count, future in enumerate(as_completed(future_to_idx), 1):
            idx = future_to_idx[future]
            results[idx] = future.result()
            if progress_callback is not None:
                progress_callback(done_count, total)

    return results  # type: ignore[return-value]


def batch_execute(
    circuits: list[Any],
    backend: Backend,
    shots: int = 1024,
    batch_size: int = 10,
    progress_callback: Callable[[int, int], None] | None = None,
) -> list[CircuitResult]:
    """Execute circuits in sequential batches for memory management.

    Parameters
    ----------
    circuits : list
        Quantum circuits to execute.
    backend : Backend
        qufin backend instance.
    shots : int
        Shots per circuit.
    batch_size : int
        Number of circuits per batch.
    progress_callback : callable, optional
        Called as ``callback(completed, total)`` after each circuit finishes.

    Returns
    -------
    list[CircuitResult]
        Results in the same order as *circuits*.
    """
    if not circuits:
        return []
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")

    total = len(circuits)
    results: list[CircuitResult] = []

    for start in range(0, total, batch_size):
        batch = circuits[start : start + batch_size]
        for circ in batch:
            res = backend.run(circ, shots)
            results.append(res)
            if progress_callback is not None:
                progress_callback(len(results), total)

    return results
