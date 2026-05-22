"""Circuit compilation cache with LRU eviction.

Caches transpiled circuits keyed by (circuit hash, backend, optimization level)
to avoid redundant transpilation. Thread-safe via threading.Lock.
"""

from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any


@dataclass
class CacheStats:
    """Hit/miss statistics for the circuit cache."""

    hits: int = 0
    misses: int = 0

    @property
    def total(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        return self.hits / self.total if self.total > 0 else 0.0


def _circuit_hash(circuit: Any) -> str:
    """Compute a deterministic hash for a Qiskit QuantumCircuit."""
    try:
        qasm = circuit.qasm()
    except Exception:
        qasm = str(circuit)
    return hashlib.sha256(qasm.encode()).hexdigest()


class CachedTranspiler:
    """LRU-cached wrapper around ``qiskit.compiler.transpile``.

    Parameters
    ----------
    max_size : int
        Maximum number of cached transpiled circuits (default 128).
    """

    def __init__(self, max_size: int = 128) -> None:
        if max_size < 1:
            raise ValueError("max_size must be >= 1")
        self._max_size = max_size
        self._cache: OrderedDict[str, Any] = OrderedDict()
        self._lock = threading.Lock()
        self._stats = CacheStats()

    @property
    def stats(self) -> CacheStats:
        return self._stats

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._cache)

    @property
    def max_size(self) -> int:
        return self._max_size

    def _make_key(self, circuit: Any, backend_id: str, optimization_level: int) -> str:
        h = _circuit_hash(circuit)
        return f"{h}:{backend_id}:{optimization_level}"

    def transpile(
        self,
        circuit: Any,
        backend_id: str = "aer",
        optimization_level: int = 1,
        **transpile_kwargs: Any,
    ) -> Any:
        """Transpile a circuit, returning a cached version if available.

        Parameters
        ----------
        circuit : QuantumCircuit
            The circuit to transpile.
        backend_id : str
            Backend identifier for cache keying.
        optimization_level : int
            Qiskit transpilation optimization level (0-3).
        **transpile_kwargs
            Extra keyword arguments forwarded to ``qiskit.compiler.transpile``.
        """
        key = self._make_key(circuit, backend_id, optimization_level)

        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                self._stats.hits += 1
                return self._cache[key]

        # Transpile outside the lock to avoid blocking
        from qiskit.compiler import transpile

        transpiled = transpile(
            circuit,
            optimization_level=optimization_level,
            **transpile_kwargs,
        )

        with self._lock:
            self._cache[key] = transpiled
            self._cache.move_to_end(key)
            self._stats.misses += 1
            while len(self._cache) > self._max_size:
                self._cache.popitem(last=False)

        return transpiled

    def clear(self) -> None:
        """Clear the cache and reset statistics."""
        with self._lock:
            self._cache.clear()
            self._stats = CacheStats()
