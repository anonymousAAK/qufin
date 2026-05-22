"""Backend abstraction layer for quantum framework dispatch."""

from __future__ import annotations

from qufin.backends.auto_select import (
    BackendRegistry,
    CircuitAnalysis,
    analyze_circuit,
    auto_select_backend,
    get_available_backends,
)
from qufin.backends.base import Backend, CircuitResult
from qufin.backends.mock import MockBackend

__all__ = [
    "Backend",
    "BackendRegistry",
    "CircuitAnalysis",
    "CircuitResult",
    "MockBackend",
    "analyze_circuit",
    "auto_select_backend",
    "get_available_backends",
]


def get_noisy_backend(**kwargs):
    """Lazy import for noisy Aer backend with configurable noise profiles."""
    from qufin.backends.noise_models import NoisyAerBackend

    return NoisyAerBackend(**kwargs)


def get_ibm_backend(**kwargs):
    """Lazy import for IBM Runtime backend (requires qufin[ibm])."""
    from qufin.backends.ibm_runtime import IBMRuntimeBackend

    return IBMRuntimeBackend(**kwargs)


def get_pennylane_backend(**kwargs):
    """Lazy import for PennyLane backend (requires qufin[pennylane])."""
    from qufin.backends.pennylane_backend import PennyLaneBackend

    return PennyLaneBackend(**kwargs)


def get_cirq_backend(**kwargs):
    """Lazy import for Cirq backend (requires qufin[cirq])."""
    from qufin.backends.cirq_backend import CirqBackend

    return CirqBackend(**kwargs)


def get_braket_backend(**kwargs):
    """Lazy import for Braket backend (requires qufin[braket])."""
    from qufin.backends.braket_backend import BraketBackend

    return BraketBackend(**kwargs)


def get_cudaq_backend(**kwargs):
    """Lazy import for CUDA-Q backend (requires cuda-quantum)."""
    from qufin.backends.cudaq_backend import CudaQBackend

    return CudaQBackend(**kwargs)


def get_dwave_backend(**kwargs):
    """Lazy import for D-Wave annealing backend (requires dwave-ocean-sdk)."""
    from qufin.backends.dwave_backend import DWaveBackend

    return DWaveBackend(**kwargs)


def get_ionq_backend(**kwargs):
    """Lazy import for IonQ backend (requires qufin[ionq])."""
    from qufin.backends.ionq_backend import IonQBackend

    return IonQBackend(**kwargs)


def get_quantinuum_backend(**kwargs):
    """Lazy import for Quantinuum backend (requires qufin[quantinuum])."""
    from qufin.backends.quantinuum_backend import QuantinuumBackend

    return QuantinuumBackend(**kwargs)
