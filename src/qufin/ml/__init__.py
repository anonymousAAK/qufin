"""Quantum ML for finance: kernels, reservoir computing, classifiers, qGANs."""

from __future__ import annotations

from qufin.ml.classifiers import VariationalQuantumClassifier, VQCConfig
from qufin.ml.kernels import (
    QuantumKernelClassifier,
    ZZFeatureMap,
    quantum_kernel,
    quantum_kernel_matrix,
)
from qufin.ml.qgan import QGANConfig, QGANResult, QuantumGAN
from qufin.ml.reservoir import QuantumReservoir, QuantumReservoirConfig

__all__ = [
    "QGANConfig",
    "QGANResult",
    "QuantumGAN",
    "QuantumKernelClassifier",
    "QuantumReservoir",
    "QuantumReservoirConfig",
    "VQCConfig",
    "VariationalQuantumClassifier",
    "ZZFeatureMap",
    "quantum_kernel",
    "quantum_kernel_matrix",
]
