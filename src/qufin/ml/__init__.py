"""Quantum ML for finance: kernels, reservoir computing, classifiers, qGANs,
Boltzmann machines, and transfer learning."""

from __future__ import annotations

from qufin.ml.classifiers import VariationalQuantumClassifier, VQCConfig
from qufin.ml.kernels import (
    QuantumKernelClassifier,
    ZZFeatureMap,
    quantum_kernel,
    quantum_kernel_matrix,
)
from qufin.ml.qgan import QGANConfig, QGANResult, QuantumGAN
from qufin.ml.quantum_boltzmann import (
    ClassicalRBM,
    HMMRegimeDetector,
    MarketRegime,
    RegimeBacktestResult,
    RestrictedQuantumBoltzmannMachine,
    RQBMConfig,
    RQBMResult,
)
from qufin.ml.quantum_credit_scoring import (
    CreditDataset,
    FairnessMetrics,
    IQPFeatureMap,
    ProjectedKernelConfig,
    compute_fairness_metrics,
    projected_kernel_matrix,
    projected_quantum_state,
)
from qufin.ml.quantum_gan_finance import (
    HQGAN,
    HQGANConfig,
    HQGANResult,
    StylizedFactsResult,
    evaluate_stylized_facts,
    privacy_preserving_synthetic,
    train_on_synthetic_validate_on_real,
)
from qufin.ml.quantum_transfer import (
    ClassicalTransferLearner,
    ClassicalTransferResult,
    PCAFeatureExtractor,
    QuantumTransferLearner,
    TradingSignal,
    TransferConfig,
    TransferResult,
)
from qufin.ml.reservoir import QuantumReservoir, QuantumReservoirConfig

__all__ = [
    "HQGAN",
    "ClassicalRBM",
    "ClassicalTransferLearner",
    "ClassicalTransferResult",
    "CreditDataset",
    "FairnessMetrics",
    "HMMRegimeDetector",
    "HQGANConfig",
    "HQGANResult",
    "IQPFeatureMap",
    "MarketRegime",
    "PCAFeatureExtractor",
    "ProjectedKernelConfig",
    "QGANConfig",
    "QGANResult",
    "QuantumGAN",
    "QuantumKernelClassifier",
    "QuantumReservoir",
    "QuantumReservoirConfig",
    "QuantumTransferLearner",
    "RQBMConfig",
    "RQBMResult",
    "RegimeBacktestResult",
    "RestrictedQuantumBoltzmannMachine",
    "StylizedFactsResult",
    "TradingSignal",
    "TransferConfig",
    "TransferResult",
    "VQCConfig",
    "VariationalQuantumClassifier",
    "ZZFeatureMap",
    "compute_fairness_metrics",
    "evaluate_stylized_facts",
    "privacy_preserving_synthetic",
    "projected_kernel_matrix",
    "projected_quantum_state",
    "quantum_kernel",
    "quantum_kernel_matrix",
    "train_on_synthetic_validate_on_real",
]
