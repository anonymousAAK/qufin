"""Quantum Amplitude Estimation family: canonical, MLAE, IQAE, FQAE."""

from __future__ import annotations

from qufin.options.amplitude_estimation.canonical import (
    CanonicalAmplitudeEstimation,
    CanonicalQAEConfig,
    CanonicalQAEResult,
)
from qufin.options.amplitude_estimation.estimation_problem import EstimationProblem
from qufin.options.amplitude_estimation.fqae import (
    FaithfulAmplitudeEstimation,
    FQAEConfig,
    FQAEResult,
)
from qufin.options.amplitude_estimation.iqae import (
    IQAEConfig,
    IQAEResult,
    IterativeAmplitudeEstimation,
)
from qufin.options.amplitude_estimation.mlae import (
    MaximumLikelihoodAmplitudeEstimation,
    MLAEConfig,
    MLAEResult,
)

__all__ = [
    "CanonicalAmplitudeEstimation",
    "CanonicalQAEConfig",
    "CanonicalQAEResult",
    "EstimationProblem",
    "FQAEConfig",
    "FQAEResult",
    "FaithfulAmplitudeEstimation",
    "IQAEConfig",
    "IQAEResult",
    "IterativeAmplitudeEstimation",
    "MLAEConfig",
    "MLAEResult",
    "MaximumLikelihoodAmplitudeEstimation",
]
