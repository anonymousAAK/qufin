"""Quantum Amplitude Estimation family: canonical, MLAE, IQAE, FQAE."""

from __future__ import annotations

from qufin.options.amplitude_estimation.american_qae import (
    AmericanQAEResult,
    AmericanQAESpec,
    BasisType,
    QuantumLSM,
    ResourceEstimate,
    american_binomial,
    estimate_resources,
    price_american_classical,
    price_american_qae,
)
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
from qufin.options.amplitude_estimation.mrqae import (
    ModifiedRealQAE,
    MRQAEConfig,
    MRQAEResult,
    direct_encode_distribution,
    price_european_mrqae,
)
from qufin.options.amplitude_estimation.multi_asset_qae import (
    MultiAssetQAEResult,
    MultiAssetSpec,
    build_basket_payoff_oracle,
    build_multi_asset_distribution,
    build_multi_asset_estimation_problem,
    price_multi_asset_mc,
    price_multi_asset_qae,
)
from qufin.options.amplitude_estimation.path_dependent_qae import (
    PathDependentAsianSpec,
    PathDependentQAEResult,
    build_asian_payoff_oracle,
    build_path_dependent_estimation_problem,
    build_path_state_preparation,
    compute_path_averages,
    price_asian_mc,
    price_asian_qae,
)

__all__ = [
    "AmericanQAEResult",
    "AmericanQAESpec",
    "BasisType",
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
    "MRQAEConfig",
    "MRQAEResult",
    "MaximumLikelihoodAmplitudeEstimation",
    "ModifiedRealQAE",
    "MultiAssetQAEResult",
    "MultiAssetSpec",
    "PathDependentAsianSpec",
    "PathDependentQAEResult",
    "QuantumLSM",
    "ResourceEstimate",
    "american_binomial",
    "build_asian_payoff_oracle",
    "build_basket_payoff_oracle",
    "build_multi_asset_distribution",
    "build_multi_asset_estimation_problem",
    "build_path_dependent_estimation_problem",
    "build_path_state_preparation",
    "compute_path_averages",
    "direct_encode_distribution",
    "estimate_resources",
    "price_american_classical",
    "price_american_qae",
    "price_asian_mc",
    "price_asian_qae",
    "price_european_mrqae",
    "price_multi_asset_mc",
    "price_multi_asset_qae",
]
