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
from qufin.options.amplitude_estimation.mrqae import (
    ModifiedRealQAE,
    direct_encode_distribution,
    MRQAEConfig,
    MRQAEResult,
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
    "ModifiedRealQAE",
    "MultiAssetQAEResult",
    "MultiAssetSpec",
    "direct_encode_distribution",
    "build_basket_payoff_oracle",
    "build_multi_asset_distribution",
    "build_multi_asset_estimation_problem",
    "MRQAEConfig",
    "MRQAEResult",
    "price_european_mrqae",
    "price_multi_asset_mc",
    "price_multi_asset_qae",
]
