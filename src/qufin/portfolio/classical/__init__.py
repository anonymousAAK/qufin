"""Classical portfolio optimization baselines."""

from __future__ import annotations

from qufin.portfolio.classical.black_litterman import black_litterman
from qufin.portfolio.classical.factor_models import (
    FactorExposureResult,
    FactorModelResult,
    build_factor_model,
    estimate_factor_exposures,
    factor_expected_returns,
    factor_model_cov,
    risk_decomposition,
)
from qufin.portfolio.classical.hrp import hrp
from qufin.portfolio.classical.mean_variance import Objective, mean_variance
from qufin.portfolio.classical.risk_parity import inverse_volatility_weights, risk_parity
from qufin.portfolio.classical.robust_cov import (
    CovEstimateResult,
    constant_correlation,
    ledoit_wolf,
    oracle_approx_shrinkage,
    select_estimator,
)

__all__ = [
    "CovEstimateResult",
    "FactorExposureResult",
    "FactorModelResult",
    "Objective",
    "black_litterman",
    "build_factor_model",
    "constant_correlation",
    "estimate_factor_exposures",
    "factor_expected_returns",
    "factor_model_cov",
    "hrp",
    "inverse_volatility_weights",
    "ledoit_wolf",
    "mean_variance",
    "oracle_approx_shrinkage",
    "risk_decomposition",
    "risk_parity",
    "select_estimator",
]
