"""Risk analysis: VaR, CVaR, credit risk (classical + quantum)."""

from __future__ import annotations

from qufin.risk.classical_var import (
    VaRResult,
    historical_var,
    monte_carlo_var,
    parametric_var,
    portfolio_var,
)
from qufin.risk.counterparty import (
    CounterpartyExposure,
    compute_cva,
    compute_ead_sa_ccr,
    portfolio_cva,
)
from qufin.risk.cvar import (
    AscendingCVaR,
    CVaRObjective,
    cvar_from_samples,
)
from qufin.risk.cvar import (
    portfolio_cvar as portfolio_cvar_risk,
)
from qufin.risk.cornish_fisher import (
    cornish_fisher_es,
    cornish_fisher_var,
    fractional_kelly,
    kelly_criterion,
)
from qufin.risk.garch import (
    GARCHResult,
    fit_garch,
    forecast_evaluate,
    rolling_garch_forecast,
)
from qufin.risk.quantum_risk_pipeline import (
    PortfolioRiskSpec,
    QuantumRiskResult,
    build_loss_loading_circuit,
    build_portfolio_loss_distribution,
    build_tail_oracle,
    quantum_stress_var,
    quantum_var_pipeline,
)
from qufin.risk.quantum_var import (
    QuantumVaRConfig,
    QuantumVaRResult,
    build_loss_distribution,
    quantum_var,
)

__all__ = [
    "AscendingCVaR",
    "CVaRObjective",
    "CounterpartyExposure",
    "GARCHResult",
    "PortfolioRiskSpec",
    "QuantumRiskResult",
    "QuantumVaRConfig",
    "QuantumVaRResult",
    "VaRResult",
    "build_loss_distribution",
    "build_loss_loading_circuit",
    "build_portfolio_loss_distribution",
    "build_tail_oracle",
    "compute_cva",
    "compute_ead_sa_ccr",
    "cornish_fisher_es",
    "cornish_fisher_var",
    "cvar_from_samples",
    "fit_garch",
    "forecast_evaluate",
    "fractional_kelly",
    "historical_var",
    "kelly_criterion",
    "monte_carlo_var",
    "parametric_var",
    "portfolio_cva",
    "portfolio_cvar_risk",
    "portfolio_var",
    "quantum_stress_var",
    "quantum_var",
    "quantum_var_pipeline",
    "rolling_garch_forecast",
]
