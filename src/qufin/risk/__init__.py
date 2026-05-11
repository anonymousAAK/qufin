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
    "QuantumVaRConfig",
    "QuantumVaRResult",
    "VaRResult",
    "build_loss_distribution",
    "compute_cva",
    "compute_ead_sa_ccr",
    "cvar_from_samples",
    "historical_var",
    "monte_carlo_var",
    "parametric_var",
    "portfolio_cva",
    "portfolio_cvar_risk",
    "portfolio_var",
    "quantum_var",
]
