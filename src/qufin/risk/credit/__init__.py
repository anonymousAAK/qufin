"""Credit risk models: Gaussian copula, NIG copula, Egger reproduction."""

from __future__ import annotations

from qufin.risk.credit.egger import (
    EggerConfig,
    EggerResult,
    build_expected_loss_problem,
    egger_classical_reference,
    egger_expected_loss,
)
from qufin.risk.credit.gaussian_copula import (
    CreditPortfolio,
    CreditRiskResult,
    gaussian_copula_mc,
    vasicek_analytical,
)
from qufin.risk.credit.nig_copula import (
    NIGParams,
    nig_copula_mc,
    nig_pdf,
)

__all__ = [
    "CreditPortfolio",
    "CreditRiskResult",
    "EggerConfig",
    "EggerResult",
    "NIGParams",
    "build_expected_loss_problem",
    "egger_classical_reference",
    "egger_expected_loss",
    "gaussian_copula_mc",
    "nig_copula_mc",
    "nig_pdf",
    "vasicek_analytical",
]
