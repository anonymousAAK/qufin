"""Counterparty credit risk.

Implements exposure-at-default models and CVA (Credit Valuation
Adjustment) computations for OTC derivative counterparty risk.

Also provides stress scenario libraries for historical backtesting.

References
----------
Gregory, "Counterparty Credit Risk and Credit Value Adjustment",
2nd ed., Wiley (2012).

Basel III, "The standardised approach for measuring counterparty
credit risk exposures" (SA-CCR), BCBS 279 (2014).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class CounterpartyExposure:
    """Counterparty exposure profile.

    Parameters
    ----------
    name : str
        Counterparty name/ID.
    notional : float
        Notional amount.
    pd : float
        Default probability (annual).
    lgd : float
        Loss given default (1 - recovery).
    exposure_profile : NDArray[np.float64]
        Expected exposure over time, shape (n_periods,).
    """

    name: str
    notional: float
    pd: float
    lgd: float
    exposure_profile: NDArray[np.float64]

    def __post_init__(self) -> None:
        self.exposure_profile = np.asarray(self.exposure_profile, dtype=np.float64)

    @property
    def epe(self) -> float:
        """Expected Positive Exposure (time-averaged)."""
        return float(np.mean(self.exposure_profile))

    @property
    def peak_exposure(self) -> float:
        """Maximum exposure over the profile."""
        return float(np.max(self.exposure_profile))


def compute_cva(
    exposure: CounterpartyExposure,
    risk_free_rate: float = 0.03,
    n_periods: int | None = None,
) -> float:
    """Compute unilateral CVA (Credit Valuation Adjustment).

    CVA = LGD * sum_t [ DF(t) * EE(t) * PD(t-1, t) ]

    where DF is the discount factor, EE is expected exposure,
    and PD(t-1, t) is the marginal default probability.

    Parameters
    ----------
    exposure : CounterpartyExposure
        Counterparty exposure profile.
    risk_free_rate : float
        Risk-free discount rate.
    n_periods : int or None
        Number of periods. Uses length of exposure profile if None.
    """
    if n_periods is None:
        n_periods = len(exposure.exposure_profile)

    ee = exposure.exposure_profile[:n_periods]
    dt = 1.0 / n_periods  # assume 1-year total horizon

    cva = 0.0
    survival = 1.0
    for t in range(n_periods):
        df = np.exp(-risk_free_rate * (t + 1) * dt)
        marginal_pd = survival * (1 - np.exp(-exposure.pd * dt))
        cva += exposure.lgd * df * ee[t] * marginal_pd
        survival *= np.exp(-exposure.pd * dt)

    return float(cva)


def compute_ead_sa_ccr(
    notional: float,
    mtm: float,
    add_on_factor: float = 0.01,
    collateral: float = 0.0,
    alpha: float = 1.4,
) -> float:
    """Exposure at Default under SA-CCR (Basel III).

    EAD = alpha * (RC + PFE)

    where RC = max(V - C, 0) is replacement cost,
    and PFE = multiplier * add-on is potential future exposure.

    Parameters
    ----------
    notional : float
        Trade notional.
    mtm : float
        Current mark-to-market value (positive = in the money).
    add_on_factor : float
        Supervisory add-on factor (depends on asset class).
    collateral : float
        Collateral held.
    alpha : float
        Supervisory scaling factor (1.4 per Basel III).
    """
    rc = max(mtm - collateral, 0)
    add_on = add_on_factor * notional
    # Multiplier: reduces PFE when MTM is negative
    v_minus_c = mtm - collateral
    if v_minus_c < 0:
        multiplier = min(1.0, 0.05 + 0.95 * np.exp(v_minus_c / (2 * add_on)) if add_on > 0 else 1.0)
    else:
        multiplier = 1.0

    pfe = multiplier * add_on
    return alpha * (rc + pfe)


def portfolio_cva(
    exposures: list[CounterpartyExposure],
    risk_free_rate: float = 0.03,
) -> dict[str, float]:
    """Compute CVA for a portfolio of counterparties.

    Returns per-counterparty and total CVA.
    """
    results: dict[str, float] = {}
    total = 0.0
    for exp in exposures:
        cva = compute_cva(exp, risk_free_rate)
        results[exp.name] = cva
        total += cva

    results["total"] = total
    return results
