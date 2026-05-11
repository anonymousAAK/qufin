"""Stress testing scenarios: 1987 Black Monday, 2008 GFC, 2020 COVID, 2022 rates."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StressScenario:
    """A single stress scenario with market-factor shocks.

    Parameters
    ----------
    name : str
        Human-readable label (e.g. "Black Monday 1987").
    date : str
        Reference date or period (e.g. "1987-10-19").
    equity_shock : float
        Equity market shock as a fraction (e.g. -0.226 for -22.6%).
    rates_shock : float
        Interest-rate shock in basis points (e.g. -50.0).
    vol_shock : float
        Implied-volatility shock as a fraction (e.g. 1.50 for +150%).
    spread_shock : float
        Credit-spread shock in basis points (e.g. 300.0).
    description : str
        Short narrative of the event.
    """

    name: str
    date: str
    equity_shock: float        # fraction, e.g. -0.226
    rates_shock: float         # basis points
    vol_shock: float           # fraction, e.g. 1.50
    spread_shock: float = 0.0  # basis points (not all events had spread moves)
    description: str = ""


# ---------------------------------------------------------------------------
# Pre-defined historical scenarios
# ---------------------------------------------------------------------------

BLACK_MONDAY_1987 = StressScenario(
    name="Black Monday 1987",
    date="1987-10-19",
    equity_shock=-0.226,
    rates_shock=-50.0,
    vol_shock=1.50,
    spread_shock=0.0,
    description=(
        "Single-day equity crash of 22.6% on 19 Oct 1987. "
        "Volatility spiked and rates fell as investors fled to safety."
    ),
)

GFC_2008 = StressScenario(
    name="GFC 2008",
    date="2008-09-15",
    equity_shock=-0.38,
    rates_shock=-200.0,
    vol_shock=2.00,
    spread_shock=300.0,
    description=(
        "Global Financial Crisis triggered by Lehman Brothers collapse. "
        "Severe equity drawdown, massive spread widening, and aggressive rate cuts."
    ),
)

COVID_2020 = StressScenario(
    name="COVID 2020",
    date="2020-03-16",
    equity_shock=-0.34,
    rates_shock=-150.0,
    vol_shock=4.00,
    spread_shock=200.0,
    description=(
        "COVID-19 pandemic sell-off. Record volatility spike, "
        "flight-to-quality rate rally, and credit spread blowout."
    ),
)

RATES_2022 = StressScenario(
    name="Rates 2022",
    date="2022-06-13",
    equity_shock=-0.19,
    rates_shock=300.0,
    vol_shock=0.50,
    spread_shock=100.0,
    description=(
        "Aggressive Fed tightening cycle. Rising rates dragged equities lower, "
        "moderate vol increase, and spread widening."
    ),
)

SCENARIO_LIBRARY: dict[str, StressScenario] = {
    s.name: s
    for s in [BLACK_MONDAY_1987, GFC_2008, COVID_2020, RATES_2022]
}


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def apply_stress(
    portfolio_value: float,
    weights: np.ndarray,
    scenario: StressScenario,
) -> dict[str, float | str]:
    """Compute stressed P&L for a portfolio under a single scenario.

    The *weights* array describes the portfolio's sensitivity to each risk
    factor in the following order:

        [equity, rates, volatility, spreads]

    Each weight represents the fraction of *portfolio_value* exposed to
    that factor.  The stressed P&L for each factor is:

        equity   -> weight * portfolio_value * equity_shock
        rates    -> weight * portfolio_value * (rates_shock / 10_000)
        vol      -> weight * portfolio_value * vol_shock
        spreads  -> weight * portfolio_value * (-spread_shock / 10_000)

    Spread widening is treated as a loss (negative P&L).

    Parameters
    ----------
    portfolio_value : float
        Total portfolio market value.
    weights : array-like, shape (4,)
        Sensitivity weights ``[equity, rates, vol, spreads]``.
    scenario : StressScenario
        The scenario to apply.

    Returns
    -------
    dict
        ``scenario`` name, individual factor P&Ls, ``total_pnl``, and
        ``pct_loss`` (total P&L as a fraction of portfolio value).
    """
    w = np.asarray(weights, dtype=float)
    if w.shape != (4,):
        raise ValueError(f"weights must have shape (4,), got {w.shape}")

    shocks = np.array([
        scenario.equity_shock,                    # fraction
        scenario.rates_shock / 10_000,            # bps -> fraction
        scenario.vol_shock,                       # fraction
        -scenario.spread_shock / 10_000,          # bps -> fraction (loss)
    ])

    factor_pnl = portfolio_value * w * shocks
    total_pnl = float(np.sum(factor_pnl))

    return {
        "scenario": scenario.name,
        "equity_pnl": float(factor_pnl[0]),
        "rates_pnl": float(factor_pnl[1]),
        "vol_pnl": float(factor_pnl[2]),
        "spread_pnl": float(factor_pnl[3]),
        "total_pnl": total_pnl,
        "pct_loss": total_pnl / portfolio_value if portfolio_value != 0 else 0.0,
    }


def stress_test_suite(
    portfolio_value: float,
    weights: np.ndarray,
    scenarios: Sequence[StressScenario] | None = None,
) -> dict[str, dict[str, float | str]]:
    """Run a suite of stress scenarios and return a summary.

    Parameters
    ----------
    portfolio_value : float
        Total portfolio market value.
    weights : array-like, shape (4,)
        Sensitivity weights ``[equity, rates, vol, spreads]``.
    scenarios : sequence of StressScenario, optional
        Scenarios to run.  Defaults to all entries in
        :data:`SCENARIO_LIBRARY`.

    Returns
    -------
    dict
        Mapping of scenario name to the result dict produced by
        :func:`apply_stress`.
    """
    if scenarios is None:
        scenarios = list(SCENARIO_LIBRARY.values())

    return {
        sc.name: apply_stress(portfolio_value, weights, sc)
        for sc in scenarios
    }
