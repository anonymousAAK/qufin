"""Classical delta hedging.

Simulates dynamic delta-hedging of a European call option under GBM,
rebalancing at discrete intervals using the Black-Scholes delta.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def bs_delta(
    spot: float,
    strike: float,
    r: float,
    sigma: float,
    T: float,
    is_call: bool = True,
) -> float:
    """Black-Scholes delta for a European option.

    Parameters
    ----------
    spot : float
        Current spot price.
    strike : float
        Strike price.
    r : float
        Risk-free rate (annualised).
    sigma : float
        Volatility (annualised).
    T : float
        Time to expiry in years.  Must be > 0.
    is_call : bool
        ``True`` for call, ``False`` for put.
    """
    if T <= 0:
        # At expiry: intrinsic delta
        if is_call:
            return 1.0 if spot > strike else 0.0
        return -1.0 if spot < strike else 0.0

    d1 = (np.log(spot / strike) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    if is_call:
        return float(norm.cdf(d1))
    return float(norm.cdf(d1) - 1.0)


def _bs_call_price(
    spot: float, strike: float, r: float, sigma: float, T: float,
) -> float:
    """Black-Scholes European call price."""
    d1 = (np.log(spot / strike) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return float(spot * norm.cdf(d1) - strike * np.exp(-r * T) * norm.cdf(d2))


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class HedgeResult:
    """Result of a discrete delta-hedging simulation.

    Attributes
    ----------
    pnl : float
        Final hedging P&L (hedge portfolio value minus option payoff).
    hedging_error : float
        Absolute hedging error at expiry.
    rebalance_dates : NDArray
        Array of rebalance times (fractions of *T*).
    deltas : NDArray
        Delta held at each rebalance date.
    spot_path : NDArray
        Simulated spot path at rebalance dates (length *n_rebalances + 1*).
    option_price : float
        Initial Black-Scholes call price used to set up the hedge.
    """

    pnl: float
    hedging_error: float
    rebalance_dates: NDArray[np.float64]
    deltas: NDArray[np.float64]
    spot_path: NDArray[np.float64]
    option_price: float


# ---------------------------------------------------------------------------
# Hedger
# ---------------------------------------------------------------------------

class DeltaHedger:
    """Discrete delta-hedging simulator for a European call under GBM.

    Parameters
    ----------
    is_call : bool
        Hedge a call (``True``) or put (``False``).
    """

    def __init__(self, is_call: bool = True) -> None:
        self.is_call = is_call

    # ------------------------------------------------------------------
    def hedge(
        self,
        spot: float,
        strike: float,
        r: float,
        sigma: float,
        T: float,
        n_rebalances: int = 100,
        seed: int | None = 42,
    ) -> HedgeResult:
        """Run a single-path discrete delta-hedging simulation.

        Parameters
        ----------
        spot : float
            Initial spot price.
        strike : float
            Strike price.
        r : float
            Risk-free rate.
        sigma : float
            Volatility.
        T : float
            Time to expiry (years).
        n_rebalances : int
            Number of rebalance steps.
        seed : int | None
            Random seed.

        Returns
        -------
        HedgeResult
        """
        rng = np.random.default_rng(seed)
        dt = T / n_rebalances

        # --- simulate GBM path ---
        z = rng.standard_normal(n_rebalances)
        log_increments = (r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z
        log_path = np.concatenate([[np.log(spot)], np.log(spot) + np.cumsum(log_increments)])
        spot_path = np.exp(log_path)

        # --- rebalance dates ---
        rebalance_dates = np.linspace(0.0, T, n_rebalances + 1)

        # --- initial hedge ---
        option_price = _bs_call_price(spot, strike, r, sigma, T)

        deltas = np.empty(n_rebalances + 1)
        cash = option_price  # premium received
        shares = 0.0

        for i in range(n_rebalances + 1):
            s_i = spot_path[i]
            tau = T - rebalance_dates[i]
            d_i = bs_delta(s_i, strike, r, sigma, tau, is_call=self.is_call)
            deltas[i] = d_i

            if i == 0:
                # buy initial delta shares
                shares = d_i
                cash -= d_i * s_i
            elif i < n_rebalances:
                # rebalance
                trade = d_i - shares
                cash -= trade * s_i
                shares = d_i
                # accrue risk-free interest on cash
                cash *= np.exp(r * dt)

        # --- expiry ---
        s_T = spot_path[-1]
        payoff = max(s_T - strike, 0.0) if self.is_call else max(strike - s_T, 0.0)

        portfolio_value = shares * s_T + cash
        pnl = portfolio_value - payoff
        hedging_error = abs(pnl)

        return HedgeResult(
            pnl=float(pnl),
            hedging_error=float(hedging_error),
            rebalance_dates=rebalance_dates,
            deltas=deltas,
            spot_path=spot_path,
            option_price=float(option_price),
        )
