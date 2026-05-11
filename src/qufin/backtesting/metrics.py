"""Portfolio performance metrics for backtesting.

Provides standard quant metrics: Sharpe, Sortino, max drawdown,
Calmar, turnover, tracking error, information ratio, and hit rate.

All functions accept numpy arrays of returns (not prices) and
assume daily frequency unless otherwise specified.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class PerformanceSummary:
    """Complete performance summary for a backtest."""

    total_return: float
    annualized_return: float
    annualized_volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    avg_turnover: float
    hit_rate: float
    skewness: float
    kurtosis: float
    var_95: float
    cvar_95: float
    n_periods: int
    periods_per_year: int


def annualized_return(returns: NDArray[np.float64], periods_per_year: int = 252) -> float:
    """Compute annualized return from a series of periodic returns."""
    total = np.prod(1 + returns) - 1
    n_years = len(returns) / periods_per_year
    if n_years <= 0:
        return 0.0
    return float((1 + total) ** (1 / n_years) - 1)


def annualized_volatility(returns: NDArray[np.float64], periods_per_year: int = 252) -> float:
    """Annualized volatility (standard deviation of returns)."""
    return float(np.std(returns, ddof=1) * np.sqrt(periods_per_year))


def sharpe_ratio(
    returns: NDArray[np.float64],
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Annualized Sharpe ratio."""
    excess = returns - risk_free_rate / periods_per_year
    if np.std(excess) < 1e-12:
        return 0.0
    return float(np.mean(excess) / np.std(excess, ddof=1) * np.sqrt(periods_per_year))


def sortino_ratio(
    returns: NDArray[np.float64],
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Annualized Sortino ratio (penalizes only downside volatility)."""
    excess = returns - risk_free_rate / periods_per_year
    downside = excess[excess < 0]
    if len(downside) == 0 or np.std(downside) < 1e-12:
        return 0.0
    return float(np.mean(excess) / np.std(downside, ddof=1) * np.sqrt(periods_per_year))


def max_drawdown(returns: NDArray[np.float64]) -> float:
    """Maximum drawdown from peak equity."""
    cum = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cum)
    drawdowns = (cum - running_max) / running_max
    return float(np.min(drawdowns))


def calmar_ratio(returns: NDArray[np.float64], periods_per_year: int = 252) -> float:
    """Calmar ratio = annualized return / |max drawdown|."""
    mdd = max_drawdown(returns)
    ann_ret = annualized_return(returns, periods_per_year)
    if abs(mdd) < 1e-12:
        return 0.0
    return float(ann_ret / abs(mdd))


def turnover(weights_history: NDArray[np.float64]) -> float:
    """Average one-way turnover from a weight matrix (T x N)."""
    if len(weights_history) < 2:
        return 0.0
    diffs = np.abs(np.diff(weights_history, axis=0))
    return float(np.mean(np.sum(diffs, axis=1)) / 2)


def tracking_error(
    returns: NDArray[np.float64],
    benchmark_returns: NDArray[np.float64],
    periods_per_year: int = 252,
) -> float:
    """Annualized tracking error relative to a benchmark."""
    diff = returns - benchmark_returns
    return float(np.std(diff, ddof=1) * np.sqrt(periods_per_year))


def information_ratio(
    returns: NDArray[np.float64],
    benchmark_returns: NDArray[np.float64],
    periods_per_year: int = 252,
) -> float:
    """Information ratio = active return / tracking error."""
    te = tracking_error(returns, benchmark_returns, periods_per_year)
    if te < 1e-12:
        return 0.0
    active = np.mean(returns - benchmark_returns) * periods_per_year
    return float(active / te)


def hit_rate(returns: NDArray[np.float64]) -> float:
    """Fraction of periods with positive returns."""
    if len(returns) == 0:
        return 0.0
    return float(np.mean(returns > 0))


def var_historical(returns: NDArray[np.float64], confidence: float = 0.95) -> float:
    """Historical Value at Risk (loss, positive number)."""
    return float(-np.percentile(returns, 100 * (1 - confidence)))


def cvar_historical(returns: NDArray[np.float64], confidence: float = 0.95) -> float:
    """Historical Conditional VaR (Expected Shortfall)."""
    var = var_historical(returns, confidence)
    tail = returns[returns <= -var]
    if len(tail) == 0:
        return var
    return float(-np.mean(tail))


def performance_summary(
    returns: NDArray[np.float64],
    weights_history: NDArray[np.float64] | None = None,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> PerformanceSummary:
    """Compute a complete performance summary.

    Parameters
    ----------
    returns : NDArray
        Array of portfolio returns (T,).
    weights_history : NDArray | None
        Weight matrix (T x N). Used for turnover calculation.
    risk_free_rate : float
        Annual risk-free rate for Sharpe/Sortino.
    periods_per_year : int
        Trading periods per year (252 for daily, 52 for weekly, 12 for monthly).
    """
    from scipy.stats import kurtosis as kurt
    from scipy.stats import skew

    avg_to = turnover(weights_history) if weights_history is not None else 0.0

    return PerformanceSummary(
        total_return=float(np.prod(1 + returns) - 1),
        annualized_return=annualized_return(returns, periods_per_year),
        annualized_volatility=annualized_volatility(returns, periods_per_year),
        sharpe_ratio=sharpe_ratio(returns, risk_free_rate, periods_per_year),
        sortino_ratio=sortino_ratio(returns, risk_free_rate, periods_per_year),
        max_drawdown=max_drawdown(returns),
        calmar_ratio=calmar_ratio(returns, periods_per_year),
        avg_turnover=avg_to,
        hit_rate=hit_rate(returns),
        skewness=float(skew(returns)),
        kurtosis=float(kurt(returns)),
        var_95=var_historical(returns),
        cvar_95=cvar_historical(returns),
        n_periods=len(returns),
        periods_per_year=periods_per_year,
    )
