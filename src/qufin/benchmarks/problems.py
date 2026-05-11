"""Standardized benchmark problem definitions.

Provides pre-built portfolio, option, and credit risk problems at
small/medium/large scales with known classical reference solutions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
from numpy.typing import NDArray


class ProblemType(str, Enum):
    PORTFOLIO = "portfolio"
    OPTION = "option"
    CREDIT = "credit"


@dataclass
class Problem:
    """A benchmark problem specification."""

    problem_id: str
    problem_type: ProblemType
    description: str
    params: dict[str, Any]
    reference_value: float | None = None
    reference_source: str = ""


@dataclass
class PortfolioProblem(Problem):
    """Portfolio optimization benchmark problem."""

    mu: NDArray[np.float64] = field(default_factory=lambda: np.zeros(0))
    cov: NDArray[np.float64] = field(default_factory=lambda: np.zeros((0, 0)))
    cardinality: int | None = None
    sector_map: dict[int, int] | None = None
    sector_caps: dict[int, int] | None = None
    tickers: list[str] = field(default_factory=list)


def _generate_realistic_data(
    n_assets: int, seed: int = 42
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Generate realistic-looking return statistics.

    Uses random correlation matrix with clustered sector structure.
    """
    rng = np.random.default_rng(seed)

    # Expected returns: slight positive bias with dispersion
    mu = rng.normal(0.0003, 0.0005, n_assets)  # daily returns

    # Generate random correlation via factor model
    n_factors = min(5, n_assets // 3 + 1)
    F = rng.normal(0, 1, (n_assets, n_factors)) * 0.3
    specific = np.diag(rng.uniform(0.01, 0.04, n_assets))
    cov = F @ F.T + specific

    # Ensure symmetric positive definite
    cov = (cov + cov.T) / 2
    eigvals = np.linalg.eigvalsh(cov)
    if eigvals.min() < 1e-8:
        cov += np.eye(n_assets) * (1e-8 - eigvals.min())

    return mu, cov


def portfolio_small(seed: int = 42) -> PortfolioProblem:
    """15-asset benchmark: S&P sector ETFs + indices.

    Cardinality K=5, 3 sectors with caps.
    """
    tickers = [
        "XLB", "XLC", "XLE", "XLF", "XLI",
        "XLK", "XLP", "XLRE", "XLU", "XLV",
        "XLY", "SPY", "QQQ", "DIA", "IWM",
    ]
    mu, cov = _generate_realistic_data(15, seed)

    # 3 sectors: cyclical (0-4), growth (5-8), defensive (9-14)
    sector_map = {}
    for i in range(5):
        sector_map[i] = 0
    for i in range(5, 9):
        sector_map[i] = 1
    for i in range(9, 15):
        sector_map[i] = 2
    sector_caps = {0: 2, 1: 2, 2: 3}

    return PortfolioProblem(
        problem_id="portfolio_small_15",
        problem_type=ProblemType.PORTFOLIO,
        description="15 assets, K=5, 3 sectors with caps [2,2,3]",
        params={"n_assets": 15, "cardinality": 5, "gamma": 1.0},
        mu=mu,
        cov=cov,
        cardinality=5,
        sector_map=sector_map,
        sector_caps=sector_caps,
        tickers=tickers,
    )


def portfolio_medium(seed: int = 42) -> PortfolioProblem:
    """25-asset benchmark: NIFTY50 subset. K=8."""
    mu, cov = _generate_realistic_data(25, seed)

    return PortfolioProblem(
        problem_id="portfolio_medium_25",
        problem_type=ProblemType.PORTFOLIO,
        description="25 assets, K=8, no sector constraints",
        params={"n_assets": 25, "cardinality": 8, "gamma": 1.0},
        mu=mu,
        cov=cov,
        cardinality=8,
    )


def portfolio_large(seed: int = 42) -> PortfolioProblem:
    """50-asset benchmark: SP500 subset. K=15."""
    mu, cov = _generate_realistic_data(50, seed)

    return PortfolioProblem(
        problem_id="portfolio_large_50",
        problem_type=ProblemType.PORTFOLIO,
        description="50 assets, K=15, no sector constraints",
        params={"n_assets": 50, "cardinality": 15, "gamma": 1.0},
        mu=mu,
        cov=cov,
        cardinality=15,
    )


def option_european_atm(seed: int = 42) -> Problem:
    """ATM European call for QAE benchmarking."""
    return Problem(
        problem_id="option_european_atm",
        problem_type=ProblemType.OPTION,
        description="ATM European call: S=100, K=100, sigma=0.2, r=0.05, T=1",
        params={"s0": 100, "k": 100, "sigma": 0.2, "r": 0.05, "T": 1.0},
        reference_value=10.4506,
        reference_source="Black-Scholes closed-form",
    )


def all_problems() -> list[Problem]:
    """Return all registered benchmark problems."""
    return [
        portfolio_small(),
        portfolio_medium(),
        portfolio_large(),
        option_european_atm(),
    ]
