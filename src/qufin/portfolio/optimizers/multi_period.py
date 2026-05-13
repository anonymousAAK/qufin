"""Multi-period portfolio optimization with turnover penalties.

Extends single-period QUBO/MVO to T-period rebalancing with turnover
and holding-cost penalties. Supports sequential QAOA (quantum) and
sequential classical MVO modes.

References
----------
Boyd et al., "Multi-Period Trading via Convex Optimization" (2017).
Brandhofer et al., arXiv:2207.10555 — portfolio QAOA benchmarking.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from qufin.portfolio.optimizers.qaoa import QAOAConfig, QAOAPortfolio
from qufin.portfolio.qubo import PortfolioQUBO


@dataclass
class MultiPeriodConfig:
    """Configuration for multi-period portfolio optimization.

    Parameters
    ----------
    n_periods : int
        Number of rebalancing periods.
    turnover_penalty : float
        Penalty coefficient for changing positions between periods.
    holding_cost : float
        Per-period holding cost applied to the portfolio.
    method : str
        "sequential" (solve period-by-period, QAOA if backend provided,
        else classical MVO) or "classical" (always classical MVO).
    risk_aversion : float
        Risk aversion parameter for the Markowitz objective.
    cardinality : int | None
        Maximum number of assets to hold (None = unconstrained).
    qaoa_depth : int
        Number of QAOA layers (p parameter) when using quantum solver.
    shots : int
        Number of measurement shots for QAOA.
    seed : int
        Random seed for reproducibility.
    """

    n_periods: int = 4
    turnover_penalty: float = 0.01
    holding_cost: float = 0.0
    method: str = "sequential"
    risk_aversion: float = 0.5
    cardinality: int | None = None
    qaoa_depth: int = 2
    shots: int = 4096
    seed: int = 42


@dataclass
class MultiPeriodResult:
    """Result from multi-period portfolio optimization.

    Attributes
    ----------
    allocations : list[NDArray[np.float64]]
        Portfolio weights for each period, length T.
    objectives : list[float]
        Objective value achieved at each period.
    turnovers : list[float]
        Turnover between consecutive periods, length T-1.
    total_turnover : float
        Sum of all inter-period turnovers.
    total_objective : float
        Sum of all per-period objectives.
    """

    allocations: list[NDArray[np.float64]] = field(default_factory=list)
    objectives: list[float] = field(default_factory=list)
    turnovers: list[float] = field(default_factory=list)
    total_turnover: float = 0.0
    total_objective: float = 0.0


def compute_turnover(w_old: NDArray[np.float64], w_new: NDArray[np.float64]) -> float:
    """Compute turnover between two weight vectors.

    Turnover is defined as half the L1 distance between the two
    weight vectors, i.e. ``sum(|w_new - w_old|) / 2``.

    Parameters
    ----------
    w_old : NDArray
        Previous portfolio weights.
    w_new : NDArray
        New portfolio weights.

    Returns
    -------
    float
        Turnover value in [0, 1].
    """
    return float(np.sum(np.abs(w_new - w_old)) / 2)


def _solve_single_period_classical(
    mu: NDArray[np.float64],
    cov: NDArray[np.float64],
    config: MultiPeriodConfig,
    previous_weights: NDArray[np.float64] | None = None,
) -> tuple[NDArray[np.float64], float]:
    """Solve a single period using classical MVO with optional turnover penalty.

    When previous_weights is provided, the turnover penalty is added as a
    regularisation term to the minimum-variance objective via CVXPY.
    """
    import cvxpy as cp

    n = len(mu)
    w = cp.Variable(n)
    ret = mu @ w
    risk = cp.quad_form(w, cov)

    # Markowitz objective: gamma * risk - return
    obj_expr = config.risk_aversion * risk - ret

    # Add holding cost
    if config.holding_cost > 0:
        obj_expr += config.holding_cost * cp.sum(w)

    # Add turnover penalty
    if previous_weights is not None and config.turnover_penalty > 0:
        obj_expr += config.turnover_penalty * cp.norm1(w - previous_weights)

    constraints = [cp.sum(w) == 1, w >= 0]

    if config.cardinality is not None:
        z = cp.Variable(n, boolean=True)
        constraints.append(w <= z)
        constraints.append(cp.sum(z) <= config.cardinality)

    prob = cp.Problem(cp.Minimize(obj_expr), constraints)

    if config.cardinality is not None:
        solved = False
        for solver_name in [cp.SCIP, cp.GLPK_MI]:
            try:
                prob.solve(solver=solver_name)
                if w.value is not None:
                    solved = True
                    break
            except (cp.error.SolverError, Exception):
                continue
        if not solved:
            # Fallback: solve without cardinality, keep top-K
            prob_cont = cp.Problem(
                cp.Minimize(obj_expr),
                [cp.sum(w) == 1, w >= 0],
            )
            prob_cont.solve(solver=cp.CLARABEL)
            if w.value is not None:
                w_val = np.array(w.value).flatten()
                top_k = np.argsort(w_val)[-config.cardinality:]
                mask = np.zeros(n)
                mask[top_k] = 1
                w_val = w_val * mask
                w_sum = w_val.sum()
                if w_sum > 0:
                    w_val = w_val / w_sum
                w.value = w_val.reshape(w.shape)
    else:
        prob.solve(solver=cp.CLARABEL)

    if w.value is None:
        weights = np.ones(n, dtype=np.float64) / n
    else:
        weights = np.array(w.value, dtype=np.float64).flatten()

    obj_val = float(config.risk_aversion * (weights @ cov @ weights) - mu @ weights)
    return weights, obj_val


def _solve_single_period_qaoa(
    mu: NDArray[np.float64],
    cov: NDArray[np.float64],
    config: MultiPeriodConfig,
    backend: object,
    previous_weights: NDArray[np.float64] | None = None,
) -> tuple[NDArray[np.float64], float]:
    """Solve a single period using QAOA with optional turnover terms in QUBO."""
    qubo = PortfolioQUBO(
        mu=mu,
        cov=cov,
        gamma=config.risk_aversion,
        cardinality=config.cardinality,
        turnover_penalty=config.turnover_penalty if previous_weights is not None else 0.0,
        previous_weights=previous_weights,
    )

    qaoa_cfg = QAOAConfig(
        p=config.qaoa_depth,
        mixer="xy_ring" if config.cardinality is not None else "x",
        cardinality=config.cardinality,
        shots=config.shots,
        seed=config.seed,
    )

    solver = QAOAPortfolio(qubo=qubo, config=qaoa_cfg, backend=backend)  # type: ignore[arg-type]
    result = solver.run()
    return result.weights, result.best_objective


def multi_period_optimize(
    mu_series: list[NDArray[np.float64]],
    cov_series: list[NDArray[np.float64]],
    config: MultiPeriodConfig | None = None,
    backend: object | None = None,
) -> MultiPeriodResult:
    """Run multi-period portfolio optimization.

    Parameters
    ----------
    mu_series : list[NDArray]
        List of T expected-return vectors, one per rebalancing period.
    cov_series : list[NDArray]
        List of T covariance matrices, one per rebalancing period.
    config : MultiPeriodConfig | None
        Optimisation configuration.  Defaults to ``MultiPeriodConfig()``.
    backend : object | None
        Quantum backend (e.g. ``QiskitAerBackend``).  When ``None`` and
        method is ``"sequential"``, classical MVO is used instead.

    Returns
    -------
    MultiPeriodResult
    """
    if config is None:
        config = MultiPeriodConfig()

    T = len(mu_series)
    if len(cov_series) != T:
        raise ValueError(
            f"mu_series length ({T}) != cov_series length ({len(cov_series)})"
        )
    if T == 0:
        return MultiPeriodResult()

    allocations: list[NDArray[np.float64]] = []
    objectives: list[float] = []
    turnovers: list[float] = []

    use_quantum = config.method == "sequential" and backend is not None

    for t in range(T):
        prev_w = allocations[t - 1] if t > 0 else None

        if use_quantum:
            weights, obj = _solve_single_period_qaoa(
                mu_series[t], cov_series[t], config, backend, prev_w
            )
        else:
            weights, obj = _solve_single_period_classical(
                mu_series[t], cov_series[t], config, prev_w
            )

        allocations.append(weights)
        objectives.append(obj)

        if t > 0:
            turnovers.append(compute_turnover(allocations[t - 1], weights))

    return MultiPeriodResult(
        allocations=allocations,
        objectives=objectives,
        turnovers=turnovers,
        total_turnover=float(np.sum(turnovers)) if turnovers else 0.0,
        total_objective=float(np.sum(objectives)),
    )


def multi_period_backtest(
    prices: NDArray[np.float64],
    allocations: list[NDArray[np.float64]],
    rebalance_dates: list[int],
) -> dict:
    """Backtest a multi-period allocation schedule against realised prices.

    Parameters
    ----------
    prices : NDArray
        Price matrix of shape (T_total, N_assets).  Each row is a
        trading day, each column an asset price.
    allocations : list[NDArray]
        Portfolio weight vectors, one per rebalancing event.
    rebalance_dates : list[int]
        Row indices into *prices* at which each rebalancing occurs.
        Must be sorted in ascending order and have the same length as
        *allocations*.

    Returns
    -------
    dict
        Keys: ``"portfolio_values"``, ``"returns"``, ``"sharpe"``,
        ``"max_drawdown"``, ``"total_turnover"``.
    """
    if len(allocations) != len(rebalance_dates):
        raise ValueError(
            f"allocations length ({len(allocations)}) != "
            f"rebalance_dates length ({len(rebalance_dates)})"
        )

    n_days = prices.shape[0]
    portfolio_values = np.ones(n_days, dtype=np.float64)
    current_weights = np.zeros(prices.shape[1], dtype=np.float64)
    total_turnover = 0.0

    # Build a map: rebalance_date -> allocation index
    rebal_map: dict[int, int] = {d: i for i, d in enumerate(rebalance_dates)}

    for day in range(1, n_days):
        # Rebalance if this day is a rebalancing date
        if day in rebal_map:
            new_weights = allocations[rebal_map[day]]
            total_turnover += compute_turnover(current_weights, new_weights)
            current_weights = new_weights.copy()
        elif day == 0:
            # Handle day 0 rebalance
            if 0 in rebal_map:
                current_weights = allocations[rebal_map[0]].copy()

        # Daily return: weighted sum of asset returns
        if np.any(current_weights > 0):
            asset_returns = (prices[day] - prices[day - 1]) / np.maximum(
                prices[day - 1], 1e-12
            )
            port_return = float(current_weights @ asset_returns)
            portfolio_values[day] = portfolio_values[day - 1] * (1 + port_return)
        else:
            portfolio_values[day] = portfolio_values[day - 1]

    # Handle day-0 rebalance
    if 0 in rebal_map:
        current_weights_init = allocations[rebal_map[0]]
        total_turnover += compute_turnover(
            np.zeros_like(current_weights_init), current_weights_init
        )

    # Compute returns series
    returns = np.diff(portfolio_values) / np.maximum(portfolio_values[:-1], 1e-12)

    # Sharpe ratio (annualised, assuming 252 trading days)
    mean_ret = float(np.mean(returns))
    std_ret = float(np.std(returns, ddof=1)) if len(returns) > 1 else 1e-12
    sharpe = (mean_ret * np.sqrt(252)) / max(std_ret * np.sqrt(252), 1e-12)

    # Maximum drawdown
    cummax = np.maximum.accumulate(portfolio_values)
    drawdowns = (cummax - portfolio_values) / np.maximum(cummax, 1e-12)
    max_drawdown = float(np.max(drawdowns))

    return {
        "portfolio_values": portfolio_values,
        "returns": returns,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "total_turnover": total_turnover,
    }
