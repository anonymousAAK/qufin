"""Walk-forward backtesting engine for portfolio strategies.

Supports rolling-window optimization with configurable:
- Train/test window sizes
- Rebalance frequency
- Multiple strategy types (classical + quantum)
- Transaction costs

The engine is strategy-agnostic: pass any callable that takes
(mu, cov) and returns a weight vector.

References
----------
de Prado, "Advances in Financial Machine Learning" (2018), Ch. 12 —
  Backtesting through cross-validation.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from qufin.backtesting.metrics import PerformanceSummary, performance_summary


@dataclass
class BacktestResult:
    """Result from a walk-forward backtest.

    Attributes
    ----------
    portfolio_returns : NDArray
        Time series of portfolio returns.
    weights_history : NDArray
        Weight matrix (n_rebalances x n_assets).
    rebalance_dates : list
        Dates when rebalancing occurred.
    strategy_name : str
        Name of the strategy.
    summary : PerformanceSummary
        Full performance metrics.
    metadata : dict
        Additional info (train_window, test_window, etc.).
    """

    portfolio_returns: NDArray[np.float64]
    weights_history: NDArray[np.float64]
    rebalance_dates: list
    strategy_name: str = ""
    summary: PerformanceSummary | None = None
    metadata: dict = field(default_factory=dict)


StrategyFn = Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]]


class BacktestEngine:
    """Walk-forward backtesting engine.

    Parameters
    ----------
    returns : NDArray
        Asset return matrix (T x N).
    dates : list | NDArray | None
        Date labels for each row. If None, uses integer indices.
    train_window : int
        Number of periods for the training (estimation) window.
    test_window : int
        Number of periods between rebalances (test/hold window).
    transaction_cost : float
        One-way transaction cost as a fraction (e.g., 0.001 = 10 bps).
    risk_free_rate : float
        Annual risk-free rate for performance metrics.
    """

    def __init__(
        self,
        returns: NDArray[np.float64],
        dates: Any = None,
        train_window: int = 252,
        test_window: int = 21,
        transaction_cost: float = 0.001,
        risk_free_rate: float = 0.0,
    ) -> None:
        self.returns = np.asarray(returns, dtype=np.float64)
        self.n_periods, self.n_assets = self.returns.shape
        self.dates = dates if dates is not None else list(range(self.n_periods))
        self.train_window = train_window
        self.test_window = test_window
        self.transaction_cost = transaction_cost
        self.risk_free_rate = risk_free_rate

    def run(
        self,
        strategy: StrategyFn,
        strategy_name: str = "unnamed",
    ) -> BacktestResult:
        """Run a walk-forward backtest with the given strategy.

        Parameters
        ----------
        strategy : callable
            Function(mu, cov) -> weights (NDArray of shape (n_assets,)).
            Called at each rebalance point with estimated parameters
            from the training window.
        strategy_name : str
            Label for the strategy.

        Returns
        -------
        BacktestResult
        """
        all_returns = []
        weights_history = []
        rebalance_dates = []

        t = self.train_window  # Start after first training window
        prev_weights = np.zeros(self.n_assets)

        while t < self.n_periods:
            # Training window: [t - train_window, t)
            train_rets = self.returns[t - self.train_window : t]
            mu = np.mean(train_rets, axis=0)
            cov = np.cov(train_rets, rowvar=False)
            if cov.ndim == 0:
                cov = np.array([[float(cov)]])

            # Compute new weights
            try:
                weights = strategy(mu, cov)
            except Exception:
                # If strategy fails, hold previous weights or equal-weight
                fallback = np.ones(self.n_assets) / self.n_assets
                weights = prev_weights if np.sum(prev_weights) > 0 else fallback

            weights = np.asarray(weights, dtype=np.float64).flatten()

            # Normalize weights to sum to 1
            w_sum = np.sum(np.abs(weights))
            if w_sum > 1e-12:
                weights = weights / w_sum

            # Transaction costs
            tc = self.transaction_cost * np.sum(np.abs(weights - prev_weights))

            rebalance_dates.append(self.dates[t] if t < len(self.dates) else t)
            weights_history.append(weights.copy())

            # Hold period: [t, t + test_window)
            end = min(t + self.test_window, self.n_periods)
            hold_returns = self.returns[t:end]

            for day_ret in hold_returns:
                port_ret = float(np.dot(weights, day_ret))
                # Deduct transaction cost on first day of hold period
                if tc > 0:
                    port_ret -= tc
                    tc = 0
                all_returns.append(port_ret)

            prev_weights = weights
            t += self.test_window

        portfolio_returns = np.array(all_returns, dtype=np.float64)
        weight_matrix = np.array(weights_history, dtype=np.float64)

        summary = performance_summary(
            portfolio_returns,
            weights_history=weight_matrix,
            risk_free_rate=self.risk_free_rate,
        )

        return BacktestResult(
            portfolio_returns=portfolio_returns,
            weights_history=weight_matrix,
            rebalance_dates=rebalance_dates,
            strategy_name=strategy_name,
            summary=summary,
            metadata={
                "train_window": self.train_window,
                "test_window": self.test_window,
                "transaction_cost": self.transaction_cost,
                "n_rebalances": len(rebalance_dates),
                "n_assets": self.n_assets,
            },
        )

    def compare(
        self,
        strategies: dict[str, StrategyFn],
    ) -> dict[str, BacktestResult]:
        """Run multiple strategies and return results for comparison.

        Parameters
        ----------
        strategies : dict
            Mapping of strategy_name -> strategy callable.

        Returns
        -------
        Dict of strategy_name -> BacktestResult.
        """
        results = {}
        for name, strategy in strategies.items():
            results[name] = self.run(strategy, strategy_name=name)
        return results

    def comparison_table(
        self, results: dict[str, BacktestResult]
    ) -> list[dict[str, Any]]:
        """Generate a comparison table from backtest results.

        Returns
        -------
        List of dicts suitable for pd.DataFrame or markdown rendering.
        """
        rows = []
        for name, res in results.items():
            assert res.summary is not None
            s = res.summary
            rows.append({
                "Strategy": name,
                "Ann. Return": f"{s.annualized_return:.2%}",
                "Ann. Vol": f"{s.annualized_volatility:.2%}",
                "Sharpe": f"{s.sharpe_ratio:.2f}",
                "Sortino": f"{s.sortino_ratio:.2f}",
                "Max DD": f"{s.max_drawdown:.2%}",
                "Calmar": f"{s.calmar_ratio:.2f}",
                "Turnover": f"{s.avg_turnover:.4f}",
                "VaR 95": f"{s.var_95:.4f}",
                "Hit Rate": f"{s.hit_rate:.2%}",
            })
        return rows
