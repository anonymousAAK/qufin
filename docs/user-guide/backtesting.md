# Backtesting

qufin includes a walk-forward backtesting engine for comparing portfolio strategies with realistic assumptions.

## Basic Usage

```python
import numpy as np
from qufin.backtesting import BacktestEngine

# Simulated returns: 2 years, 10 assets
rng = np.random.default_rng(42)
returns = rng.normal(0.0003, 0.01, (504, 10))

# Create engine with 6-month train window, monthly rebalance
engine = BacktestEngine(
    returns,
    train_window=126,     # 6 months lookback
    test_window=21,       # rebalance monthly
    transaction_cost=0.001,  # 10 bps per trade
)
```

## Defining Strategies

A strategy is any callable that takes `(mu, cov)` and returns a weight vector:

```python
def equal_weight(mu, cov):
    n = len(mu)
    return np.ones(n) / n

def min_variance(mu, cov):
    from qufin.portfolio.classical.mean_variance import mean_variance
    return mean_variance(mu, cov).weights

def risk_parity(mu, cov):
    from qufin.portfolio.classical.risk_parity import risk_parity as rp
    return rp(cov).weights
```

## Running a Backtest

```python
result = engine.run(equal_weight, strategy_name="Equal Weight")

print(f"Annualized Return: {result.summary.annualized_return:.2%}")
print(f"Annualized Vol:    {result.summary.annualized_volatility:.2%}")
print(f"Sharpe Ratio:      {result.summary.sharpe_ratio:.2f}")
print(f"Max Drawdown:      {result.summary.max_drawdown:.2%}")
print(f"Calmar Ratio:      {result.summary.calmar_ratio:.2f}")
print(f"VaR 95%:           {result.summary.var_95:.4f}")
print(f"Hit Rate:          {result.summary.hit_rate:.2%}")
```

## Comparing Strategies

```python
strategies = {
    "Equal Weight": equal_weight,
    "Min Variance": min_variance,
    "Risk Parity": risk_parity,
}

results = engine.compare(strategies)
table = engine.comparison_table(results)

# Print formatted comparison
for row in table:
    print(f"{row['Strategy']:15s} | Sharpe: {row['Sharpe']} | MaxDD: {row['Max DD']}")
```

## Performance Metrics

The `PerformanceSummary` dataclass includes:

| Metric | Description |
|--------|-------------|
| `total_return` | Cumulative return |
| `annualized_return` | CAGR |
| `annualized_volatility` | Annualized standard deviation |
| `sharpe_ratio` | Risk-adjusted return |
| `sortino_ratio` | Downside-risk-adjusted return |
| `max_drawdown` | Peak-to-trough decline |
| `calmar_ratio` | Return / max drawdown |
| `avg_turnover` | Average portfolio turnover |
| `hit_rate` | Fraction of positive-return days |
| `var_95` | 95% Value at Risk |
| `cvar_95` | 95% Conditional VaR |
| `skewness` | Return distribution skewness |
| `kurtosis` | Return distribution kurtosis |

## Standalone Metrics

All metrics are also available as standalone functions:

```python
from qufin.backtesting.metrics import (
    sharpe_ratio, max_drawdown, tracking_error, information_ratio,
)

sr = sharpe_ratio(returns, risk_free_rate=0.04)
te = tracking_error(portfolio_returns, benchmark_returns)
ir = information_ratio(portfolio_returns, benchmark_returns)
```
