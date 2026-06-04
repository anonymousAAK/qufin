# Quickstart

## Option Pricing in 5 Lines

```python
from qufin.options.european import EuropeanOption

opt = EuropeanOption(s0=100, k=105, sigma=0.2, r=0.05, T=1.0)
print(f"Price: ${opt.bs_price():.2f}")
print(f"Delta: {opt.bs_delta():.4f}")
print(f"Vega:  {opt.bs_vega():.4f}")
```

## Portfolio Optimization

### Classical (Mean-Variance)

```python
import numpy as np
from qufin.portfolio.classical.mean_variance import mean_variance, Objective

# 5-asset example
mu = np.array([0.10, 0.08, 0.12, 0.06, 0.15])
cov = np.diag([0.04, 0.03, 0.05, 0.02, 0.06])

result = mean_variance(mu, cov, objective=Objective.MAX_SHARPE)
print(f"Weights: {result.weights}")
print(f"Sharpe:  {result.sharpe_ratio:.2f}")
```

### Quantum (QAOA)

```python
from qufin.portfolio.qubo import PortfolioQUBO
from qufin.portfolio.optimizers.qaoa import QAOAPortfolio, QAOAConfig
from qufin.backends.qiskit_backend import QiskitAerBackend

# Build QUBO problem
qubo = PortfolioQUBO(mu=mu, cov=cov, gamma=1.0, cardinality=3, budget_penalty=1e4)

# Configure QAOA with XY mixer (preserves cardinality)
config = QAOAConfig(p=2, mixer="xy_ring", cardinality=3, shots=4096)
backend = QiskitAerBackend(seed=42)

solver = QAOAPortfolio(qubo, config, backend)
result = solver.run()
print(f"Selected assets: {result.best_bitstring}")
print(f"Weights: {result.weights}")
```

## Risk Analysis

```python
from qufin.risk.classical_var import historical_var, portfolio_var

# Portfolio VaR
returns = np.random.default_rng(42).normal(0.0003, 0.015, (252, 5))
weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])

var_result = portfolio_var(returns, weights, confidence=0.95)
print(f"95% VaR: {var_result.var:.4f}")
print(f"Expected Shortfall: {var_result.expected_shortfall:.4f}")
```

## Backtesting

```python
from qufin.backtesting import BacktestEngine

# Walk-forward backtest
engine = BacktestEngine(returns, train_window=126, test_window=21)

def equal_weight(mu, cov):
    return np.ones(len(mu)) / len(mu)

result = engine.run(equal_weight, strategy_name="Equal Weight")
print(f"Sharpe Ratio: {result.summary.sharpe_ratio:.2f}")
print(f"Max Drawdown: {result.summary.max_drawdown:.2%}")
```

## Monte Carlo Pricing

```python
from qufin.options.classical.monte_carlo import european_mc

result = european_mc(s=100, k=105, r=0.05, sigma=0.2, T=1.0, n_paths=1_000_000)
print(f"MC Price: ${result.price:.4f}")
print(f"Std Error: {result.std_err:.6f}")
```

## Stress Testing

```python
from qufin.risk.stress import stress_test_suite

results = stress_test_suite(
    portfolio_value=10_000_000,
    weights=[0.6, 0.2, 0.1, 0.1],  # equity, rates, vol, spreads
)
for scenario, res in results.items():
    print(f"{scenario}: {res['pct_loss']:.2%} loss")
```
