# Portfolio Optimization

qufin provides both classical and quantum portfolio optimization methods that share a common interface.

## Classical Methods

### Mean-Variance (Markowitz)

Solves the classic Markowitz problem using CVXPY with support for cardinality constraints.

```python
from qufin.portfolio.classical.mean_variance import mean_variance, Objective

result = mean_variance(
    mu, cov,
    objective=Objective.MIN_VARIANCE,  # or MAX_SHARPE, MAX_RETURN
    cardinality=10,     # max 10 assets
    long_only=True,     # no short selling
    max_weight=0.15,    # position limit
)
```

**Objectives:**

- `MIN_VARIANCE` — Minimum variance portfolio
- `MAX_SHARPE` — Maximum Sharpe ratio (Cornuejols-Tutuncu transformation)
- `MAX_RETURN` — Maximum return subject to variance constraint

### Black-Litterman

Combines market equilibrium with investor views.

```python
from qufin.portfolio.classical.black_litterman import black_litterman

result = black_litterman(
    cov=cov,
    market_caps=caps,
    P=np.array([[1, -1, 0, 0, 0]]),  # Asset 0 outperforms Asset 1
    Q=np.array([0.02]),               # by 2%
)
# result.posterior_mu, result.posterior_cov
```

### Hierarchical Risk Parity (HRP)

Machine-learning-based allocation using hierarchical clustering.

```python
from qufin.portfolio.classical.hrp import hrp

result = hrp(returns, linkage_method="single")
# result.weights, result.cluster_order
```

### Risk Parity

Equal risk contribution portfolio.

```python
from qufin.portfolio.classical.risk_parity import risk_parity

result = risk_parity(cov, budget=None)  # None = equal budget
# result.weights, result.risk_contributions
```

## Quantum Methods

### QUBO Formulation

All quantum portfolio optimizers work on a QUBO (Quadratic Unconstrained Binary Optimization) matrix.

```python
from qufin.portfolio.qubo import PortfolioQUBO

qubo = PortfolioQUBO(
    mu=mu, cov=cov,
    gamma=1.0,              # risk aversion
    cardinality=5,          # select 5 assets
    sector_map={0: [0,1,2], 1: [3,4]},
    sector_caps={0: 3, 1: 2},
    turnover_penalty=0.01,
    budget_penalty=1e4,
    encoding="one_hot",     # or "binary"
)

Q = qubo.build_matrix()  # (n_qubits x n_qubits) QUBO matrix
```

### QAOA Portfolio Optimizer

```python
from qufin.portfolio.optimizers.qaoa import QAOAPortfolio, QAOAConfig
from qufin.backends.qiskit_backend import QiskitAerBackend

config = QAOAConfig(
    p=3,                    # QAOA depth (layers)
    mixer="xy_ring",        # Hamming-weight-preserving mixer
    cardinality=5,          # must match QUBO cardinality
    cvar_alpha=0.5,         # CVaR tail optimization
    shots=8192,
    seed=42,
)

solver = QAOAPortfolio(qubo, config, QiskitAerBackend(seed=42))
result = solver.run()
```

**Mixer options:**

| Mixer | Preserves Hamming Weight | Circuit Depth | Best For |
|-------|-------------------------|---------------|----------|
| `x` | No | O(n) | Unconstrained |
| `xy_ring` | Yes | O(n) | Cardinality-constrained |
| `xy_full` | Yes | O(n^2) | Small problems, better mixing |
| `grover` | Yes | O(n) | Hard constraints |

### VQE Portfolio Optimizer

Hardware-efficient ansatz with CVaR objective.

```python
from qufin.portfolio.optimizers.vqe import VQEPortfolio, VQEConfig

config = VQEConfig(
    reps=3,
    entanglement="linear",
    cvar_alpha=0.5,
    shots=8192,
)

solver = VQEPortfolio(qubo, config, backend)
result = solver.run()
```

### Exhaustive Solver (Small Problems)

Brute-force optimal solution for validation (up to 20 qubits).

```python
from qufin.portfolio.optimizers.exhaustive import exhaustive_solve

result = exhaustive_solve(qubo, return_all=True)
# result.best_bitstring, result.best_objective, result.all_objectives
```

## Comparison Example

```python
from qufin.backtesting import BacktestEngine

engine = BacktestEngine(returns, train_window=252, test_window=63)

strategies = {
    "Min Variance": lambda mu, cov: mean_variance(mu, cov).weights,
    "Risk Parity": lambda mu, cov: risk_parity(cov).weights,
    "HRP": lambda mu, cov: hrp_from_cov(cov).weights,
    "Equal Weight": lambda mu, cov: np.ones(len(mu)) / len(mu),
}

results = engine.compare(strategies)
table = engine.comparison_table(results)
```
