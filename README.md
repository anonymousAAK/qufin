# qufin

![CI](https://github.com/anonymousAAK/qufin/actions/workflows/ci.yml/badge.svg)
![PyPI](https://img.shields.io/pypi/v/qufin)
![Python](https://img.shields.io/pypi/pyversions/qufin)
![License](https://img.shields.io/badge/license-Apache--2.0-blue)
![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)

**Research-grade quantum algorithms for production-grade quant finance.**

qufin bridges the gap between quantum computing research and quantitative finance practice.
Every quantum algorithm ships with a classical baseline on the same problem, and a standardized
benchmark harness for honest quantum-vs-classical comparison. No hype -- just reproducible results.

---

## Features

| Module | Classical | Quantum | Status |
|---|---|---|---|
| **Portfolio Optimization** | Mean-Variance (CVXPY), Black-Litterman, Risk Parity, HRP | QAOA (X / XY-ring / XY-full / Grover mixers, CVaR objective, Dicke init), VQE, Warm-Start | Active |
| **Option Pricing** | Black-Scholes (full Greeks + implied vol), Monte Carlo (European / Asian / Barrier, antithetic), CRR Binomial (European + American) | QAE: Canonical, MLAE, IQAE, FQAE | In Progress |
| **Risk Management** | VaR (historical / parametric / MC), CVaR, stress testing, counterparty CVA/DVA, credit (Gaussian / NIG copula) | Quantum VaR, Egger credit-risk | In Progress |
| **Hedging** | Delta hedging, deep hedging (PyTorch) | Quantum deep hedging, RL-quantum hedging | Planned |
| **Machine Learning** | Classical classifiers | Quantum kernels, VQC, qGAN, reservoir computing | Planned |
| **Derivatives** | Bermudan (LSM), basket, autocallable, path-dependent | -- | Planned |
| **Backtesting** | Walk-forward engine, 15+ performance metrics | -- | Active |
| **Data** | Yahoo Finance, FRED, synthetic generators (GBM, Heston, Merton), asset universes | -- | Active |
| **Benchmarks** | Standardized problem sets (15 / 25 / 50 assets), runner, metrics, leaderboard, reproducibility manifests | -- | Active |

---

## Quickstart

### Install

```bash
pip install qufin
```

### Price a European call with Black-Scholes

```python
from qufin.options.classical.black_scholes import bs_price, bs_greeks

price = bs_price(s=100, k=105, sigma=0.2, r=0.05, T=1.0)
greeks = bs_greeks(s=100, k=105, sigma=0.2, r=0.05, T=1.0)
print(f"Price: {price:.4f}  Delta: {greeks['delta']:.4f}  Gamma: {greeks['gamma']:.6f}")
```

### Optimize a portfolio with QAOA

```python
from qufin.benchmarks.problems import make_benchmark
from qufin.portfolio.qubo import build_qubo
from qufin.portfolio.optimizers.qaoa import QAOAOptimizer
from qufin.backends.qiskit_backend import QiskitAerBackend

problem = make_benchmark(15)  # 15-asset benchmark
Q = build_qubo(problem.mu, problem.sigma, risk_aversion=0.5, k=5)
backend = QiskitAerBackend(shots=4096)
optimizer = QAOAOptimizer(backend=backend, p=2, mixer="xy_ring")
result = optimizer.solve(Q, n_assets=15, k=5)
print(f"Selected assets: {result.selected}  Objective: {result.objective:.4f}")
```

### Run a backtest

```python
from qufin.backtesting.engine import BacktestEngine
from qufin.backtesting.metrics import compute_metrics

engine = BacktestEngine(rebalance_freq="monthly", window=252)
portfolio_values = engine.run(prices_df, strategy_fn)
metrics = compute_metrics(portfolio_values)
print(f"Sharpe: {metrics['sharpe']:.2f}  Max Drawdown: {metrics['max_drawdown']:.2%}")
```

---

## Architecture

```
src/qufin/
    backends/           # Pluggable backend abstraction (ABC)
        base.py         #   Backend protocol + registry
        mock.py         #   MockBackend (deterministic, no simulator)
        qiskit_backend.py  QiskitAerBackend + NoisyAerBackend (4 noise profiles)
        ibm_runtime.py  #   IBM Quantum Runtime (Sampler/Estimator)
        pennylane_backend.py
        cirq_backend.py
        braket_backend.py
        noise_models.py #   Depolarizing, thermal, device-calibrated
        error_mitigation.py
    backtesting/        # Walk-forward engine + metrics
    benchmarks/         # Standardized problems, runner, leaderboard
    data/               # Yahoo Finance, FRED, synthetic generators, caching
    derivatives/        # Exotic derivatives (basket, autocallable, Bermudan LSM)
    hedging/            # Delta, deep hedging, quantum deep hedging
    ml/                 # Quantum kernels, VQC, qGAN, reservoir
    options/            # European, Asian, barrier, Bermudan
        classical/      #   Black-Scholes, CRR binomial, Monte Carlo
        amplitude_estimation/  # Canonical, MLAE, IQAE, FQAE
    portfolio/          # Portfolio optimization
        classical/      #   Mean-Variance, Black-Litterman, HRP, Risk Parity
        optimizers/     #   QAOA, VQE, warm-start, exhaustive, hybrid
        qubo.py         #   QUBO builder (cardinality, sector, turnover, tx cost)
        encodings.py    #   One-hot, binary encoding
        mixers.py       #   X, XY-ring, XY-full, Grover mixer circuits
    risk/               # VaR, CVaR, stress, counterparty
        credit/         #   Egger quantum credit risk, Gaussian/NIG copula
    utils/              # Settings, logging, visualization, encoders
```

---

## Installation

```bash
# Core (includes Qiskit Aer, NumPy, SciPy, pandas, CVXPY)
pip install qufin

# With IBM Quantum Runtime
pip install "qufin[ibm]"

# With PyTorch for deep hedging and ML
pip install "qufin[ml]"

# With Plotly for interactive visualization
pip install "qufin[viz]"

# Development (pytest, ruff, mypy, pre-commit)
pip install "qufin[dev]"

# Everything
pip install "qufin[all]"
```

Requires Python 3.10 or later.

---

## Backends

qufin uses a pluggable backend architecture. All quantum algorithms accept any backend that
implements the `Backend` protocol defined in `qufin.backends.base`.

| Backend | Description | Use Case |
|---|---|---|
| `MockBackend` | Deterministic, no simulator dependency | Unit tests, CI |
| `QiskitAerBackend` | Qiskit Aer statevector and QASM simulator | Local development, research |
| `NoisyAerBackend` | Aer with configurable noise (4 built-in profiles) | Noise-aware algorithm development |
| `IBMRuntimeBackend` | IBM Quantum Platform via Sampler/Estimator | Cloud execution, real hardware |
| `PennyLaneBackend` | PennyLane Lightning | Cross-framework interop |
| `CirqBackend` | Google Cirq | Cirq ecosystem integration |
| `BraketBackend` | Amazon Braket | AWS quantum hardware access |

---

## Benchmarks

qufin includes standardized benchmark problem sets (15, 25, and 50 assets) with real covariance
matrices and expected return vectors. The benchmark runner measures solution quality, time-to-solution,
circuit depth, and approximation ratio against the exact classical optimum.

Results are tracked via reproducibility manifests that record hardware, software versions, and
random seeds. The framework is designed for honest assessment of quantum advantage -- every
quantum result is compared head-to-head against the best available classical solver on the same
problem instance.

---

## Documentation

Full documentation: [https://anonymousAAK.github.io/qufin](https://anonymousAAK.github.io/qufin) *(coming soon)*

---

## Contributing

Contributions are welcome. Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on
code style, testing, and pull request process.

---

## License

Apache 2.0. See [LICENSE](LICENSE) for details.

---

## Citation

```bibtex
@software{qufin,
  author       = {Adarsh},
  title        = {qufin: Research-Grade Quantum Algorithms for Quant Finance},
  year         = {2025},
  url          = {https://github.com/anonymousAAK/qufin},
  license      = {Apache-2.0}
}
```
