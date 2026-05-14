<div align="center">

# qufin - Quantum Finance

### Research-Grade Quantum Algorithms for Production-Grade Quant Finance

[![CI](https://github.com/anonymousAAK/qufin/actions/workflows/ci.yml/badge.svg)](https://github.com/anonymousAAK/qufin/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/qufin)](https://pypi.org/project/qufin/)
[![Python](https://img.shields.io/pypi/pyversions/qufin)](https://pypi.org/project/qufin/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Tests](https://img.shields.io/badge/tests-1011%20passing-brightgreen)]()
[![Downloads](https://img.shields.io/pypi/dm/qufin)](https://pypi.org/project/qufin/)
[![Coverage](https://img.shields.io/badge/coverage-90%25-brightgreen)]()

---

**109 modules** &bull; **10 subpackages** &bull; **1011 tests** &bull; **7 backends** &bull; **4 QAE variants**

*Every quantum algorithm ships with a classical baseline. No hype -- just reproducible results.*

[Installation](#installation) &bull; [Quickstart](#quickstart) &bull; [Modules](#modules) &bull; [Architecture](#architecture) &bull; [Backends](#backends) &bull; [Benchmarks](#benchmarks) &bull; [Docs](https://anonymousAAK.github.io/qufin/) &bull; [Contributing](#contributing)

</div>

---

## Why qufin?

Most quantum finance libraries are either toy demos or tightly coupled to a single framework. qufin is different:

- **Honest benchmarking** -- every quantum result is compared head-to-head against the best classical solver on the same problem instance. No cherry-picked examples.
- **Backend-agnostic** -- swap between Qiskit Aer, IBM Runtime, PennyLane, Cirq, and Amazon Braket with one line. Your algorithms don't change.
- **Production patterns** -- pluggable backends, typed configs, reproducibility manifests, noise-aware simulation with real device profiles, and error mitigation built in.
- **Research-ready** -- implements algorithms from Brassard et al., Egger et al., and Farhi et al. with correct mathematical details (Grover global phase, IQAE multi-branch enumeration, canonical QPE).

---

## Installation

```bash
pip install qufin
```

<details>
<summary><strong>Optional extras</strong></summary>

```bash
pip install "qufin[ibm]"        # IBM Quantum Runtime (Sampler/Estimator)
pip install "qufin[ml]"         # PyTorch for deep hedging and ML
pip install "qufin[viz]"        # Plotly interactive visualization
pip install "qufin[pennylane]"  # PennyLane Lightning backend
pip install "qufin[cirq]"       # Google Cirq backend
pip install "qufin[braket]"     # Amazon Braket backend
pip install "qufin[dev]"        # pytest, ruff, mypy, pre-commit
pip install "qufin[all]"        # Everything
```

</details>

> Requires Python 3.10+

---

## Quickstart

### Price a European call with Black-Scholes

```python
from qufin.options.classical.black_scholes import bs_price, bs_greeks

price = bs_price(s=100, k=105, sigma=0.2, r=0.05, T=1.0)
greeks = bs_greeks(s=100, k=105, sigma=0.2, r=0.05, T=1.0)
print(f"Price: {price:.4f}  Delta: {greeks['delta']:.4f}")
# Price: 8.0214  Delta: 0.5374
```

### Price the same option with Quantum Amplitude Estimation

```python
from qufin.options.amplitude_estimation.european_qae import european_qae_price
from qufin.options.amplitude_estimation.iqae import IQAEConfig
from qufin.backends.qiskit_backend import QiskitAerBackend

backend = QiskitAerBackend(shots=4096)
config = IQAEConfig(epsilon_target=0.01, shots_per_round=1024)
qae_price = european_qae_price(
    s=100, k=105, sigma=0.2, r=0.05, T=1.0,
    backend=backend, config=config
)
print(f"QAE Price: {qae_price:.4f}")
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
print(f"Selected: {result.selected}  Objective: {result.objective:.4f}")
```

### Run a backtest

```python
from qufin.backtesting.engine import BacktestEngine
from qufin.backtesting.metrics import compute_metrics

engine = BacktestEngine(rebalance_freq="monthly", window=252)
portfolio_values = engine.run(prices_df, strategy_fn)
metrics = compute_metrics(portfolio_values)
print(f"Sharpe: {metrics['sharpe']:.2f}  Max DD: {metrics['max_drawdown']:.2%}")
```

### Generate synthetic market data

```python
from qufin.data.synthetic import gbm_paths, heston_paths, merton_jump_paths

# Geometric Brownian Motion
paths = gbm_paths(s0=100, mu=0.08, sigma=0.2, T=1.0, n_steps=252, n_paths=10_000)

# Heston stochastic volatility
from qufin.data.synthetic import HestonParams
params = HestonParams(v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7)
paths = heston_paths(s0=100, mu=0.08, params=params, T=1.0, n_steps=252, n_paths=10_000)
```

---

## Modules

<table>
<tr>
<td width="50%" valign="top">

### Portfolio Optimization
**Classical**: Mean-Variance (CVXPY), Black-Litterman, Risk Parity, HRP, Multi-Period, ADMM, Hybrid, Factor Models<br>
**Quantum**: QAOA (X / XY-ring / XY-full / Grover mixers), VQE, Warm-Start, Szegedy Quantum Walk, Robust CVaR QUBO, Sector Rotation (VQC)<br>
**Features**: CVaR objective, Dicke state init, QUBO with cardinality + sector + turnover + transaction cost constraints

### Option Pricing
**Classical**: Black-Scholes (full Greeks + implied vol), Monte Carlo (European / Asian / Barrier, antithetic variates), CRR Binomial (European + American), American (LSM), Implied Vol Surface (SABR/SVI)<br>
**Quantum**: Canonical QAE, IQAE, MLAE, FQAE, Path-Dependent QAE, American QAE (Quantum LSM)<br>
**Exotics**: Bermudan (LSM), lookback, cliquet, autocallable, basket, path-dependent

### Risk Management
**Classical**: VaR (historical / parametric / MC), CVaR, stress testing, counterparty CVA/DVA<br>
**Quantum**: Quantum VaR, Egger credit-risk, Quantum Stress Testing<br>
**Credit**: Gaussian copula, NIG copula models

</td>
<td width="50%" valign="top">

### Hedging
**Classical**: Delta hedging, deep hedging (PyTorch)<br>
**Quantum**: Quantum deep hedging, RL-quantum hedging

### Machine Learning
**Classical**: Standard classifiers<br>
**Quantum**: Kernel methods, VQC, qGAN, reservoir computing

### Data & Infrastructure
**Market Data**: Yahoo Finance, FRED macroeconomic<br>
**Synthetic**: GBM, Heston stochastic vol, Merton jump-diffusion<br>
**Backtesting**: Walk-forward engine, 15+ performance metrics<br>
**Benchmarks**: Standardized problem sets (15 / 25 / 50 assets), reproducibility manifests, leaderboard

</td>
</tr>
</table>

---

## Architecture

```
src/qufin/
    backends/                  # Pluggable backend abstraction
        base.py                #   Backend ABC + registry
        mock.py                #   MockBackend (deterministic, no simulator)
        qiskit_backend.py      #   QiskitAerBackend + NoisyAerBackend
        ibm_runtime.py         #   IBM Quantum Runtime (Sampler/Estimator)
        noise_models.py        #   Depolarizing, thermal, device-calibrated
        error_mitigation.py    #   ZNE, TREX, readout calibration
    options/
        classical/             #   Black-Scholes, CRR binomial, Monte Carlo
        amplitude_estimation/  #   Canonical, MLAE, IQAE, FQAE
            path_dependent_qae.py  #   Path-dependent QAE pricing
            american_qae.py    #   American option QAE (Quantum LSM)
        implied_vol_surface.py #   Implied vol surface (SABR/SVI)
    portfolio/
        classical/             #   Mean-Variance, Black-Litterman, HRP, Risk Parity
            factor_models.py   #   Factor-based portfolio models
        optimizers/            #   QAOA, VQE, warm-start, exhaustive, hybrid
            multi_period.py    #   Multi-period portfolio optimization
            robust.py          #   Robust CVaR QUBO optimizer
            quantum_walk.py    #   Szegedy quantum walk optimizer
            admm.py            #   ADMM decomposition optimizer
            hybrid.py          #   Hybrid classical-quantum optimizer
        sector_rotation.py     #   VQC-based sector rotation
        qubo.py                #   QUBO builder with constraints
        mixers.py              #   X, XY-ring, XY-full, Grover mixer circuits
    risk/                      #   VaR, CVaR, stress, counterparty
        quantum_stress.py      #   Quantum stress testing
        credit/                #   Egger quantum credit, Gaussian/NIG copula
    hedging/                   #   Delta, deep hedging, quantum RL
    ml/                        #   Quantum kernels, VQC, qGAN, reservoir
    derivatives/               #   Basket, autocallable, Bermudan LSM
    data/                      #   Yahoo Finance, FRED, synthetic generators
    backtesting/               #   Walk-forward engine + metrics
    benchmarks/                #   Problem sets, runner, leaderboard, manifests
    utils/                     #   Settings, logging, visualization, encoders
```

---

## Backends

All quantum algorithms accept any backend implementing the `Backend` protocol. Swap backends without changing algorithm code:

```python
from qufin.backends.qiskit_backend import QiskitAerBackend, NoisyAerBackend
from qufin.backends.mock import MockBackend

# Noiseless simulation
backend = QiskitAerBackend(shots=4096)

# Noisy simulation with a real device profile
backend = NoisyAerBackend(shots=4096, profile="melbourne")

# Deterministic mock for testing
backend = MockBackend()
```

| Backend | Use Case |
|---|---|
| `MockBackend` | Unit tests, CI, deterministic results |
| `QiskitAerBackend` | Local simulation, research |
| `NoisyAerBackend` | Noise-aware development (4 device profiles) |
| `IBMRuntimeBackend` | Real quantum hardware via IBM Quantum |
| `PennyLaneBackend` | PennyLane ecosystem |
| `CirqBackend` | Google Cirq ecosystem |
| `BraketBackend` | Amazon Braket / AWS hardware |

---

## Benchmarks

qufin includes standardized benchmark suites for honest quantum-vs-classical comparison:

- **Problem sets**: 15, 25, and 50 asset portfolios with real covariance matrices
- **Metrics**: Solution quality, time-to-solution, circuit depth, approximation ratio vs exact classical optimum
- **Reproducibility**: Every run generates a manifest recording hardware, software versions, and random seeds
- **Leaderboard**: Track and compare results across backends and algorithm configurations

```python
from qufin.benchmarks.runner import BenchmarkRunner
from qufin.benchmarks.problems import make_benchmark

runner = BenchmarkRunner()
results = runner.run(make_benchmark(15), algorithms=["qaoa", "vqe", "mean_variance"])
runner.summary(results)
```

---

## Testing

```bash
# Run all tests
pytest

# Run specific suites
pytest tests/unit/              # Unit tests
pytest tests/integration/       # Integration tests (requires Qiskit Aer)
pytest tests/property/          # Property-based tests (Hypothesis)
pytest tests/stress/            # Stress tests
pytest -m "not slow"            # Skip slow tests
pytest -m "not hardware"        # Skip IBM hardware tests
```

---

## Project Status

| Component | Status |
|---|---|
| Portfolio Optimization (classical + QAOA/VQE) | Stable |
| Option Pricing (BS + QAE) | Stable |
| Risk Management (VaR/CVaR + quantum) | Stable |
| Noise Models + Error Mitigation | Stable |
| Backtesting Engine | Stable |
| Benchmarks & Leaderboard | Stable |
| Data Layer (Yahoo, FRED, synthetic) | Stable |
| Hedging (delta + deep) | Beta |
| ML (kernels, VQC, qGAN) | Beta |
| Documentation Site | Planned |

---

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on code style, testing, and pull request process.

---

## License

Apache 2.0. See [LICENSE](LICENSE) for details.

---

## Citation

```bibtex
@software{qufin,
  author       = {Adarsh Keshri},
  title        = {qufin: Research-Grade Quantum Algorithms for Quant Finance},
  year         = {2025},
  url          = {https://github.com/anonymousAAK/qufin},
  license      = {Apache-2.0}
}
```
