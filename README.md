<div align="center">

<br>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://img.shields.io/badge/qufin-Quantum%20Finance-white?style=for-the-badge&labelColor=000000&color=ffffff">
  <source media="(prefers-color-scheme: light)" srcset="https://img.shields.io/badge/qufin-Quantum%20Finance-black?style=for-the-badge&labelColor=ffffff&color=000000">
  <img alt="qufin" src="https://img.shields.io/badge/qufin-Quantum%20Finance-black?style=for-the-badge&labelColor=ffffff&color=000000" height="40">
</picture>

<br><br>

**The open-source framework for quantum-enhanced quantitative finance.**<br>
Research-grade algorithms. Production-grade engineering. Honest benchmarks.

<br>

[![CI](https://github.com/anonymousAAK/qufin/actions/workflows/ci.yml/badge.svg)](https://github.com/anonymousAAK/qufin/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/qufin?color=blue)](https://pypi.org/project/qufin/)
[![Python](https://img.shields.io/pypi/pyversions/qufin)](https://pypi.org/project/qufin/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-2499%20passing-brightgreen)]()
[![Coverage](https://img.shields.io/badge/coverage-91%25-brightgreen)]()

<br>

<sub>
159 modules &middot; 14 subpackages &middot; 2,499 tests &middot; 11 backends &middot; 5 error mitigation strategies &middot; 4 QAE variants
</sub>

<br>

[Installation](#installation) &middot;
[Quickstart](#quickstart) &middot;
[Capabilities](#capabilities) &middot;
[Backends](#backends) &middot;
[Benchmarks](#benchmarks) &middot;
[Docs](https://anonymousAAK.github.io/qufin/)

</div>

<br>

---

<br>

## Why qufin

Most quantum finance libraries are toy demos or locked to one framework. qufin is different.

Every quantum algorithm ships alongside the best classical solver for the same problem. Results are compared head-to-head on identical inputs, with identical metrics, on standardized benchmark sets.

- **Backend-agnostic** &mdash; Write once, run on 11 backends (Aer, IBM, PennyLane, Cirq, Braket, CUDA-Q, D-Wave, IonQ, Quantinuum, noisy sim, mock)
- **Mathematically correct** &mdash; Grover global phase, IQAE multi-branch theta, canonical QPE. Details matter for derivative pricing.
- **Production patterns** &mdash; Typed configs, reproducibility manifests, noise-aware simulation, 5 error mitigation strategies, finance-optimized transpilation.

<br>

## Installation

```bash
pip install qufin
```

Requires Python 3.10+

<details>
<summary><b>Optional backends and extras</b></summary>
<br>

| Extra | What it adds |
|:------|:-------------|
| `qufin[ibm]` | IBM Quantum Runtime |
| `qufin[pennylane]` | PennyLane Lightning |
| `qufin[cirq]` | Google Cirq |
| `qufin[braket]` | Amazon Braket |
| `qufin[cudaq]` | NVIDIA CUDA-Q |
| `qufin[dwave]` | D-Wave Ocean |
| `qufin[ionq]` | IonQ via Braket |
| `qufin[quantinuum]` | Quantinuum H-Series |
| `qufin[ml]` | PyTorch |
| `qufin[viz]` | Plotly |
| `qufin[api]` | FastAPI + Celery + Redis |
| `qufin[dev]` | pytest, ruff, mypy |
| `qufin[all]` | Everything above |

</details>

<br>

## Quickstart

### Option pricing: Black-Scholes price and Greeks

```python
from qufin.options.classical.black_scholes import call_price, delta, vega

price = call_price(s=100, k=105, sigma=0.2, r=0.05, T=1.0)
print(f"Call price: {price:.4f}")
print(f"Delta:      {delta(s=100, k=105, sigma=0.2, r=0.05, T=1.0):.4f}")
print(f"Vega:       {vega(s=100, k=105, sigma=0.2, r=0.05, T=1.0):.4f}")

# Quantum amplitude-estimation pricers live in
# qufin.options.amplitude_estimation: IterativeAmplitudeEstimation (IQAE),
# MaximumLikelihoodAmplitudeEstimation, QMC, QSP, Asian / American QAE.
```

### Portfolio optimization with QAOA

```python
import numpy as np
from qufin.portfolio.qubo import PortfolioQUBO
from qufin.portfolio.optimizers.qaoa import QAOAConfig, QAOAPortfolio
from qufin.backends.qiskit_backend import QiskitAerBackend

# Expected returns and covariance for 6 assets
rng = np.random.default_rng(0)
mu = rng.normal(0.001, 0.0005, 6)
factor = rng.normal(0, 1, (6, 6)) * 0.1
cov = factor @ factor.T + np.eye(6) * 0.02

# Cardinality-constrained Markowitz QUBO: pick exactly K=3 assets
qubo = PortfolioQUBO(mu=mu, cov=cov, gamma=1.0, cardinality=3)

optimizer = QAOAPortfolio(
    qubo,
    QAOAConfig(p=2, mixer="xy_ring", cardinality=3, shots=4096, seed=42),
    QiskitAerBackend(seed=42),
)
result = optimizer.run()
selected = [i for i, bit in enumerate(result.best_bitstring) if bit == "1"]
print(f"Selected assets: {selected}")
print(f"Feasible (==K):  {result.feasible}")
```

<details>
<summary><b>More examples</b></summary>
<br>

**Synthetic market data**

```python
from qufin.data.synthetic import gbm_paths, heston_paths

# Geometric Brownian motion -> array of shape (n_paths, n_steps + 1)
paths = gbm_paths(s0=100, mu=0.08, sigma=0.2, T=1.0,
                  n_steps=252, n_paths=10_000)

# Heston stochastic volatility -> (prices, variances)
prices, variances = heston_paths(
    s0=100, v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7,
    mu=0.08, T=1.0, n_steps=252, n_paths=10_000,
)
```

**Backtesting**

```python
import numpy as np
from qufin.backtesting.engine import BacktestEngine
from qufin.backtesting.metrics import performance_summary

returns = np.random.default_rng(0).normal(0.0004, 0.01, (600, 5))
engine = BacktestEngine(returns, train_window=252, test_window=21)

def equal_weight(mu, cov):          # strategy: (mu, cov) -> weights
    return np.ones(len(mu)) / len(mu)

result = engine.run(equal_weight, strategy_name="equal_weight")
summary = performance_summary(result.portfolio_returns)
print(f"Sharpe: {summary.sharpe_ratio:.2f}")
```

**Automatic backend selection**

```python
from qiskit.circuit import QuantumCircuit
from qufin.backends.auto_select import auto_select_backend

circuit = QuantumCircuit(3); circuit.h(0); circuit.cx(0, 1); circuit.measure_all()
backend = auto_select_backend(circuit)  # GPU -> Aer -> Mock
```

</details>

<br>

## Capabilities

### Portfolio Optimization

- **Classical**: Mean-Variance, Black-Litterman, Risk Parity, HRP, Multi-Period, ADMM, Factor Models
- **Quantum**: QAOA (4 mixers), VQE, Warm-Start, Szegedy Walk, Robust CVaR QUBO, Grover Search, Quantum IPM, Simulated Quantum Annealing
- **Annealing**: D-Wave QUBO solver (Pegasus/Zephyr/Chimera)
- **Constraints**: Cardinality, sector, turnover, transaction cost, budget

### Option Pricing

- **Classical**: Black-Scholes (full Greeks), Monte Carlo, CRR Binomial, LSM American, Implied Vol (SABR/SVI)
- **Quantum**: Canonical QAE, IQAE, MLAE, FQAE, Path-Dependent QAE, American QAE, QMC (Montanaro), QSP, Asian QAE
- **Exotics**: Bermudan, lookback, cliquet, autocallable, basket

### Risk Management

- **Classical**: VaR (historical/parametric/MC), CVaR, stress testing, CVA/DVA, tail risk (EVaR, spectral)
- **Quantum**: Quantum VaR, Egger credit-risk, Quantum Stress Testing, HHL Linear Systems, Quantum Entropy
- **Credit**: Gaussian copula, NIG copula

### Hedging

- **Classical**: Delta hedging, deep hedging (PyTorch)
- **Quantum**: Quantum deep hedging, RL-quantum hedging, PPO with VQC policy

### Machine Learning

- **Classical**: Standard classifiers, PCA anomaly detection
- **Quantum**: Kernel methods, VQC, qGAN, HQGAN, reservoir computing, Boltzmann machine, credit scoring, transfer learning, quantum autoencoder

### Error Mitigation

- **Level 1**: Readout calibration, TREX
- **Level 2**: ZNE (Richardson), Dynamical Decoupling (XY4/CPMG/Uhrig)
- **Level 3**: PEC, CDR, M3 (matrix-free)
- **Adaptive**: Noise-aware variational optimization

### Data & Infrastructure

- **Market Data**: Yahoo Finance, FRED, Bloomberg, Refinitiv, CoinGecko crypto
- **Streaming**: Alpaca, Polygon, IEX WebSocket
- **Synthetic**: GBM, Heston, Merton jump-diffusion
- **Warehouse**: Parquet storage, PyArrow, auto-compaction
- **Backtesting**: Walk-forward engine, permutation test, CSCV overfitting detection

### Enterprise

- **REST API**: FastAPI (optimize, price, risk)
- **Job Queue**: Celery + Redis with priority routing
- **Caching**: SQLite/Redis with TTL
- **Deployment**: Docker, Kubernetes Helm chart
- **Compliance**: Audit trail, SR 11-7/SS1/23, Shapley explainability

<br>

## Backends

All quantum algorithms accept any `Backend` implementation. Swap without changing algorithm code.

```python
from qiskit.circuit import QuantumCircuit
from qufin.backends.auto_select import auto_select_backend

circuit = QuantumCircuit(3); circuit.h(0); circuit.cx(0, 1); circuit.measure_all()
backend = auto_select_backend(circuit)
```

| Backend | Target |
|:--------|:-------|
| `MockBackend` | Deterministic testing |
| `QiskitAerBackend` | Statevector + QASM sim |
| `NoisyAerBackend` | Device noise profiles |
| `IBMRuntimeBackend` | IBM QPU (156 qubits) |
| `PennyLaneBackend` | PennyLane Lightning |
| `CirqBackend` | Google Sycamore/Willow |
| `BraketBackend` | AWS (IonQ, Rigetti, IQM) |
| `CudaQBackend` | NVIDIA GPU simulation |
| `DWaveBackend` | Quantum annealing |
| `IonQBackend` | IonQ Aria/Forte |
| `QuantinuumBackend` | Quantinuum H-Series |

<br>

## Benchmarks

Standardized suites for honest quantum-vs-classical comparison.

```python
from qufin.benchmarks.problems import portfolio_small, portfolio_medium, portfolio_large

# Standardized benchmark problems (15 / 25 / 50 assets, with cardinality + sector caps)
problem = portfolio_small()
print(problem.problem_id, "| assets:", problem.mu.shape[0], "| K:", problem.cardinality)
```

- **Problem sets**: 15, 25, 50 asset portfolios
- **Metrics**: Approximation ratio, time-to-solution, circuit depth
- **Reproducibility**: Hardware, versions, seeds manifest
- **Transpiler**: QUBO-aware ZZ optimization, 30-50% CNOT reduction

<br>

## Architecture

<details>
<summary><b>Source tree (159 modules)</b></summary>

```
src/qufin/
  backends/           11 backends + error mitigation + transpiler
  options/
    classical/        Black-Scholes, binomial, Monte Carlo
    amplitude_estimation/  QAE, IQAE, MLAE, FQAE, Asian, QMC, QSP
  portfolio/
    classical/        MVO, Black-Litterman, HRP, Risk Parity
    optimizers/       QAOA, VQE, warm-start, annealing, Grover, IPM
  risk/               VaR, CVaR, stress, entropy, tail risk, HHL
    credit/           Egger, Gaussian copula, NIG copula
  hedging/            Delta, deep, quantum RL (PPO + VQC)
  ml/                 Kernels, VQC, qGAN, Boltzmann, autoencoder
  derivatives/        Bermudan, lookback, cliquet, autocallable
  data/               Yahoo, FRED, Bloomberg, Refinitiv, crypto
  backtesting/        Walk-forward, permutation test, CSCV
  benchmarks/         Problem sets, runner, resource estimation
  viz/                Plotly widgets, Dash dashboard
  api/                FastAPI + Celery + Redis cache
  compliance/         Audit trail, validation, explainability
  utils/              Circuit cache, parallel exec, sparse Pauli
  cli.py              CLI: optimize, price, risk, benchmark
  plugins.py          Entry-point plugin discovery
```

</details>

<br>

## Testing

```bash
pytest                             # Full suite (2499 tests)
pytest tests/unit/                 # Unit tests (fast)
pytest -m "not slow"               # Skip slow tests
pytest -m "not hardware"           # Skip hardware tests
```

<br>

## Contributing

Contributions welcome. See [CONTRIBUTING.md](CONTRIBUTING.md).

<br>

## License

Apache 2.0. See [LICENSE](LICENSE).

<br>

## Citation

```bibtex
@software{qufin,
  author  = {Adarsh Keshri},
  title   = {qufin: Quantum Algorithms for Quant Finance},
  year    = {2026},
  url     = {https://github.com/anonymousAAK/qufin},
  license = {Apache-2.0}
}
```

<br>

<div align="center">
<sub>Built for researchers who ship.</sub>
</div>
