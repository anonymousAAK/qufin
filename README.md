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
[![codecov](https://codecov.io/gh/anonymousAAK/qufin/branch/main/graph/badge.svg)](https://codecov.io/gh/anonymousAAK/qufin)
[![Status](https://img.shields.io/badge/status-1.x%20active--development-blue)]()

<br>

<sub>
159 modules &middot; 14 subpackages &middot; 11 backends &middot; 8 error-mitigation strategies &middot; 6 QAE algorithms
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
- **Production patterns** &mdash; Typed configs, reproducibility manifests, noise-aware simulation, 8 error-mitigation strategies, finance-optimized transpilation.

> **Project status: 1.x, actively developed.** The classical quant core
> (Black-Scholes/Greeks, Monte Carlo, VaR/CVaR, mean-variance, HRP, GARCH,
> backtesting) is well-tested and matches textbook values. Several quantum
> paths are research-stage and evolving, and may change between minor versions. The
> Quickstart below is exercised end-to-end by
> [`examples/quickstart.py`](examples/quickstart.py).

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

### Option pricing: classical vs. quantum

```python
import numpy as np
from qiskit.circuit import QuantumCircuit

from qufin.options.classical.black_scholes import call_price
from qufin.options.amplitude_estimation.estimation_problem import EstimationProblem
from qufin.options.amplitude_estimation.iqae import (
    IterativeAmplitudeEstimation, IQAEConfig,
)
from qufin.backends.qiskit_backend import QiskitAerBackend

# Classical: Black-Scholes closed form
classical = call_price(s=100, k=105, r=0.05, sigma=0.2, T=1.0)

# Quantum: Iterative Quantum Amplitude Estimation of a = sin^2(theta)
theta = np.pi / 5
oracle = QuantumCircuit(1)
oracle.ry(2 * theta, 0)  # A|0> = cos(theta)|0> + sin(theta)|1>
problem = EstimationProblem(state_preparation=oracle, objective_qubits=[0], n_qubits=1)

backend = QiskitAerBackend(method="automatic", seed=42)
result = IterativeAmplitudeEstimation(
    problem, IQAEConfig(epsilon_target=0.01, shots_per_round=2048), backend,
).estimate()

print(f"Black-Scholes call: {classical:.4f}")
print(f"IQAE estimate:      {result.estimate:.4f}  (true {np.sin(theta) ** 2:.4f})")
```

### Portfolio optimization with QAOA

```python
import numpy as np
from qufin.portfolio.qubo import PortfolioQUBO
from qufin.portfolio.optimizers.qaoa import QAOAPortfolio, QAOAConfig
from qufin.backends.qiskit_backend import QiskitAerBackend

rng = np.random.default_rng(42)
n_assets = 6
mu = rng.uniform(0.05, 0.15, n_assets)
factor = rng.standard_normal((n_assets, n_assets))
cov = (factor @ factor.T) / n_assets

qubo = PortfolioQUBO(mu=mu, cov=cov, gamma=0.5, cardinality=3, encoding="one_hot")
config = QAOAConfig(p=2, mixer="xy_ring", cardinality=3, shots=2048, seed=42)

result = QAOAPortfolio(qubo, config, QiskitAerBackend(seed=42)).run()
print(f"Selected (bitstring): {result.best_bitstring}")
print(f"Objective:            {result.best_objective:.6f}")
```

<details>
<summary><b>More examples</b></summary>
<br>

**Synthetic market data**

```python
from qufin.data.synthetic import gbm_paths, heston_paths

# GBM: shape (n_paths, n_steps + 1)
gbm = gbm_paths(s0=100, mu=0.08, sigma=0.2, T=1.0,
                n_steps=252, n_paths=10_000, seed=42)

# Heston returns (prices, variances), each (n_paths, n_steps + 1)
prices, variances = heston_paths(
    s0=100, v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7,
    mu=0.08, T=1.0, n_steps=252, n_paths=10_000, seed=42,
)
```

**Backtesting**

```python
import numpy as np
from qufin.backtesting.engine import BacktestEngine

returns = np.random.default_rng(0).normal(0.0004, 0.01, size=(800, 5))

def equal_weight(mu, cov):           # strategy: (mu, cov) -> weights
    return np.ones(len(mu)) / len(mu)

engine = BacktestEngine(returns, train_window=252, test_window=21)
result = engine.run(equal_weight, strategy_name="equal_weight")
print(f"Sharpe: {result.summary.sharpe_ratio:.2f}")
```

**Automatic backend selection**

```python
from qufin.backends.auto_select import auto_select_backend

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
- **Quantum (6 QAE algorithms)**: Canonical QAE, IQAE, MLAE, FQAE, MRQAE, QMC (Montanaro)
- **Option wrappers**: European, Asian, American, Path-Dependent, Multi-Asset, QSP pricing
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

### Error Mitigation (8 strategies)

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
from qufin.backends.auto_select import auto_select_backend
backend = auto_select_backend(circuit)
```

| Backend | Target |
|:--------|:-------|
| `MockBackend` | Deterministic testing |
| `QiskitAerBackend` | Statevector + QASM sim |
| `NoisyAerBackend` | Device noise profiles |
| `IBMRuntimeBackend` | IBM QPU (default `ibm_brisbane`, 127q; Heron r2 up to 156q) |
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
from qufin.benchmarks.runner import BenchmarkRunner, SolverEntry
from qufin.benchmarks.problems import portfolio_small

runner = BenchmarkRunner()
runner.register(SolverEntry(name="mean_variance", family="classical",
                            solve_fn=my_mean_variance_fn))
runner.register(SolverEntry(name="qaoa-p2", family="quantum",
                            solve_fn=my_qaoa_fn))

rows = runner.run_problem(portfolio_small())   # 15-asset benchmark
for row in rows:
    print(row.solver_name, row.family, row.objective, row.wall_seconds)
```

Each registered solver is a callable `problem -> {"objective": ..., ...}`; the
runner returns a list of `BenchmarkRow` records.

- **Problem sets**: 15, 25, 50 asset portfolios (`portfolio_small/medium/large`)
- **Metrics**: objective, relative error vs. reference, wall time, circuit depth
- **Reproducibility**: Hardware, versions, seeds manifest
- **Transpiler**: QUBO-aware ZZ optimization via Qiskit `optimization_level=3`
  (gate cancellation, commutation, template matching). CNOT reduction depends
  on circuit structure; dense QAOA cost layers see little reduction (see the
  measured benchmark in [`docs/`](https://anonymousAAK.github.io/qufin/)).

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
pytest                             # Full suite (2,507 tests collected)
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
  version = {0.1.dev},
  url     = {https://github.com/anonymousAAK/qufin},
  license = {Apache-2.0}
}
```

<br>

<div align="center">
<sub>Built for researchers who ship.</sub>
</div>
