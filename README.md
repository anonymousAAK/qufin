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

### Option pricing: classical vs. quantum

```python
from qufin.options.classical.black_scholes import bs_price
from qufin.options.amplitude_estimation.european_qae import european_qae_price
from qufin.options.amplitude_estimation.iqae import IQAEConfig
from qufin.backends.qiskit_backend import QiskitAerBackend

# Classical: Black-Scholes
classical = bs_price(s=100, k=105, sigma=0.2, r=0.05, T=1.0)

# Quantum: Iterative Quantum Amplitude Estimation
backend = QiskitAerBackend(shots=4096)
quantum = european_qae_price(
    s=100, k=105, sigma=0.2, r=0.05, T=1.0,
    backend=backend, config=IQAEConfig(epsilon_target=0.01)
)

print(f"Black-Scholes: {classical:.4f}")
print(f"IQAE:          {quantum:.4f}")
```

### Portfolio optimization with QAOA

```python
from qufin.benchmarks.problems import make_benchmark
from qufin.portfolio.qubo import build_qubo
from qufin.portfolio.optimizers.qaoa import QAOAOptimizer
from qufin.backends.qiskit_backend import QiskitAerBackend

problem = make_benchmark(15)
Q = build_qubo(problem.mu, problem.sigma, risk_aversion=0.5, k=5)

optimizer = QAOAOptimizer(
    backend=QiskitAerBackend(shots=4096),
    p=2, mixer="xy_ring",
)
result = optimizer.solve(Q, n_assets=15, k=5)
print(f"Selected assets: {result.selected}")
print(f"Objective:       {result.objective:.6f}")
```

<details>
<summary><b>More examples</b></summary>
<br>

**Synthetic market data**

```python
from qufin.data.synthetic import gbm_paths, heston_paths, HestonParams

paths = gbm_paths(s0=100, mu=0.08, sigma=0.2, T=1.0,
                  n_steps=252, n_paths=10_000)

params = HestonParams(v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7)
paths = heston_paths(s0=100, mu=0.08, params=params, T=1.0,
                     n_steps=252, n_paths=10_000)
```

**Backtesting**

```python
from qufin.backtesting.engine import BacktestEngine
from qufin.backtesting.metrics import compute_metrics

engine = BacktestEngine(rebalance_freq="monthly", window=252)
portfolio_values = engine.run(prices_df, strategy_fn)
metrics = compute_metrics(portfolio_values)
print(f"Sharpe: {metrics['sharpe']:.2f}")
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
from qufin.backends.auto_select import auto_select_backend
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
from qufin.benchmarks.runner import BenchmarkRunner
from qufin.benchmarks.problems import make_benchmark

runner = BenchmarkRunner()
results = runner.run(make_benchmark(15),
                     algorithms=["qaoa", "vqe", "mean_variance"])
runner.summary(results)
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
