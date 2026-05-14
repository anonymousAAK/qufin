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
[![Tests](https://img.shields.io/badge/tests-1724%20passing-brightgreen)]()
[![Coverage](https://img.shields.io/badge/coverage-91%25-brightgreen)]()
[![Downloads](https://img.shields.io/pypi/dm/qufin)](https://pypi.org/project/qufin/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

<br>

<sub>
129 modules &nbsp;&middot;&nbsp; 12 subpackages &nbsp;&middot;&nbsp; 1,724 tests &nbsp;&middot;&nbsp; 8 backends &nbsp;&middot;&nbsp; 5 error mitigation strategies &nbsp;&middot;&nbsp; 4 QAE variants
</sub>

<br>

[Installation](#installation) &nbsp;&middot;&nbsp;
[Quickstart](#quickstart) &nbsp;&middot;&nbsp;
[Capabilities](#capabilities) &nbsp;&middot;&nbsp;
[Backends](#backends) &nbsp;&middot;&nbsp;
[Benchmarks](#benchmarks) &nbsp;&middot;&nbsp;
[Architecture](#architecture) &nbsp;&middot;&nbsp;
[Docs](https://anonymousAAK.github.io/qufin/)

</div>

<br>

---

<br>

## Why qufin

Most quantum finance libraries are toy demos or locked to a single framework. qufin takes a different approach.

Every quantum algorithm ships alongside the best classical solver for the same problem. Results are compared head-to-head on identical inputs, with identical metrics, on standardized benchmark sets. If the quantum method doesn't win, the data shows it.

<table>
<tr>
<td width="33%" valign="top">

**Backend-agnostic**

Write your algorithm once. Run it on Qiskit Aer, IBM hardware, PennyLane, Cirq, Amazon Braket, or NVIDIA CUDA-Q. One interface, eight backends.

</td>
<td width="33%" valign="top">

**Mathematically correct**

Grover operator with the correct global phase. IQAE with multi-branch theta enumeration. Canonical QPE. Details matter when you're pricing derivatives.

</td>
<td width="33%" valign="top">

**Production patterns**

Typed configs. Reproducibility manifests. Noise-aware simulation. Five error mitigation strategies. Finance-optimized transpilation. Not a notebook demo.

</td>
</tr>
</table>

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
| `qufin[ibm]` | IBM Quantum Runtime &mdash; Sampler / Estimator primitives |
| `qufin[pennylane]` | PennyLane Lightning &mdash; parameter-shift gradients |
| `qufin[cirq]` | Google Cirq &mdash; Sycamore / Willow target support |
| `qufin[braket]` | Amazon Braket &mdash; IonQ, Rigetti, IQM hardware |
| `qufin[cudaq]` | NVIDIA CUDA-Q &mdash; GPU-accelerated simulation |
| `qufin[ml]` | PyTorch &mdash; deep hedging and quantum ML |
| `qufin[viz]` | Plotly &mdash; interactive visualization |
| `qufin[api]` | FastAPI, Celery, Redis &mdash; REST API and job queue |
| `qufin[dev]` | pytest, ruff, mypy, pre-commit |
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
    p=2,
    mixer="xy_ring",
)
result = optimizer.solve(Q, n_assets=15, k=5)
print(f"Selected assets: {result.selected}")
print(f"Objective:       {result.objective:.6f}")
```

### Backtesting

```python
from qufin.backtesting.engine import BacktestEngine
from qufin.backtesting.metrics import compute_metrics

engine = BacktestEngine(rebalance_freq="monthly", window=252)
portfolio_values = engine.run(prices_df, strategy_fn)
metrics = compute_metrics(portfolio_values)
print(f"Sharpe: {metrics['sharpe']:.2f}  Max DD: {metrics['max_drawdown']:.2%}")
```

<details>
<summary><b>More examples: synthetic data, factor models, noise-aware optimization</b></summary>
<br>

**Synthetic market data generation**

```python
from qufin.data.synthetic import gbm_paths, heston_paths, HestonParams

# Geometric Brownian Motion
paths = gbm_paths(s0=100, mu=0.08, sigma=0.2, T=1.0, n_steps=252, n_paths=10_000)

# Heston stochastic volatility
params = HestonParams(v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7)
paths = heston_paths(s0=100, mu=0.08, params=params, T=1.0, n_steps=252, n_paths=10_000)
```

**Factor model risk decomposition**

```python
from qufin.portfolio.classical.factor_models import build_factor_model, risk_decomposition

model = build_factor_model(asset_returns, factor_returns, window=252)
decomp = risk_decomposition(weights, model.exposures, model.factor_cov)
print(f"Systematic: {decomp['systematic_pct']:.1%}")
```

**Automatic backend selection**

```python
from qufin.backends.auto_select import auto_select_backend

backend = auto_select_backend(circuit)  # GPU -> Aer -> Mock fallback
```

</details>

<br>

## Capabilities

<table>
<tr>
<td width="50%" valign="top">

### Portfolio Optimization

| | Methods |
|:--|:--------|
| **Classical** | Mean-Variance (CVXPY), Black-Litterman, Risk Parity, HRP, Multi-Period, ADMM, Factor Models |
| **Quantum** | QAOA (4 mixers), VQE, Warm-Start, Szegedy Walk, Robust CVaR QUBO, Sector Rotation (VQC) |
| **Constraints** | Cardinality, sector, turnover, transaction cost, budget |

### Option Pricing

| | Methods |
|:--|:--------|
| **Classical** | Black-Scholes (full Greeks), Monte Carlo (European/Asian/Barrier), CRR Binomial, LSM American, Implied Vol (SABR/SVI) |
| **Quantum** | Canonical QAE, IQAE, MLAE, FQAE, Path-Dependent QAE, American QAE (Quantum LSM) |
| **Exotics** | Bermudan, lookback, cliquet, autocallable, basket |

### Risk Management

| | Methods |
|:--|:--------|
| **Classical** | VaR (historical/parametric/MC), CVaR, stress testing, CVA/DVA |
| **Quantum** | Quantum VaR, Egger credit-risk, Quantum Stress Testing |
| **Credit** | Gaussian copula, NIG copula |

</td>
<td width="50%" valign="top">

### Hedging

| | Methods |
|:--|:--------|
| **Classical** | Delta hedging, deep hedging (PyTorch) |
| **Quantum** | Quantum deep hedging, RL-quantum hedging |

### Machine Learning

| | Methods |
|:--|:--------|
| **Classical** | Standard classifiers |
| **Quantum** | Kernel methods, VQC, qGAN, reservoir computing |

### Error Mitigation

| | Methods |
|:--|:--------|
| **Level 1** | Readout calibration, TREX |
| **Level 2** | ZNE (Richardson extrapolation), Dynamical Decoupling (XY4/CPMG/Uhrig) |
| **Level 3** | PEC (unbiased), CDR (Clifford regression), M3 (matrix-free) |
| **Adaptive** | Noise-aware variational optimization, robust to calibration drift |

### Data & Infrastructure

| | |
|:--|:--|
| **Market Data** | Yahoo Finance, FRED, Bloomberg, Refinitiv/LSEG |
| **Streaming** | Alpaca, Polygon, IEX WebSocket with rebalance triggers |
| **Synthetic** | GBM, Heston, Merton jump-diffusion |
| **Warehouse** | Parquet partitioned storage, PyArrow, auto-compaction |
| **Quality** | Gap detection, outlier flagging, data lineage, quality scoring |
| **Backtesting** | Walk-forward engine, 15+ metrics |
| **Benchmarks** | 15/25/50-asset problem sets, hardware validation, reproducibility manifests |

### Enterprise

| | |
|:--|:--|
| **REST API** | FastAPI endpoints for optimization, pricing, risk (OpenAPI, auth, rate limiting) |
| **Job Queue** | Celery + Redis async jobs with priority routing and timeouts |
| **Caching** | Result caching (SQLite/Redis) with TTL and invalidation |
| **Deployment** | Docker, docker-compose, Kubernetes Helm chart with auto-scaling |
| **Audit** | Immutable audit trail (SQLite WAL/Postgres), CSV/JSON export |
| **Compliance** | SR 11-7 / SS1/23 checklist, champion-challenger, sensitivity analysis |
| **Explainability** | QUBO decomposition, Shapley attribution, quantum-vs-classical comparison |

</td>
</tr>
</table>

<br>

## Backends

All quantum algorithms accept any backend implementing the `Backend` protocol. Swap backends without changing algorithm code.

```python
from qufin.backends.auto_select import auto_select_backend

# Automatic: selects the best available backend for your circuit
backend = auto_select_backend(circuit)
```

| Backend | Target | Key Feature |
|:--------|:-------|:------------|
| `MockBackend` | Testing | Deterministic, zero dependencies |
| `QiskitAerBackend` | Simulation | Statevector + QASM, local |
| `NoisyAerBackend` | Noise R&D | 4 device profiles (Eagle, Heron) |
| `IBMRuntimeBackend` | IBM QPU | Sampler/Estimator primitives, up to 156 qubits |
| `PennyLaneBackend` | PennyLane | Parameter-shift gradients, Lightning |
| `CirqBackend` | Google QPU | Sycamore/Willow, XEB noise characterization |
| `BraketBackend` | AWS QPU | IonQ Aria/Forte, Rigetti Ankaa, cost estimation |
| `CudaQBackend` | GPU | NVIDIA CUDA-Q, multi-GPU for 30+ qubits |

<br>

## Benchmarks

Standardized benchmark suites for honest quantum-vs-classical comparison. No cherry-picked results.

```python
from qufin.benchmarks.runner import BenchmarkRunner
from qufin.benchmarks.problems import make_benchmark

runner = BenchmarkRunner()
results = runner.run(make_benchmark(15), algorithms=["qaoa", "vqe", "mean_variance"])
runner.summary(results)
```

- **Problem sets** &mdash; 15, 25, and 50 asset portfolios with real covariance structure
- **Metrics** &mdash; Approximation ratio, time-to-solution, circuit depth, success probability
- **Reproducibility** &mdash; Every run generates a manifest (hardware, versions, seeds, calibration data)
- **Hardware validation** &mdash; QAOA/QAE runners for IBM and IonQ with statistical analysis (mean, std, 95% CI)
- **Finance transpiler** &mdash; QUBO-aware ZZ optimization, commuting gate parallelization, 30-50% CNOT reduction

<br>

## Architecture

```
src/qufin/
    backends/
        base.py                    Backend ABC + CircuitResult
        mock.py                    Deterministic mock (no simulator)
        qiskit_backend.py          QiskitAerBackend
        ibm_runtime.py             IBM Quantum Runtime
        pennylane_backend.py       PennyLane + parameter-shift
        cirq_backend.py            Cirq + Sycamore/Willow
        braket_backend.py          Braket + IonQ/Rigetti
        cudaq_backend.py           CUDA-Q GPU simulation
        auto_select.py             Auto-selection + registry
        transpiler.py              Finance circuit transpiler
        noise_models.py            Device noise profiles
        error_mitigation.py        ZNE, TREX, PEC, CDR, readout
        dynamical_decoupling.py    XY4, CPMG, Uhrig DD
        m3_mitigation.py           Matrix-free measurement mitigation
        noise_aware_optimizer.py   Noise-aware variational optimization
    options/
        classical/                 Black-Scholes, binomial, Monte Carlo
        amplitude_estimation/      QAE, IQAE, MLAE, FQAE, path-dep, American
        implied_vol_surface.py     SABR, SVI, QSVM regression
    portfolio/
        classical/                 MVO, Black-Litterman, HRP, Risk Parity, Factors
        optimizers/                QAOA, VQE, warm-start, ADMM, hybrid, robust, walk
        sector_rotation.py         VQC regime detection (11 GICS sectors)
        qubo.py                    QUBO builder + constraints
        mixers.py                  X, XY-ring, XY-full, Grover mixers
    risk/                          VaR, CVaR, stress testing, CVA/DVA
        quantum_stress.py          Quantum stress testing (GFC, COVID, rate hike)
        credit/                    Egger, Gaussian copula, NIG copula
    hedging/                       Delta, deep hedging, quantum RL
    ml/                            Kernels, VQC, qGAN, reservoir computing
    derivatives/                   Bermudan, lookback, cliquet, autocallable, basket
    data/                          Yahoo Finance, FRED, GBM/Heston/Merton synthetic
        bloomberg.py               Bloomberg Terminal connector (blpapi)
        refinitiv.py               Refinitiv/LSEG Eikon connector
        streaming.py               WebSocket streaming (Alpaca, Polygon, IEX)
        warehouse.py               Parquet data warehouse with compaction
        quality.py                 Data quality, lineage, scoring
    backtesting/                   Walk-forward engine, 15+ performance metrics
    benchmarks/                    Problem sets, runner, leaderboard, hardware validation
    api/                           REST API + async job queue
        server.py                  FastAPI endpoints (optimize, price, risk)
        jobs.py                    Celery job queue with priority routing
        cache.py                   Result caching (SQLite/Redis)
    compliance/                    Regulatory compliance tooling
        audit.py                   Immutable audit trail (SQLite WAL/Postgres)
        validation.py              SR 11-7/SS1/23, champion-challenger, sensitivity
        explainability.py          QUBO decomposition, Shapley, comparison reports
```

<br>

## Testing

```bash
pytest                             # Full suite
pytest tests/unit/                 # Unit tests (fast)
pytest tests/integration/          # Integration (requires Qiskit Aer)
pytest tests/property/             # Property-based (Hypothesis)
pytest tests/stress/               # Stress tests
pytest -m "not slow"               # Skip slow tests
pytest -m "not hardware"           # Skip hardware-dependent tests
```

<br>

## Project Status

| Component | Status | Tests |
|:----------|:-------|:------|
| Portfolio Optimization | **Stable** | 280+ |
| Option Pricing (BS + 4 QAE variants) | **Stable** | 200+ |
| Risk Management (VaR/CVaR + quantum) | **Stable** | 120+ |
| Backends (8 targets) | **Stable** | 180+ |
| Error Mitigation (ZNE, TREX, PEC, CDR, DD, M3) | **Stable** | 150+ |
| Backtesting Engine | **Stable** | 40+ |
| Benchmarks & Hardware Validation | **Stable** | 50+ |
| Data Layer (Yahoo, FRED, Bloomberg, Refinitiv, streaming) | **Stable** | 200+ |
| REST API & Job Queue | **Stable** | 78 |
| Compliance & Audit | **Stable** | 109 |
| Hedging (delta + deep + quantum) | **Beta** | 50+ |
| ML (kernels, VQC, qGAN) | **Beta** | 70+ |

<br>

## Contributing

Contributions welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for code style, testing, and PR guidelines.

<br>

## License

Apache 2.0. See [LICENSE](LICENSE).

<br>

## Citation

```bibtex
@software{qufin,
  author  = {Adarsh Keshri},
  title   = {qufin: Research-Grade Quantum Algorithms for Quant Finance},
  year    = {2025},
  url     = {https://github.com/anonymousAAK/qufin},
  license = {Apache-2.0}
}
```

<br>

<div align="center">
<sub>Built for researchers who ship.</sub>
</div>
