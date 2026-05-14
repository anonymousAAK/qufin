---
title: 'qufin: A Research-Grade Framework for Quantum-Enhanced Quantitative Finance'
tags:
  - Python
  - quantum computing
  - quantitative finance
  - portfolio optimization
  - option pricing
  - amplitude estimation
  - risk analysis
authors:
  - name: Adarsh Keshri
    orcid: 0000-0000-0000-0000
    affiliation: 1
affiliations:
  - name: Independent Researcher
    index: 1
date: 14 May 2026
bibliography: paper.bib
---

# Summary

`qufin` is an open-source Python framework that implements quantum algorithms for quantitative finance alongside their best-available classical counterparts, enabling rigorous head-to-head comparison on identical inputs with identical metrics. The library spans 129 modules across 12 subpackages, covering portfolio optimization, option pricing, risk management, hedging, machine learning, derivatives pricing, backtesting, and enterprise deployment. It provides 8 pluggable quantum backends (Qiskit Aer, IBM Runtime, PennyLane, Cirq, Amazon Braket, NVIDIA CUDA-Q, noisy simulation, and a deterministic mock), 5 error mitigation strategies, dynamical decoupling sequences, matrix-free measurement mitigation, and 4 variants of quantum amplitude estimation. The framework is validated by 1,724 tests---unit, integration, property-based, and stress---with 91% code coverage.

# Statement of Need

Quantum computing is frequently cited as a transformative technology for computational finance, with theoretical quadratic speedups for Monte Carlo simulation [@montanaro2015], combinatorial speedups for portfolio optimization [@farhi2014; @brandhofer2023], and improved risk analysis [@woerneregger2019]. However, a significant gap exists between quantum research and finance practice:

**Library fragmentation and deprecation.** Qiskit Finance, the most widely used quantum finance library, was deprecated in 2024 with no replacement. PennyLane provides no dedicated finance modules. Cirq and Braket offer general-purpose quantum tooling but no financial algorithms. Researchers must build from scratch.

**Reproducibility crisis.** An estimated 80% of quantum finance papers ship without runnable code [@stamatopoulos2020]. Results cannot be independently verified, and implementation details critical to algorithm correctness---such as the Grover operator global phase [@brassard2002], IQAE multi-branch theta enumeration [@grinko2021], and canonical QPE eigenvalue extraction---are often omitted from publications.

**Missing classical baselines.** Most quantum finance demonstrations compare against trivially weak baselines or none at all, making it impossible to assess whether quantum methods offer any practical advantage on current or near-term hardware [@herman2023].

**Backend lock-in.** Existing quantum finance code is typically written for a single quantum framework, preventing cross-platform comparison and hardware-agnostic research.

`qufin` addresses these gaps by providing a unified, backend-agnostic framework where every quantum algorithm is paired with the best available classical solver for the same problem, and results are compared with identical metrics on standardized benchmark sets.

# Key Design Decisions

## Backend Abstraction

All quantum algorithms in `qufin` accept any object implementing the `Backend` abstract base class, which defines a minimal interface: `run(circuit, shots)` returning a `CircuitResult`, and `statevector(circuit)`. This design decouples algorithm logic from hardware concerns and allows users to develop algorithms once, then deploy them across 8 backends without code changes:

```python
from qufin.backends.qiskit_backend import QiskitAerBackend
from qufin.backends.noise_models import NoisyAerBackend, NoiseProfile

# Ideal simulation
ideal = QiskitAerBackend(shots=4096)

# Noisy simulation with IBM Eagle r3 profile
noisy = NoisyAerBackend(profile=NoiseProfile.EAGLE_R3, shots=4096)
```

The `auto_select_backend()` function automatically selects the best available backend for a given circuit, preferring GPU acceleration when available, falling back through Aer to the deterministic mock.

## Honest Benchmarking with Classical Baselines

Every quantum algorithm module in `qufin` contains a classical baseline solving the same problem. The benchmarking framework provides standardized problem sets (15, 25, and 50-asset portfolios with realistic covariance structure), a runner that dispatches to all registered solvers, and reports approximation ratio, time-to-solution, circuit depth, and success probability. Reproducibility manifests record hardware specifications, software versions, random seeds, and calibration data for every run. This design philosophy---classical baselines on every problem---is unique among quantum computing frameworks.

## Error Mitigation

Production use of near-term quantum hardware requires error mitigation. `qufin` implements five strategies:

- **Zero-Noise Extrapolation (ZNE)** [@temme2017]: Richardson extrapolation from multiple noise-scaled circuit executions.
- **Twirled Readout Error eXtinction (TREX)**: Randomized readout twirling to suppress measurement bias.
- **Readout calibration**: Full confusion matrix calibration with matrix inversion.
- **Probabilistic Error Cancellation (PEC)** [@temme2017]: Unbiased noise cancellation via quasi-probability decomposition.
- **Clifford Data Regression (CDR)** [@czarnik2021]: Learning a correction from near-Clifford circuits with known classical simulability.

Additionally, dynamical decoupling sequences (XY4, CPMG, Uhrig) suppress idle-time decoherence, and matrix-free measurement mitigation (M3) [@bravyi2021] handles multi-qubit readout errors efficiently. A noise-aware variational optimizer adapts circuit parameters to the current device calibration, robust to drift.

## Mathematically Correct Implementations

Several subtle implementation details are critical for correct quantum finance algorithms:

- The Grover diffusion operator requires a global phase correction of $\pi$ (the Brassard minus sign [@brassard2002]) to produce correct eigenphases for amplitude estimation.
- Iterative QAE must enumerate all theta branches from $\sin^2((2k+1)\theta)$ measurements [@grinko2021], not just the principal value, to avoid convergence to incorrect amplitudes.
- Canonical QPE uses standard phase estimation with proper eigenvalue mapping from phase to amplitude.

These details are implemented, tested against known analytical solutions, and documented in `qufin`.

# Functionality Overview

## Option Pricing with Quantum Amplitude Estimation

`qufin` implements four QAE variants for derivative pricing: canonical (QPE-based) [@brassard2002], Iterative (IQAE) [@grinko2021], Maximum Likelihood (MLAE) [@suzuki2020], and Fourier (FQAE) [@giurgicatiron2022]. The pricing pipeline encodes a discretized payoff distribution into quantum amplitudes and uses amplitude estimation to compute the expected discounted payoff. Supported contract types include European, Asian, barrier, lookback, cliquet, autocallable, basket, and Bermudan options, with Bermudan pricing using quantum extensions of Longstaff-Schwartz [@chakrabarti2021]:

```python
from qufin.options.amplitude_estimation.european_qae import (
    EuropeanQAESpec, build_european_estimation_problem,
)
from qufin.options.amplitude_estimation.iqae import (
    IQAEConfig, IterativeAmplitudeEstimation,
)
from qufin.backends.qiskit_backend import QiskitAerBackend

spec = EuropeanQAESpec(
    s=100, k=105, sigma=0.2, r=0.05, T=1.0,
    n_qubits=5, option_type="call",
)
problem = build_european_estimation_problem(spec)
backend = QiskitAerBackend(shots=4096)

iqae = IterativeAmplitudeEstimation(
    problem=problem, backend=backend,
    config=IQAEConfig(epsilon_target=0.01),
)
result = iqae.estimate()
print(f"IQAE price: {result.value:.4f}")
```

## Portfolio Optimization with QAOA

The portfolio optimization module formulates the cardinality-constrained Markowitz problem as a Quadratic Unconstrained Binary Optimization (QUBO) [@brandhofer2023] and solves it using QAOA [@farhi2014] with four mixer types (X, XY-ring, XY-full, Grover). The XY-ring mixer preserves the Hamming-weight subspace, enforcing cardinality constraints without penalty terms. Warm-start strategies initialize QAOA parameters from continuous relaxations:

```python
from qufin.portfolio.qubo import PortfolioQUBO
from qufin.portfolio.optimizers.qaoa import QAOAPortfolio, QAOAConfig
from qufin.backends.qiskit_backend import QiskitAerBackend
import numpy as np

mu = np.array([0.12, 0.10, 0.07, 0.03, 0.15])
cov = np.diag([0.04, 0.03, 0.02, 0.01, 0.05])

qubo = PortfolioQUBO(mu=mu, cov=cov, gamma=0.5, cardinality=3)
config = QAOAConfig(p=2, mixer="xy_ring", shots=4096)
backend = QiskitAerBackend(shots=4096)

solver = QAOAPortfolio(qubo, config, backend)
result = solver.run()
print(f"Selected: {result.best_bitstring}")
print(f"Objective: {result.best_objective:.6f}")
```

## Risk Analysis with Quantum VaR

Quantum Value-at-Risk implements the Woerner and Egger [@woerneregger2019] approach: a loss distribution is encoded into quantum amplitudes, and amplitude estimation computes tail probabilities to determine VaR via bisection. Conditional VaR (Expected Shortfall) is computed similarly by estimating the conditional tail expectation:

```python
from qufin.risk.quantum_var import quantum_var, QuantumVaRConfig
from qufin.backends.qiskit_backend import QiskitAerBackend
import numpy as np

returns = np.random.normal(0.0005, 0.02, 252)
backend = QiskitAerBackend(shots=4096)

result = quantum_var(
    returns=returns, confidence=0.99,
    backend=backend,
    config=QuantumVaRConfig(n_qubits=5),
)
print(f"Quantum VaR (99%): {result.value:.4f}")
```

# Additional Modules

Beyond the core algorithms above, `qufin` provides:

- **Hedging**: Delta hedging, deep hedging (PyTorch), quantum deep hedging [@cherrat2023], and RL-quantum hedging.
- **Machine Learning**: Quantum kernel methods, variational quantum classifiers (VQC), quantum GANs [@zoufal2019], and quantum reservoir computing.
- **Data Layer**: Connectors for Yahoo Finance, FRED, Bloomberg, and Refinitiv/LSEG; WebSocket streaming from Alpaca, Polygon, and IEX; synthetic data generation via GBM, Heston, and Merton jump-diffusion.
- **Backtesting**: Walk-forward engine with 15+ performance metrics.
- **Enterprise**: REST API (FastAPI), async job queue (Celery), result caching, Docker/Kubernetes deployment, and regulatory compliance tooling (audit trails, SR 11-7/SS1/23 validation, explainability).

# Comparison with Alternatives

| Feature | qufin | Qiskit Finance | PennyLane | Cirq |
|:--------|:------|:---------------|:----------|:-----|
| Finance modules | 129 | ~20 | 0 | 0 |
| Classical baselines | Every problem | None | N/A | N/A |
| Backend-agnostic | 8 backends | Qiskit only | PennyLane only | Cirq only |
| QAE variants | 4 | 3 | 0 | 0 |
| Error mitigation | 5 + DD + M3 | ZNE only | None | None |
| Benchmarks | 15/25/50-asset | No | No | No |
| Status | Active | Deprecated | Active (no finance) | Active (no finance) |

# Testing and Quality

`qufin` is validated by 1,724 tests spanning four categories: unit tests for individual function correctness, integration tests executing end-to-end quantum circuits on Qiskit Aer, property-based tests using Hypothesis for numerical routine fuzzing, and stress tests for large-scale problems. Code coverage is 91%, enforced by CI. Static analysis uses `ruff` for linting and formatting. The package follows a src-layout with `hatchling` as the build backend and `hatch-vcs` for version management.

# Acknowledgements

This work builds upon the Qiskit ecosystem [@qiskit2024] and the extensive quantum finance literature from IBM Research, Goldman Sachs, JPMorgan, and academic groups worldwide.

# References
