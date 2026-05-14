# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0] - 2026-05-14

Backend expansion release: 7 new modules, 1149 total tests, 116 source files, 8 backends.

### Added

#### Backend Expansion
- PennyLane backend: parameter-shift gradient integration, cross-framework verification, enhanced Qiskit conversion
- Amazon Braket backend: IonQ Aria/Forte trapped-ion targets, Rigetti Ankaa target, hybrid job support, cost estimation ($/shot)
- Cirq backend: Google Sycamore/Willow hardware config, XEB noise characterization, sqrt-iSWAP decomposition, noise model integration
- CUDA-Q backend: GPU-accelerated state vector simulation, multi-GPU support for 30+ qubit circuits
- Backend auto-selection: circuit analysis, backend registry, automatic fallback chain (GPU → Aer → Mock)

#### Hardware Validation
- Finance circuit transpiler: QUBO-aware ZZ optimization, commuting gate parallelization, connectivity-aware routing, transpilation benchmarking
- Hardware benchmark framework: QAOA/QAE benchmark runners, error mitigation comparison, reproducibility manifests, statistical analysis
- IonQ benchmark runner with cost analysis and 2Q gate comparison

#### Error Mitigation v2
- Probabilistic Error Cancellation (PEC): quasi-probability decomposition, Monte Carlo PEC estimator, overhead estimation
- Clifford Data Regression (CDR): near-Clifford training circuit generation, linear/ridge regression correction
- Dynamical Decoupling: XY4, CPMG, Uhrig DD sequences, compound DD+ZNE mitigation, T2 extension estimation
- M3 matrix-free measurement mitigation: tensored calibration (linear scaling), iterative Bayesian correction
- Noise-aware variational optimizer: noise penalty in cost function, robust optimization over calibration drift

### Changed
- Source modules increased from 109 to 116
- Test count increased from 1011 to 1149
- Backend count increased from 7 to 8

## [0.2.0] - 2026-05-14

Algorithm expansion release: 7 new modules, 1011 total tests, 109 source files.

### Added

#### Portfolio Optimization
- Multi-period portfolio optimization with turnover penalties and holding costs
- ADMM (Alternating Direction Method of Multipliers) hybrid quantum-classical optimizer
- Hybrid optimizer combining classical warm-start with quantum refinement
- Szegedy quantum walk portfolio optimizer with Markov chain mixing
- Robust portfolio optimization with worst-case CVaR QUBO and ellipsoidal uncertainty sets
- Sector rotation strategy with VQC-based regime detection (11 GICS sectors)
- Fama-French factor model integration (OLS exposure estimation, factor covariance, risk decomposition)

#### Option Pricing
- Path-dependent QAE for Asian option pricing with quantum amplitude estimation
- American option pricing via Quantum Least-Squares Monte Carlo (QuantumLSM)
- Implied volatility surface construction with QSVM regression, SABR and SVI models

#### Risk Management
- Quantum stress testing framework with predefined crisis scenarios (GFC 2008, COVID 2020, Rate Hike 2022)

### Changed
- Test count increased from 635 to 1011
- Source modules increased from 90 to 109
- Coverage maintained at 90%

## [0.1.0] - 2026-05-11

Initial public release with 90 modules, 635 tests, and 92% coverage.

### Added

#### Portfolio Optimization
- Mean-variance optimization via CVXPY (MIQP with GLPK_MI fallback for cardinality constraints)
- Black-Litterman model with view blending
- Risk parity optimization
- Hierarchical Risk Parity (HRP) with dendrogram-based clustering
- QAOA optimizer with 4 mixer types: X, XY-ring, XY-full, Grover
- CVaR-weighted objective for tail-risk-aware QAOA
- Dicke state initialization for feasible-subspace QAOA
- VQE optimizer with hardware-efficient ansatz
- Warm-start QAOA from classical relaxation
- QUBO formulation with cardinality, sector, turnover, and transaction cost constraints
- One-hot and binary variable encodings
- Exhaustive QUBO solver for exact reference solutions

#### Option Pricing
- Black-Scholes analytical pricing with full Greeks (delta, gamma, theta, vega, rho)
- Implied volatility solver (Newton-Raphson)
- CRR binomial tree for European and American exercise
- Monte Carlo engine with antithetic variates (European, Asian, barrier)
- Bermudan options via Longstaff-Schwartz (LSM)
- Exotic derivatives: lookback, cliquet, autocallable, basket, path-dependent

#### Quantum Amplitude Estimation
- Canonical QAE with correct Grover global phase (Brassard minus sign)
- Iterative QAE (IQAE) with multi-branch theta enumeration
- Maximum Likelihood QAE (MLAE)
- Fourier QAE (FQAE)

#### Risk Management
- Historical, parametric, and Monte Carlo VaR
- Conditional VaR (CVaR / Expected Shortfall)
- Stress testing framework with scenario generation
- Counterparty credit valuation adjustment (CVA/DVA)
- Quantum VaR estimation
- Credit risk: Gaussian copula, NIG copula, Egger quantum credit-risk

#### Hedging
- Delta hedging engine
- Deep hedging with PyTorch
- Quantum deep hedging
- RL-quantum hedging agent

#### Machine Learning
- Quantum kernel methods
- Variational Quantum Classifier (VQC)
- Quantum GAN (qGAN)
- Quantum reservoir computing

#### Data Layer
- Yahoo Finance integration via yfinance
- FRED macroeconomic data via fredapi
- Synthetic generators: GBM, Heston stochastic volatility, Merton jump-diffusion
- Local caching with TTL and invalidation
- Pre-built asset universes (S&P sectors, indices)

#### Backtesting
- Walk-forward backtesting engine with configurable rebalance frequency
- 15+ performance metrics (Sharpe, Sortino, max drawdown, Calmar, etc.)
- Transaction cost modeling

#### Noise Simulation & Error Mitigation
- 4 device noise profiles (depolarizing, thermal, device-calibrated)
- Zero-noise extrapolation (ZNE)
- Twirled readout error extinction (TREX)
- Readout error calibration matrices

#### Backends
- `MockBackend` -- deterministic, no simulator dependency
- `QiskitAerBackend` -- statevector and QASM simulation
- `NoisyAerBackend` -- configurable noise with device profiles
- `IBMRuntimeBackend` -- IBM Quantum via Sampler/Estimator primitives
- `PennyLaneBackend`, `CirqBackend`, `BraketBackend` interfaces

#### Benchmarks
- Standardized problem sets: 15, 25, and 50 asset portfolios
- Benchmark runner with timing, circuit depth, and approximation ratio
- Classical comparison baselines
- Quantum scaling analysis
- Reproducibility manifests (hardware, versions, seeds)
- Leaderboard tracking

#### Infrastructure
- GitHub Actions CI (Python 3.10-3.12, Ubuntu/macOS/Windows)
- Linting (ruff), formatting (ruff), type checking (mypy)
- Security scanning (bandit), dependency audit (pip-audit)
- Pre-commit hooks
- Apache 2.0 license

[Unreleased]: https://github.com/anonymousAAK/qufin/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/anonymousAAK/qufin/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/anonymousAAK/qufin/releases/tag/v0.2.0
[0.1.0]: https://github.com/anonymousAAK/qufin/releases/tag/v0.1.0
