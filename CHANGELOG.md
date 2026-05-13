# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/anonymousAAK/qufin/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/anonymousAAK/qufin/releases/tag/v0.1.0
