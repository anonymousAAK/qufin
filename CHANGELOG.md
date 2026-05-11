# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-05-11

### Added

#### Portfolio Optimization
- Mean-variance optimization with CVXPY (MIQP with GLPK_MI fallback)
- Black-Litterman model
- Risk parity optimization
- Hierarchical Risk Parity (HRP)
- QAOA optimizer with X, XY-ring, XY-full, and Grover mixers (CVaR objective, Dicke init)
- QUBO formulation with cardinality, sector, turnover, and transaction cost constraints
- One-hot and binary variable encoding
- Exhaustive QUBO solver for exact reference solutions

#### Option Pricing
- Black-Scholes analytical pricing with full Greeks (delta, gamma, theta, vega, rho)
- Black-Scholes functional API with implied volatility solver
- CRR binomial tree (European and American exercise)
- Monte Carlo engine with antithetic variates (European, Asian, barrier)
- Bermudan, lookback, cliquet, and autocallable exotic options

#### Quantum Amplitude Estimation
- Canonical QAE
- Iterative QAE (IQAE)
- Maximum Likelihood QAE (MLAE)
- Fourier QAE (FQAE)

#### Risk Management
- Historical, parametric, and Monte Carlo VaR
- Conditional VaR (CVaR / Expected Shortfall)
- Stress testing framework
- Counterparty credit valuation adjustment (CVA/DVA)
- Credit risk modeling with Gaussian and NIG copula

#### Backtesting
- Walk-forward backtesting engine
- 15 performance metrics
- Transaction cost modeling

#### Data Layer
- Yahoo Finance integration
- FRED macroeconomic data
- Synthetic data generators (GBM, Heston, Merton jump-diffusion)
- Local caching system

#### Noise Simulation
- 4 device noise profiles
- Zero-noise extrapolation (ZNE)
- Twirled readout error extinction (TREX)
- Readout error calibration

#### Machine Learning
- Quantum kernel methods
- Variational Quantum Classifier (VQC)
- Quantum reservoir computing

#### Backends
- MockBackend for unit testing
- Qiskit Aer simulator backend
- Noisy Aer backend with device profiles
- IBM Runtime backend interface

#### Benchmarks
- 150+ benchmark entries across 15, 25, and 50 asset universes
- Classical comparison baselines
- Quantum scaling analysis
- Quantum advantage metrics

#### Infrastructure
- MkDocs Material documentation site
- GitHub Actions CI/CD pipeline
- Linting with ruff, type checking with mypy, security with bandit, dependency audit with pip-audit

[0.1.0]: https://github.com/anonymousAAK/qufin/releases/tag/v0.1.0
