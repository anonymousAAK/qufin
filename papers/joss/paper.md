---
title: 'qufin: A Reproducibility-First Quantum Finance Toolkit'
tags:
  - Python
  - quantum computing
  - finance
  - option pricing
  - portfolio optimization
  - amplitude estimation
  - risk analysis
authors:
  - name: Adarsh
    orcid: 0000-0000-0000-0000
    affiliation: 1
  - name: anonymousAAK
    affiliation: 1
affiliations:
  - name: Independent Researcher
    index: 1
date: 2026-05-09
bibliography: paper.bib
---

# Summary

`qufin` is an open-source Python package that pairs research-grade quantum algorithms with production-grade classical baselines for quantitative finance. It provides implementations of quantum amplitude estimation (QAE) for option pricing, QAOA and VQE for portfolio optimization, and quantum credit risk analysis, alongside classical methods including Black-Scholes, Monte Carlo simulation, mean-variance optimization (CVXPY), and Longstaff-Schwartz for Bermudan options. The package is backend-pluggable, supporting Qiskit Aer, IBM Runtime, PennyLane, Cirq, and Amazon Braket through a unified `Backend` abstraction.

# Statement of Need

Quantum computing for finance has produced over 80 peer-reviewed papers since 2019, yet fewer than 20% ship reproducible code. The primary reference implementation—Qiskit Finance—entered community-maintenance mode following IBM's 2023 reorganization, receiving no feature updates since. PennyLane, the other major quantum framework, contains no finance modules. Meanwhile, practitioners at banks, hedge funds, and research labs need a single toolkit to (1) reproduce published quantum finance results, (2) benchmark quantum approaches against classical baselines on identical problem instances, and (3) scale experiments from toy examples (4 qubits) to realistic sizes (25–100 assets).

`qufin` fills this gap by implementing 15 published algorithms under one API with a **standardized benchmark harness**—the first of its kind for quantum finance. Every algorithm is tested against known analytical results or published paper figures, with documented tolerances and reproducibility manifests capturing RNG seeds, dependency versions, and hardware identifiers.

# Key Features

**Option Pricing.** `qufin` implements four QAE variants—canonical (QPE-based) [@brassard2002], iterative (IQAE) [@grinko2021], maximum-likelihood (MLAE) [@suzuki2020], and faithful/low-depth (FQAE) [@giurgicatiron2022]—for pricing European, Asian, barrier, and basket options. Heston stochastic volatility is supported via the weak-Euler scheme of Wang & Kan [@wangkan2024], and Bermudan options use Longstaff-Schwartz with quantum extensions following Chakrabarti et al. [@chakrabarti2021].

**Portfolio Optimization.** QAOA with cardinality, sector, and turnover constraints scales to 25+ assets using XY-ring mixers and Dicke state initialization [@brandhofer2023]. VQE with CVaR objectives [@barkoutsos2020] and warm-start strategies (Egger–Goemans-Williamson) are provided alongside classical baselines (CVXPY mean-variance, Black-Litterman, HRP, risk parity).

**Risk Analysis.** Quantum VaR via bisection + QAE [@woerneregger2019] and quantum credit risk following Egger et al. [@egger2021credit] are implemented with Gaussian and NIG copula models, counterparty CVA, and SA-CCR exposure calculations.

**Benchmark Harness.** Standardized problem sets (15/25/50-asset portfolios, option suites) with a runner that dispatches to all registered solvers, computes metrics (relative error, approximation ratio, time-to-solution), and generates leaderboard tables with full reproducibility manifests.

# Acknowledgments

This work builds upon and cites the Qiskit ecosystem, PennyLane, and the extensive quantum finance literature from IBM Research, Goldman Sachs, JPMorgan, and academic groups worldwide.

# References
