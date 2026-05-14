# qufin

**Research-grade quantum algorithms for quantitative finance.**

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

---

## What is qufin?

`qufin` is a Python library that brings quantum amplitude estimation, QAOA/VQE portfolio optimization, quantum credit-risk analysis, and quantum deep hedging into one package — with **classical baselines on every problem** and a **standardized benchmark harness** for honest quantum-vs-classical comparison.

## Why qufin?

- **Qiskit Finance** is in community-maintenance mode since 2023
- **PennyLane** has no finance modules
- ~80% of 2024-2025 quantum-finance papers ship without code

qufin fills this gap with production-grade implementations that run on simulators today and real quantum hardware tomorrow.

## Architecture

```mermaid
graph TB
    subgraph Data["Data Layer"]
        EQ[Equities - Yahoo/Bloomberg/Refinitiv]
        MACRO[Macro - FRED API]
        SYNTH[Synthetic - GBM/Heston/Merton]
        STREAM[Real-Time Streaming]
        WH[Parquet Warehouse]
        DQ[Data Quality]
    end

    subgraph Classical["Classical Algorithms"]
        MV[Mean-Variance / CVXPY]
        BL[Black-Litterman]
        HRP[Hierarchical Risk Parity]
        RP[Risk Parity]
        BS[Black-Scholes]
        MC[Monte Carlo]
        BIN[Binomial Tree]
    end

    subgraph Quantum["Quantum Algorithms"]
        QAOA[QAOA Portfolio]
        VQE[VQE Portfolio]
        QAE[Amplitude Estimation]
        QVAR[Quantum VaR/CVaR]
    end

    subgraph Backends["Backend Abstraction"]
        MOCK[MockBackend]
        AER[Qiskit Aer]
        IBM[IBM Runtime]
        PL[PennyLane]
        BRAKET[Amazon Braket]
        CIRQ[Cirq / Google]
        CUDAQ[CUDA-Q]
        NOISE[Noisy Aer + Mitigation]
        AUTO[Auto-Selection + Transpiler]
    end

    subgraph Analysis["Analysis"]
        BT[Backtesting Engine]
        BENCH[Benchmark Runner]
        METRICS[Performance Metrics]
    end

    subgraph Enterprise["Enterprise (v0.4.0)"]
        API[REST API - FastAPI]
        JOBS[Async Job Queue]
        CACHE[Result Caching]
        AUDIT[Audit Trail]
        VALID[Model Validation]
        EXPLAIN[Explainability]
    end

    Data --> Classical
    Data --> Quantum
    Quantum --> Backends
    Classical --> Analysis
    Quantum --> Analysis
    Analysis --> Enterprise
    Classical --> Enterprise
    Quantum --> Enterprise
```

## Quick Example

```python
from qufin.options.european import EuropeanOption

opt = EuropeanOption(s0=100, k=105, sigma=0.2, r=0.05, T=1.0)
print(f"Price: ${opt.bs_price():.2f}")
print(f"Delta: {opt.bs_delta():.4f}")
print(f"Gamma: {opt.bs_gamma():.4f}")
```

## Package Overview

| Module | Classical | Quantum |
|--------|-----------|---------|
| `portfolio` | Mean-Variance, Black-Litterman, HRP, Risk Parity, Multi-Period, Factor Models, Sector Rotation | QAOA (X/XY/Grover mixers), VQE (CVaR), Szegedy Walk, Robust CVaR QUBO, ADMM, Hybrid |
| `options` | Black-Scholes, Monte Carlo, Binomial (CRR), American (LSM), Implied Vol Surface | Canonical QAE, IQAE, MLAE, FQAE, Path-Dependent QAE, American QAE |
| `risk` | Historical/Parametric VaR, CVaR, Stress Testing | Quantum VaR, Credit Risk (Egger), Quantum Stress Testing |
| `hedging` | Delta Hedging | Deep Hedging, Quantum Deep Hedging |
| `backends` | — | Qiskit Aer, IBM Runtime, PennyLane, Amazon Braket, Cirq, CUDA-Q, Noise Models, Error Mitigation (ZNE, TREX, PEC, CDR), M3 Mitigation, Dynamical Decoupling, Finance Transpiler, Noise-Aware Optimizer, Auto-Selection |
| `data` | Yahoo Finance, FRED, Bloomberg, Refinitiv, Streaming, Parquet Warehouse, Quality Scoring | — |
| `backtesting` | Walk-Forward Engine, 15+ Performance Metrics | — |
| `benchmarks` | Standardized Problems, Leaderboard | Scaling Analysis, Hardware Benchmarks (IonQ, IBM) |
| `api` | REST API (FastAPI), Async Job Queue (Celery), Result Caching | — |
| `compliance` | Audit Trail, Model Validation (SR 11-7/SS1/23), Champion-Challenger, Explainability (SHAP, QUBO decomposition) | — |
