# qufin: research-grade quantum algorithms for quant finance

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

`qufin` brings quantum amplitude estimation, QAOA/VQE portfolio optimization,
quantum credit-risk analysis, and quantum deep hedging into one Python package
with **classical baselines on the same problems** and a **standardized benchmark
harness** for honest quantum-vs-classical comparison.

## Why

- Qiskit Finance is in community-maintenance mode since 2023.
- PennyLane has no finance modules.
- ~80% of 2024-2025 quantum-finance papers ship without code.

## Install

```bash
pip install qufin              # core
pip install "qufin[all]"       # + IBM, PennyLane, Cirq, Braket, Torch
```

## 30-second example: price a European call with Black-Scholes

```python
import qufin as qf

opt = qf.options.EuropeanOption(s0=100, k=105, sigma=0.2, r=0.05, T=1.0)
print(f"Black-Scholes price: {opt.bs_price():.4f}")
print(f"Delta: {opt.bs_delta():.4f}")
```

## What's in the box

| Module | Highlights |
| --- | --- |
| `qufin.portfolio` | QAOA (X / XY-ring / Dicke), VQE (CVaR), QUBO with cardinality+sector+turnover constraints, classical baselines (mean-var, BL, HRP, risk parity) |
| `qufin.options` | Canonical/MLAE/IQAE/FQAE; European/Asian/barrier/Bermudan; Heston (weak-Euler) |
| `qufin.risk` | Quantum VaR/CVaR; Egger credit risk; CDO; counterparty; stress |
| `qufin.hedging` | Delta, deep hedging (Torch), quantum deep hedging (revival of JPM/QCWare) |
| `qufin.ml` | Quantum kernels, qGANs, quantum reservoir computing |
| `qufin.benchmarks` | Standardized problem sets + leaderboard + reproducibility manifests |
| `qufin.backends` | Qiskit Aer/IBM Runtime, PennyLane Lightning, Cirq, AWS Braket, Mock |

## Cite

`CITATION.cff` in this repository.

## License

Apache-2.0.
