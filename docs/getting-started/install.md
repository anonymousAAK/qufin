# Installation

## Basic Install

```bash
pip install qufin
```

This installs the core package with numpy, scipy, and qiskit dependencies.

## Optional Extras

```bash
# Full install with all backends
pip install "qufin[all]"

# Individual backends
pip install "qufin[ibm]"        # IBM Quantum Runtime
pip install "qufin[pennylane]"  # PennyLane Lightning
pip install "qufin[braket]"     # AWS Braket
pip install "qufin[cirq]"       # Google Cirq

# Enterprise (v0.4.0)
pip install "qufin[api]"        # FastAPI, Celery, Redis

# Development
pip install "qufin[dev]"        # pytest, ruff, mypy
```

## Requirements

- **Python**: >= 3.10
- **Core**: numpy, scipy, qiskit >= 2.0, qiskit-aer
- **Portfolio**: cvxpy (for classical mean-variance)
- **Data**: yfinance, fredapi (for market data)
- **ML**: torch (for deep hedging)

## Verify Installation

```python
import qufin
print(qufin.__version__)

# Quick smoke test
from qufin.options.european import EuropeanOption
opt = EuropeanOption(s0=100, k=100, r=0.05, sigma=0.2, T=1.0)
print(f"BS Price: {opt.bs_price():.4f}")  # ~10.45
```

## Development Setup

```bash
git clone https://github.com/anonymousAAK/qufin.git
cd qufin
pip install -e ".[dev]"
pytest tests/unit/  # Run tests
```
