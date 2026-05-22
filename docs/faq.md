# FAQ & Troubleshooting

## Installation

### `pip install qufin` fails with Qiskit errors

Qiskit requires a C compiler for some extensions. On different platforms:

=== "Linux"
    ```bash
    sudo apt install gcc g++ python3-dev
    pip install qufin
    ```

=== "macOS"
    ```bash
    xcode-select --install
    pip install qufin
    ```

=== "Windows"
    Install [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/), then:
    ```bash
    pip install qufin
    ```

### `ImportError: No module named 'qiskit_aer'`

Qiskit Aer is a core dependency and should install automatically. If it doesn't:

```bash
pip install qiskit-aer>=0.14
```

### CVXPY solver errors (GLPK, SCIP not found)

qufin's mean-variance optimizer uses GLPK_MI as a fallback for cardinality-constrained problems. Install it:

```bash
pip install cvxopt   # provides GLPK_MI bindings
```

For best results with integer programming, install SCIP:

```bash
pip install pyscipopt
```

### Python 3.13+ compatibility

qufin requires Python 3.10+. Python 3.13 and 3.14 work but some optional dependencies (PennyLane, Cirq) may not yet support the latest Python. Check each dependency's PyPI page for version support.

---

## Quantum Algorithms

### My QAOA results are random / not converging

Common causes:

1. **Too few shots**: Increase shots to at least 4096 for reliable expectation values.
2. **Too few QAOA layers**: Try `p=2` or `p=3` instead of `p=1`.
3. **Wrong mixer**: For cardinality-constrained problems, use `mixer="xy_ring"` instead of the default X-mixer.
4. **Penalty too low**: If the QUBO budget penalty is too small, the optimizer finds infeasible solutions. Try increasing `budget_penalty`.

```python
# Recommended settings for reliable QAOA
optimizer = QAOAOptimizer(
    backend=QiskitAerBackend(shots=4096),
    p=2,
    mixer="xy_ring",
    optimizer="cobyla",
    max_iter=200,
)
```

### QAE option pricing gives wrong results

Check these common issues:

1. **Precision qubits**: More qubits = better accuracy. Use at least 4 evaluation qubits.
2. **IQAE config**: Ensure `epsilon_target` is small enough (e.g., 0.01) and `shots_per_round` is at least 1024.
3. **Price range**: QAE encodes prices in a fixed range. If the option price is outside this range, results will be incorrect.

### What's the difference between QAE variants?

| Variant | Pros | Cons | Best For |
|---|---|---|---|
| Canonical | Exact, well-understood | Deep circuits, many qubits | Simulator, research |
| IQAE | Shallow circuits, adaptive | More total shots | NISQ hardware |
| MLAE | No extra qubits | Requires many circuit evaluations | Limited qubit count |
| FQAE | Fourier analysis approach | Less mature | Research |

---

## Data

### Yahoo Finance returns empty DataFrame

1. Check the ticker symbol is correct (use Yahoo Finance website to verify).
2. Yahoo Finance may rate-limit or block automated requests. Wait and retry.
3. Check date range — weekends and holidays have no data.

### FRED API key not working

Set it as an environment variable:

```bash
export FRED_API_KEY="your_key_here"    # Linux/macOS
set FRED_API_KEY=your_key_here         # Windows CMD
$env:FRED_API_KEY="your_key_here"      # PowerShell
```

---

## Performance

### Simulations are slow

1. **Reduce shots**: For development, 1024 shots is often sufficient. Use 4096+ for final results.
2. **Use MockBackend**: For unit testing and debugging, `MockBackend` is instant.
3. **Reduce problem size**: Start with 5-10 assets before scaling to 15+.
4. **Use statevector**: For noiseless simulation, statevector is faster than QASM for small circuits.

### Memory issues with large portfolios

For 50+ asset problems:

- Use sparse QUBO representations
- Reduce simulation shot count during parameter search
- Consider the ADMM decomposition approach (available since v0.2.0)

---

## Contributing

### Tests fail on Windows with encoding errors

Set UTF-8 encoding:

```bash
set PYTHONUTF8=1
pytest
```

### Pre-commit hooks reject my code

Run the formatters manually:

```bash
ruff check --fix src/ tests/
ruff format src/ tests/
```

---

## General

### Is quantum advantage real for finance?

Honest answer: **not yet on current hardware** for production-scale problems. See our [Benchmarks](benchmarks.md) page for detailed analysis. qufin provides the tools to track progress honestly as hardware improves.

### How does qufin compare to Qiskit Finance?

Qiskit Finance was deprecated in 2024. qufin fills that gap with:

- Active maintenance and Qiskit 2.x compatibility
- Classical baselines alongside every quantum algorithm
- Honest benchmarking (quantum vs classical on same problems)
- Production features (backtesting, data pipeline, noise models)

### Can I use qufin in production?

qufin is at v1.1.0. The classical algorithms (Black-Scholes, MVO, VaR) are production-ready. Quantum algorithms are research-grade — suitable for experimentation and benchmarking, not yet for production trading decisions.
