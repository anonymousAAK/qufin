# Benchmark Results

qufin includes a comprehensive benchmark suite comparing classical and quantum implementations.

## Running Benchmarks

```bash
# All benchmarks
python benchmarks/run_all.py

# Individual suites
python benchmarks/classical_comparison.py
python benchmarks/quantum_scaling.py
python benchmarks/quantum_advantage.py
```

Results are saved to `benchmarks/results/` as JSON and Markdown.

## Classical Performance

### Black-Scholes Pricing

| Method | N | Time (s) | Notes |
|--------|---|----------|-------|
| qufin BS analytical | 1 | ~95 us | Matches numpy to machine precision |
| numpy vectorized | 10,000 | ~1 ms | Vectorized over spot prices |
| qufin loop | 10,000 | ~1.3 s | Per-object overhead; use vectorized for batch |

### Monte Carlo Convergence

| Paths | Time (s) | Rel Error vs BS |
|-------|----------|-----------------|
| 10,000 | 0.4 ms | ~0.4% |
| 100,000 | 4 ms | ~0.2% |
| 1,000,000 | 39 ms | ~0.06% |

Error scales as $O(1/\sqrt{N})$ as expected.

### Portfolio Optimization Scaling

| Method | 10 assets | 50 assets | 100 assets | 200 assets |
|--------|-----------|-----------|------------|------------|
| Mean-Variance (CVXPY) | 12 ms | 14 ms | 23 ms | 81 ms |
| Risk Parity | 69 ms | 191 ms | 800 ms | 5.1 s |
| HRP | 1.2 ms | 3.2 ms | 6.8 ms | 16 ms |
| scipy SLSQP | 0.9 ms | 3.5 ms | 10 ms | 72 ms |

## Quantum Scaling

### QUBO Build Time

QUBO matrix construction scales as $O(N^2)$:

| Encoding | 10 assets | 50 assets | 100 assets | 200 assets |
|----------|-----------|-----------|------------|------------|
| One-hot | 0.1 ms | 1.2 ms | 5 ms | 17 ms |
| Binary (3-bit) | 1.8 ms | 59 ms | — | — |

### Exhaustive Solver

Exponential scaling (reference optimal solutions):

| Assets | Qubits | Time | States Evaluated |
|--------|--------|------|-----------------|
| 10 | 10 | 14 ms | 1,024 |
| 14 | 14 | 209 ms | 16,384 |
| 18 | 18 | 3.4 s | 262,144 |
| 20 | 20 | 14.1 s | 1,048,576 |

### QAOA Performance

Wall-clock time for QAOA with COBYLA optimizer (50 iterations):

| Assets | p=1 X | p=1 XY-ring | p=2 XY-ring | p=3 XY-ring |
|--------|-------|-------------|-------------|-------------|
| 4 | 8.1 s | 6.1 s | 12.2 s | 12.4 s |
| 8 | 8.9 s | 5.8 s | 13.6 s | 13.1 s |
| 12 | 8.8 s | 4.7 s | 10.0 s | 13.1 s |

Key finding: **XY-ring mixer consistently finds better solutions** than X mixer, especially at higher $p$.

### Circuit Depth (per mixer layer)

| Qubits | X Mixer | XY-Ring | XY-Full |
|--------|---------|---------|---------|
| 4 | 5 | 35 | 45 |
| 8 | 5 | 65 | 97 |
| 12 | 5 | 93 | 145 |
| 16 | 5 | 121 | 193 |

## Quantum Advantage Assessment

!!! info "Honest Assessment"
    There is **no quantum advantage** on current NISQ hardware for any finance application. qufin's value is providing quantum-ready infrastructure that delivers competitive classical results today.

### When Will Quantum Help?

| Application | Theoretical Speedup | Hardware Required | Estimated Timeline |
|-------------|--------------------|--------------------|-------------------|
| Option Pricing (QAE) | Quadratic ($1/\epsilon$ vs $1/\epsilon^2$) | ~100+ logical qubits | 2030+ |
| Portfolio (QAOA) | Unproven | ~1000+ logical qubits | 2030-2035 |
| Risk (VaR/CVaR) | Quadratic | ~50+ logical qubits | 2028-2032 |

### qufin's Value Proposition

1. **Today**: Production-grade classical implementations competitive with industry tools
2. **Near-term**: Noise-aware quantum algorithms validated on NISQ hardware
3. **Future**: Same codebase scales to fault-tolerant quantum hardware
4. **Always**: Honest benchmarks — no overclaiming quantum advantage
