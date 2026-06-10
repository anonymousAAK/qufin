# Option Pricing

qufin provides multiple pricing engines for European, Asian, barrier, and Bermudan options.

## European Options

### Black-Scholes (Analytical)

```python
from qufin.options.european import EuropeanOption

opt = EuropeanOption(s0=100, k=105, sigma=0.2, r=0.05, T=1.0, is_call=True)

price = opt.bs_price()      # 8.02
delta = opt.bs_delta()       # 0.57
gamma = opt.bs_gamma()       # 0.019
vega  = opt.bs_vega()        # 37.52
theta = opt.bs_theta()       # -6.41
```

### Monte Carlo

```python
from qufin.options.classical.monte_carlo import european_mc, asian_mc, barrier_mc

# European
result = european_mc(s=100, k=105, r=0.05, sigma=0.2, T=1.0, n_paths=1_000_000)

# Asian (arithmetic average)
result = asian_mc(s=100, k=100, r=0.05, sigma=0.2, T=1.0, n_paths=100_000)

# Barrier (up-and-out)
result = barrier_mc(s=100, k=100, r=0.05, sigma=0.2, T=1.0,
                    barrier=120, barrier_type="up-and-out", n_paths=100_000)
```

### Binomial Tree (CRR)

```python
from qufin.options.classical.binomial import crr_tree

# crr_tree returns a BinomialResult; use .price.
result = crr_tree(s=100, k=105, r=0.05, sigma=0.2, T=1.0, n_steps=500)
price = result.price
```

## Exotic Options

### Asian Options

Geometric Asian has a closed-form solution (Kemna-Vorst):

```python
from qufin.options.asian import geometric_asian_closed_form

price = geometric_asian_closed_form(s0=100, k=100, r=0.05, sigma=0.2, T=1.0, n_monitoring=12)
```

### Barrier Options

Closed-form (Reiner-Rubinstein):

```python
from qufin.options.barrier import barrier_closed_form

price = barrier_closed_form(
    s0=100, k=100, r=0.05, sigma=0.2, T=1.0,
    barrier=120, barrier_type="up-and-out",
)
```

### Bermudan Options

Binomial tree with discrete exercise dates:

```python
from qufin.options.bermudan import bermudan_binomial, BermudanOptionSpec

spec = BermudanOptionSpec(
    s0=100, k=100, r=0.05, sigma=0.2, T=1.0,
    exercise_dates=[0.25, 0.5, 0.75, 1.0],
    is_call=False,
    n_steps=500,
)
result = bermudan_binomial(spec)
# result.price, result.exercise_boundary
```

## Quantum Amplitude Estimation

QAE can price options with quadratic speedup over Monte Carlo: $O(1/\epsilon)$ vs $O(1/\epsilon^2)$ queries.

### Estimation Problem Setup

```python
from qufin.options.amplitude_estimation.estimation_problem import EstimationProblem
from qiskit.circuit import QuantumCircuit

# State preparation: encode option payoff into amplitude
A = QuantumCircuit(3)
# ... build state preparation circuit ...

problem = EstimationProblem(
    state_preparation=A,
    objective_qubits=[2],  # qubit encoding the payoff
)
```

### Canonical QAE (QPE-based)

```python
from qufin.options.amplitude_estimation.canonical import (
    CanonicalAmplitudeEstimation, CanonicalQAEConfig,
)
from qufin.backends.qiskit_backend import QiskitAerBackend

config = CanonicalQAEConfig(n_eval_qubits=5, shots=4096)
solver = CanonicalAmplitudeEstimation(problem, config, QiskitAerBackend())
result = solver.estimate()
print(f"Amplitude: {result.estimation:.4f}")
```

### Iterative QAE (IQAE)

No QPE overhead — uses adaptive Grover iterations:

```python
from qufin.options.amplitude_estimation.iqae import (
    IterativeAmplitudeEstimation, IQAEConfig,
)

config = IQAEConfig(epsilon=0.01, alpha=0.05, shots=4096)
solver = IterativeAmplitudeEstimation(problem, config, backend)
result = solver.estimate()
```

### QAE Variants Comparison

| Variant | Qubits | Depth | Accuracy | Best For |
|---------|--------|-------|----------|----------|
| Canonical | $n_{eval} + n_{state}$ | $O(2^{n_{eval}})$ | $O(2^{-n_{eval}})$ | High accuracy, deep circuits |
| IQAE | $n_{state}$ | Adaptive | $\epsilon$ | NISQ-friendly, no QPE |
| MLAE | $n_{state}$ | Fixed schedule | Statistical | Multiple circuit depths |
| FQAE | $n_{state}$ | Adaptive | $\epsilon$ | Low circuit depth |

## Advanced Option Pricing (v0.2.0)

### Path-Dependent QAE (Asian Options)

Price Asian options using quantum amplitude estimation:

```python
from qufin.options.amplitude_estimation.path_dependent_qae import (
    price_asian_qae, PathDependentAsianSpec,
)

spec = PathDependentAsianSpec(s0=100, k=100, r=0.05, sigma=0.2, T=1.0, n_steps=12)
price = price_asian_qae(spec, backend=backend, n_qubits=4)
```

### American Options (Quantum LSM)

Price American options with quantum least-squares Monte Carlo:

```python
from qufin.options.amplitude_estimation.american_qae import (
    QuantumLSM, AmericanQAESpec,
)

spec = AmericanQAESpec(s0=100, k=100, r=0.05, sigma=0.2, T=1.0, is_call=False)
result = QuantumLSM(spec, backend=backend).price()
price = result.price
```

### Implied Volatility Surface

Build and evaluate implied vol surfaces with SABR/SVI models:

```python
from qufin.options.implied_vol_surface import QuantumIVSurface

surface = QuantumIVSurface(market_strikes, market_expiries, market_vols)
vol = surface.evaluate(strike=105, expiry=0.5)
```
