# Exotic Derivatives

qufin supports pricing and analysis of exotic derivatives beyond vanilla European options.

## Bermudan Options (Longstaff-Schwartz)

Bermudan options can be exercised at specific dates before expiration. Priced via Least-Squares Monte Carlo (LSM).

```python
from qufin.derivatives.bermudan_lsm import lsm_price

# lsm_price returns a dict with a "price" key.
result = lsm_price(
    s0=100,
    k=105,
    r=0.05,
    sigma=0.2,
    T=1.0,
    n_steps=12,          # monthly exercise opportunities
    n_paths=100_000,
    exercise_dates=[i / 12 for i in range(1, 13)],
    is_call=False,
)
print(f"Bermudan put price: {result['price']:.4f}")
```

## Basket Options

Options on a weighted basket of multiple underlying assets.

```python
import numpy as np
from qufin.derivatives.basket import basket_mc, BasketOptionSpec

corr = np.eye(3)  # 3x3 correlation matrix
spec = BasketOptionSpec(
    s0=np.array([100.0, 110.0, 95.0]),
    k=100.0,
    r=0.05,
    sigma=np.array([0.2, 0.25, 0.18]),
    corr=corr,
    T=1.0,
    weights=np.array([0.4, 0.35, 0.25]),
    is_call=True,
)
result = basket_mc(spec, n_paths=100_000, seed=42)
print(f"Basket price: {result.price:.4f}")
```

## Autocallable Notes

Structured products that automatically redeem if the underlying exceeds a barrier on observation dates.

```python
from qufin.derivatives.autocallable import autocallable_mc, AutocallableSpec

spec = AutocallableSpec(
    s0=100,
    k=100,
    barrier=105,              # autocall barrier (absolute level)
    coupon=0.08,              # 8% coupon
    observation_dates=[0.25, 0.5, 0.75, 1.0],  # quarterly
    r=0.05,
    sigma=0.25,
    T=1.0,
)
result = autocallable_mc(spec, seed=42)
print(f"Autocallable price: {result['price']:.4f}")
```

## Path-Dependent Options

### Lookback Options

The payoff depends on the maximum or minimum price during the option's life.

```python
from qufin.derivatives.path_dependent import lookback_mc, LookbackOptionSpec

spec = LookbackOptionSpec(s0=100, r=0.05, sigma=0.2, T=1.0, is_call=True, n_steps=252)
price = lookback_mc(spec, n_paths=100_000, seed=42)
print(f"Lookback price: {price:.4f}")
```

### Cliquet Options

Options with periodic resets that lock in gains at each observation date.

```python
from qufin.derivatives.path_dependent import cliquet_mc

price = cliquet_mc(
    s0=100,
    r=0.05,
    sigma=0.2,
    T=1.0,
    n_periods=4,         # quarterly resets
    cap=0.05,            # 5% cap per period
    floor=0.0,           # 0% floor per period
    n_paths=100_000,
    seed=42,
)
print(f"Cliquet price: {price:.4f}")
```

## Pricing Methods Comparison

| Derivative | Method | Convergence | Notes |
|---|---|---|---|
| Bermudan | LSM (Monte Carlo) | O(1/sqrt(N)) | Basis function choice matters |
| Basket | Monte Carlo | O(1/sqrt(N)) | Correlation structure critical |
| Autocallable | Monte Carlo | O(1/sqrt(N)) | Path-dependent, many barriers |
| Lookback | Monte Carlo | O(1/sqrt(N)) | Needs fine time discretization |
| Cliquet | Monte Carlo | O(1/sqrt(N)) | Cap/floor interactions complex |

!!! info "Quantum pricing"
    Multi-asset and path-dependent QAE pricing is planned for v0.2.0. Currently, quantum amplitude estimation is available for single-asset European options via `qufin.options.amplitude_estimation`.
