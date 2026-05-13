# Exotic Derivatives

qufin supports pricing and analysis of exotic derivatives beyond vanilla European options.

## Bermudan Options (Longstaff-Schwartz)

Bermudan options can be exercised at specific dates before expiration. Priced via Least-Squares Monte Carlo (LSM).

```python
from qufin.derivatives.bermudan_lsm import price_bermudan_put

price = price_bermudan_put(
    s0=100,
    strike=105,
    r=0.05,
    sigma=0.2,
    T=1.0,
    n_steps=12,          # monthly exercise dates
    n_paths=100_000,
    basis_functions=3,   # polynomial degree for continuation value
)
print(f"Bermudan put price: {price:.4f}")
```

## Basket Options

Options on a weighted basket of multiple underlying assets.

```python
from qufin.derivatives.basket import price_basket_option

price = price_basket_option(
    spots=[100, 110, 95],
    weights=[0.4, 0.35, 0.25],
    strike=100,
    r=0.05,
    sigma=[0.2, 0.25, 0.18],
    corr_matrix=corr,    # 3x3 correlation matrix
    T=1.0,
    n_paths=100_000,
    option_type="call",
)
```

## Autocallable Notes

Structured products that automatically redeem if the underlying exceeds a barrier on observation dates.

```python
from qufin.derivatives.autocallable import price_autocallable

price = price_autocallable(
    s0=100,
    autocall_barrier=1.05,    # 105% of initial
    coupon_rate=0.08,         # 8% annual coupon
    ki_barrier=0.70,          # 70% knock-in barrier
    observation_dates=[0.25, 0.5, 0.75, 1.0],  # quarterly
    r=0.05,
    sigma=0.25,
    T=1.0,
    n_paths=100_000,
)
```

## Path-Dependent Options

### Lookback Options

The payoff depends on the maximum or minimum price during the option's life.

```python
from qufin.derivatives.path_dependent import price_lookback

price = price_lookback(
    s0=100,
    r=0.05,
    sigma=0.2,
    T=1.0,
    n_steps=252,
    n_paths=100_000,
    lookback_type="floating_strike",  # or "fixed_strike"
    option_type="call",
)
```

### Cliquet Options

Options with periodic resets that lock in gains at each observation date.

```python
from qufin.derivatives.path_dependent import price_cliquet

price = price_cliquet(
    s0=100,
    r=0.05,
    sigma=0.2,
    T=1.0,
    reset_dates=[0.25, 0.5, 0.75, 1.0],
    local_cap=0.05,      # 5% cap per period
    local_floor=0.0,     # 0% floor per period
    global_floor=0.0,
    n_paths=100_000,
)
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
