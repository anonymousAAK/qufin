# Data Sources

qufin provides three data sources for market data, plus a local caching layer. No network calls are made unless you explicitly call a data-fetching function.

## Yahoo Finance (Equities)

Fetch historical price data for any ticker available on Yahoo Finance.

```python
from qufin.data.equities import fetch_prices, fetch_returns

# Daily closing prices
prices = fetch_prices(["AAPL", "MSFT", "GOOGL"], start="2020-01-01", end="2024-12-31")
print(prices.head())

# Log returns
returns = fetch_returns(["AAPL", "MSFT", "GOOGL"], start="2020-01-01", end="2024-12-31")

# Covariance matrix for portfolio optimization
cov = returns.cov() * 252  # annualized
mu = returns.mean() * 252
```

## FRED (Macroeconomic Data)

Access Federal Reserve Economic Data for interest rates, inflation, and macro indicators.

!!! note "API Key Required"
    Set the `FRED_API_KEY` environment variable. Get a free key at [fred.stlouisfed.org](https://fred.stlouisfed.org/docs/api/api_key.html).

```python
from qufin.data.macro import fetch_fred

# 10-year Treasury yield
rates = fetch_fred("GS10", start="2020-01-01")

# Multiple series
data = fetch_fred(["GS10", "GS2", "CPIAUCSL"], start="2020-01-01")
```

## Synthetic Data Generators

Generate simulated price paths for testing and development without any network dependency.

### Geometric Brownian Motion (GBM)

```python
from qufin.data.synthetic import gbm_paths

paths = gbm_paths(
    s0=100,       # initial price
    mu=0.08,      # annual drift
    sigma=0.2,    # annual volatility
    T=1.0,        # time horizon in years
    n_steps=252,  # trading days
    n_paths=10_000,
)
# paths.shape: (10000, 253)
```

### Heston Stochastic Volatility

```python
from qufin.data.synthetic import heston_paths, HestonParams

params = HestonParams(
    v0=0.04,      # initial variance
    kappa=2.0,    # mean reversion speed
    theta=0.04,   # long-run variance
    xi=0.3,       # vol of vol
    rho=-0.7,     # correlation (price-vol)
)

paths = heston_paths(
    s0=100, mu=0.08, params=params,
    T=1.0, n_steps=252, n_paths=10_000,
)
```

### Merton Jump-Diffusion

```python
from qufin.data.synthetic import merton_jump_paths

paths = merton_jump_paths(
    s0=100,
    mu=0.08,
    sigma=0.15,
    lam=0.5,           # jump intensity (jumps/year)
    jump_mean=-0.02,    # mean jump size
    jump_std=0.03,      # jump size std
    T=1.0,
    n_steps=252,
    n_paths=10_000,
)
```

## Asset Universes

Pre-built asset lists for benchmarks and quick experimentation.

```python
from qufin.data.universes import get_universe

sp500_tech = get_universe("sp500_tech")      # Tech sector of S&P 500
sp500_health = get_universe("sp500_health")  # Healthcare sector
```

## Caching

Data fetched from Yahoo Finance and FRED is cached locally to avoid redundant API calls.

```python
from qufin.data.cache import get_cache_dir, clear_cache

# Cache location
print(get_cache_dir())  # ~/.cache/qufin/

# Clear cached data
clear_cache()
```

Cache files are stored as Parquet in `~/.cache/qufin/` and are excluded from version control.

## Preparing Data for Portfolio Optimization

End-to-end example: fetch data → compute returns → optimize.

```python
from qufin.data.equities import fetch_returns
from qufin.portfolio.classical.mean_variance import mean_variance_optimize

# Fetch 3 years of data
tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
returns = fetch_returns(tickers, start="2022-01-01", end="2024-12-31")

# Annualize
mu = returns.mean() * 252
cov = returns.cov() * 252

# Optimize
weights = mean_variance_optimize(mu.values, cov.values, target_return=0.15)
print(dict(zip(tickers, weights.round(4))))
```
