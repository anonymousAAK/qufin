# Data Sources

qufin provides three data sources for market data, plus a local caching layer. No network calls are made unless you explicitly call a data-fetching function.

## Yahoo Finance (Equities)

Fetch historical price data for any ticker available on Yahoo Finance.

```python
from qufin.data.equities import YahooEquityProvider

provider = YahooEquityProvider()

# Daily closing prices
prices = provider.get_prices(["AAPL", "MSFT", "GOOGL"], start="2020-01-01", end="2024-12-31")
print(prices.head())

# Returns
returns = provider.get_returns(["AAPL", "MSFT", "GOOGL"], start="2020-01-01", end="2024-12-31")

# Covariance matrix for portfolio optimization
cov = returns.cov() * 252  # annualized
mu = returns.mean() * 252
```

## FRED (Macroeconomic Data)

Access Federal Reserve Economic Data for interest rates, inflation, and macro indicators.

!!! note "API Key Required"
    Set the `FRED_API_KEY` environment variable. Get a free key at [fred.stlouisfed.org](https://fred.stlouisfed.org/docs/api/api_key.html).

```python
from qufin.data.macro import FREDProvider

fred = FREDProvider()

# 10-year Treasury yield
rates = fred.get_series("GS10", start="2020-01-01")

# Risk-free rate / yield curve helpers
rf = fred.get_risk_free_rate()
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
from qufin.data.synthetic import heston_paths

# heston_paths takes individual keyword args and returns (prices, variances),
# each of shape (n_paths, n_steps + 1).
prices, variances = heston_paths(
    s0=100,
    v0=0.04,       # initial variance
    kappa=2.0,     # mean reversion speed
    theta=0.04,    # long-run variance
    xi=0.3,        # vol of vol
    rho=-0.7,      # correlation (price-vol)
    mu=0.08,
    T=1.0, n_steps=252, n_paths=10_000, seed=42,
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
from qufin.data.universes import SP500_SECTOR_ETFS, DOW_JONES_30, NIFTY50_SAMPLE

sectors = SP500_SECTOR_ETFS   # S&P 500 sector ETFs
dow = DOW_JONES_30            # Dow Jones 30 tickers
nifty = NIFTY50_SAMPLE       # NIFTY 50 sample
```

## Caching

Data fetched from Yahoo Finance and FRED is cached locally to avoid redundant API calls.

```python
from qufin.data.cache import get_cached, put_cache, clear_cache

# Look up a cached DataFrame (returns None on a miss)
cached = get_cached("prices", "AAPL", "2020-01-01")

# Clear cached data (optionally filtered by prefix); returns count removed
clear_cache()
```

Cache files are stored as Parquet in `~/.cache/qufin/` and are excluded from version control.

## Bloomberg (v0.4.0)

!!! note "License Required"
    Bloomberg data requires a Bloomberg Terminal license and the `blpapi` Python SDK.

```python
from qufin.data.bloomberg import BloombergDataSource, BloombergConfig

source = BloombergDataSource(BloombergConfig())
prices = source.get_prices(["AAPL US Equity", "MSFT US Equity"], start="2020-01-01")
returns = source.get_returns(["AAPL US Equity"], start="2020-01-01")
dividends = source.get_dividends("AAPL US Equity", start="2020-01-01")
```

## Refinitiv / LSEG (v0.4.0)

```python
from qufin.data.refinitiv import RefinitivDataSource, RefinitivConfig

source = RefinitivDataSource(RefinitivConfig(app_key="YOUR_KEY"))
prices = source.get_equity_prices(["AAPL.O", "MSFT.O"], start="2020-01-01")
bonds = source.get_bond_data(["US10YT=RR"])
curve = source.get_yield_curve(currency="USD")
```

## Real-Time Streaming (v0.4.0)

Stream live prices from Alpaca, Polygon, or IEX via WebSocket.

```python
import asyncio
from qufin.data.streaming import PriceStream, StreamConfig, Provider, RebalanceConfig

config = StreamConfig(provider=Provider.ALPACA, api_key="YOUR_KEY")
rebal = RebalanceConfig(drift_threshold=0.05)

stream = PriceStream(
    config,
    rebalance_config=rebal,
    target_weights={"AAPL": 0.5, "MSFT": 0.5},
    holdings={"AAPL": 100, "MSFT": 100},
)

asyncio.run(stream.connect(["AAPL", "MSFT"], max_messages=1000))
```

## Parquet Data Warehouse (v0.4.0)

Store and query market data locally in a partitioned Parquet warehouse.

```python
from qufin.data.warehouse import ParquetWarehouse, WarehouseConfig

wh = ParquetWarehouse(WarehouseConfig(root_dir="./data_warehouse"))
wh.write(prices_df, asset_class="equity", ticker="AAPL")
df = wh.read(asset_class="equity", ticker="AAPL", start_date="2023-01-01")
wh.compact(asset_class="equity", ticker="AAPL")  # merge small files
```

## Data Quality (v0.4.0)

Validate and score data quality before using it in models.

```python
from qufin.data.quality import detect_gaps, detect_outliers, compute_quality_score

gaps = detect_gaps(prices_series)
outliers = detect_outliers(prices_series, sigma_threshold=5.0)
score = compute_quality_score(prices_series)
print(f"Quality: {score.overall:.1%}")
```

## Preparing Data for Portfolio Optimization

End-to-end example: fetch data → compute returns → optimize.

```python
from qufin.data.equities import YahooEquityProvider
from qufin.portfolio.classical.mean_variance import mean_variance, Objective

# Fetch 3 years of data
tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
returns = YahooEquityProvider().get_returns(tickers, start="2022-01-01", end="2024-12-31")

# Annualize
mu = returns.mean() * 252
cov = returns.cov() * 252

# Optimize (target an annualized return floor)
result = mean_variance(mu.values, cov.values,
                       objective=Objective.MIN_VARIANCE, min_return=0.15)
print(dict(zip(tickers, result.weights.round(4))))
```
