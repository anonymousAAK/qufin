# Data

This directory holds small example datasets and asset universe definitions shipped with qufin.

## What's here

- **Asset universes** -- pre-defined stock lists for benchmarks (S&P sectors, indices)
- **Example datasets** -- small CSVs for tests and documentation examples

## What's not here

Large downloads, API responses, and computed caches are stored locally at `~/.cache/qufin/` and excluded from version control via `.gitignore`.

## Data sources

qufin supports three data sources, all opt-in:

| Source | Module | Requires |
|---|---|---|
| Yahoo Finance | `qufin.data.equities` | `yfinance` (included) |
| FRED | `qufin.data.macro` | `fredapi` (included) + [API key](https://fred.stlouisfed.org/docs/api/api_key.html) |
| Synthetic | `qufin.data.synthetic` | Nothing -- generates GBM, Heston, and Merton paths locally |

No network calls are made unless you explicitly call a data-fetching function.
