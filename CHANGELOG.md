# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Project scaffolding: pyproject.toml, src-layout, CI pipeline
- `qufin.backends`: Backend ABC, MockBackend, QiskitAerBackend
- `qufin.data.synthetic`: GBM, Heston, Merton jump-diffusion path generators
- `qufin.data.equities`: Yahoo Finance equity data provider
- `qufin.data.universes`: S&P 500 sector ETFs, Dow Jones 30, NIFTY 50 sample
- `qufin.options.european`: Black-Scholes pricing with full Greeks
- `qufin.portfolio.qubo`: Portfolio QUBO with cardinality constraints
- `qufin.portfolio.mixers`: X-mixer, XY-ring mixer, Dicke initial state
- `qufin.portfolio.optimizers.qaoa`: QAOA portfolio optimizer
- `qufin.portfolio.classical.mean_variance`: CVXPY-based mean-variance optimization
- `qufin.utils`: Result dataclass, Settings (pydantic), encoders, logging
- Unit tests for synthetic data, European options, backends, encoders, QUBO
