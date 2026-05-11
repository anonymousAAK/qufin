"""qufin: Research-grade quantum algorithms for production-grade quant finance."""

from __future__ import annotations

try:
    from qufin._version import __version__
except ModuleNotFoundError:  # editable install without VCS
    __version__ = "0.0.0.dev0"

from qufin import backends, benchmarks, data, derivatives, hedging, ml, options, portfolio, risk

__all__ = [
    "__version__",
    "backends",
    "benchmarks",
    "data",
    "derivatives",
    "hedging",
    "ml",
    "options",
    "portfolio",
    "risk",
]
