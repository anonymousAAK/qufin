"""Option pricing: classical baselines + quantum amplitude estimation."""

from __future__ import annotations

from qufin.options.european import EuropeanOption

__all__ = ["EuropeanOption"]

# Lazy imports for quantum components (require qiskit)
def get_distribution_loaders():
    from qufin.options import distributions
    return distributions

def get_amplitude_estimation():
    from qufin.options import amplitude_estimation
    return amplitude_estimation
