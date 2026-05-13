"""Option pricing: classical baselines + quantum amplitude estimation."""

from __future__ import annotations

from qufin.options.european import EuropeanOption
from qufin.options.implied_vol_surface import (
    IVSurfaceData,
    QuantumIVSurface,
    QuantumIVSurfaceConfig,
    SABRModel,
    SVIModel,
    SurfaceMetrics,
    evaluate_surface,
    generate_synthetic_iv_surface,
)

__all__ = [
    "EuropeanOption",
    "IVSurfaceData",
    "QuantumIVSurface",
    "QuantumIVSurfaceConfig",
    "SABRModel",
    "SVIModel",
    "SurfaceMetrics",
    "evaluate_surface",
    "generate_synthetic_iv_surface",
]

# Lazy imports for quantum components (require qiskit)
def get_distribution_loaders():
    from qufin.options import distributions
    return distributions

def get_amplitude_estimation():
    from qufin.options import amplitude_estimation
    return amplitude_estimation
