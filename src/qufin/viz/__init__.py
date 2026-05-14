"""Interactive visualization: widgets, charts, and optional Dash dashboard."""

from __future__ import annotations

from qufin.viz.widgets import (
    asset_selector,
    efficient_frontier,
    parameter_sensitivity,
    quantum_vs_classical,
)

__all__ = [
    "asset_selector",
    "efficient_frontier",
    "parameter_sensitivity",
    "quantum_vs_classical",
]
