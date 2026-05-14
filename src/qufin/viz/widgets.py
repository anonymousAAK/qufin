"""Section 6.2.1 -- Jupyter widget helpers (Plotly-based).

All public functions return ``plotly.graph_objects.Figure`` when *plotly* is
installed, or plain ``dict`` representations otherwise.  No ipywidgets
dependency -- the helpers are pure-data so they work in any notebook runtime.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

try:
    import plotly.graph_objects as go

    HAS_PLOTLY = True
except ImportError:  # pragma: no cover
    HAS_PLOTLY = False

__all__ = [
    "ChartConfig",
    "asset_selector",
    "efficient_frontier",
    "parameter_sensitivity",
    "quantum_vs_classical",
]


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class ChartConfig:
    """Shared visual configuration for all chart helpers."""

    width: int = 900
    height: int = 600
    template: str = "plotly_white"
    title_font_size: int = 16
    colorway: list[str] = field(
        default_factory=lambda: [
            "#636EFA", "#EF553B", "#00CC96", "#AB63FA",
            "#FFA15A", "#19D3F3", "#FF6692", "#B6E880",
        ]
    )


_DEFAULT_CFG = ChartConfig()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_figure(data: list, layout: dict) -> Any:
    """Return a Plotly Figure when available, else a plain dict."""
    if HAS_PLOTLY:
        fig = go.Figure(data=data, layout=go.Layout(**layout))
        return fig
    return {"data": data, "layout": layout}


def _trace_dict(**kwargs: Any) -> Any:
    """Build a trace as a go.Scatter (or dict fallback)."""
    if HAS_PLOTLY:
        trace_type = kwargs.pop("trace_type", "scatter")
        cls = getattr(go, trace_type.capitalize(), go.Scatter)
        return cls(**kwargs)
    kwargs.pop("trace_type", None)
    return kwargs


# ---------------------------------------------------------------------------
# 1. Asset selector
# ---------------------------------------------------------------------------

def asset_selector(
    available: Sequence[str],
    selected: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Return a dict with *available* tickers and the *selected* subset.

    This is a pure-data helper so downstream widgets / UI code can consume
    the selection without coupling to ipywidgets.

    Parameters
    ----------
    available:
        Full universe of ticker strings.
    selected:
        Pre-selected tickers.  Defaults to all *available*.

    Returns
    -------
    dict with keys ``available``, ``selected``.
    """
    available = list(available)
    if selected is None:
        selected = list(available)
    else:
        selected = [t for t in selected if t in available]
    return {"available": available, "selected": selected}


# ---------------------------------------------------------------------------
# 2. Efficient frontier
# ---------------------------------------------------------------------------

def efficient_frontier(
    returns: np.ndarray,
    volatilities: np.ndarray,
    *,
    sharpe_ratios: np.ndarray | None = None,
    labels: Sequence[str] | None = None,
    optimal_idx: int | None = None,
    config: ChartConfig | None = None,
) -> Any:
    """Plot the mean-variance efficient frontier.

    Parameters
    ----------
    returns:
        1-D array of portfolio expected returns (one per frontier point).
    volatilities:
        Matching 1-D array of portfolio standard deviations.
    sharpe_ratios:
        Optional Sharpe ratios used for colour-mapping.
    labels:
        Optional hover labels for each point.
    optimal_idx:
        Index of the optimal portfolio to highlight.
    config:
        Visual configuration.

    Returns
    -------
    Plotly Figure or dict.
    """
    cfg = config or _DEFAULT_CFG
    returns = np.asarray(returns, dtype=float)
    volatilities = np.asarray(volatilities, dtype=float)

    marker: dict[str, Any] = {"size": 6}
    if sharpe_ratios is not None:
        sharpe_ratios = np.asarray(sharpe_ratios, dtype=float)
        marker["color"] = sharpe_ratios.tolist()
        marker["colorscale"] = "Viridis"
        marker["colorbar"] = {"title": "Sharpe"}
        marker["showscale"] = True

    hover = list(labels) if labels is not None else None

    traces = [
        _trace_dict(
            x=volatilities.tolist(),
            y=returns.tolist(),
            mode="markers",
            marker=marker,
            text=hover,
            name="Frontier",
            trace_type="scatter",
        )
    ]

    if optimal_idx is not None and 0 <= optimal_idx < len(returns):
        traces.append(
            _trace_dict(
                x=[float(volatilities[optimal_idx])],
                y=[float(returns[optimal_idx])],
                mode="markers",
                marker={"size": 14, "color": "red", "symbol": "star"},
                name="Optimal",
                trace_type="scatter",
            )
        )

    layout = {
        "title": "Efficient Frontier",
        "xaxis_title": "Volatility",
        "yaxis_title": "Expected Return",
        "template": cfg.template,
        "width": cfg.width,
        "height": cfg.height,
        "title_font_size": cfg.title_font_size,
    }
    return _make_figure(traces, layout)


# ---------------------------------------------------------------------------
# 3. Quantum vs classical comparison
# ---------------------------------------------------------------------------

def quantum_vs_classical(
    metric_names: Sequence[str],
    classical_values: Sequence[float],
    quantum_values: Sequence[float],
    *,
    title: str = "Quantum vs Classical",
    config: ChartConfig | None = None,
) -> Any:
    """Side-by-side grouped bar chart comparing quantum and classical metrics.

    Parameters
    ----------
    metric_names:
        Labels for the x-axis categories.
    classical_values, quantum_values:
        Numeric values for each metric.
    title:
        Chart title.
    config:
        Visual configuration.

    Returns
    -------
    Plotly Figure or dict.
    """
    cfg = config or _DEFAULT_CFG
    metric_names = list(metric_names)
    classical_values = [float(v) for v in classical_values]
    quantum_values = [float(v) for v in quantum_values]

    traces = [
        _trace_dict(
            x=metric_names,
            y=classical_values,
            name="Classical",
            trace_type="bar",
        ),
        _trace_dict(
            x=metric_names,
            y=quantum_values,
            name="Quantum",
            trace_type="bar",
        ),
    ]

    layout = {
        "title": title,
        "barmode": "group",
        "xaxis_title": "Metric",
        "yaxis_title": "Value",
        "template": cfg.template,
        "width": cfg.width,
        "height": cfg.height,
        "title_font_size": cfg.title_font_size,
    }
    return _make_figure(traces, layout)


# ---------------------------------------------------------------------------
# 4. Parameter sensitivity
# ---------------------------------------------------------------------------

def parameter_sensitivity(
    param_name: str,
    param_values: Sequence[float],
    objective_values: Sequence[float],
    *,
    series_name: str = "Objective",
    extra_series: dict[str, Sequence[float]] | None = None,
    title: str | None = None,
    config: ChartConfig | None = None,
) -> Any:
    """Line chart showing how an objective changes with a parameter.

    Useful for sweeping risk-aversion, cardinality, mixer type index, etc.

    Parameters
    ----------
    param_name:
        Name of the parameter being swept (x-axis label).
    param_values:
        Numeric parameter values.
    objective_values:
        Corresponding objective / metric values.
    series_name:
        Legend name for the primary series.
    extra_series:
        Optional mapping ``{legend_name: values}`` for overlay lines.
    title:
        Chart title (defaults to ``"Sensitivity: {param_name}"``).
    config:
        Visual configuration.

    Returns
    -------
    Plotly Figure or dict.
    """
    cfg = config or _DEFAULT_CFG
    param_values = [float(v) for v in param_values]
    objective_values = [float(v) for v in objective_values]
    title = title or f"Sensitivity: {param_name}"

    traces = [
        _trace_dict(
            x=param_values,
            y=objective_values,
            mode="lines+markers",
            name=series_name,
            trace_type="scatter",
        )
    ]

    if extra_series:
        for name, vals in extra_series.items():
            traces.append(
                _trace_dict(
                    x=param_values,
                    y=[float(v) for v in vals],
                    mode="lines+markers",
                    name=name,
                    trace_type="scatter",
                )
            )

    layout = {
        "title": title,
        "xaxis_title": param_name,
        "yaxis_title": "Value",
        "template": cfg.template,
        "width": cfg.width,
        "height": cfg.height,
        "title_font_size": cfg.title_font_size,
    }
    return _make_figure(traces, layout)
