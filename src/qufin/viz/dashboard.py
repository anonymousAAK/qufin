"""Section 6.2.2 -- Portfolio dashboard charts and optional Dash app.

Every public function returns a ``plotly.graph_objects.Figure`` (or plain
``dict`` when *plotly* is not installed).  The ``create_dash_app`` factory
builds a full Dash application but is gated behind an optional ``dash``
import.
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
    "DashboardConfig",
    "create_dash_app",
    "drawdown_chart",
    "portfolio_value_chart",
    "rebalancing_schedule_chart",
    "risk_decomposition_chart",
    "transaction_cost_chart",
]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class DashboardConfig:
    """Layout configuration shared across dashboard charts."""

    width: int = 1000
    height: int = 500
    template: str = "plotly_white"
    title_font_size: int = 16
    colorway: list[str] = field(
        default_factory=lambda: [
            "#636EFA", "#EF553B", "#00CC96", "#AB63FA",
            "#FFA15A", "#19D3F3", "#FF6692", "#B6E880",
        ]
    )


_DEFAULT_CFG = DashboardConfig()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _make_figure(data: list, layout: dict) -> Any:
    if HAS_PLOTLY:
        return go.Figure(data=data, layout=go.Layout(**layout))
    return {"data": data, "layout": layout}


def _trace(trace_type: str = "scatter", **kwargs: Any) -> Any:
    if HAS_PLOTLY:
        cls = getattr(go, trace_type.capitalize(), go.Scatter)
        return cls(**kwargs)
    kwargs["type"] = trace_type
    return kwargs


# ---------------------------------------------------------------------------
# 1. Portfolio value over time
# ---------------------------------------------------------------------------

def portfolio_value_chart(
    dates: Sequence[Any],
    values: Sequence[float],
    *,
    benchmark_values: Sequence[float] | None = None,
    title: str = "Portfolio Value Over Time",
    config: DashboardConfig | None = None,
) -> Any:
    """Line chart of portfolio NAV with optional benchmark overlay.

    Parameters
    ----------
    dates:
        Sequence of date-like x-axis values.
    values:
        Portfolio values corresponding to each date.
    benchmark_values:
        Optional benchmark series for comparison.
    title:
        Chart title.
    config:
        Visual configuration.
    """
    cfg = config or _DEFAULT_CFG
    dates = list(dates)
    values = [float(v) for v in values]

    traces = [
        _trace(
            x=dates, y=values, mode="lines", name="Portfolio",
        ),
    ]
    if benchmark_values is not None:
        traces.append(
            _trace(
                x=dates,
                y=[float(v) for v in benchmark_values],
                mode="lines",
                name="Benchmark",
                line={"dash": "dash"},
            ),
        )

    layout = {
        "title": title,
        "xaxis_title": "Date",
        "yaxis_title": "Value",
        "template": cfg.template,
        "width": cfg.width,
        "height": cfg.height,
        "title_font_size": cfg.title_font_size,
    }
    return _make_figure(traces, layout)


# ---------------------------------------------------------------------------
# 2. Drawdown waterfall
# ---------------------------------------------------------------------------

def _compute_drawdown(values: np.ndarray) -> np.ndarray:
    """Return drawdown series (always <= 0)."""
    cummax = np.maximum.accumulate(values)
    return (values - cummax) / np.where(cummax == 0, 1.0, cummax)


def drawdown_chart(
    dates: Sequence[Any],
    values: Sequence[float],
    *,
    title: str = "Drawdown",
    config: DashboardConfig | None = None,
) -> Any:
    """Area chart of portfolio drawdown.

    Parameters
    ----------
    dates:
        Date-like x-axis values.
    values:
        Portfolio values (not returns).
    title:
        Chart title.
    config:
        Visual configuration.
    """
    cfg = config or _DEFAULT_CFG
    vals = np.asarray(values, dtype=float)
    dd = _compute_drawdown(vals)

    traces = [
        _trace(
            x=list(dates),
            y=dd.tolist(),
            fill="tozeroy",
            mode="lines",
            name="Drawdown",
            line={"color": "crimson"},
        ),
    ]

    layout = {
        "title": title,
        "xaxis_title": "Date",
        "yaxis_title": "Drawdown",
        "template": cfg.template,
        "width": cfg.width,
        "height": cfg.height,
        "title_font_size": cfg.title_font_size,
        "yaxis_tickformat": ".1%",
    }
    return _make_figure(traces, layout)


# ---------------------------------------------------------------------------
# 3. Risk decomposition
# ---------------------------------------------------------------------------

def risk_decomposition_chart(
    categories: Sequence[str],
    contributions: Sequence[float],
    *,
    level: str = "asset",
    title: str | None = None,
    config: DashboardConfig | None = None,
) -> Any:
    """Horizontal bar chart of risk contributions by factor / sector / asset.

    Parameters
    ----------
    categories:
        Labels (factor names, sector names, or asset tickers).
    contributions:
        Risk contribution values (e.g. variance contribution, VaR
        decomposition).
    level:
        One of ``"factor"``, ``"sector"``, ``"asset"`` -- used only for
        the default title.
    title:
        Chart title override.
    config:
        Visual configuration.
    """
    cfg = config or _DEFAULT_CFG
    title = title or f"Risk Decomposition ({level.title()} Level)"
    categories = list(categories)
    contributions = [float(c) for c in contributions]

    # Sort descending for readability
    order = np.argsort(contributions)[::-1]
    sorted_cats = [categories[i] for i in order]
    sorted_vals = [contributions[i] for i in order]

    colors = [
        cfg.colorway[i % len(cfg.colorway)] for i in range(len(sorted_cats))
    ]

    traces = [
        _trace(
            trace_type="bar",
            x=sorted_vals,
            y=sorted_cats,
            orientation="h",
            marker={"color": colors},
            name="Contribution",
        ),
    ]

    layout = {
        "title": title,
        "xaxis_title": "Risk Contribution",
        "yaxis_title": level.title(),
        "template": cfg.template,
        "width": cfg.width,
        "height": max(cfg.height, 40 * len(categories)),
        "title_font_size": cfg.title_font_size,
        "yaxis_autorange": "reversed",
    }
    return _make_figure(traces, layout)


# ---------------------------------------------------------------------------
# 4. Rebalancing schedule
# ---------------------------------------------------------------------------

def rebalancing_schedule_chart(
    dates: Sequence[Any],
    turnover: Sequence[float],
    *,
    title: str = "Rebalancing Schedule",
    config: DashboardConfig | None = None,
) -> Any:
    """Bar chart of portfolio turnover at each rebalancing date.

    Parameters
    ----------
    dates:
        Rebalancing dates.
    turnover:
        Turnover fraction at each rebalancing (0-1 or 0-100 scale).
    title:
        Chart title.
    config:
        Visual configuration.
    """
    cfg = config or _DEFAULT_CFG
    traces = [
        _trace(
            trace_type="bar",
            x=list(dates),
            y=[float(t) for t in turnover],
            name="Turnover",
            marker={"color": cfg.colorway[0]},
        ),
    ]

    layout = {
        "title": title,
        "xaxis_title": "Rebalancing Date",
        "yaxis_title": "Turnover",
        "template": cfg.template,
        "width": cfg.width,
        "height": cfg.height,
        "title_font_size": cfg.title_font_size,
    }
    return _make_figure(traces, layout)


# ---------------------------------------------------------------------------
# 5. Transaction cost tracking
# ---------------------------------------------------------------------------

def transaction_cost_chart(
    dates: Sequence[Any],
    costs: Sequence[float],
    *,
    cumulative: bool = True,
    title: str = "Transaction Costs",
    config: DashboardConfig | None = None,
) -> Any:
    """Line/bar chart of transaction costs over time.

    Parameters
    ----------
    dates:
        Trade / rebalancing dates.
    costs:
        Per-period transaction costs.
    cumulative:
        If True, also overlay the cumulative cost line.
    title:
        Chart title.
    config:
        Visual configuration.
    """
    cfg = config or _DEFAULT_CFG
    costs_list = [float(c) for c in costs]
    dates_list = list(dates)

    traces = [
        _trace(
            trace_type="bar",
            x=dates_list,
            y=costs_list,
            name="Period Cost",
            marker={"color": cfg.colorway[1], "opacity": 0.6},
        ),
    ]

    if cumulative:
        cum = np.cumsum(costs_list).tolist()
        traces.append(
            _trace(
                x=dates_list,
                y=cum,
                mode="lines+markers",
                name="Cumulative",
                yaxis="y2",
                line={"color": cfg.colorway[0]},
            ),
        )

    layout: dict[str, Any] = {
        "title": title,
        "xaxis_title": "Date",
        "yaxis_title": "Period Cost",
        "template": cfg.template,
        "width": cfg.width,
        "height": cfg.height,
        "title_font_size": cfg.title_font_size,
    }

    if cumulative:
        layout["yaxis2"] = {
            "title": "Cumulative Cost",
            "overlaying": "y",
            "side": "right",
        }

    return _make_figure(traces, layout)


# ---------------------------------------------------------------------------
# 6. Dash app factory (optional)
# ---------------------------------------------------------------------------

def create_dash_app(
    portfolio_dates: Sequence[Any],
    portfolio_values: Sequence[float],
    *,
    benchmark_values: Sequence[float] | None = None,
    risk_categories: Sequence[str] | None = None,
    risk_contributions: Sequence[float] | None = None,
    rebalance_dates: Sequence[Any] | None = None,
    rebalance_turnover: Sequence[float] | None = None,
    transaction_dates: Sequence[Any] | None = None,
    transaction_costs: Sequence[float] | None = None,
    config: DashboardConfig | None = None,
    title: str = "qufin Portfolio Dashboard",
) -> Any:
    """Build a Dash app that renders the full dashboard.

    Returns a ``dash.Dash`` application instance.  If *dash* is not
    installed, raises ``ImportError`` with a helpful message.
    """
    try:
        from dash import Dash, dcc, html  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "dash is required for the dashboard app.  "
            "Install it with:  pip install dash"
        ) from exc

    cfg = config or _DEFAULT_CFG

    figures: list[Any] = []
    figures.append(
        portfolio_value_chart(
            portfolio_dates, portfolio_values,
            benchmark_values=benchmark_values, config=cfg,
        )
    )
    figures.append(
        drawdown_chart(portfolio_dates, portfolio_values, config=cfg)
    )
    if risk_categories is not None and risk_contributions is not None:
        figures.append(
            risk_decomposition_chart(
                risk_categories, risk_contributions, config=cfg,
            )
        )
    if rebalance_dates is not None and rebalance_turnover is not None:
        figures.append(
            rebalancing_schedule_chart(
                rebalance_dates, rebalance_turnover, config=cfg,
            )
        )
    if transaction_dates is not None and transaction_costs is not None:
        figures.append(
            transaction_cost_chart(
                transaction_dates, transaction_costs, config=cfg,
            )
        )

    app = Dash(__name__)
    children = [html.H1(title)]
    for fig in figures:
        children.append(dcc.Graph(figure=fig))

    app.layout = html.Div(children=children)
    return app
