"""Visualization helpers.

Provides standardized plotting functions for qufin results,
including efficient frontier, convergence plots, loss distributions,
and benchmark comparisons.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


def plot_efficient_frontier(
    returns: NDArray[np.float64],
    cov: NDArray[np.float64],
    weights_list: list[NDArray[np.float64]] | None = None,
    labels: list[str] | None = None,
    n_points: int = 100,
    ax: Any = None,
    **kwargs,
) -> Any:
    """Plot the mean-variance efficient frontier.

    Parameters
    ----------
    returns : NDArray
        Expected returns, shape (N,).
    cov : NDArray
        Covariance matrix, shape (N, N).
    weights_list : list of NDArray or None
        Portfolio weight vectors to highlight on the frontier.
    labels : list of str or None
        Labels for each highlighted portfolio.
    n_points : int
        Number of frontier points to compute.
    ax : matplotlib Axes or None
        Axes to plot on. Creates new figure if None.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(8, 5))

    # Generate random portfolios for the cloud
    rng = np.random.default_rng(42)
    n_assets = len(returns)
    port_returns = []
    port_risks = []
    for _ in range(n_points * 10):
        w = rng.dirichlet(np.ones(n_assets))
        r = float(w @ returns)
        risk = float(np.sqrt(w @ cov @ w))
        port_returns.append(r)
        port_risks.append(risk)

    ax.scatter(port_risks, port_returns, c="lightblue", alpha=0.3, s=5, label="Random portfolios")

    if weights_list:
        for i, w in enumerate(weights_list):
            r = float(w @ returns)
            risk = float(np.sqrt(w @ cov @ w))
            label = labels[i] if labels and i < len(labels) else f"Portfolio {i}"
            ax.scatter([risk], [r], s=100, zorder=5, label=label)

    ax.set_xlabel("Risk (Std Dev)")
    ax.set_ylabel("Expected Return")
    ax.set_title("Efficient Frontier")
    ax.legend(fontsize=8)
    return ax


def plot_convergence(
    values: list[float],
    reference: float | None = None,
    xlabel: str = "Iteration",
    ylabel: str = "Value",
    title: str = "Convergence",
    ax: Any = None,
) -> Any:
    """Plot convergence of an iterative algorithm.

    Parameters
    ----------
    values : list of float
        Values at each iteration.
    reference : float or None
        Reference/target value to show as horizontal line.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(8, 4))

    ax.plot(values, "b-", linewidth=1.5, label="Estimate")
    if reference is not None:
        ax.axhline(y=reference, color="r", linestyle="--", label=f"Reference: {reference:.4f}")
        ax.legend()

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    return ax


def plot_loss_distribution(
    losses: NDArray[np.float64],
    var: float | None = None,
    es: float | None = None,
    n_bins: int = 50,
    ax: Any = None,
) -> Any:
    """Plot loss distribution with VaR and ES markers.

    Parameters
    ----------
    losses : NDArray
        Loss samples.
    var : float or None
        Value at Risk to mark.
    es : float or None
        Expected Shortfall to mark.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(8, 4))

    ax.hist(losses, bins=n_bins, density=True, alpha=0.7, color="steelblue", edgecolor="white")

    if var is not None:
        ax.axvline(x=var, color="red", linestyle="--", linewidth=2, label=f"VaR: {var:.4f}")
    if es is not None:
        ax.axvline(x=es, color="darkred", linestyle="-.", linewidth=2, label=f"ES: {es:.4f}")

    if var is not None or es is not None:
        ax.legend()

    ax.set_xlabel("Loss")
    ax.set_ylabel("Density")
    ax.set_title("Loss Distribution")
    return ax


def plot_benchmark_comparison(
    solver_names: list[str],
    values: list[float],
    reference: float | None = None,
    metric_name: str = "Solution Quality",
    ax: Any = None,
) -> Any:
    """Bar chart comparing solver performance.

    Parameters
    ----------
    solver_names : list of str
        Names of solvers.
    values : list of float
        Metric values per solver.
    reference : float or None
        Reference value line.
    metric_name : str
        Y-axis label.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(8, 4))

    colors = plt.cm.Set2(np.linspace(0, 1, len(solver_names)))
    ax.bar(solver_names, values, color=colors, edgecolor="gray")

    if reference is not None:
        ax.axhline(y=reference, color="red", linestyle="--", label=f"Reference: {reference:.4f}")
        ax.legend()

    ax.set_ylabel(metric_name)
    ax.set_title(f"Benchmark: {metric_name}")
    ax.tick_params(axis="x", rotation=30)
    return ax
