"""Explainability module for quantum portfolio optimization decisions.

Provides tools to decompose, attribute, and compare QUBO-based portfolio
selections so that quantum optimization decisions can be understood,
audited, and justified to stakeholders and regulators.

Features
--------
- QUBO coefficient decomposition: identify strongest asset interactions
- Marginal contribution analysis: effect of removing each asset on objective
- SHAP-like attribution for QAOA/quantum optimizer decisions
- Comparison report: quantum selection rationale vs classical optimizer
- Visualization: heatmap of asset interaction strengths
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class QUBODecomposition:
    """Result of decomposing a QUBO matrix into interpretable components.

    Attributes
    ----------
    linear_terms : NDArray
        Diagonal elements of Q — per-asset linear bias (shape ``(N,)``).
    interaction_matrix : NDArray
        Off-diagonal interaction strengths (shape ``(N, N)``), zeroed diagonal.
    top_interactions : list[tuple[int, int, float]]
        Strongest pairwise interactions sorted by absolute magnitude,
        each entry is ``(asset_i, asset_j, strength)``.
    asset_total_interaction : NDArray
        Sum of absolute interaction strengths per asset (shape ``(N,)``).
    """

    linear_terms: NDArray[np.float64]
    interaction_matrix: NDArray[np.float64]
    top_interactions: list[tuple[int, int, float]]
    asset_total_interaction: NDArray[np.float64]


@dataclass
class MarginalContribution:
    """Marginal contribution of each asset to the QUBO objective.

    Attributes
    ----------
    base_objective : float
        Objective value with all selected assets present.
    contributions : NDArray
        Change in objective when each asset is removed (shape ``(N,)``).
        Positive means removing the asset *increases* the objective
        (i.e., the asset was beneficial for minimization).
    asset_rankings : list[int]
        Asset indices sorted from most to least beneficial.
    """

    base_objective: float
    contributions: NDArray[np.float64]
    asset_rankings: list[int]


@dataclass
class SHAPAttribution:
    """SHAP-like attribution values for a quantum portfolio decision.

    Uses a permutation-based approximation of Shapley values over the
    binary asset-selection vector.

    Attributes
    ----------
    shapley_values : NDArray
        Approximate Shapley value for each asset (shape ``(N,)``).
    base_value : float
        Objective value with no assets selected (empty coalition).
    selection : NDArray
        The binary selection vector being explained.
    """

    shapley_values: NDArray[np.float64]
    base_value: float
    selection: NDArray[np.float64]


@dataclass
class ComparisonReport:
    """Side-by-side comparison of quantum vs classical portfolio selections.

    Attributes
    ----------
    quantum_selection : NDArray
        Binary vector of quantum-selected assets.
    classical_selection : NDArray
        Binary vector of classical-selected assets.
    quantum_objective : float
        QUBO objective value for the quantum solution.
    classical_objective : float
        QUBO objective value for the classical solution.
    agreement_mask : NDArray
        Boolean mask — True where both agree on inclusion/exclusion.
    agreement_ratio : float
        Fraction of assets where quantum and classical agree.
    quantum_unique : list[int]
        Asset indices selected only by quantum optimizer.
    classical_unique : list[int]
        Asset indices selected only by classical optimizer.
    quantum_marginals : MarginalContribution
        Marginal contribution analysis of quantum selection.
    classical_marginals : MarginalContribution
        Marginal contribution analysis of classical selection.
    summary : str
        Human-readable summary of differences.
    """

    quantum_selection: NDArray[np.float64]
    classical_selection: NDArray[np.float64]
    quantum_objective: float
    classical_objective: float
    agreement_mask: NDArray[np.bool_]
    agreement_ratio: float
    quantum_unique: list[int]
    classical_unique: list[int]
    quantum_marginals: MarginalContribution
    classical_marginals: MarginalContribution
    summary: str


@dataclass
class InteractionHeatmapData:
    """Data payload for the asset-interaction heatmap visualization.

    Attributes
    ----------
    matrix : NDArray
        Symmetric interaction matrix (shape ``(N, N)``).
    labels : list[str]
        Asset labels for axis ticks.
    title : str
        Plot title.
    """

    matrix: NDArray[np.float64]
    labels: list[str] = field(default_factory=list)
    title: str = "Asset Interaction Strengths"


# ---------------------------------------------------------------------------
# Core analysis functions
# ---------------------------------------------------------------------------


def decompose_qubo(
    Q: NDArray[np.float64],
    *,
    top_k: int = 10,
    asset_labels: list[str] | None = None,
) -> QUBODecomposition:
    """Decompose a QUBO matrix into linear and interaction components.

    Parameters
    ----------
    Q : NDArray
        Square QUBO matrix of shape ``(N, N)``.
    top_k : int
        Number of top interactions to return (by absolute magnitude).
    asset_labels : list[str] | None
        Optional human-readable labels (unused in computation, stored
        for downstream visualization).

    Returns
    -------
    QUBODecomposition
    """
    n = Q.shape[0]
    if Q.shape != (n, n):
        raise ValueError(f"Q must be square, got shape {Q.shape}")

    linear = np.diag(Q).copy()

    # Interaction matrix: upper + lower triangular, zero diagonal
    interaction = Q.copy()
    np.fill_diagonal(interaction, 0.0)

    # Collect upper-triangle interactions (avoid double-counting)
    interactions_list: list[tuple[int, int, float]] = []
    for i in range(n):
        for j in range(i + 1, n):
            strength = Q[i, j] + Q[j, i]  # total coupling
            interactions_list.append((i, j, float(strength)))

    interactions_list.sort(key=lambda t: abs(t[2]), reverse=True)
    top_interactions = interactions_list[:top_k]

    # Per-asset total absolute interaction
    asset_total = np.zeros(n, dtype=np.float64)
    for i in range(n):
        for j in range(n):
            if i != j:
                asset_total[i] += abs(Q[i, j])

    return QUBODecomposition(
        linear_terms=linear,
        interaction_matrix=interaction,
        top_interactions=top_interactions,
        asset_total_interaction=asset_total,
    )


def marginal_contribution(
    Q: NDArray[np.float64],
    selection: NDArray[np.float64],
) -> MarginalContribution:
    """Compute marginal contribution of each selected asset.

    For each selected asset *i*, computes the change in QUBO objective
    ``x^T Q x`` when asset *i* is removed from the selection.

    Parameters
    ----------
    Q : NDArray
        Square QUBO matrix, shape ``(N, N)``.
    selection : NDArray
        Binary selection vector, shape ``(N,)``.

    Returns
    -------
    MarginalContribution
    """
    n = Q.shape[0]
    x = np.asarray(selection, dtype=np.float64).copy()
    base_obj = float(x @ Q @ x)

    contributions = np.zeros(n, dtype=np.float64)
    for i in range(n):
        if x[i] < 0.5:
            # Asset not selected — contribution is the change if we *add* it
            x_mod = x.copy()
            x_mod[i] = 1.0
            contributions[i] = float(x_mod @ Q @ x_mod) - base_obj
        else:
            # Asset selected — contribution is the change if we *remove* it
            x_mod = x.copy()
            x_mod[i] = 0.0
            contributions[i] = float(x_mod @ Q @ x_mod) - base_obj

    # Rank: most beneficial for minimization first (largest positive
    # contribution = removing it hurts the most, i.e., it was helping).
    # For selected assets, positive contribution means objective goes UP
    # when removed, so asset was beneficial.
    rankings = sorted(range(n), key=lambda i: contributions[i], reverse=True)

    return MarginalContribution(
        base_objective=base_obj,
        contributions=contributions,
        asset_rankings=rankings,
    )


def shapley_attribution(
    Q: NDArray[np.float64],
    selection: NDArray[np.float64],
    *,
    n_permutations: int = 100,
    seed: int | None = None,
) -> SHAPAttribution:
    """Compute SHAP-like Shapley values for a QUBO portfolio decision.

    Uses a permutation-based Monte Carlo approximation of Shapley values.
    Each asset's Shapley value quantifies its average marginal contribution
    across all possible orderings of asset inclusion.

    Parameters
    ----------
    Q : NDArray
        Square QUBO matrix, shape ``(N, N)``.
    selection : NDArray
        Binary selection vector, shape ``(N,)``.
    n_permutations : int
        Number of random permutations to sample.
    seed : int | None
        Random seed for reproducibility.

    Returns
    -------
    SHAPAttribution
    """
    n = Q.shape[0]
    x = np.asarray(selection, dtype=np.float64)
    rng = np.random.default_rng(seed)

    base_value = 0.0  # objective with empty set
    shapley = np.zeros(n, dtype=np.float64)

    for _ in range(n_permutations):
        perm = rng.permutation(n)
        coalition = np.zeros(n, dtype=np.float64)
        prev_val = base_value

        for idx in perm:
            # Only include assets that are in the selection
            coalition[idx] = x[idx]
            curr_val = float(coalition @ Q @ coalition)
            shapley[idx] += curr_val - prev_val
            prev_val = curr_val

    shapley /= n_permutations

    return SHAPAttribution(
        shapley_values=shapley,
        base_value=base_value,
        selection=x.copy(),
    )


def compare_selections(
    Q: NDArray[np.float64],
    quantum_selection: NDArray[np.float64],
    classical_selection: NDArray[np.float64],
    *,
    asset_labels: list[str] | None = None,
) -> ComparisonReport:
    """Compare quantum and classical portfolio selections.

    Parameters
    ----------
    Q : NDArray
        Square QUBO matrix, shape ``(N, N)``.
    quantum_selection : NDArray
        Binary selection vector from quantum optimizer.
    classical_selection : NDArray
        Binary selection vector from classical optimizer.
    asset_labels : list[str] | None
        Optional human-readable asset labels for the summary.

    Returns
    -------
    ComparisonReport
    """
    n = Q.shape[0]
    q_sel = np.asarray(quantum_selection, dtype=np.float64)
    c_sel = np.asarray(classical_selection, dtype=np.float64)

    q_obj = float(q_sel @ Q @ q_sel)
    c_obj = float(c_sel @ Q @ c_sel)

    q_binary = q_sel > 0.5
    c_binary = c_sel > 0.5
    agreement = q_binary == c_binary
    agreement_ratio = float(agreement.sum()) / n if n > 0 else 1.0

    q_unique = [int(i) for i in range(n) if q_binary[i] and not c_binary[i]]
    c_unique = [int(i) for i in range(n) if c_binary[i] and not q_binary[i]]

    q_marginals = marginal_contribution(Q, q_sel)
    c_marginals = marginal_contribution(Q, c_sel)

    labels = asset_labels or [f"Asset_{i}" for i in range(n)]

    # Build human-readable summary
    lines = [
        "=== Quantum vs Classical Portfolio Comparison ===",
        f"Agreement: {agreement_ratio:.1%} ({int(agreement.sum())}/{n} assets)",
        f"Quantum objective:   {q_obj:.6f}",
        f"Classical objective:  {c_obj:.6f}",
    ]

    if q_obj < c_obj:
        lines.append("Quantum solution is BETTER (lower objective).")
    elif c_obj < q_obj:
        lines.append("Classical solution is BETTER (lower objective).")
    else:
        lines.append("Both solutions have EQUAL objective values.")

    if q_unique:
        lines.append(
            f"Quantum-only assets: {[labels[i] for i in q_unique]}"
        )
    if c_unique:
        lines.append(
            f"Classical-only assets: {[labels[i] for i in c_unique]}"
        )

    summary = "\n".join(lines)

    return ComparisonReport(
        quantum_selection=q_sel,
        classical_selection=c_sel,
        quantum_objective=q_obj,
        classical_objective=c_obj,
        agreement_mask=agreement,
        agreement_ratio=agreement_ratio,
        quantum_unique=q_unique,
        classical_unique=c_unique,
        quantum_marginals=q_marginals,
        classical_marginals=c_marginals,
        summary=summary,
    )


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def build_interaction_heatmap_data(
    Q: NDArray[np.float64],
    *,
    asset_labels: list[str] | None = None,
    title: str = "Asset Interaction Strengths",
    symmetrize: bool = True,
) -> InteractionHeatmapData:
    """Prepare heatmap data from a QUBO interaction matrix.

    Parameters
    ----------
    Q : NDArray
        Square QUBO matrix, shape ``(N, N)``.
    asset_labels : list[str] | None
        Tick labels for each asset.
    title : str
        Plot title.
    symmetrize : bool
        If True, return ``(Q + Q^T) / 2`` with zeroed diagonal.

    Returns
    -------
    InteractionHeatmapData
    """
    n = Q.shape[0]
    mat = Q.copy()
    np.fill_diagonal(mat, 0.0)

    if symmetrize:
        mat = (mat + mat.T) / 2.0

    labels = asset_labels or [f"Asset_{i}" for i in range(n)]

    return InteractionHeatmapData(matrix=mat, labels=labels, title=title)


def plot_interaction_heatmap(
    data: InteractionHeatmapData,
    **kwargs: Any,
) -> Any:
    """Render an asset-interaction heatmap using plotly (optional dependency).

    Parameters
    ----------
    data : InteractionHeatmapData
        Prepared heatmap data.
    **kwargs
        Additional keyword arguments forwarded to ``plotly.figure_factory``
        or ``plotly.graph_objects.Heatmap``.

    Returns
    -------
    plotly.graph_objects.Figure or None
        The figure object, or None if plotly is not installed.
    """
    try:
        import plotly.graph_objects as go  # type: ignore[import-untyped]
    except ImportError:
        return None

    colorscale = kwargs.pop("colorscale", "RdBu_r")
    fig = go.Figure(
        data=go.Heatmap(
            z=data.matrix,
            x=data.labels,
            y=data.labels,
            colorscale=colorscale,
            zmid=0,
            **kwargs,
        )
    )
    fig.update_layout(title=data.title, xaxis_title="Asset", yaxis_title="Asset")
    return fig


def plot_marginal_contributions(
    mc: MarginalContribution,
    *,
    asset_labels: list[str] | None = None,
    title: str = "Marginal Contributions",
) -> Any:
    """Bar chart of marginal contributions (requires plotly).

    Parameters
    ----------
    mc : MarginalContribution
        Result from :func:`marginal_contribution`.
    asset_labels : list[str] | None
        Tick labels.
    title : str
        Plot title.

    Returns
    -------
    plotly.graph_objects.Figure or None
    """
    try:
        import plotly.graph_objects as go  # type: ignore[import-untyped]
    except ImportError:
        return None

    n = len(mc.contributions)
    labels = asset_labels or [f"Asset_{i}" for i in range(n)]
    order = mc.asset_rankings
    ordered_labels = [labels[i] for i in order]
    ordered_vals = [float(mc.contributions[i]) for i in order]

    colors = ["green" if v > 0 else "red" for v in ordered_vals]
    fig = go.Figure(
        data=go.Bar(x=ordered_labels, y=ordered_vals, marker_color=colors)
    )
    fig.update_layout(title=title, xaxis_title="Asset", yaxis_title="Delta Objective")
    return fig


def plot_shapley_values(
    attr: SHAPAttribution,
    *,
    asset_labels: list[str] | None = None,
    title: str = "Shapley Attribution",
) -> Any:
    """Bar chart of Shapley attribution values (requires plotly).

    Parameters
    ----------
    attr : SHAPAttribution
        Result from :func:`shapley_attribution`.
    asset_labels : list[str] | None
        Tick labels.
    title : str
        Plot title.

    Returns
    -------
    plotly.graph_objects.Figure or None
    """
    try:
        import plotly.graph_objects as go  # type: ignore[import-untyped]
    except ImportError:
        return None

    n = len(attr.shapley_values)
    labels = asset_labels or [f"Asset_{i}" for i in range(n)]
    order = np.argsort(attr.shapley_values)
    ordered_labels = [labels[i] for i in order]
    ordered_vals = [float(attr.shapley_values[i]) for i in order]

    colors = ["green" if v < 0 else "red" for v in ordered_vals]
    fig = go.Figure(
        data=go.Bar(x=ordered_labels, y=ordered_vals, marker_color=colors)
    )
    fig.update_layout(
        title=title, xaxis_title="Asset", yaxis_title="Shapley Value"
    )
    return fig
