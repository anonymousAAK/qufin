"""Resource Estimation Suite for quantum finance algorithms.

Provides per-algorithm resource tables, surface code overhead calculation,
break-even timeline analysis, and optional interactive visualisation.

Covers:
- QAE option pricing
- QAOA portfolio optimisation
- Quantum VaR estimation
- Quantum credit risk analysis

References
----------
Chakrabarti et al., "A threshold for quantum advantage in derivative
    pricing", Quantum 5:463 (2021), arXiv:2012.03819.
Egger et al., "Quantum computing for finance: state-of-the-art and
    future prospects", IEEE TQCE 1:1 (2020).
Gidney & Ekera, "How to factor 2048 bit RSA integers in 8 hours using
    20 million noisy qubits" (2021), arXiv:1905.09749.
Litinski, "Magic State Distillation: Not as Costly as You Think" (2019).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class AlgorithmResource:
    """Resource estimate for a single algorithm at a given problem size.

    Parameters
    ----------
    algorithm : str
        Algorithm name (e.g. "QAE pricing", "QAOA optimisation").
    problem_size : int
        Problem size parameter (e.g. number of assets, qubits).
    n_logical_qubits : int
        Number of logical qubits required.
    t_gate_count : int
        Total T-gate count.
    t_depth : int
        T-depth (parallelised T-gate layers).
    n_physical_qubits : int
        Physical qubits with surface code overhead (default d=17).
    circuit_depth : int
        Total circuit depth.
    runtime_seconds : float
        Estimated wall-clock runtime in seconds.
    metadata : dict[str, Any]
        Additional metadata.
    """

    algorithm: str = ""
    problem_size: int = 0
    n_logical_qubits: int = 0
    t_gate_count: int = 0
    t_depth: int = 0
    n_physical_qubits: int = 0
    circuit_depth: int = 0
    runtime_seconds: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SurfaceCodeOverhead:
    """Surface code overhead calculation result.

    Parameters
    ----------
    n_logical_qubits : int
        Input logical qubit count.
    code_distance : int
        Required code distance.
    n_physical_qubits : int
        Total physical qubits (data + syndrome + distillation).
    physical_per_logical : int
        Physical qubits per logical qubit.
    logical_error_rate : float
        Achieved logical error rate per round.
    n_rounds : int
        Number of error correction rounds.
    runtime_seconds : float
        Estimated runtime in seconds.
    distillation_qubits : int
        Qubits dedicated to magic state distillation.
    """

    n_logical_qubits: int = 0
    code_distance: int = 0
    n_physical_qubits: int = 0
    physical_per_logical: int = 0
    logical_error_rate: float = 0.0
    n_rounds: int = 0
    runtime_seconds: float = 0.0
    distillation_qubits: int = 0


@dataclass
class BreakEvenPoint:
    """Break-even analysis result for a quantum algorithm.

    Parameters
    ----------
    algorithm : str
        Algorithm name.
    break_even_year : int
        Estimated year when quantum becomes practical.
    required_physical_qubits : int
        Physical qubits needed.
    required_error_rate : float
        Required physical error rate.
    hardware_target : str
        Hardware platform (e.g. "IBM", "IonQ", "QuEra").
    is_practical : bool
        Whether the algorithm is practical with projected hardware.
    notes : str
        Additional context.
    """

    algorithm: str = ""
    break_even_year: int = 0
    required_physical_qubits: int = 0
    required_error_rate: float = 0.0
    hardware_target: str = ""
    is_practical: bool = False
    notes: str = ""


# ---------------------------------------------------------------------------
# Hardware roadmap data
# ---------------------------------------------------------------------------

# Projected hardware capabilities by year and vendor
# Based on public roadmaps (IBM, IonQ, QuEra) as of 2024
HARDWARE_ROADMAP: dict[str, list[dict[str, Any]]] = {
    "IBM": [
        {"year": 2024, "qubits": 1121, "error_rate": 1e-3, "name": "Condor"},
        {"year": 2025, "qubits": 5000, "error_rate": 5e-4, "name": "Kookaburra"},
        {"year": 2026, "qubits": 10000, "error_rate": 2e-4, "name": "Starling"},
        {"year": 2028, "qubits": 100000, "error_rate": 1e-4, "name": "Blue Jay"},
        {"year": 2030, "qubits": 1000000, "error_rate": 5e-5, "name": "Projected"},
    ],
    "IonQ": [
        {"year": 2024, "qubits": 35, "error_rate": 3e-4, "name": "Forte"},
        {"year": 2025, "qubits": 64, "error_rate": 1e-4, "name": "Forte Enterprise"},
        {"year": 2026, "qubits": 256, "error_rate": 5e-5, "name": "Gen2"},
        {"year": 2028, "qubits": 1024, "error_rate": 1e-5, "name": "Gen3"},
        {"year": 2030, "qubits": 4096, "error_rate": 5e-6, "name": "Projected"},
    ],
    "QuEra": [
        {"year": 2024, "qubits": 256, "error_rate": 1e-2, "name": "Aquila"},
        {"year": 2025, "qubits": 1000, "error_rate": 5e-3, "name": "Gen2"},
        {"year": 2026, "qubits": 3000, "error_rate": 1e-3, "name": "Gen3"},
        {"year": 2028, "qubits": 10000, "error_rate": 5e-4, "name": "Gen4"},
        {"year": 2030, "qubits": 100000, "error_rate": 1e-4, "name": "Projected"},
    ],
}


# ---------------------------------------------------------------------------
# Per-algorithm resource estimation
# ---------------------------------------------------------------------------


def estimate_qae_pricing(
    n_price_qubits: int = 4,
    n_precision_qubits: int = 8,
    n_assets: int = 1,
    surface_code_distance: int = 17,
) -> AlgorithmResource:
    """Estimate resources for QAE-based option pricing.

    Parameters
    ----------
    n_price_qubits : int
        Qubits for price discretisation per asset.
    n_precision_qubits : int
        QAE evaluation qubits.
    n_assets : int
        Number of underlying assets.
    surface_code_distance : int
        Surface code distance for physical qubit count.

    Returns
    -------
    AlgorithmResource
        Resource estimate for QAE pricing.
    """
    # Logical qubits
    price_reg = n_assets * n_price_qubits
    eval_reg = n_precision_qubits
    ancillae = 1 + n_assets + 2 * n_price_qubits * n_assets  # payoff + comparators + arithmetic
    n_logical = price_reg + eval_reg + ancillae

    # T-gates per oracle call
    t_per_adder = 8 * n_price_qubits
    t_per_multiplier = 8 * n_price_qubits**2
    t_per_comparator = 8 * n_price_qubits
    t_per_rotation = 8 * n_precision_qubits
    t_per_oracle = (
        n_assets * (t_per_adder + t_per_multiplier + t_per_comparator)
        + t_per_rotation
    )

    # Total: O(2^n_precision) Grover iterations
    n_iterations = 2**n_precision_qubits
    t_total = t_per_oracle * n_iterations

    # T-depth
    t_depth = max(1, t_total // max(n_logical, 1))

    # Circuit depth
    circuit_depth = t_depth * 3

    # Physical qubits
    d = surface_code_distance
    phys_per_logical = 2 * d**2
    distillation = 15 * d**2
    n_physical = n_logical * phys_per_logical + distillation

    # Runtime: assume 1 us per T-gate layer
    runtime = t_depth * 1e-6

    return AlgorithmResource(
        algorithm="QAE pricing",
        problem_size=n_assets * n_price_qubits,
        n_logical_qubits=n_logical,
        t_gate_count=t_total,
        t_depth=t_depth,
        n_physical_qubits=n_physical,
        circuit_depth=circuit_depth,
        runtime_seconds=runtime,
        metadata={
            "n_price_qubits": n_price_qubits,
            "n_precision_qubits": n_precision_qubits,
            "n_assets": n_assets,
            "t_per_oracle": t_per_oracle,
            "n_iterations": n_iterations,
        },
    )


def estimate_qaoa_optimization(
    n_assets: int = 4,
    p_layers: int = 3,
    surface_code_distance: int = 17,
) -> AlgorithmResource:
    """Estimate resources for QAOA portfolio optimisation.

    Parameters
    ----------
    n_assets : int
        Number of assets (= number of qubits).
    p_layers : int
        Number of QAOA layers.
    surface_code_distance : int
        Surface code distance.

    Returns
    -------
    AlgorithmResource
        Resource estimate for QAOA.
    """
    # QAOA: 1 qubit per asset
    n_logical = n_assets

    # Per layer:
    # - Cost Hamiltonian: n*(n-1)/2 ZZ interactions, each ~4 T-gates
    # - Mixer: n RX gates, each ~0 T-gates (Clifford + rotation)
    # - Rotation synthesis: ~50 T-gates per arbitrary rotation (Solovay-Kitaev)
    n_zz = n_assets * (n_assets - 1) // 2
    t_per_zz = 4  # CNOT-RZ-CNOT decomposition
    t_per_rotation = 50  # rotation synthesis

    t_per_layer = n_zz * t_per_zz + n_assets * t_per_rotation + n_zz * t_per_rotation
    t_total = t_per_layer * p_layers

    # T-depth: ZZ gates can be partially parallelised
    t_depth_per_layer = max(1, (n_zz * t_per_zz + n_assets * t_per_rotation) // max(n_assets, 1))
    t_depth = t_depth_per_layer * p_layers

    circuit_depth = t_depth * 3

    # Physical qubits
    d = surface_code_distance
    phys_per_logical = 2 * d**2
    distillation = 15 * d**2
    n_physical = n_logical * phys_per_logical + distillation

    runtime = t_depth * 1e-6

    return AlgorithmResource(
        algorithm="QAOA optimisation",
        problem_size=n_assets,
        n_logical_qubits=n_logical,
        t_gate_count=t_total,
        t_depth=t_depth,
        n_physical_qubits=n_physical,
        circuit_depth=circuit_depth,
        runtime_seconds=runtime,
        metadata={
            "n_assets": n_assets,
            "p_layers": p_layers,
            "n_zz_interactions": n_zz,
            "t_per_layer": t_per_layer,
        },
    )


def estimate_quantum_var(
    n_assets: int = 4,
    n_price_qubits: int = 4,
    n_precision_qubits: int = 8,
    surface_code_distance: int = 17,
) -> AlgorithmResource:
    """Estimate resources for quantum Value-at-Risk computation.

    VaR via quantum requires:
    1. Loading the joint distribution of portfolio returns
    2. Amplitude estimation to find the VaR threshold
    3. Conditional value estimation for CVaR

    Parameters
    ----------
    n_assets : int
        Number of assets in the portfolio.
    n_price_qubits : int
        Qubits per asset for return discretisation.
    n_precision_qubits : int
        QAE precision qubits.
    surface_code_distance : int
        Surface code distance.

    Returns
    -------
    AlgorithmResource
        Resource estimate for quantum VaR.
    """
    # Logical qubits: multi-asset distribution + QAE + ancillae
    dist_reg = n_assets * n_price_qubits
    eval_reg = n_precision_qubits
    # Need arithmetic for portfolio value computation
    portfolio_ancillae = n_price_qubits + int(np.ceil(np.log2(max(n_assets, 2))))
    comparator_ancillae = n_price_qubits + 1
    arithmetic_ancillae = 3 * n_price_qubits * n_assets
    n_logical = dist_reg + eval_reg + portfolio_ancillae + comparator_ancillae + arithmetic_ancillae

    # T-gates:
    # - Distribution loading (via QROM): ~n_assets * 2^n_price * n_price T-gates
    n_points = 2**n_price_qubits
    t_distribution = n_assets * n_points * n_price_qubits
    # - Portfolio value computation: n_assets multiplications + additions
    t_portfolio = n_assets * (8 * n_price_qubits**2 + 8 * n_price_qubits)
    # - Comparison: 8 * n_price_qubits
    t_comparison = 8 * n_price_qubits
    # - QAE amplification
    t_per_oracle = t_distribution + t_portfolio + t_comparison
    n_iterations = 2**n_precision_qubits
    t_total = t_per_oracle * n_iterations

    t_depth = max(1, t_total // max(n_logical, 1))
    circuit_depth = t_depth * 3

    d = surface_code_distance
    phys_per_logical = 2 * d**2
    distillation = 15 * d**2
    n_physical = n_logical * phys_per_logical + distillation

    runtime = t_depth * 1e-6

    return AlgorithmResource(
        algorithm="Quantum VaR",
        problem_size=n_assets,
        n_logical_qubits=n_logical,
        t_gate_count=t_total,
        t_depth=t_depth,
        n_physical_qubits=n_physical,
        circuit_depth=circuit_depth,
        runtime_seconds=runtime,
        metadata={
            "n_assets": n_assets,
            "n_price_qubits": n_price_qubits,
            "n_precision_qubits": n_precision_qubits,
            "t_per_oracle": t_per_oracle,
        },
    )


def estimate_quantum_credit(
    n_obligors: int = 10,
    n_default_qubits: int = 3,
    n_precision_qubits: int = 8,
    surface_code_distance: int = 17,
) -> AlgorithmResource:
    """Estimate resources for quantum credit risk analysis.

    Models correlated defaults using a quantum circuit and estimates
    expected loss via amplitude estimation.

    Parameters
    ----------
    n_obligors : int
        Number of obligors in the credit portfolio.
    n_default_qubits : int
        Qubits per obligor for default probability encoding.
    n_precision_qubits : int
        QAE precision qubits.
    surface_code_distance : int
        Surface code distance.

    Returns
    -------
    AlgorithmResource
        Resource estimate for quantum credit risk.
    """
    # Logical qubits
    # - 1 qubit per obligor for default indicator
    # - Correlation loading: ~n_obligors ancillae
    # - Loss aggregation: log2(n_obligors) + n_default_qubits
    # - QAE evaluation register
    obligor_reg = n_obligors
    correlation_ancillae = n_obligors
    loss_reg = int(np.ceil(np.log2(max(n_obligors, 2)))) + n_default_qubits
    eval_reg = n_precision_qubits
    arithmetic_ancillae = 2 * n_obligors
    n_logical = obligor_reg + correlation_ancillae + loss_reg + eval_reg + arithmetic_ancillae

    # T-gates:
    # - Correlated Bernoulli loading: n_obligors controlled rotations
    t_loading = n_obligors * 50  # rotation synthesis per obligor
    # - Loss aggregation: adder tree
    n_additions = n_obligors - 1
    t_additions = n_additions * 8 * n_default_qubits
    # - Threshold comparison
    t_comparison = 8 * (int(np.ceil(np.log2(max(n_obligors, 2)))) + n_default_qubits)
    t_per_oracle = t_loading + t_additions + t_comparison
    n_iterations = 2**n_precision_qubits
    t_total = t_per_oracle * n_iterations

    t_depth = max(1, t_total // max(n_logical, 1))
    circuit_depth = t_depth * 3

    d = surface_code_distance
    phys_per_logical = 2 * d**2
    distillation = 15 * d**2
    n_physical = n_logical * phys_per_logical + distillation

    runtime = t_depth * 1e-6

    return AlgorithmResource(
        algorithm="Quantum credit risk",
        problem_size=n_obligors,
        n_logical_qubits=n_logical,
        t_gate_count=t_total,
        t_depth=t_depth,
        n_physical_qubits=n_physical,
        circuit_depth=circuit_depth,
        runtime_seconds=runtime,
        metadata={
            "n_obligors": n_obligors,
            "n_default_qubits": n_default_qubits,
            "n_precision_qubits": n_precision_qubits,
            "t_per_oracle": t_per_oracle,
        },
    )


# ---------------------------------------------------------------------------
# Full resource table
# ---------------------------------------------------------------------------


def generate_resource_table(
    problem_sizes: list[int] | None = None,
    surface_code_distance: int = 17,
) -> list[AlgorithmResource]:
    """Generate a comprehensive resource table for all algorithms.

    Parameters
    ----------
    problem_sizes : list[int] | None
        Problem sizes to evaluate. Defaults to [4, 8, 16, 32].
    surface_code_distance : int
        Surface code distance for physical qubit estimation.

    Returns
    -------
    List of AlgorithmResource, one per (algorithm, problem_size) pair.
    """
    if problem_sizes is None:
        problem_sizes = [4, 8, 16, 32]

    table: list[AlgorithmResource] = []

    for size in problem_sizes:
        # QAE pricing (single-asset, varying precision)
        qae = estimate_qae_pricing(
            n_price_qubits=size,
            n_precision_qubits=min(size * 2, 16),
            n_assets=1,
            surface_code_distance=surface_code_distance,
        )
        table.append(qae)

        # QAOA optimisation
        qaoa = estimate_qaoa_optimization(
            n_assets=size,
            p_layers=min(size, 8),
            surface_code_distance=surface_code_distance,
        )
        table.append(qaoa)

        # Quantum VaR
        qvar = estimate_quantum_var(
            n_assets=min(size, 16),
            n_price_qubits=max(4, size // 2),
            n_precision_qubits=min(size * 2, 12),
            surface_code_distance=surface_code_distance,
        )
        table.append(qvar)

        # Quantum credit risk
        qcredit = estimate_quantum_credit(
            n_obligors=size,
            n_default_qubits=3,
            n_precision_qubits=min(size * 2, 12),
            surface_code_distance=surface_code_distance,
        )
        table.append(qcredit)

    return table


def resource_table_to_dicts(
    table: list[AlgorithmResource],
) -> list[dict[str, Any]]:
    """Convert resource table to list of dicts for tabular display.

    Parameters
    ----------
    table : list[AlgorithmResource]
        Resource table from generate_resource_table.

    Returns
    -------
    List of dicts with standard columns.
    """
    rows = []
    for r in table:
        rows.append({
            "algorithm": r.algorithm,
            "problem_size": r.problem_size,
            "logical_qubits": r.n_logical_qubits,
            "t_gates": r.t_gate_count,
            "t_depth": r.t_depth,
            "physical_qubits_d17": r.n_physical_qubits,
            "circuit_depth": r.circuit_depth,
            "runtime_s": r.runtime_seconds,
        })
    return rows


# ---------------------------------------------------------------------------
# Surface code overhead calculator
# ---------------------------------------------------------------------------


def compute_surface_code_overhead(
    n_logical_qubits: int,
    t_gate_count: int,
    physical_error_rate: float = 1e-3,
    target_logical_error_rate: float = 1e-10,
    t_gate_time_us: float = 1.0,
) -> SurfaceCodeOverhead:
    """Calculate surface code overhead for a logical circuit.

    Determines the code distance needed to achieve the target logical
    error rate and computes total physical qubit count including magic
    state distillation factories.

    Parameters
    ----------
    n_logical_qubits : int
        Number of logical qubits in the circuit.
    t_gate_count : int
        Total number of T-gates.
    physical_error_rate : float
        Physical error rate of the hardware.
    target_logical_error_rate : float
        Target logical error rate per round.
    t_gate_time_us : float
        Time per T-gate layer in microseconds.

    Returns
    -------
    SurfaceCodeOverhead
        Detailed overhead calculation.
    """
    # Code distance calculation:
    # Logical error rate per round ~ 0.1 * (100 * p_phys)^((d+1)/2)
    # Solve for d such that p_logical < target
    # => (100 * p_phys)^((d+1)/2) < target / 0.1
    # => (d+1)/2 > log(target/0.1) / log(100 * p_phys)

    threshold_ratio = 100 * physical_error_rate
    if threshold_ratio >= 1.0:
        # Below threshold: error correction won't help
        # Use minimum distance
        code_distance = 3
    else:
        log_target = math.log(target_logical_error_rate / 0.1)
        log_threshold = math.log(threshold_ratio)
        if log_threshold == 0:
            code_distance = 3
        else:
            d_half = log_target / log_threshold
            code_distance = max(3, int(2 * d_half - 1))

    # Ensure odd code distance
    if code_distance % 2 == 0:
        code_distance += 1

    # Physical qubits per logical qubit: 2 * d^2
    phys_per_logical = 2 * code_distance**2

    # Magic state distillation factory:
    # Level-1 distillation: ~15 * d^2 qubits per factory
    # Need enough factories for T-gate throughput
    # One factory produces ~1 magic state per d code cycles
    n_factories = max(1, min(n_logical_qubits, 4))
    distillation_qubits = n_factories * 15 * code_distance**2

    # Total physical qubits
    n_physical = n_logical_qubits * phys_per_logical + distillation_qubits

    # Logical error rate achieved
    achieved_logical_error = 0.1 * threshold_ratio ** ((code_distance + 1) / 2)

    # Number of error correction rounds ~ t_depth
    t_depth = max(1, t_gate_count // max(n_logical_qubits, 1))
    n_rounds = t_depth

    # Runtime: t_depth * t_gate_time
    runtime = t_depth * t_gate_time_us * 1e-6

    return SurfaceCodeOverhead(
        n_logical_qubits=n_logical_qubits,
        code_distance=code_distance,
        n_physical_qubits=n_physical,
        physical_per_logical=phys_per_logical,
        logical_error_rate=achieved_logical_error,
        n_rounds=n_rounds,
        runtime_seconds=runtime,
        distillation_qubits=distillation_qubits,
    )


# ---------------------------------------------------------------------------
# Break-even timeline
# ---------------------------------------------------------------------------


def compute_break_even_timeline(
    algorithms: list[AlgorithmResource] | None = None,
    hardware_roadmap: dict[str, list[dict[str, Any]]] | None = None,
) -> list[BreakEvenPoint]:
    """Determine when each quantum finance algorithm becomes practical.

    Overlays algorithm requirements against hardware roadmap projections
    to estimate break-even years.

    Parameters
    ----------
    algorithms : list[AlgorithmResource] | None
        Algorithm resource estimates. If None, uses defaults.
    hardware_roadmap : dict | None
        Hardware roadmap data. If None, uses built-in HARDWARE_ROADMAP.

    Returns
    -------
    List of BreakEvenPoint, one per (algorithm, vendor) pair.
    """
    if algorithms is None:
        algorithms = [
            estimate_qae_pricing(n_price_qubits=8, n_precision_qubits=10),
            estimate_qaoa_optimization(n_assets=20, p_layers=5),
            estimate_quantum_var(n_assets=10, n_price_qubits=6, n_precision_qubits=10),
            estimate_quantum_credit(n_obligors=50, n_precision_qubits=10),
        ]

    if hardware_roadmap is None:
        hardware_roadmap = HARDWARE_ROADMAP

    results: list[BreakEvenPoint] = []

    for algo in algorithms:
        for vendor, roadmap in hardware_roadmap.items():
            # Sort roadmap by year
            sorted_roadmap = sorted(roadmap, key=lambda x: x["year"])

            break_even_year = 0
            is_practical = False
            notes = ""

            for hw_point in sorted_roadmap:
                year = hw_point["year"]
                available_qubits = hw_point["qubits"]
                error_rate = hw_point["error_rate"]

                # Can we run this algorithm on this hardware?
                # Compute surface code overhead at this error rate
                overhead = compute_surface_code_overhead(
                    n_logical_qubits=algo.n_logical_qubits,
                    t_gate_count=algo.t_gate_count,
                    physical_error_rate=error_rate,
                    target_logical_error_rate=1e-10,
                )

                if overhead.n_physical_qubits <= available_qubits:
                    break_even_year = year
                    is_practical = True
                    notes = (
                        f"Feasible on {vendor} {hw_point['name']} "
                        f"({available_qubits} qubits, "
                        f"error rate {error_rate:.1e})"
                    )
                    break

            if not is_practical:
                last_hw = sorted_roadmap[-1]
                overhead = compute_surface_code_overhead(
                    n_logical_qubits=algo.n_logical_qubits,
                    t_gate_count=algo.t_gate_count,
                    physical_error_rate=last_hw["error_rate"],
                )
                notes = (
                    f"Requires {overhead.n_physical_qubits} physical qubits, "
                    f"exceeds {vendor} 2030 projection of {last_hw['qubits']}"
                )

            results.append(BreakEvenPoint(
                algorithm=algo.algorithm,
                break_even_year=break_even_year,
                required_physical_qubits=algo.n_physical_qubits,
                required_error_rate=1e-3,  # baseline assumption
                hardware_target=vendor,
                is_practical=is_practical,
                notes=notes,
            ))

    return results


def break_even_summary(
    timeline: list[BreakEvenPoint] | None = None,
) -> dict[str, Any]:
    """Summarise break-even analysis into a concise report.

    Parameters
    ----------
    timeline : list[BreakEvenPoint] | None
        Break-even timeline. If None, computes defaults.

    Returns
    -------
    Dict with per-algorithm summaries and overall assessment.
    """
    if timeline is None:
        timeline = compute_break_even_timeline()

    # Group by algorithm
    by_algo: dict[str, list[BreakEvenPoint]] = {}
    for bp in timeline:
        by_algo.setdefault(bp.algorithm, []).append(bp)

    summary: dict[str, Any] = {}
    for algo, points in by_algo.items():
        practical_points = [p for p in points if p.is_practical]
        if practical_points:
            earliest = min(p.break_even_year for p in practical_points)
            earliest_vendor = next(
                p.hardware_target for p in practical_points
                if p.break_even_year == earliest
            )
        else:
            earliest = 0
            earliest_vendor = "none"

        summary[algo] = {
            "earliest_break_even": earliest,
            "earliest_vendor": earliest_vendor,
            "n_vendors_feasible": len(practical_points),
            "vendors": {
                p.hardware_target: {
                    "year": p.break_even_year,
                    "practical": p.is_practical,
                    "notes": p.notes,
                }
                for p in points
            },
        }

    return summary


# ---------------------------------------------------------------------------
# Visualisation (optional, Plotly-guarded)
# ---------------------------------------------------------------------------


def plot_resource_table(
    table: list[AlgorithmResource],
) -> Any:
    """Create interactive resource comparison plot.

    Parameters
    ----------
    table : list[AlgorithmResource]
        Resource table to visualise.

    Returns
    -------
    Plotly Figure or None if Plotly is not available.
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        return None

    rows = resource_table_to_dicts(table)

    algorithms = sorted({r["algorithm"] for r in rows})
    colors = {
        "QAE pricing": "#1f77b4",
        "QAOA optimisation": "#ff7f0e",
        "Quantum VaR": "#2ca02c",
        "Quantum credit risk": "#d62728",
    }

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[
            "Logical Qubits",
            "T-Gate Count",
            "Physical Qubits (d=17)",
            "Circuit Depth",
        ],
    )

    for algo in algorithms:
        algo_rows = [r for r in rows if r["algorithm"] == algo]
        sizes = [r["problem_size"] for r in algo_rows]
        color = colors.get(algo, "#7f7f7f")

        fig.add_trace(
            go.Scatter(
                x=sizes,
                y=[r["logical_qubits"] for r in algo_rows],
                name=algo,
                marker={"color": color},
                legendgroup=algo,
            ),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=sizes,
                y=[r["t_gates"] for r in algo_rows],
                name=algo,
                marker={"color": color},
                legendgroup=algo,
                showlegend=False,
            ),
            row=1, col=2,
        )
        fig.add_trace(
            go.Scatter(
                x=sizes,
                y=[r["physical_qubits_d17"] for r in algo_rows],
                name=algo,
                marker={"color": color},
                legendgroup=algo,
                showlegend=False,
            ),
            row=2, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=sizes,
                y=[r["circuit_depth"] for r in algo_rows],
                name=algo,
                marker={"color": color},
                legendgroup=algo,
                showlegend=False,
            ),
            row=2, col=2,
        )

    fig.update_layout(
        title="Quantum Finance Algorithm Resource Comparison",
        height=800,
    )
    fig.update_xaxes(title_text="Problem Size")
    fig.update_yaxes(type="log")

    return fig


def plot_break_even_timeline(
    timeline: list[BreakEvenPoint] | None = None,
) -> Any:
    """Create interactive break-even timeline plot.

    Parameters
    ----------
    timeline : list[BreakEvenPoint] | None
        Break-even data. If None, computes defaults.

    Returns
    -------
    Plotly Figure or None if Plotly is not available.
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        return None

    if timeline is None:
        timeline = compute_break_even_timeline()

    # Group by vendor
    by_vendor: dict[str, list[BreakEvenPoint]] = {}
    for bp in timeline:
        by_vendor.setdefault(bp.hardware_target, []).append(bp)

    fig = go.Figure()

    vendor_colors = {"IBM": "#1f77b4", "IonQ": "#ff7f0e", "QuEra": "#2ca02c"}

    for vendor, points in by_vendor.items():
        practical = [p for p in points if p.is_practical]
        color = vendor_colors.get(vendor, "#7f7f7f")

        if practical:
            fig.add_trace(go.Scatter(
                x=[p.break_even_year for p in practical],
                y=[p.algorithm for p in practical],
                mode="markers",
                name=f"{vendor} (feasible)",
                marker={
                    "size": 15,
                    "color": color,
                    "symbol": "circle",
                },
                text=[p.notes for p in practical],
                hoverinfo="text",
            ))

    fig.update_layout(
        title="Quantum Finance Algorithm Break-Even Timeline",
        xaxis_title="Year",
        yaxis_title="Algorithm",
        height=500,
    )

    return fig
