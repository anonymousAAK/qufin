"""D-Wave quantum annealing backend for combinatorial optimization.

Provides a qufin backend interface wrapping D-Wave Ocean SDK, supporting
QUBO submission to Advantage2 QPU, hybrid CQM solver for mixed-integer
problems, and minor-embedding onto Pegasus topology.

Includes a benchmark framework for comparing annealing vs gate-based
portfolio optimization at 15/25/50 asset scales.

Requires: pip install dwave-ocean-sdk
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
from numpy.typing import NDArray

from qufin.backends.base import Backend, CircuitResult

# ---------------------------------------------------------------------------
# Optional dependency guard
# ---------------------------------------------------------------------------

try:
    import dimod as _dimod
    import dwave.system as _dwave_system
    import minorminer as _minorminer
    from dwave.system import DWaveSampler, EmbeddingComposite

    _HAS_DWAVE = True
except ImportError:
    _dimod = None  # type: ignore[assignment]
    _dwave_system = None  # type: ignore[assignment]
    _minorminer = None  # type: ignore[assignment]
    _HAS_DWAVE = False

try:
    from dwave.system import LeapHybridCQMSampler as _LeapHybridCQMSampler

    _HAS_HYBRID = True
except ImportError:
    _LeapHybridCQMSampler = None  # type: ignore[assignment]
    _HAS_HYBRID = False


def _require_dwave() -> None:
    """Raise if dwave-ocean-sdk is not installed."""
    if not _HAS_DWAVE:
        raise ImportError(
            "D-Wave Ocean SDK is required. Install with: pip install dwave-ocean-sdk"
        )


# ---------------------------------------------------------------------------
# Topology and solver enums
# ---------------------------------------------------------------------------


class DWaveTopology(Enum):
    """D-Wave QPU topology."""

    PEGASUS = "pegasus"
    ZEPHYR = "zephyr"
    CHIMERA = "chimera"


class SolverType(Enum):
    """D-Wave solver type."""

    QPU = "qpu"
    HYBRID_CQM = "hybrid_cqm"
    HYBRID_BQM = "hybrid_bqm"
    SIMULATED = "simulated"


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------


@dataclass
class DWaveConfig:
    """Configuration for D-Wave backend.

    Parameters
    ----------
    solver_type : SolverType
        Which solver to use.
    topology : DWaveTopology
        QPU topology (for embedding calculations).
    num_reads : int
        Number of annealing reads (samples).
    annealing_time : float
        Annealing time in microseconds.
    chain_strength : float | None
        Chain strength for embedding. None = auto-scale.
    auto_scale : bool
        Whether to auto-scale QUBO coefficients.
    token : str | None
        D-Wave API token. None = use environment variable.
    endpoint : str | None
        D-Wave Leap endpoint URL.
    solver_name : str | None
        Specific solver name (e.g., "Advantage2_prototype2.3").
    """

    solver_type: SolverType = SolverType.SIMULATED
    topology: DWaveTopology = DWaveTopology.PEGASUS
    num_reads: int = 1000
    annealing_time: float = 20.0
    chain_strength: float | None = None
    auto_scale: bool = True
    token: str | None = None
    endpoint: str | None = None
    solver_name: str | None = None


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class AnnealingTiming:
    """Timing breakdown for an annealing run.

    Attributes
    ----------
    total_seconds : float
        Total wall-clock time.
    embedding_seconds : float
        Time spent on minor-embedding.
    sampling_seconds : float
        Time spent sampling on the QPU.
    post_processing_seconds : float
        Time spent on post-processing.
    """

    total_seconds: float = 0.0
    embedding_seconds: float = 0.0
    sampling_seconds: float = 0.0
    post_processing_seconds: float = 0.0


@dataclass
class AnnealingResult:
    """Result from a D-Wave annealing run.

    Attributes
    ----------
    best_sample : dict[int, int]
        Best solution found (variable -> value).
    best_energy : float
        Energy of the best solution.
    all_samples : list[dict[int, int]]
        All samples returned.
    all_energies : NDArray[np.float64]
        Energy of each sample.
    num_occurrences : NDArray[np.int64]
        How many times each sample was observed.
    timing : AnnealingTiming
        Timing breakdown.
    chain_break_fraction : float
        Fraction of samples with broken chains.
    metadata : dict[str, Any]
        Additional solver metadata.
    """

    best_sample: dict[int, int] = field(default_factory=dict)
    best_energy: float = 0.0
    all_samples: list[dict[int, int]] = field(default_factory=list)
    all_energies: NDArray[np.float64] = field(
        default_factory=lambda: np.array([], dtype=np.float64)
    )
    num_occurrences: NDArray[np.int64] = field(
        default_factory=lambda: np.array([], dtype=np.int64)
    )
    timing: AnnealingTiming = field(default_factory=AnnealingTiming)
    chain_break_fraction: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EmbeddingInfo:
    """Information about the minor-embedding of a QUBO onto hardware.

    Attributes
    ----------
    logical_qubits : int
        Number of logical variables.
    physical_qubits : int
        Number of physical qubits used.
    max_chain_length : int
        Maximum chain length in the embedding.
    avg_chain_length : float
        Average chain length.
    embedding_time_seconds : float
        Time to find the embedding.
    embedding : dict[int, list[int]] | None
        The actual embedding map (logical -> physical qubits).
    """

    logical_qubits: int = 0
    physical_qubits: int = 0
    max_chain_length: int = 0
    avg_chain_length: float = 0.0
    embedding_time_seconds: float = 0.0
    embedding: dict[int, list[int]] | None = None


# ---------------------------------------------------------------------------
# D-Wave Backend
# ---------------------------------------------------------------------------


class DWaveBackend(Backend):
    """D-Wave quantum annealing backend.

    Supports QUBO submission to Advantage2 QPU, hybrid CQM solver,
    and simulated annealing for local testing.
    """

    def __init__(self, config: DWaveConfig | None = None) -> None:
        self._config = config or DWaveConfig()
        self._sampler: Any = None
        self._last_embedding_info: EmbeddingInfo | None = None

    @property
    def backend_id(self) -> str:
        return f"dwave_{self._config.solver_type.value}"

    @property
    def config(self) -> DWaveConfig:
        return self._config

    @property
    def last_embedding_info(self) -> EmbeddingInfo | None:
        """Return embedding info from the last QPU solve."""
        return self._last_embedding_info

    def is_simulator(self) -> bool:
        return self._config.solver_type == SolverType.SIMULATED

    def run(self, circuit: Any, shots: int = 1024) -> CircuitResult:
        """Not applicable for annealing — raises NotImplementedError."""
        raise NotImplementedError(
            "DWaveBackend is an annealing backend; use solve_qubo() or "
            "solve_cqm() instead of run()."
        )

    def statevector(self, circuit: Any) -> NDArray[np.complex128]:
        """Not applicable for annealing — raises NotImplementedError."""
        raise NotImplementedError(
            "DWaveBackend is an annealing backend; statevector is not supported."
        )

    def _get_sampler(self) -> Any:
        """Get or create the D-Wave sampler."""
        if self._sampler is not None:
            return self._sampler

        if self._config.solver_type == SolverType.SIMULATED:
            _require_dwave()
            self._sampler = _dimod.SimulatedAnnealingSampler()
            return self._sampler

        if self._config.solver_type == SolverType.QPU:
            _require_dwave()
            kwargs: dict[str, Any] = {}
            if self._config.token:
                kwargs["token"] = self._config.token
            if self._config.endpoint:
                kwargs["endpoint"] = self._config.endpoint
            if self._config.solver_name:
                kwargs["solver"] = self._config.solver_name

            qpu = DWaveSampler(**kwargs)
            self._sampler = EmbeddingComposite(qpu)
            return self._sampler

        if self._config.solver_type == SolverType.HYBRID_BQM:
            _require_dwave()
            from dwave.system import LeapHybridSampler

            kwargs = {}
            if self._config.token:
                kwargs["token"] = self._config.token
            self._sampler = LeapHybridSampler(**kwargs)
            return self._sampler

        raise ValueError(f"Unsupported solver type: {self._config.solver_type}")

    def solve_qubo(
        self,
        Q: NDArray[np.float64] | dict[tuple[int, int], float],
        num_reads: int | None = None,
    ) -> AnnealingResult:
        """Submit a QUBO to the D-Wave solver.

        Parameters
        ----------
        Q : QUBO matrix (numpy array) or dictionary {(i,j): value}.
        num_reads : Number of annealing reads. Defaults to config value.

        Returns
        -------
        AnnealingResult
        """
        _require_dwave()

        if num_reads is None:
            num_reads = self._config.num_reads

        # Convert numpy array to dict format
        if isinstance(Q, np.ndarray):
            Q_dict = _matrix_to_qubo_dict(Q)
        else:
            Q_dict = Q

        bqm = _dimod.BinaryQuadraticModel.from_qubo(Q_dict)

        sampler = self._get_sampler()
        total_start = time.perf_counter()

        # Embedding timing (for QPU)
        embed_start = time.perf_counter()
        if self._config.solver_type == SolverType.QPU:
            self._last_embedding_info = self._compute_embedding_info(bqm)
        embed_time = time.perf_counter() - embed_start

        # Sampling
        sample_start = time.perf_counter()
        kwargs: dict[str, Any] = {"num_reads": num_reads}

        if self._config.solver_type == SolverType.QPU:
            if self._config.chain_strength is not None:
                kwargs["chain_strength"] = self._config.chain_strength
            kwargs["annealing_time"] = self._config.annealing_time
            if self._config.auto_scale:
                kwargs["auto_scale"] = True

        sampleset = sampler.sample(bqm, **kwargs)
        sample_time = time.perf_counter() - sample_start

        total_time = time.perf_counter() - total_start

        return self._parse_sampleset(
            sampleset,
            AnnealingTiming(
                total_seconds=total_time,
                embedding_seconds=embed_time,
                sampling_seconds=sample_time,
                post_processing_seconds=total_time - embed_time - sample_time,
            ),
        )

    def solve_cqm(
        self,
        objective: dict[tuple[int, int], float] | NDArray[np.float64],
        constraints: list[dict[str, Any]] | None = None,
        variable_types: dict[int, str] | None = None,
        time_limit: float = 5.0,
    ) -> AnnealingResult:
        """Solve a constrained quadratic model using the hybrid CQM solver.

        Parameters
        ----------
        objective : QUBO matrix or dict for the objective function.
        constraints : List of constraint dicts with keys:
            - "lhs": dict of {var: coefficient}
            - "sense": "<=", ">=", or "=="
            - "rhs": float
            - "label": str (optional)
        variable_types : {var_index: "BINARY" | "INTEGER"}.
            Defaults to all BINARY.
        time_limit : Time limit in seconds for the hybrid solver.

        Returns
        -------
        AnnealingResult
        """
        _require_dwave()
        if not _HAS_HYBRID:
            raise ImportError(
                "LeapHybridCQMSampler is required. "
                "Install with: pip install dwave-ocean-sdk"
            )

        if isinstance(objective, np.ndarray):
            obj_dict = _matrix_to_qubo_dict(objective)
        else:
            obj_dict = objective

        cqm = _dimod.ConstrainedQuadraticModel()

        # Determine variables
        all_vars: set[int] = set()
        for (i, j), _ in obj_dict.items():
            all_vars.add(i)
            all_vars.add(j)

        var_types = variable_types or {}
        cqm_vars: dict[int, Any] = {}
        for v in sorted(all_vars):
            vtype = var_types.get(v, "BINARY")
            if vtype == "INTEGER":
                cqm_vars[v] = _dimod.Integer(f"x{v}")
            else:
                cqm_vars[v] = _dimod.Binary(f"x{v}")

        # Build objective
        obj_expr = 0
        for (i, j), val in obj_dict.items():
            if i == j:
                obj_expr += val * cqm_vars[i]
            else:
                obj_expr += val * cqm_vars[i] * cqm_vars[j]

        cqm.set_objective(obj_expr)

        # Add constraints
        if constraints:
            for idx, constr in enumerate(constraints):
                lhs = constr["lhs"]
                sense = constr["sense"]
                rhs = constr["rhs"]
                label = constr.get("label", f"c{idx}")

                expr = sum(coef * cqm_vars[v] for v, coef in lhs.items())

                if sense == "<=":
                    cqm.add_constraint(expr <= rhs, label=label)
                elif sense == ">=":
                    cqm.add_constraint(expr >= rhs, label=label)
                elif sense == "==":
                    cqm.add_constraint(expr == rhs, label=label)

        # Solve
        total_start = time.perf_counter()
        kwargs: dict[str, Any] = {}
        if self._config.token:
            kwargs["token"] = self._config.token
        sampler = _LeapHybridCQMSampler(**kwargs)
        sampleset = sampler.sample_cqm(cqm, time_limit=time_limit)
        total_time = time.perf_counter() - total_start

        # Parse results — CQM returns variable names like "x0", "x1"
        feasible = sampleset.filter(lambda s: s.is_feasible)
        if len(feasible) > 0:
            working_set = feasible
        else:
            working_set = sampleset

        samples = []
        energies = []
        occurrences = []

        for datum in working_set.data():
            sample_dict = {}
            for var_name, val in datum.sample.items():
                # Extract integer index from variable name
                idx = int(var_name[1:])
                sample_dict[idx] = int(val)
            samples.append(sample_dict)
            energies.append(float(datum.energy))
            occurrences.append(int(datum.num_occurrences))

        if not samples:
            return AnnealingResult(
                timing=AnnealingTiming(total_seconds=total_time),
            )

        best_idx = int(np.argmin(energies))
        return AnnealingResult(
            best_sample=samples[best_idx],
            best_energy=energies[best_idx],
            all_samples=samples,
            all_energies=np.array(energies, dtype=np.float64),
            num_occurrences=np.array(occurrences, dtype=np.int64),
            timing=AnnealingTiming(total_seconds=total_time),
        )

    def find_embedding(
        self, Q: NDArray[np.float64] | dict[tuple[int, int], float]
    ) -> EmbeddingInfo:
        """Find a minor-embedding for the given QUBO onto the target topology.

        Parameters
        ----------
        Q : QUBO matrix or dictionary.

        Returns
        -------
        EmbeddingInfo
        """
        _require_dwave()

        if isinstance(Q, np.ndarray):
            Q_dict = _matrix_to_qubo_dict(Q)
        else:
            Q_dict = Q

        bqm = _dimod.BinaryQuadraticModel.from_qubo(Q_dict)
        return self._compute_embedding_info(bqm)

    def _compute_embedding_info(self, bqm: Any) -> EmbeddingInfo:
        """Compute embedding info for a BQM."""
        _require_dwave()

        source_graph = list(bqm.quadratic) + [(v, v) for v in bqm.linear]
        n_logical = len(bqm.variables)

        # Get target graph
        target_graph = self._get_target_graph()

        embed_start = time.perf_counter()
        try:
            embedding = _minorminer.find_embedding(source_graph, target_graph)
            embed_time = time.perf_counter() - embed_start

            if not embedding:
                return EmbeddingInfo(
                    logical_qubits=n_logical,
                    embedding_time_seconds=embed_time,
                )

            chain_lengths = [len(chain) for chain in embedding.values()]
            n_physical = sum(chain_lengths)

            return EmbeddingInfo(
                logical_qubits=n_logical,
                physical_qubits=n_physical,
                max_chain_length=max(chain_lengths) if chain_lengths else 0,
                avg_chain_length=float(np.mean(chain_lengths)) if chain_lengths else 0.0,
                embedding_time_seconds=embed_time,
                embedding=embedding,
            )
        except Exception:
            embed_time = time.perf_counter() - embed_start
            return EmbeddingInfo(
                logical_qubits=n_logical,
                embedding_time_seconds=embed_time,
            )

    def _get_target_graph(self) -> Any:
        """Get the target hardware graph for the configured topology."""
        _require_dwave()
        import dwave_networkx as dnx

        if self._config.topology == DWaveTopology.PEGASUS:
            return dnx.pegasus_graph(16)
        elif self._config.topology == DWaveTopology.ZEPHYR:
            return dnx.zephyr_graph(4)
        elif self._config.topology == DWaveTopology.CHIMERA:
            return dnx.chimera_graph(16, 16, 4)
        else:
            raise ValueError(f"Unknown topology: {self._config.topology}")

    def _parse_sampleset(
        self, sampleset: Any, timing: AnnealingTiming
    ) -> AnnealingResult:
        """Parse a dimod SampleSet into an AnnealingResult."""
        samples = []
        energies = []
        occurrences = []
        chain_breaks = 0
        total_samples = 0

        for datum in sampleset.data():
            sample_dict = {int(k): int(v) for k, v in datum.sample.items()}
            samples.append(sample_dict)
            energies.append(float(datum.energy))
            occ = int(datum.num_occurrences)
            occurrences.append(occ)
            total_samples += occ

            # Check chain breaks if available
            if hasattr(datum, "chain_break_fraction") and datum.chain_break_fraction > 0:
                    chain_breaks += occ

        if not samples:
            return AnnealingResult(timing=timing)

        energies_arr = np.array(energies, dtype=np.float64)
        best_idx = int(np.argmin(energies_arr))

        cbf = chain_breaks / total_samples if total_samples > 0 else 0.0

        return AnnealingResult(
            best_sample=samples[best_idx],
            best_energy=energies[best_idx],
            all_samples=samples,
            all_energies=energies_arr,
            num_occurrences=np.array(occurrences, dtype=np.int64),
            timing=timing,
            chain_break_fraction=cbf,
        )


# ---------------------------------------------------------------------------
# Simulated annealing fallback (no D-Wave SDK needed)
# ---------------------------------------------------------------------------


def simulated_annealing_solve(
    Q: NDArray[np.float64],
    num_reads: int = 100,
    num_sweeps: int = 1000,
    beta_range: tuple[float, float] = (0.1, 10.0),
    seed: int | None = None,
) -> AnnealingResult:
    """Pure-numpy simulated annealing QUBO solver (no D-Wave SDK needed).

    Useful for local testing and benchmarking without D-Wave access.

    Parameters
    ----------
    Q : QUBO matrix, shape (n, n).
    num_reads : Number of independent runs.
    num_sweeps : Number of sweeps per run.
    beta_range : (beta_start, beta_end) for the annealing schedule.
    seed : Random seed.

    Returns
    -------
    AnnealingResult
    """
    n = Q.shape[0]
    rng = np.random.default_rng(seed)

    beta_start, beta_end = beta_range
    betas = np.linspace(beta_start, beta_end, num_sweeps)

    samples = []
    energies = []

    total_start = time.perf_counter()

    for _ in range(num_reads):
        x = rng.integers(0, 2, size=n, dtype=np.int64)
        energy = float(x @ Q @ x)

        for sweep in range(num_sweeps):
            beta = betas[sweep]
            for bit in range(n):
                # Flip bit and compute energy change
                x[bit] = 1 - x[bit]
                new_energy = float(x @ Q @ x)
                delta = new_energy - energy

                if delta < 0 or rng.random() < np.exp(-beta * delta):
                    energy = new_energy
                else:
                    x[bit] = 1 - x[bit]  # Reject flip

        samples.append({i: int(x[i]) for i in range(n)})
        energies.append(energy)

    total_time = time.perf_counter() - total_start

    energies_arr = np.array(energies, dtype=np.float64)
    best_idx = int(np.argmin(energies_arr))

    return AnnealingResult(
        best_sample=samples[best_idx],
        best_energy=energies[best_idx],
        all_samples=samples,
        all_energies=energies_arr,
        num_occurrences=np.ones(len(samples), dtype=np.int64),
        timing=AnnealingTiming(total_seconds=total_time),
    )


# ---------------------------------------------------------------------------
# Benchmark framework: annealing vs gate-based
# ---------------------------------------------------------------------------


@dataclass
class PortfolioBenchmarkConfig:
    """Configuration for portfolio optimization benchmarks.

    Parameters
    ----------
    n_assets : int
        Number of assets (15, 25, or 50).
    gamma : float
        Risk aversion parameter.
    cardinality : int | None
        Cardinality constraint (select exactly K assets).
    budget_penalty : float
        Penalty for budget constraint.
    seed : int
        Random seed for generating synthetic data.
    """

    n_assets: int = 15
    gamma: float = 1.0
    cardinality: int | None = None
    budget_penalty: float = 10.0
    seed: int = 42


@dataclass
class BenchmarkMetrics:
    """Metrics for comparing optimization approaches.

    Attributes
    ----------
    method : str
        Optimization method name.
    n_assets : int
        Number of assets in the problem.
    best_energy : float
        Best objective value found.
    approximation_ratio : float
        Ratio of solution quality to best known.
    feasibility_rate : float
        Fraction of solutions satisfying constraints.
    time_to_solution : float
        Total wall-clock time in seconds.
    embedding_overhead : float
        Time spent on embedding (annealing only).
    cost_estimate : float
        Estimated cost in USD.
    best_bitstring : str
        Best solution bitstring.
    n_feasible : int
        Number of feasible solutions found.
    n_total : int
        Total solutions sampled.
    """

    method: str = ""
    n_assets: int = 0
    best_energy: float = float("inf")
    approximation_ratio: float = 0.0
    feasibility_rate: float = 0.0
    time_to_solution: float = 0.0
    embedding_overhead: float = 0.0
    cost_estimate: float = 0.0
    best_bitstring: str = ""
    n_feasible: int = 0
    n_total: int = 0


def generate_portfolio_problem(
    config: PortfolioBenchmarkConfig,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Generate a synthetic portfolio optimization problem.

    Returns
    -------
    mu : Expected returns, shape (n_assets,).
    cov : Covariance matrix, shape (n_assets, n_assets).
    """
    rng = np.random.default_rng(config.seed)
    n = config.n_assets

    # Generate realistic expected returns (annualized, ~5-15%)
    mu = rng.uniform(0.05, 0.15, size=n)

    # Generate a positive-definite covariance matrix
    # Use factor model: cov = B @ B^T + diag(sigma^2)
    n_factors = min(5, n)
    B = rng.standard_normal((n, n_factors)) * 0.1
    idiosyncratic = rng.uniform(0.01, 0.05, size=n)
    cov = B @ B.T + np.diag(idiosyncratic**2)

    # Ensure symmetry
    cov = (cov + cov.T) / 2

    return mu, cov


def benchmark_annealing(
    Q: NDArray[np.float64],
    config: PortfolioBenchmarkConfig,
    dwave_config: DWaveConfig | None = None,
    feasibility_fn: Any = None,
) -> BenchmarkMetrics:
    """Benchmark D-Wave annealing on a portfolio QUBO.

    Parameters
    ----------
    Q : QUBO matrix.
    config : Portfolio benchmark config.
    dwave_config : D-Wave configuration. Defaults to simulated.
    feasibility_fn : Optional callable(bitstring) -> bool for feasibility check.

    Returns
    -------
    BenchmarkMetrics
    """
    if dwave_config is None:
        dwave_config = DWaveConfig(solver_type=SolverType.SIMULATED)

    if dwave_config.solver_type == SolverType.SIMULATED and not _HAS_DWAVE:
        # Use pure-numpy fallback
        result = simulated_annealing_solve(
            Q, num_reads=dwave_config.num_reads, seed=config.seed
        )
    else:
        backend = DWaveBackend(dwave_config)
        result = backend.solve_qubo(Q, num_reads=dwave_config.num_reads)

    return _compute_benchmark_metrics(
        result=result,
        Q=Q,
        config=config,
        method=f"dwave_{dwave_config.solver_type.value}",
        feasibility_fn=feasibility_fn,
    )


def benchmark_exhaustive(
    Q: NDArray[np.float64],
    config: PortfolioBenchmarkConfig,
    feasibility_fn: Any = None,
    max_qubits: int = 20,
) -> BenchmarkMetrics:
    """Exhaustive search benchmark (only for small problems).

    Parameters
    ----------
    Q : QUBO matrix.
    config : Benchmark config.
    feasibility_fn : Optional feasibility checker.
    max_qubits : Maximum number of qubits for exhaustive search.

    Returns
    -------
    BenchmarkMetrics
    """
    n = Q.shape[0]
    if n > max_qubits:
        raise ValueError(
            f"Exhaustive search not feasible for {n} qubits (max {max_qubits})."
        )

    start = time.perf_counter()

    best_energy = float("inf")
    best_bitstring = ""
    n_feasible = 0
    n_total = 2**n

    for i in range(n_total):
        bits = format(i, f"0{n}b")
        x = np.array([int(b) for b in bits], dtype=np.float64)
        energy = float(x @ Q @ x)

        is_feasible = True
        if feasibility_fn is not None:
            is_feasible = feasibility_fn(bits)

        if is_feasible:
            n_feasible += 1

        if energy < best_energy:
            best_energy = energy
            best_bitstring = bits

    elapsed = time.perf_counter() - start

    return BenchmarkMetrics(
        method="exhaustive",
        n_assets=config.n_assets,
        best_energy=best_energy,
        approximation_ratio=1.0,
        feasibility_rate=n_feasible / n_total if n_total > 0 else 0.0,
        time_to_solution=elapsed,
        best_bitstring=best_bitstring,
        n_feasible=n_feasible,
        n_total=n_total,
    )


def compare_portfolio_methods(
    config: PortfolioBenchmarkConfig,
    dwave_config: DWaveConfig | None = None,
    include_exhaustive: bool = False,
) -> list[BenchmarkMetrics]:
    """Compare annealing and gate-based methods on portfolio optimization.

    Parameters
    ----------
    config : Portfolio benchmark configuration.
    dwave_config : D-Wave configuration.
    include_exhaustive : Whether to include exhaustive search (small problems only).

    Returns
    -------
    list of BenchmarkMetrics for each method tested.
    """
    from qufin.portfolio.qubo import PortfolioQUBO

    mu, cov = generate_portfolio_problem(config)

    qubo = PortfolioQUBO(
        mu=mu,
        cov=cov,
        gamma=config.gamma,
        cardinality=config.cardinality,
        budget_penalty=config.budget_penalty,
    )
    Q = qubo.build_matrix()

    def feasibility_fn(bits: str) -> bool:
        checks = qubo.feasibility_check(bits)
        return all(checks.values()) if checks else True

    results: list[BenchmarkMetrics] = []

    # Annealing benchmark
    annealing_result = benchmark_annealing(
        Q, config, dwave_config, feasibility_fn
    )
    results.append(annealing_result)

    # Exhaustive (if small enough)
    if include_exhaustive and config.n_assets <= 20:
        try:
            exact_result = benchmark_exhaustive(Q, config, feasibility_fn)
            results.append(exact_result)

            # Update approximation ratios relative to exact solution
            if exact_result.best_energy != 0:
                for r in results:
                    if r.method != "exhaustive":
                        r.approximation_ratio = (
                            exact_result.best_energy / r.best_energy
                            if r.best_energy != 0
                            else 0.0
                        )
        except ValueError:
            pass

    return results


def estimate_dwave_cost(
    n_qubits: int,
    num_reads: int = 1000,
    solver_type: SolverType = SolverType.QPU,
) -> float:
    """Estimate the cost in USD for a D-Wave job.

    Based on D-Wave Leap pricing as of 2025:
    - QPU: ~$0.00019 per second of QPU access time
    - Hybrid: ~$0.05 per second of solver time

    Parameters
    ----------
    n_qubits : Problem size.
    num_reads : Number of samples.
    solver_type : Solver type.

    Returns
    -------
    Estimated cost in USD.
    """
    if solver_type == SolverType.QPU:
        # QPU time ≈ 20us * num_reads + overhead
        qpu_time_s = (20e-6 * num_reads) + 0.01  # 10ms overhead
        return qpu_time_s * 0.00019
    elif solver_type in (SolverType.HYBRID_CQM, SolverType.HYBRID_BQM):
        # Hybrid: minimum 5s, scales with problem size
        min_time = max(5.0, n_qubits * 0.1)
        return min_time * 0.05
    else:
        return 0.0  # Simulated is free


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _matrix_to_qubo_dict(
    Q: NDArray[np.float64],
) -> dict[tuple[int, int], float]:
    """Convert a numpy QUBO matrix to dict format."""
    n = Q.shape[0]
    Q_dict: dict[tuple[int, int], float] = {}
    for i in range(n):
        for j in range(i, n):
            val = Q[i, j]
            if i != j:
                val += Q[j, i]  # Combine upper and lower triangular
            if val != 0:
                Q_dict[(i, j)] = float(val)
    return Q_dict


def _compute_benchmark_metrics(
    result: AnnealingResult,
    Q: NDArray[np.float64],
    config: PortfolioBenchmarkConfig,
    method: str,
    feasibility_fn: Any = None,
) -> BenchmarkMetrics:
    """Compute benchmark metrics from an annealing result."""
    n = Q.shape[0]

    # Count feasible solutions
    n_feasible = 0
    n_total = len(result.all_samples)

    for sample in result.all_samples:
        bits = "".join(str(sample.get(i, 0)) for i in range(n))
        if feasibility_fn is None or feasibility_fn(bits):
            n_feasible += 1

    best_bits = "".join(str(result.best_sample.get(i, 0)) for i in range(n))

    return BenchmarkMetrics(
        method=method,
        n_assets=config.n_assets,
        best_energy=result.best_energy,
        approximation_ratio=0.0,  # Set later when exact solution is known
        feasibility_rate=n_feasible / n_total if n_total > 0 else 0.0,
        time_to_solution=result.timing.total_seconds,
        embedding_overhead=result.timing.embedding_seconds,
        cost_estimate=estimate_dwave_cost(
            n, config.n_assets, SolverType.QPU
        ),
        best_bitstring=best_bits,
        n_feasible=n_feasible,
        n_total=n_total,
    )
