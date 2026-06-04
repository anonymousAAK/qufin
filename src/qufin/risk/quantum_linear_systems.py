"""Quantum linear systems (HHL) for risk factor analysis.

Encodes a covariance matrix as a Hamiltonian and solves
Sigma * x = b to determine factor exposures, portfolio sensitivities,
and risk decompositions. Provides condition number sensitivity analysis
for well-conditioned vs ill-conditioned portfolios, and comparison
against classical Cholesky decomposition.

References
----------
Harrow, Hassidim, Lloyd, "Quantum Algorithm for Linear Systems of
    Equations", PRL 103, 150502 (2009).
Rebentrost, Gupt, Bromley, "Quantum computational finance: Monte Carlo
    pricing of financial derivatives", PRA 98, 022321 (2018).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from qufin.backends.base import Backend
from qufin.utils.results import Result

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class HHLConfig:
    """Configuration for HHL-based linear system solver.

    Parameters
    ----------
    n_clock_qubits : int
        Number of clock qubits for phase estimation precision.
    shots : int
        Number of measurement shots.
    regularisation : float | None
        Tikhonov regularisation parameter. If None, auto-determined.
    seed : int | None
        Random seed for reproducibility.
    """

    n_clock_qubits: int = 4
    shots: int = 4096
    regularisation: float | None = None
    seed: int | None = 42


@dataclass
class ConditionNumberReport:
    """Condition number sensitivity report for a covariance matrix.

    Attributes
    ----------
    condition_number : float
        Condition number kappa(Sigma).
    eigenvalues : NDArray
        Sorted eigenvalues (descending).
    rank : int
        Effective rank.
    is_well_conditioned : bool
        True if kappa < 1e4.
    is_ill_conditioned : bool
        True if kappa > 1e6.
    spectral_gap : float
        Ratio of largest to second-largest eigenvalue.
    regularisation_applied : float
        Amount of regularisation applied (0 if none).
    effective_condition_after_reg : float
        Condition number after regularisation.
    """

    condition_number: float = 0.0
    eigenvalues: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(0)
    )
    rank: int = 0
    is_well_conditioned: bool = True
    is_ill_conditioned: bool = False
    spectral_gap: float = 0.0
    regularisation_applied: float = 0.0
    effective_condition_after_reg: float = 0.0


@dataclass
class LinearSystemResult(Result):
    """Result from solving a quantum linear system Sigma * x = b.

    Attributes
    ----------
    solution : NDArray
        Solution vector x.
    residual_norm : float
        ||Sigma x - b|| / ||b||.
    method : str
        ``"quantum_hhl"`` or ``"classical_cholesky"``.
    condition_report : ConditionNumberReport
        Condition number analysis of the system matrix.
    n_qubits_used : int
        Number of qubits used (0 for classical).
    circuit_depth_estimate : int
        Estimated circuit depth for HHL.
    """

    solution: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(0)
    )
    residual_norm: float = 0.0
    method: str = "quantum_hhl"
    condition_report: ConditionNumberReport = field(
        default_factory=ConditionNumberReport
    )
    n_qubits_used: int = 0
    circuit_depth_estimate: int = 0


@dataclass
class FactorExposureResult:
    """Factor exposure analysis result.

    Attributes
    ----------
    exposures : NDArray
        Factor exposure vector.
    risk_contributions : NDArray
        Per-factor risk contributions.
    total_risk : float
        Total portfolio risk (std dev).
    diversification_ratio : float
        Sum of individual risks / portfolio risk.
    method : str
        Solver method used.
    wall_time_s : float
        Wall-clock time.
    """

    exposures: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(0)
    )
    risk_contributions: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(0)
    )
    total_risk: float = 0.0
    diversification_ratio: float = 0.0
    method: str = "quantum_hhl"
    wall_time_s: float = 0.0


# ---------------------------------------------------------------------------
# Condition number analysis
# ---------------------------------------------------------------------------


def analyse_condition_number(
    sigma: NDArray[np.float64],
    regularisation: float = 0.0,
) -> ConditionNumberReport:
    """Analyse the condition number of a covariance matrix.

    Parameters
    ----------
    sigma : NDArray, shape (n, n)
        Covariance matrix.
    regularisation : float
        Tikhonov regularisation to apply before analysis.

    Returns
    -------
    ConditionNumberReport
    """
    sigma = np.asarray(sigma, dtype=np.float64)

    eigvals = np.linalg.eigvalsh(sigma)
    eigvals_sorted = np.sort(eigvals)[::-1]

    max_eig = float(eigvals_sorted[0]) if len(eigvals_sorted) > 0 else 1.0
    eps_threshold = max(max_eig * 1e-12, 1e-30)
    rank = int(np.sum(eigvals > eps_threshold))

    min_positive = float(eigvals_sorted[rank - 1]) if rank > 0 else 1e-30
    cond = max_eig / max(min_positive, 1e-30)

    # Spectral gap
    spectral_gap = 0.0
    if rank >= 2:
        spectral_gap = float(eigvals_sorted[0] / max(eigvals_sorted[1], 1e-30))

    # Condition after regularisation
    if regularisation > 0:
        eigvals_reg = eigvals + regularisation
        eigvals_reg_sorted = np.sort(eigvals_reg)[::-1]
        max_reg = float(eigvals_reg_sorted[0])
        min_reg = float(eigvals_reg_sorted[-1])
        cond_reg = max_reg / max(min_reg, 1e-30)
    else:
        cond_reg = cond

    return ConditionNumberReport(
        condition_number=cond,
        eigenvalues=eigvals_sorted,
        rank=rank,
        is_well_conditioned=cond < 1e4,
        is_ill_conditioned=cond > 1e6,
        spectral_gap=spectral_gap,
        regularisation_applied=regularisation,
        effective_condition_after_reg=cond_reg,
    )


def auto_regularise(
    sigma: NDArray[np.float64],
    target_condition: float = 1e4,
) -> tuple[NDArray[np.float64], float]:
    """Automatically regularise a covariance matrix.

    Adds epsilon * I to bring the condition number below the target.

    Parameters
    ----------
    sigma : NDArray, shape (n, n)
        Covariance matrix.
    target_condition : float
        Target maximum condition number.

    Returns
    -------
    (regularised_sigma, epsilon)
    """
    sigma = np.asarray(sigma, dtype=np.float64)
    eigvals = np.linalg.eigvalsh(sigma)
    max_eig = float(np.max(eigvals))
    min_eig = float(np.min(eigvals))

    current_cond = max_eig / max(min_eig, 1e-30)
    if current_cond <= target_condition and min_eig > 0:
        return sigma, 0.0

    # epsilon such that (max_eig + eps) / (min_eig + eps) <= target
    # => eps >= (max_eig - target * min_eig) / (target - 1)
    eps = max(
        (max_eig - target_condition * max(min_eig, 0.0))
        / max(target_condition - 1.0, 1.0),
        max_eig * 1e-8,
    )
    eps = max(eps, 1e-12)

    return sigma + eps * np.eye(sigma.shape[0]), eps


# ---------------------------------------------------------------------------
# Hamiltonian encoding
# ---------------------------------------------------------------------------


def encode_covariance_hamiltonian(
    sigma: NDArray[np.float64],
) -> dict[str, Any]:
    """Encode a covariance matrix as a Hamiltonian for HHL.

    Pads the matrix to power-of-2 dimension and analyses its structure
    for efficient Hamiltonian simulation.

    Parameters
    ----------
    sigma : NDArray, shape (n, n)
        Covariance matrix.

    Returns
    -------
    dict with keys:
        hamiltonian : padded Hermitian matrix
        n_qubits : qubits for the state register
        original_dim : original matrix dimension
        padded_dim : padded dimension (power of 2)
        sparsity : max nonzeros per row
        norm : spectral norm of the matrix
    """
    sigma = np.asarray(sigma, dtype=np.float64)
    n = sigma.shape[0]

    n_qubits = max(1, int(np.ceil(np.log2(n)))) if n > 1 else 1
    dim = 2 ** n_qubits

    H = np.zeros((dim, dim), dtype=np.float64)
    H[:n, :n] = sigma

    # Pad diagonal with average eigenvalue to avoid singularity
    avg_diag = float(np.mean(np.diag(sigma))) if n > 0 else 1.0
    for i in range(n, dim):
        H[i, i] = max(avg_diag, 1e-6)

    sparsity = int(np.max(np.sum(np.abs(H) > 1e-15, axis=1)))
    norm = float(np.linalg.norm(H, ord=2))

    return {
        "hamiltonian": H,
        "n_qubits": n_qubits,
        "original_dim": n,
        "padded_dim": dim,
        "sparsity": sparsity,
        "norm": norm,
    }


# ---------------------------------------------------------------------------
# HHL solver
# ---------------------------------------------------------------------------


def build_hhl_circuit(
    H: NDArray[np.float64],
    b: NDArray[np.float64],
    n_clock_qubits: int = 4,
) -> Any:
    """Build an HHL circuit for solving H x = b.

    Parameters
    ----------
    H : NDArray, shape (2^n, 2^n)
        Hermitian matrix.
    b : NDArray, shape (2^n,)
        Right-hand side vector.
    n_clock_qubits : int
        Clock qubits for phase estimation.

    Returns
    -------
    QuantumCircuit
    """
    from qiskit.circuit import QuantumCircuit
    from qiskit.circuit.library import StatePreparation

    H = np.asarray(H, dtype=np.complex128)
    b = np.asarray(b, dtype=np.complex128)

    dim = H.shape[0]
    n_state = int(np.log2(dim))
    assert 2 ** n_state == dim

    b_norm = np.linalg.norm(b)
    b_normed = b / b_norm if b_norm > 1e-15 else b

    n_total = n_clock_qubits + n_state + 1
    qc = QuantumCircuit(n_total, n_state)

    clock = list(range(n_clock_qubits))
    state = list(range(n_clock_qubits, n_clock_qubits + n_state))
    ancilla = n_clock_qubits + n_state

    # Prepare |b>
    qc.append(
        StatePreparation(b_normed.real.astype(np.float64)),
        state,
    )

    # Hadamard on clock
    for q in clock:
        qc.h(q)

    # Controlled unitaries e^{i H t_k}
    eigvals, eigvecs = np.linalg.eigh(H.real)

    for k, cq in enumerate(clock):
        t_k = 2 * np.pi * (2 ** k) / (2 ** n_clock_qubits)
        phases = np.exp(1j * eigvals * t_k)
        U_k = (eigvecs * phases) @ eigvecs.T

        gate = QuantumCircuit(n_state, name=f"e^(iHt_{k})")
        gate.unitary(U_k, range(n_state))  # type: ignore[attr-defined]
        controlled = gate.to_gate().control(1)
        qc.append(controlled, [cq, *state])

    # Inverse QFT on clock
    _inverse_qft(qc, clock)

    # Eigenvalue inversion rotations
    for k in range(1, 2 ** n_clock_qubits):
        lam = k / (2 ** n_clock_qubits)
        angle = 2 * np.arcsin(min(1.0, 1.0 / (2 ** n_clock_qubits * lam + 1e-15)))

        bits = format(k, f"0{n_clock_qubits}b")
        for idx, bit in enumerate(bits):
            if bit == "0":
                qc.x(clock[idx])

        if n_clock_qubits == 1:
            qc.cry(angle, clock[0], ancilla)
        else:
            from qiskit.circuit.library import RYGate
            gate = RYGate(angle).control(n_clock_qubits)
            qc.append(gate, [*clock, ancilla])

        for idx, bit in enumerate(bits):
            if bit == "0":
                qc.x(clock[idx])

    # QFT to undo
    _qft(qc, clock)

    qc.measure(state, list(range(n_state)))

    return qc


def _qft(qc: Any, qubits: list[int]) -> None:
    """Apply QFT in-place."""
    n = len(qubits)
    for i in range(n):
        qc.h(qubits[i])
        for j in range(i + 1, n):
            qc.cp(np.pi / (2 ** (j - i)), qubits[j], qubits[i])
    for i in range(n // 2):
        qc.swap(qubits[i], qubits[n - 1 - i])


def _inverse_qft(qc: Any, qubits: list[int]) -> None:
    """Apply inverse QFT in-place."""
    n = len(qubits)
    for i in range(n // 2):
        qc.swap(qubits[i], qubits[n - 1 - i])
    for i in range(n - 1, -1, -1):
        for j in range(n - 1, i, -1):
            qc.cp(-np.pi / (2 ** (j - i)), qubits[j], qubits[i])
        qc.h(qubits[i])


def hhl_solve(
    sigma: NDArray[np.float64],
    b: NDArray[np.float64],
    backend: Backend,
    config: HHLConfig | None = None,
) -> NDArray[np.float64]:
    """Solve ``Sigma * x = b`` (classical solve; HHL circuit for accounting only).

    .. warning::
       The returned solution is computed **classically** via
       ``numpy.linalg.solve``. This function constructs the HHL circuit
       (Harrow-Hassidim-Lloyd, 2009) purely for *resource accounting* — it is
       **not executed on** ``backend``, no quantum phase estimation is run, and
       the result does not depend on ``config.n_clock_qubits``. It therefore
       provides **no quantum speed-up** and must not be presented as a quantum
       linear-system solver; treat it as a classical reference with an HHL
       resource estimate attached. A real measurement-based HHL pipeline is
       tracked separately.

    Parameters
    ----------
    sigma : NDArray, shape (n, n)
        Covariance matrix.
    b : NDArray, shape (n,)
        Right-hand side vector (e.g. factor loadings).
    backend : Backend
        Accepted for interface compatibility and resource accounting; the
        circuit is **not** run on it.
    config : HHLConfig | None
        HHL configuration. Defaults to HHLConfig().

    Returns
    -------
    NDArray, shape (n,)
        Solution vector x.
    """
    if config is None:
        config = HHLConfig()

    sigma = np.asarray(sigma, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    n = sigma.shape[0]

    # Apply regularisation if needed
    if config.regularisation is not None and config.regularisation > 0:
        sigma_reg = sigma + config.regularisation * np.eye(n)
    else:
        report = analyse_condition_number(sigma)
        if report.is_ill_conditioned:
            sigma_reg, _ = auto_regularise(sigma)
        else:
            sigma_reg = sigma

    # Encode as Hamiltonian
    ham_info = encode_covariance_hamiltonian(sigma_reg)
    H = ham_info["hamiltonian"]
    dim = ham_info["padded_dim"]

    b_padded = np.zeros(dim, dtype=np.float64)
    b_padded[:n] = b

    # Build circuit (for resource accounting)
    _circuit = build_hhl_circuit(H, b_padded, config.n_clock_qubits)

    # Classical simulation of the HHL result
    try:
        x_padded = np.linalg.solve(H, b_padded)
    except np.linalg.LinAlgError:
        x_padded = np.linalg.lstsq(H, b_padded, rcond=None)[0]

    return x_padded[:n]


# ---------------------------------------------------------------------------
# Classical Cholesky baseline
# ---------------------------------------------------------------------------


def cholesky_solve(
    sigma: NDArray[np.float64],
    b: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Solve Sigma * x = b via Cholesky decomposition.

    Parameters
    ----------
    sigma : NDArray, shape (n, n)
        Symmetric positive definite covariance matrix.
    b : NDArray, shape (n,)
        Right-hand side vector.

    Returns
    -------
    NDArray, shape (n,)
        Solution vector x.

    Raises
    ------
    np.linalg.LinAlgError
        If sigma is not positive definite.
    """
    sigma = np.asarray(sigma, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    L = np.linalg.cholesky(sigma)
    # Solve L y = b
    y = np.linalg.solve(L, b)
    # Solve L^T x = y
    x = np.linalg.solve(L.T, y)
    return x


# ---------------------------------------------------------------------------
# High-level solver interface
# ---------------------------------------------------------------------------


def solve_linear_system(
    sigma: NDArray[np.float64],
    b: NDArray[np.float64],
    backend: Backend | None = None,
    config: HHLConfig | None = None,
    method: str = "quantum",
) -> LinearSystemResult:
    """Solve Sigma * x = b with quantum or classical method.

    Parameters
    ----------
    sigma : NDArray, shape (n, n)
        Covariance matrix.
    b : NDArray, shape (n,)
        Right-hand side vector.
    backend : Backend | None
        Quantum backend. Required if method="quantum".
    config : HHLConfig | None
        HHL configuration.
    method : str
        ``"quantum"`` or ``"classical"``.

    Returns
    -------
    LinearSystemResult
    """
    start = time.perf_counter()
    sigma = np.asarray(sigma, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    if config is None:
        config = HHLConfig()

    # Condition analysis
    reg = config.regularisation if config.regularisation is not None else 0.0
    cond_report = analyse_condition_number(sigma, regularisation=reg)

    if method == "quantum":
        if backend is None:
            from qufin.backends.mock import MockBackend
            backend = MockBackend(seed=config.seed or 42)

        x = hhl_solve(sigma, b, backend, config)
        method_name = "quantum_hhl"

        # Resource estimate
        ham = encode_covariance_hamiltonian(sigma)
        n_qubits = ham["n_qubits"] + config.n_clock_qubits + 1
        depth_est = int(
            ham["sparsity"] ** 2
            * np.log2(max(cond_report.condition_number, 2))
            * config.n_clock_qubits
        )
    else:
        try:
            x = cholesky_solve(sigma, b)
            method_name = "classical_cholesky"
        except np.linalg.LinAlgError:
            # Fall back to regularised solve
            sigma_reg, _ = auto_regularise(sigma)
            x = cholesky_solve(sigma_reg, b)
            method_name = "classical_cholesky_regularised"
        n_qubits = 0
        depth_est = 0

    # Residual
    b_norm = np.linalg.norm(b)
    residual = float(
        np.linalg.norm(sigma @ x - b) / max(b_norm, 1e-15)
    )

    wall_time = time.perf_counter() - start

    return LinearSystemResult(
        value=0.0,
        wall_time_s=wall_time,
        backend_id=backend.backend_id if backend else "classical",
        seed=config.seed,
        solution=x,
        residual_norm=residual,
        method=method_name,
        condition_report=cond_report,
        n_qubits_used=n_qubits,
        circuit_depth_estimate=depth_est,
    )


# ---------------------------------------------------------------------------
# Factor exposure analysis
# ---------------------------------------------------------------------------


def compute_factor_exposures(
    sigma: NDArray[np.float64],
    portfolio_weights: NDArray[np.float64],
    factor_loadings: NDArray[np.float64],
    backend: Backend | None = None,
    config: HHLConfig | None = None,
    method: str = "quantum",
) -> FactorExposureResult:
    """Compute factor exposures by solving Sigma * x = B^T w.

    Determines how portfolio risk decomposes across factors by solving
    the linear system Sigma_F * exposures = B^T * w, where Sigma_F is
    the factor covariance and B is the factor loading matrix.

    Parameters
    ----------
    sigma : NDArray, shape (k, k)
        Factor covariance matrix (k factors).
    portfolio_weights : NDArray, shape (n,)
        Portfolio weights for n assets.
    factor_loadings : NDArray, shape (n, k)
        Factor loading matrix (n assets x k factors).
    backend : Backend | None
        Quantum backend.
    config : HHLConfig | None
        HHL configuration.
    method : str
        ``"quantum"`` or ``"classical"``.

    Returns
    -------
    FactorExposureResult
    """
    start = time.perf_counter()

    sigma = np.asarray(sigma, dtype=np.float64)
    w = np.asarray(portfolio_weights, dtype=np.float64)
    B = np.asarray(factor_loadings, dtype=np.float64)

    # b = B^T w (factor-space RHS)
    b = B.T @ w

    # Solve Sigma * exposures = b
    result = solve_linear_system(sigma, b, backend, config, method)
    exposures = result.solution

    # Risk contributions: exposure_i * (Sigma @ exposures)_i
    sigma_x = sigma @ exposures
    risk_contribs = exposures * sigma_x

    # Total risk
    total_var = float(w @ (B @ sigma @ B.T) @ w) if B.shape[0] == len(w) else float(
        exposures @ sigma @ exposures
    )
    total_risk = float(np.sqrt(max(total_var, 0.0)))

    # Diversification ratio
    individual_risks = np.sqrt(np.maximum(np.diag(sigma), 0.0)) * np.abs(exposures)
    sum_individual = float(np.sum(individual_risks))
    div_ratio = sum_individual / max(total_risk, 1e-15)

    wall_time = time.perf_counter() - start

    return FactorExposureResult(
        exposures=exposures,
        risk_contributions=risk_contribs,
        total_risk=total_risk,
        diversification_ratio=div_ratio,
        method=result.method,
        wall_time_s=wall_time,
    )


# ---------------------------------------------------------------------------
# Sensitivity comparison
# ---------------------------------------------------------------------------


def condition_sensitivity_comparison(
    well_conditioned_cov: NDArray[np.float64],
    ill_conditioned_cov: NDArray[np.float64],
    b: NDArray[np.float64],
    backend: Backend | None = None,
    config: HHLConfig | None = None,
) -> dict[str, Any]:
    """Compare HHL performance on well-conditioned vs ill-conditioned systems.

    Parameters
    ----------
    well_conditioned_cov : NDArray, shape (n, n)
        Well-conditioned covariance matrix.
    ill_conditioned_cov : NDArray, shape (n, n)
        Ill-conditioned covariance matrix.
    b : NDArray, shape (n,)
        Right-hand side vector.
    backend : Backend | None
        Quantum backend.
    config : HHLConfig | None
        HHL configuration.

    Returns
    -------
    dict with comparison metrics.
    """
    if config is None:
        config = HHLConfig()

    # Well-conditioned
    result_well = solve_linear_system(
        well_conditioned_cov, b, backend, config, method="quantum",
    )
    result_well_classical = solve_linear_system(
        well_conditioned_cov, b, config=config, method="classical",
    )

    # Ill-conditioned
    result_ill = solve_linear_system(
        ill_conditioned_cov, b, backend, config, method="quantum",
    )
    result_ill_classical = solve_linear_system(
        ill_conditioned_cov, b, config=config, method="classical",
    )

    return {
        "well_conditioned": {
            "condition_number": result_well.condition_report.condition_number,
            "quantum_residual": result_well.residual_norm,
            "classical_residual": result_well_classical.residual_norm,
            "quantum_time": result_well.wall_time_s,
            "classical_time": result_well_classical.wall_time_s,
            "n_qubits": result_well.n_qubits_used,
        },
        "ill_conditioned": {
            "condition_number": result_ill.condition_report.condition_number,
            "quantum_residual": result_ill.residual_norm,
            "classical_residual": result_ill_classical.residual_norm,
            "quantum_time": result_ill.wall_time_s,
            "classical_time": result_ill_classical.wall_time_s,
            "n_qubits": result_ill.n_qubits_used,
        },
    }


def generate_test_covariance(
    n: int,
    condition_number: float = 10.0,
    seed: int | None = 42,
) -> NDArray[np.float64]:
    """Generate a covariance matrix with a specified condition number.

    Parameters
    ----------
    n : int
        Matrix dimension.
    condition_number : float
        Target condition number.
    seed : int | None
        Random seed.

    Returns
    -------
    NDArray, shape (n, n)
        Symmetric positive definite matrix with approximately the
        given condition number.
    """
    rng = np.random.default_rng(seed)

    # Generate random orthogonal matrix via QR
    A = rng.standard_normal((n, n))
    Q, _ = np.linalg.qr(A)

    # Eigenvalues geometrically spaced from 1 to condition_number
    if n == 1:
        eigvals = np.array([condition_number])
    else:
        eigvals = np.geomspace(1.0, condition_number, n)

    sigma = Q @ np.diag(eigvals) @ Q.T
    # Symmetrise
    sigma = (sigma + sigma.T) / 2.0

    return sigma
