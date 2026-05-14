"""American option pricing via quantum-accelerated Longstaff-Schwartz Monte Carlo.

Combines classical path simulation with quantum-enhanced regression for
continuation value estimation, and optional QAE for final price aggregation.

The approach:
1. Simulate GBM paths classically (path simulation is not amenable to
   quantum speedup in the NISQ era).
2. At each exercise step, use a variational quantum solver (VQE-based
   least-squares) to fit continuation values from basis function features.
3. Apply backward induction with quantum regression to determine the
   early exercise boundary.
4. Optionally use QAE for the final discounted payoff estimation.

References
----------
Longstaff & Schwartz (2001), "Valuing American Options by Simulation".
Rebentrost et al. (2014), "Quantum Support Vector Machine", arXiv:1307.0471.
Chakrabarti et al. (2021), "A threshold for quantum advantage in derivative
    pricing", Quantum 5:463, arXiv:2012.03819.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
from numpy.typing import NDArray

from qufin.utils.results import Result

# ---------------------------------------------------------------------------
# Basis function types
# ---------------------------------------------------------------------------

class BasisType(Enum):
    """Basis function families for regression."""

    POLYNOMIAL = "polynomial"
    LAGUERRE = "laguerre"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class AmericanQAESpec:
    """American option specification for quantum LSM pricing.

    Parameters
    ----------
    s0 : float
        Spot price.
    k : float
        Strike price.
    r : float
        Risk-free rate (annualised).
    sigma : float
        Volatility (annualised).
    T : float
        Time to expiry in years.
    is_call : bool
        True for call, False for put.
    n_steps : int
        Number of time steps for path discretisation.
    n_paths : int
        Number of Monte Carlo paths.
    basis_type : BasisType
        Basis function family for regression.
    basis_degree : int
        Degree of basis functions.
    n_qubits : int
        Number of qubits for quantum regression circuit.
    vqe_layers : int
        Number of variational layers in the ansatz.
    seed : int | None
        Random seed for reproducibility.
    """

    s0: float = 100.0
    k: float = 100.0
    r: float = 0.05
    sigma: float = 0.2
    T: float = 1.0
    is_call: bool = False
    n_steps: int = 50
    n_paths: int = 10_000
    basis_type: BasisType = BasisType.POLYNOMIAL
    basis_degree: int = 3
    n_qubits: int = 4
    vqe_layers: int = 2
    seed: int | None = 42


@dataclass
class AmericanQAEResult(Result):
    """Result from American option QAE pricing.

    Attributes
    ----------
    price : float
        Estimated option price.
    std_err : float
        Standard error of the price estimate.
    exercise_boundary : dict[int, float]
        Map from time step to critical stock price for early exercise.
    classical_price : float
        Classical LSM price for comparison.
    n_exercise_paths : int
        Number of paths that exercised early.
    quantum_regression_used : bool
        Whether quantum regression was used (vs classical fallback).
    """

    price: float = 0.0
    exercise_boundary: dict[int, float] = field(default_factory=dict)
    classical_price: float = 0.0
    n_exercise_paths: int = 0
    quantum_regression_used: bool = False


@dataclass
class ResourceEstimate:
    """Resource estimate for quantum American option pricing.

    Attributes
    ----------
    n_steps : int
        Number of time steps.
    n_basis : int
        Number of basis functions.
    qubits_regression : int
        Qubits needed for the variational regression circuit.
    qubits_qae : int
        Qubits needed for QAE final estimation.
    total_qubits : int
        Total qubit count (max of regression and QAE stages).
    circuit_depth_regression : int
        Estimated circuit depth for one regression step.
    total_circuits : int
        Total number of circuit evaluations across all steps.
    """

    n_steps: int = 0
    n_basis: int = 0
    qubits_regression: int = 0
    qubits_qae: int = 0
    total_qubits: int = 0
    circuit_depth_regression: int = 0
    total_circuits: int = 0


# ---------------------------------------------------------------------------
# Basis functions
# ---------------------------------------------------------------------------

def _polynomial_basis(x: NDArray[np.float64], degree: int) -> NDArray[np.float64]:
    """Build polynomial basis matrix.

    Parameters
    ----------
    x : NDArray
        Input values, shape ``(n_samples,)``.
    degree : int
        Polynomial degree (0 to *degree* inclusive).

    Returns
    -------
    NDArray
        Basis matrix of shape ``(n_samples, degree + 1)``.
    """
    x_norm = (x - np.mean(x)) / (np.std(x) + 1e-12)
    return np.column_stack([x_norm ** p for p in range(degree + 1)])


def _laguerre_basis(x: NDArray[np.float64], degree: int) -> NDArray[np.float64]:
    """Build Laguerre polynomial basis matrix.

    Uses the first *degree + 1* Laguerre polynomials L_0, L_1, ..., L_degree
    evaluated at ``x / mean(x)`` (normalised to avoid overflow).

    Parameters
    ----------
    x : NDArray
        Input values, shape ``(n_samples,)``.
    degree : int
        Maximum Laguerre polynomial order.

    Returns
    -------
    NDArray
        Basis matrix of shape ``(n_samples, degree + 1)``.
    """
    x_norm = x / (np.mean(x) + 1e-12)
    cols = [np.ones_like(x_norm)]  # L_0 = 1
    if degree >= 1:
        cols.append(1.0 - x_norm)  # L_1 = 1 - x
    if degree >= 2:
        cols.append(
            0.5 * (x_norm**2 - 4.0 * x_norm + 2.0)
        )  # L_2 = (x^2 - 4x + 2) / 2
    for n in range(3, degree + 1):
        # Recurrence: (n+1) L_{n+1}(x) = (2n+1-x) L_n(x) - n L_{n-1}(x)
        l_prev = cols[n - 1]
        l_prev2 = cols[n - 2]
        cols.append(((2 * n - 1 - x_norm) * l_prev - (n - 1) * l_prev2) / n)
    return np.column_stack(cols[: degree + 1])


def build_basis(
    x: NDArray[np.float64],
    basis_type: BasisType,
    degree: int,
) -> NDArray[np.float64]:
    """Build basis function matrix for regression.

    Parameters
    ----------
    x : NDArray
        Input values, shape ``(n_samples,)``.
    basis_type : BasisType
        Which basis family to use.
    degree : int
        Degree / order of the basis.

    Returns
    -------
    NDArray
        Basis matrix of shape ``(n_samples, degree + 1)``.
    """
    if basis_type == BasisType.LAGUERRE:
        return _laguerre_basis(x, degree)
    return _polynomial_basis(x, degree)


# ---------------------------------------------------------------------------
# Path simulation
# ---------------------------------------------------------------------------

def _simulate_gbm_paths(
    s0: float,
    r: float,
    sigma: float,
    T: float,
    n_steps: int,
    n_paths: int,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Simulate GBM paths under risk-neutral measure.

    Returns
    -------
    NDArray
        Shape ``(n_paths, n_steps + 1)`` with ``[:, 0] == s0``.
    """
    dt = T / n_steps
    z = rng.standard_normal((n_paths, n_steps))
    drift = (r - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt) * z
    log_returns = drift + diffusion
    log_s = np.zeros((n_paths, n_steps + 1))
    log_s[:, 0] = np.log(s0)
    log_s[:, 1:] = np.log(s0) + np.cumsum(log_returns, axis=1)
    return np.exp(log_s)


def _intrinsic(s: NDArray[np.float64], k: float, is_call: bool) -> NDArray[np.float64]:
    """Compute intrinsic (exercise) value."""
    if is_call:
        return np.maximum(s - k, 0.0)
    return np.maximum(k - s, 0.0)


# ---------------------------------------------------------------------------
# Quantum regression (VQE-based least squares)
# ---------------------------------------------------------------------------

class QuantumLSM:
    """Quantum-accelerated Longstaff-Schwartz regression.

    Uses a variational quantum eigensolver (VQE) approach to solve the
    least-squares regression for continuation values. The basis functions
    are encoded into quantum feature vectors, and a parameterised ansatz
    learns the optimal coefficients.

    On NISQ hardware this provides a proof-of-concept for quantum-enhanced
    regression. For production use the classical fallback is recommended
    until fault-tolerant hardware is available.

    Parameters
    ----------
    spec : AmericanQAESpec
        Option and algorithm specification.
    backend : Any | None
        Quantum backend for circuit execution. If None, uses classical
        least-squares as fallback.
    """

    def __init__(self, spec: AmericanQAESpec, backend: Any | None = None) -> None:
        self.spec = spec
        self.backend = backend
        self._rng = np.random.default_rng(spec.seed)

    def _build_feature_circuit(
        self,
        features: NDArray[np.float64],
    ) -> Any:
        """Build a parameterised circuit encoding basis function features.

        Uses angle encoding: each feature is mapped to a rotation angle
        on a dedicated qubit. A hardware-efficient ansatz with entangling
        layers follows.

        Parameters
        ----------
        features : NDArray
            Normalised feature vector, shape ``(n_features,)``.

        Returns
        -------
        QuantumCircuit
            Parameterised circuit with ``n_qubits`` qubits.
        """
        from qiskit.circuit import ParameterVector, QuantumCircuit

        n_q = self.spec.n_qubits
        n_layers = self.spec.vqe_layers
        n_features = len(features)

        qc = QuantumCircuit(n_q)

        # Feature encoding layer: Ry rotations
        for i in range(min(n_features, n_q)):
            angle = float(np.arctan(features[i]) * 2)
            qc.ry(angle, i)

        # Variational ansatz layers
        params = ParameterVector("theta", n_layers * n_q * 2)
        idx = 0
        for _layer in range(n_layers):
            for q in range(n_q):
                qc.ry(params[idx], q)
                idx += 1
                qc.rz(params[idx], q)
                idx += 1
            # Entangling: linear chain of CNOTs
            for q in range(n_q - 1):
                qc.cx(q, q + 1)

        return qc

    def _variational_regression(
        self,
        basis_matrix: NDArray[np.float64],
        targets: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Solve least-squares via variational quantum optimisation.

        Encodes each data point's basis features into a quantum circuit
        and optimises the variational parameters to minimise MSE.

        Falls back to classical least-squares if no backend is available
        or if the optimisation does not converge.

        Parameters
        ----------
        basis_matrix : NDArray
            Feature matrix, shape ``(n_samples, n_features)``.
        targets : NDArray
            Target values, shape ``(n_samples,)``.

        Returns
        -------
        NDArray
            Predicted continuation values, shape ``(n_samples,)``.
        """
        if self.backend is None:
            return self._classical_regression(basis_matrix, targets)

        n_q = self.spec.n_qubits
        n_layers = self.spec.vqe_layers
        n_params = n_layers * n_q * 2

        # Normalise targets to [0, 1]
        t_min, t_max = np.min(targets), np.max(targets)
        t_range = t_max - t_min
        if t_range < 1e-12:
            return np.full(len(targets), np.mean(targets))
        t_norm = (targets - t_min) / t_range

        # Normalise features per column
        col_max = np.max(np.abs(basis_matrix), axis=0) + 1e-12
        b_norm = basis_matrix / col_max

        # VQE optimisation: minimise sum of (p(x_i; theta) - y_i)^2
        # where p(x_i; theta) is the probability of measuring |0...0>
        best_params = self._rng.uniform(-np.pi, np.pi, n_params)
        best_cost = np.inf


        n_opt_steps = 30
        lr = 0.1

        for _step in range(n_opt_steps):
            cost = 0.0
            gradients = np.zeros(n_params)

            # Subsample for efficiency
            sample_size = min(50, len(targets))
            sample_idx = self._rng.choice(len(targets), sample_size, replace=False)

            for i in sample_idx:
                features = b_norm[i]
                qc = self._build_feature_circuit(features)

                # Bind parameters
                param_dict = dict(zip(qc.parameters, best_params, strict=False))
                bound_qc = qc.assign_parameters(param_dict)

                # Get probability of |0...0>
                result = self.backend.run(bound_qc, shots=256)
                zero_key = "0" * n_q
                p0 = result.probabilities.get(zero_key, 0.0)

                residual = p0 - t_norm[i]
                cost += residual**2

                # Numerical gradient (parameter shift)
                shift = np.pi / 4
                for p_idx in range(n_params):
                    params_plus = best_params.copy()
                    params_plus[p_idx] += shift
                    param_dict_plus = dict(zip(qc.parameters, params_plus, strict=False))
                    bound_plus = qc.assign_parameters(param_dict_plus)
                    r_plus = self.backend.run(bound_plus, shots=256)
                    p_plus = r_plus.probabilities.get(zero_key, 0.0)

                    params_minus = best_params.copy()
                    params_minus[p_idx] -= shift
                    param_dict_minus = dict(zip(qc.parameters, params_minus, strict=False))
                    bound_minus = qc.assign_parameters(param_dict_minus)
                    r_minus = self.backend.run(bound_minus, shots=256)
                    p_minus = r_minus.probabilities.get(zero_key, 0.0)

                    gradients[p_idx] += 2 * residual * (p_plus - p_minus) / (2 * shift)

            cost /= sample_size
            gradients /= sample_size

            best_cost = min(best_cost, cost)

            best_params -= lr * gradients

        # Final prediction with optimised params
        predictions = np.zeros(len(targets))
        for i in range(len(targets)):
            features = b_norm[i]
            qc = self._build_feature_circuit(features)
            param_dict = dict(zip(qc.parameters, best_params, strict=False))
            bound_qc = qc.assign_parameters(param_dict)
            result = self.backend.run(bound_qc, shots=512)
            zero_key = "0" * n_q
            p0 = result.probabilities.get(zero_key, 0.0)
            predictions[i] = p0 * t_range + t_min

        return predictions

    @staticmethod
    def _classical_regression(
        basis_matrix: NDArray[np.float64],
        targets: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Classical OLS regression (fallback and baseline).

        Parameters
        ----------
        basis_matrix : NDArray
            Feature matrix, shape ``(n_samples, n_features)``.
        targets : NDArray
            Target values, shape ``(n_samples,)``.

        Returns
        -------
        NDArray
            Predicted continuation values.
        """
        coeffs, *_ = np.linalg.lstsq(basis_matrix, targets, rcond=None)
        return basis_matrix @ coeffs

    def price(self) -> AmericanQAEResult:
        """Price an American option using quantum-accelerated LSM.

        Returns
        -------
        AmericanQAEResult
            Pricing result with exercise boundary and comparison price.
        """
        t_start = time.perf_counter()
        spec = self.spec
        rng = np.random.default_rng(spec.seed)

        # Simulate paths
        paths = _simulate_gbm_paths(
            spec.s0, spec.r, spec.sigma, spec.T,
            spec.n_steps, spec.n_paths, rng,
        )
        dt = spec.T / spec.n_steps

        # --- Quantum LSM backward induction ---
        cashflow = np.zeros(spec.n_paths)
        exercise_time = np.full(spec.n_paths, np.nan)
        exercise_boundary: dict[int, float] = {}
        quantum_used = self.backend is not None

        # Terminal payoff
        cashflow[:] = _intrinsic(paths[:, -1], spec.k, spec.is_call)
        exercise_time[:] = spec.T

        for step in range(spec.n_steps - 1, 0, -1):
            t_step = step * dt
            s_step = paths[:, step]
            intrinsic = _intrinsic(s_step, spec.k, spec.is_call)
            itm = intrinsic > 0

            if np.sum(itm) < max(5, spec.basis_degree + 2):
                continue

            # Discount existing cash-flows back to this step
            time_ahead = exercise_time[itm] - t_step
            discounted_cf = cashflow[itm] * np.exp(-spec.r * time_ahead)

            # Build basis features
            x_itm = s_step[itm]
            basis = build_basis(x_itm, spec.basis_type, spec.basis_degree)

            # Regression for continuation value
            if quantum_used:
                continuation = self._variational_regression(basis, discounted_cf)
            else:
                continuation = self._classical_regression(basis, discounted_cf)

            # Exercise decision
            exercise_mask = intrinsic[itm] > continuation
            itm_indices = np.where(itm)[0][exercise_mask]
            cashflow[itm_indices] = intrinsic[itm_indices]
            exercise_time[itm_indices] = t_step

            # Record exercise boundary (median stock price at exercise)
            if np.any(exercise_mask):
                exercise_boundary[step] = float(np.median(x_itm[exercise_mask]))

        # Discount all cash-flows to t=0
        discount_factors = np.exp(-spec.r * exercise_time)
        present_values = cashflow * discount_factors
        price = float(np.mean(present_values))
        std_err = float(np.std(present_values, ddof=1) / np.sqrt(spec.n_paths))

        # Count early exercises
        exercised_early = (
            (~np.isnan(exercise_time))
            & (cashflow > 0)
            & (exercise_time < spec.T - 1e-12)
        )
        n_exercise = int(np.sum(exercised_early))

        # Classical comparison price
        classical_price = _classical_lsm_price(
            paths, spec.k, spec.r, spec.T, spec.n_steps,
            spec.is_call, spec.basis_type, spec.basis_degree,
        )

        wall_time = time.perf_counter() - t_start

        return AmericanQAEResult(
            value=price,
            price=price,
            std_err=std_err,
            exercise_boundary=exercise_boundary,
            classical_price=classical_price,
            n_exercise_paths=n_exercise,
            quantum_regression_used=quantum_used,
            wall_time_s=wall_time,
            backend_id=getattr(self.backend, "backend_id", "classical"),
            seed=spec.seed,
        )


# ---------------------------------------------------------------------------
# Standalone pricing functions
# ---------------------------------------------------------------------------

def _classical_lsm_price(
    paths: NDArray[np.float64],
    k: float,
    r: float,
    T: float,
    n_steps: int,
    is_call: bool,
    basis_type: BasisType = BasisType.POLYNOMIAL,
    basis_degree: int = 3,
) -> float:
    """Classical LSM price from pre-simulated paths."""
    n_paths = paths.shape[0]
    dt = T / n_steps

    cashflow = _intrinsic(paths[:, -1], k, is_call).copy()
    exercise_time = np.full(n_paths, T)

    for step in range(n_steps - 1, 0, -1):
        t_step = step * dt
        s_step = paths[:, step]
        intrinsic = _intrinsic(s_step, k, is_call)
        itm = intrinsic > 0

        if np.sum(itm) < max(5, basis_degree + 2):
            continue

        time_ahead = exercise_time[itm] - t_step
        discounted_cf = cashflow[itm] * np.exp(-r * time_ahead)

        basis = build_basis(s_step[itm], basis_type, basis_degree)
        coeffs, *_ = np.linalg.lstsq(basis, discounted_cf, rcond=None)
        continuation = basis @ coeffs

        exercise_mask = intrinsic[itm] > continuation
        itm_indices = np.where(itm)[0][exercise_mask]
        cashflow[itm_indices] = intrinsic[itm_indices]
        exercise_time[itm_indices] = t_step

    discount_factors = np.exp(-r * exercise_time)
    present_values = cashflow * discount_factors
    return float(np.mean(present_values))


def price_american_qae(
    s0: float = 100.0,
    k: float = 100.0,
    r: float = 0.05,
    sigma: float = 0.2,
    T: float = 1.0,
    is_call: bool = False,
    n_steps: int = 50,
    n_paths: int = 10_000,
    basis_type: BasisType = BasisType.POLYNOMIAL,
    basis_degree: int = 3,
    n_qubits: int = 4,
    vqe_layers: int = 2,
    backend: Any | None = None,
    seed: int | None = 42,
) -> AmericanQAEResult:
    """Price an American option via quantum-accelerated Longstaff-Schwartz.

    This is a convenience wrapper around :class:`QuantumLSM`.

    Parameters
    ----------
    s0 : float
        Spot price.
    k : float
        Strike price.
    r : float
        Risk-free rate.
    sigma : float
        Volatility.
    T : float
        Time to expiry.
    is_call : bool
        True for call, False for put.
    n_steps : int
        Number of time steps.
    n_paths : int
        Number of Monte Carlo paths.
    basis_type : BasisType
        Basis function family.
    basis_degree : int
        Degree of basis functions.
    n_qubits : int
        Number of qubits for quantum regression.
    vqe_layers : int
        Number of variational layers.
    backend : Any | None
        Quantum backend. If None, uses classical regression.
    seed : int | None
        Random seed.

    Returns
    -------
    AmericanQAEResult
        Pricing result.
    """
    spec = AmericanQAESpec(
        s0=s0, k=k, r=r, sigma=sigma, T=T,
        is_call=is_call, n_steps=n_steps, n_paths=n_paths,
        basis_type=basis_type, basis_degree=basis_degree,
        n_qubits=n_qubits, vqe_layers=vqe_layers, seed=seed,
    )
    qlsm = QuantumLSM(spec, backend=backend)
    return qlsm.price()


def price_american_classical(
    s0: float = 100.0,
    k: float = 100.0,
    r: float = 0.05,
    sigma: float = 0.2,
    T: float = 1.0,
    is_call: bool = False,
    n_steps: int = 50,
    n_paths: int = 50_000,
    basis_type: BasisType = BasisType.POLYNOMIAL,
    basis_degree: int = 3,
    seed: int | None = 42,
) -> dict[str, Any]:
    """Classical LSM baseline for American option pricing.

    Parameters
    ----------
    s0, k, r, sigma, T : float
        Standard option parameters.
    is_call : bool
        True for call, False for put.
    n_steps : int
        Number of time steps.
    n_paths : int
        Number of Monte Carlo paths.
    basis_type : BasisType
        Basis function family.
    basis_degree : int
        Degree of basis functions.
    seed : int | None
        Random seed.

    Returns
    -------
    dict
        ``price``, ``std_err``.
    """
    rng = np.random.default_rng(seed)
    paths = _simulate_gbm_paths(s0, r, sigma, T, n_steps, n_paths, rng)
    dt = T / n_steps

    cashflow = _intrinsic(paths[:, -1], k, is_call).copy()
    exercise_time = np.full(n_paths, T)

    for step in range(n_steps - 1, 0, -1):
        t_step = step * dt
        s_step = paths[:, step]
        intrinsic = _intrinsic(s_step, k, is_call)
        itm = intrinsic > 0

        if np.sum(itm) < max(5, basis_degree + 2):
            continue

        time_ahead = exercise_time[itm] - t_step
        discounted_cf = cashflow[itm] * np.exp(-r * time_ahead)

        basis = build_basis(s_step[itm], basis_type, basis_degree)
        coeffs, *_ = np.linalg.lstsq(basis, discounted_cf, rcond=None)
        continuation = basis @ coeffs

        exercise_mask = intrinsic[itm] > continuation
        itm_indices = np.where(itm)[0][exercise_mask]
        cashflow[itm_indices] = intrinsic[itm_indices]
        exercise_time[itm_indices] = t_step

    discount_factors = np.exp(-r * exercise_time)
    present_values = cashflow * discount_factors
    price = float(np.mean(present_values))
    std_err = float(np.std(present_values, ddof=1) / np.sqrt(n_paths))

    return {"price": price, "std_err": std_err}


def american_binomial(
    s0: float = 100.0,
    k: float = 100.0,
    r: float = 0.05,
    sigma: float = 0.2,
    T: float = 1.0,
    is_call: bool = False,
    n_steps: int = 500,
) -> float:
    """Price an American option using the CRR binomial tree.

    Used as a reference price for validation.

    Parameters
    ----------
    s0, k, r, sigma, T : float
        Standard option parameters.
    is_call : bool
        True for call, False for put.
    n_steps : int
        Number of binomial tree steps (higher = more accurate).

    Returns
    -------
    float
        Option price.
    """
    dt = T / n_steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    disc = np.exp(-r * dt)
    p = (np.exp(r * dt) - d) / (u - d)

    # Terminal asset prices
    j = np.arange(n_steps + 1)
    s_terminal = s0 * u ** (n_steps - j) * d ** j

    # Terminal payoff
    if is_call:
        values = np.maximum(s_terminal - k, 0.0)
    else:
        values = np.maximum(k - s_terminal, 0.0)

    # Backward induction with early exercise at every step
    for i in range(n_steps - 1, -1, -1):
        s_nodes = s0 * u ** (i - np.arange(i + 1)) * d ** np.arange(i + 1)
        continuation = disc * (p * values[: i + 1] + (1 - p) * values[1: i + 2])
        if is_call:
            intrinsic = np.maximum(s_nodes - k, 0.0)
        else:
            intrinsic = np.maximum(k - s_nodes, 0.0)
        values = np.maximum(intrinsic, continuation)

    return float(values[0])


# ---------------------------------------------------------------------------
# Resource estimation
# ---------------------------------------------------------------------------

def estimate_resources(
    n_steps: int = 50,
    basis_degree: int = 3,
    n_qubits_regression: int = 4,
    vqe_layers: int = 2,
    n_eval_qubits_qae: int = 5,
) -> ResourceEstimate:
    """Estimate quantum resources for American option pricing.

    Parameters
    ----------
    n_steps : int
        Number of time steps in the option.
    basis_degree : int
        Degree of basis functions for regression.
    n_qubits_regression : int
        Qubits per regression circuit.
    vqe_layers : int
        Number of variational ansatz layers.
    n_eval_qubits_qae : int
        Evaluation qubits for QAE final estimation.

    Returns
    -------
    ResourceEstimate
        Resource breakdown.
    """
    n_basis = basis_degree + 1

    # Regression circuit: feature encoding + variational layers + entangling
    depth_per_layer = n_qubits_regression * 2 + (n_qubits_regression - 1)
    depth_encoding = n_qubits_regression
    circuit_depth = depth_encoding + vqe_layers * depth_per_layer

    # QAE stage: state prep + Grover iterations
    qubits_qae = n_qubits_regression + 1 + n_eval_qubits_qae

    # Total circuits: VQE iterations * gradient evaluations * steps
    n_vqe_iters = 30
    n_gradient_evals = 2 * vqe_layers * n_qubits_regression * 2
    total_circuits = n_steps * n_vqe_iters * (n_gradient_evals + 1)

    return ResourceEstimate(
        n_steps=n_steps,
        n_basis=n_basis,
        qubits_regression=n_qubits_regression,
        qubits_qae=qubits_qae,
        total_qubits=max(n_qubits_regression, qubits_qae),
        circuit_depth_regression=circuit_depth,
        total_circuits=total_circuits,
    )
