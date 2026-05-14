"""Implied volatility surface modeling via quantum and classical regression.

Provides quantum kernel (QSVM) and variational quantum circuit (VQC)
regression for interpolating / extrapolating the IV surface, plus
classical baselines (SABR, SVI parameterisation).

References
----------
- Gatheral, *The Volatility Surface*, Wiley, 2006.
- Havlicek et al., Nature 567, 209-212 (2019).
- Hagan et al., Wilmott Magazine, Jan 2002 (SABR).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from qufin.backends.base import Backend

# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class IVSurfaceData:
    """Container for implied volatility surface observations.

    Parameters
    ----------
    strikes : array of shape (n,)
        Strike prices.
    expiries : array of shape (n,)
        Times to expiry (years).
    ivs : array of shape (n,)
        Observed implied volatilities.
    spot : float
        Current spot price of the underlying.
    """

    strikes: NDArray[np.float64]
    expiries: NDArray[np.float64]
    ivs: NDArray[np.float64]
    spot: float = 100.0

    def __post_init__(self) -> None:
        self.strikes = np.asarray(self.strikes, dtype=np.float64)
        self.expiries = np.asarray(self.expiries, dtype=np.float64)
        self.ivs = np.asarray(self.ivs, dtype=np.float64)
        n = len(self.strikes)
        if len(self.expiries) != n or len(self.ivs) != n:
            raise ValueError("strikes, expiries, ivs must have equal length")

    @property
    def moneyness(self) -> NDArray[np.float64]:
        """Log-moneyness ln(K/S)."""
        return np.log(self.strikes / self.spot)

    @property
    def features(self) -> NDArray[np.float64]:
        """Feature matrix of shape (n, 2): [K/S, T]."""
        return np.column_stack([self.strikes / self.spot, self.expiries])


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

@dataclass
class SurfaceMetrics:
    """Out-of-sample evaluation metrics."""

    rmse: float
    mae: float
    max_error: float
    n_samples: int


def evaluate_surface(
    y_true: NDArray[np.float64],
    y_pred: NDArray[np.float64],
) -> SurfaceMetrics:
    """Compute RMSE, MAE, and max absolute error.

    Parameters
    ----------
    y_true : array of shape (n,)
        Ground-truth implied volatilities.
    y_pred : array of shape (n,)
        Predicted implied volatilities.

    Returns
    -------
    SurfaceMetrics
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    errors = y_true - y_pred
    return SurfaceMetrics(
        rmse=float(np.sqrt(np.mean(errors ** 2))),
        mae=float(np.mean(np.abs(errors))),
        max_error=float(np.max(np.abs(errors))),
        n_samples=len(y_true),
    )


# ---------------------------------------------------------------------------
# Classical baseline: SABR model
# ---------------------------------------------------------------------------

def _sabr_iv(
    f: float,
    k: float,
    T: float,
    alpha: float,
    beta: float,
    rho: float,
    nu: float,
) -> float:
    """Hagan et al. SABR approximate implied volatility formula.

    Parameters
    ----------
    f : float
        Forward price.
    k : float
        Strike price.
    T : float
        Time to expiry.
    alpha, beta, rho, nu : float
        SABR parameters.

    Returns
    -------
    float
        Approximate implied volatility.
    """
    eps = 1e-12
    if abs(f - k) < eps:
        # ATM formula
        fmid = f
        term1 = alpha / (fmid ** (1.0 - beta))
        bracket = (
            1.0
            + (
                ((1.0 - beta) ** 2 / 24.0) * alpha ** 2 / (fmid ** (2.0 - 2.0 * beta))
                + 0.25 * rho * beta * nu * alpha / (fmid ** (1.0 - beta))
                + (2.0 - 3.0 * rho ** 2) / 24.0 * nu ** 2
            )
            * T
        )
        return float(term1 * bracket)

    fk = f * k
    fk_beta = fk ** ((1.0 - beta) / 2.0)
    log_fk = np.log(f / k)

    z = (nu / alpha) * fk_beta * log_fk
    x_z = np.log((np.sqrt(1.0 - 2.0 * rho * z + z ** 2) + z - rho) / (1.0 - rho))

    if abs(x_z) < eps:
        x_z = 1.0
        z = 1.0

    A = alpha / (
        fk_beta
        * (
            1.0
            + (1.0 - beta) ** 2 / 24.0 * log_fk ** 2
            + (1.0 - beta) ** 4 / 1920.0 * log_fk ** 4
        )
    )

    B = 1.0 + (
        (1.0 - beta) ** 2 / 24.0 * alpha ** 2 / (fk ** (1.0 - beta))
        + 0.25 * rho * beta * nu * alpha / fk_beta
        + (2.0 - 3.0 * rho ** 2) / 24.0 * nu ** 2
    ) * T

    return float(A * (z / x_z) * B)


class SABRModel:
    """SABR stochastic volatility model calibration.

    Calibrates the four SABR parameters (alpha, beta, rho, nu)
    to market IV data for each expiry slice.
    """

    def __init__(self, beta: float = 0.5) -> None:
        self.beta = beta
        self._params: dict[str, float] = {}
        self._calibrated = False

    def calibrate(
        self,
        data: IVSurfaceData,
        r: float = 0.0,
    ) -> SABRModel:
        """Calibrate SABR parameters to observed IV surface.

        Uses a single global calibration across all expiry slices.

        Parameters
        ----------
        data : IVSurfaceData
            Observed IV surface data.
        r : float
            Risk-free rate (for computing forward).

        Returns
        -------
        self
        """
        forwards = data.spot * np.exp(r * data.expiries)

        def objective(params: NDArray[np.float64]) -> float:
            alpha, rho, nu = params
            alpha = max(alpha, 1e-8)
            nu = max(nu, 1e-8)
            rho = np.clip(rho, -0.999, 0.999)
            total = 0.0
            for i in range(len(data.strikes)):
                try:
                    iv_model = _sabr_iv(
                        forwards[i], data.strikes[i], data.expiries[i],
                        alpha, self.beta, rho, nu,
                    )
                    total += (iv_model - data.ivs[i]) ** 2
                except (ValueError, ZeroDivisionError):
                    total += 1.0
            return float(total)

        result = minimize(
            objective,
            x0=np.array([0.3, -0.5, 0.4]),
            method="Nelder-Mead",
            options={"maxiter": 2000, "xatol": 1e-8, "fatol": 1e-10},
        )

        self._params = {
            "alpha": max(float(result.x[0]), 1e-8),
            "rho": float(np.clip(result.x[1], -0.999, 0.999)),
            "nu": max(float(result.x[2]), 1e-8),
        }
        self._calibrated = True
        return self

    def predict(
        self,
        strikes: NDArray[np.float64],
        expiries: NDArray[np.float64],
        spot: float,
        r: float = 0.0,
    ) -> NDArray[np.float64]:
        """Predict IV for given strikes and expiries."""
        if not self._calibrated:
            raise RuntimeError("Must call calibrate() first")
        strikes = np.asarray(strikes, dtype=np.float64)
        expiries = np.asarray(expiries, dtype=np.float64)
        forwards = spot * np.exp(r * expiries)
        result = np.zeros(len(strikes), dtype=np.float64)
        for i in range(len(strikes)):
            result[i] = _sabr_iv(
                forwards[i], strikes[i], expiries[i],
                self._params["alpha"], self.beta,
                self._params["rho"], self._params["nu"],
            )
        return result

    @property
    def params(self) -> dict[str, float]:
        """Return calibrated parameters."""
        if not self._calibrated:
            raise RuntimeError("Must call calibrate() first")
        return {**self._params, "beta": self.beta}


# ---------------------------------------------------------------------------
# Classical baseline: SVI parameterisation
# ---------------------------------------------------------------------------

class SVIModel:
    """Stochastic Volatility Inspired (SVI) parameterisation.

    The raw SVI parameterisation models total implied variance
    w(k) = a + b * (rho * (k - m) + sqrt((k - m)^2 + sigma^2))
    where k = ln(K/S) is log-moneyness.

    Calibration is performed per-expiry slice.  For prediction at
    expiries not in the training set, parameters are linearly
    interpolated between the two nearest calibrated slices.

    References
    ----------
    Gatheral & Jacquier, Quant. Finance 14(1), 59-71 (2014).
    """

    def __init__(self) -> None:
        self._slice_params: dict[float, NDArray[np.float64]] = {}
        self._sorted_expiries: NDArray[np.float64] | None = None
        self._calibrated = False

    @staticmethod
    def _total_variance(
        k: NDArray[np.float64],
        a: float,
        b: float,
        rho: float,
        m: float,
        sigma: float,
    ) -> NDArray[np.float64]:
        """Raw SVI total implied variance."""
        return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma ** 2))

    @classmethod
    def _calibrate_slice(
        cls,
        k: NDArray[np.float64],
        w_market: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Calibrate a single expiry slice, returning [a, b, rho, m, sigma]."""

        def objective(params: NDArray[np.float64]) -> float:
            a, b, rho_raw, m, sigma = params
            b = max(b, 1e-8)
            sigma = max(sigma, 1e-8)
            rho_val = np.tanh(rho_raw)
            w_model = cls._total_variance(k, a, b, rho_val, m, sigma)
            return float(np.sum((w_model - w_market) ** 2))

        best_result = None
        best_cost = np.inf
        w_mean = float(np.mean(w_market))
        x0_list = [
            np.array([w_mean, 0.1, 0.0, 0.0, 0.1]),
            np.array([w_mean * 0.5, 0.2, -0.5, 0.0, 0.2]),
            np.array([w_mean, 0.05, 0.5, 0.0, 0.05]),
        ]
        for x0 in x0_list:
            res = minimize(
                objective, x0=x0, method="Nelder-Mead",
                options={"maxiter": 10000, "xatol": 1e-12, "fatol": 1e-14},
            )
            if res.fun < best_cost:
                best_cost = res.fun
                best_result = res

        # Refine
        res = minimize(
            objective, x0=best_result.x, method="Powell",
            options={"maxiter": 10000, "ftol": 1e-14},
        )
        if res.fun < best_cost:
            best_result = res

        p = best_result.x
        return np.array([
            p[0], max(p[1], 1e-8), np.tanh(p[2]), p[3], max(p[4], 1e-8)
        ])

    def calibrate(self, data: IVSurfaceData) -> SVIModel:
        """Calibrate SVI parameters per-expiry slice.

        Parameters
        ----------
        data : IVSurfaceData
            Observed IV surface data.

        Returns
        -------
        self
        """
        k_all = data.moneyness
        unique_expiries = np.unique(data.expiries)

        self._slice_params = {}
        for T in unique_expiries:
            mask = data.expiries == T
            k_slice = k_all[mask]
            w_slice = data.ivs[mask] ** 2 * T
            params = self._calibrate_slice(k_slice, w_slice)
            self._slice_params[float(T)] = params

        self._sorted_expiries = np.sort(unique_expiries)
        self._calibrated = True
        return self

    def _get_params_for_expiry(self, T: float) -> NDArray[np.float64]:
        """Get (possibly interpolated) SVI params for a given expiry."""
        if T in self._slice_params:
            return self._slice_params[T]

        exp_arr = self._sorted_expiries
        if exp_arr[0] >= T:
            return self._slice_params[float(exp_arr[0])]
        if exp_arr[-1] <= T:
            return self._slice_params[float(exp_arr[-1])]

        # Linear interpolation between two nearest slices
        idx = int(np.searchsorted(exp_arr, T)) - 1
        T_lo, T_hi = float(exp_arr[idx]), float(exp_arr[idx + 1])
        w = (T - T_lo) / (T_hi - T_lo)
        p_lo = self._slice_params[T_lo]
        p_hi = self._slice_params[T_hi]
        return (1 - w) * p_lo + w * p_hi

    def predict(
        self,
        strikes: NDArray[np.float64],
        expiries: NDArray[np.float64],
        spot: float,
    ) -> NDArray[np.float64]:
        """Predict IV for given strikes and expiries."""
        if not self._calibrated:
            raise RuntimeError("Must call calibrate() first")
        strikes = np.asarray(strikes, dtype=np.float64)
        expiries = np.asarray(expiries, dtype=np.float64)
        k = np.log(strikes / spot)
        result = np.zeros(len(strikes), dtype=np.float64)
        for i in range(len(strikes)):
            T = float(expiries[i])
            p = self._get_params_for_expiry(T)
            a, b, rho_val, m, sigma = p
            w = float(self._total_variance(
                np.array([k[i]]), a, b, rho_val, m, sigma,
            )[0])
            w = max(w, 1e-12)
            result[i] = np.sqrt(w / max(T, 1e-12))
        return result

    @property
    def params(self) -> NDArray[np.float64]:
        """Return calibrated parameters for the first slice [a, b, rho, m, sigma]."""
        if not self._calibrated:
            raise RuntimeError("Must call calibrate() first")
        first_key = float(self._sorted_expiries[0])
        return self._slice_params[first_key].copy()


# ---------------------------------------------------------------------------
# Quantum regression: QSVM kernel regression
# ---------------------------------------------------------------------------

class _QSVMRegressor:
    """Quantum kernel SVR for IV surface regression.

    Uses the ZZ feature map quantum kernel with sklearn's SVR.
    """

    def __init__(
        self,
        n_qubits: int,
        backend: Backend,
        reps: int = 2,
        C: float = 1.0,
        epsilon: float = 0.01,
    ) -> None:
        self.n_qubits = n_qubits
        self.backend = backend
        self.reps = reps
        self.C = C
        self.epsilon = epsilon
        self._X_train: NDArray[np.float64] | None = None
        self._svr: Any = None

    def _kernel_value(
        self, x1: NDArray[np.float64], x2: NDArray[np.float64]
    ) -> float:
        """Compute quantum kernel between two feature vectors."""
        from qufin.ml.kernels import quantum_kernel

        return quantum_kernel(x1, x2, self.n_qubits, self.backend, self.reps)

    def _kernel_matrix(
        self,
        X1: NDArray[np.float64],
        X2: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        """Compute (cross-)kernel matrix."""
        if X2 is None:
            from qufin.ml.kernels import quantum_kernel_matrix

            return quantum_kernel_matrix(
                X1, self.n_qubits, self.backend, self.reps
            )
        n1, n2 = X1.shape[0], X2.shape[0]
        K = np.zeros((n1, n2), dtype=np.float64)
        for i in range(n1):
            for j in range(n2):
                K[i, j] = self._kernel_value(X1[i], X2[j])
        return K

    def fit(
        self, X: NDArray[np.float64], y: NDArray[np.float64]
    ) -> _QSVMRegressor:
        """Fit the quantum SVR."""
        from sklearn.svm import SVR

        self._X_train = X.copy()
        K = self._kernel_matrix(X)
        self._svr = SVR(kernel="precomputed", C=self.C, epsilon=self.epsilon)
        self._svr.fit(K, y)
        return self

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Predict IV values."""
        assert self._X_train is not None, "Must call fit() first"
        K = self._kernel_matrix(X, self._X_train)
        return self._svr.predict(K)


# ---------------------------------------------------------------------------
# Quantum regression: VQC regressor
# ---------------------------------------------------------------------------

class _VQCRegressor:
    """Variational quantum circuit for IV surface regression.

    Uses angle encoding and a TwoLocal ansatz. The circuit output
    (expectation of |0...0>) is mapped to IV via a linear rescaling.
    """

    def __init__(
        self,
        n_qubits: int,
        backend: Backend,
        n_layers: int = 3,
        maxiter: int = 300,
        seed: int | None = 42,
    ) -> None:
        self.n_qubits = n_qubits
        self.backend = backend
        self.n_layers = n_layers
        self.maxiter = maxiter
        self._rng = np.random.default_rng(seed)
        self._params: NDArray[np.float64] | None = None
        self._scale_min: float = 0.0
        self._scale_range: float = 1.0
        self._feat_mean: NDArray[np.float64] | None = None
        self._feat_std: NDArray[np.float64] | None = None

    def _n_params(self) -> int:
        """Total variational parameters."""
        return 2 * self.n_qubits * self.n_layers + self.n_qubits

    def _build_circuit(
        self,
        x: NDArray[np.float64],
        params: NDArray[np.float64],
    ) -> Any:
        """Build VQC circuit with angle encoding and TwoLocal ansatz."""
        from qiskit.circuit import QuantumCircuit

        n = self.n_qubits
        qc = QuantumCircuit(n)

        # Angle encoding: R_Y(pi * x_i) for each feature
        for i in range(min(len(x), n)):
            qc.ry(float(np.pi * x[i]), i)

        # TwoLocal ansatz
        idx = 0
        for _layer in range(self.n_layers):
            for i in range(n):
                qc.ry(float(params[idx]), i)
                idx += 1
                qc.rz(float(params[idx]), i)
                idx += 1
            for i in range(n - 1):
                qc.cx(i, i + 1)
            if n > 2:
                qc.cx(n - 1, 0)

        # Final rotation
        for i in range(n):
            qc.ry(float(params[idx]), i)
            idx += 1

        return qc

    def _circuit_output(
        self, x: NDArray[np.float64], params: NDArray[np.float64]
    ) -> float:
        """Probability of |0...0> state."""
        qc = self._build_circuit(x, params)
        sv = self.backend.statevector(qc)
        return float(np.abs(sv[0]) ** 2)

    def _normalize_features(
        self, X: NDArray[np.float64], fit: bool = False
    ) -> NDArray[np.float64]:
        """Standardise features to zero mean / unit variance."""
        if fit:
            self._feat_mean = X.mean(axis=0)
            self._feat_std = X.std(axis=0)
            self._feat_std[self._feat_std < 1e-12] = 1.0
        return (X - self._feat_mean) / self._feat_std

    def fit(
        self, X: NDArray[np.float64], y: NDArray[np.float64]
    ) -> _VQCRegressor:
        """Train the VQC regressor."""
        X_norm = self._normalize_features(X, fit=True)

        # Scale targets to [0, 1] for probability output
        self._scale_min = float(y.min())
        self._scale_range = float(y.max() - y.min())
        if self._scale_range < 1e-12:
            self._scale_range = 1.0
        y_scaled = (y - self._scale_min) / self._scale_range

        def loss(params: NDArray[np.float64]) -> float:
            total = 0.0
            for i in range(X_norm.shape[0]):
                pred = self._circuit_output(X_norm[i], params)
                total += (pred - y_scaled[i]) ** 2
            return float(total / X_norm.shape[0])

        init = self._rng.uniform(0, 2 * np.pi, self._n_params())
        result = minimize(
            loss, init, method="COBYLA",
            options={"maxiter": self.maxiter},
        )
        self._params = result.x
        return self

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Predict IV values."""
        assert self._params is not None, "Must call fit() first"
        X_norm = self._normalize_features(X)
        preds = np.array([
            self._circuit_output(X_norm[i], self._params)
            for i in range(X_norm.shape[0])
        ])
        return preds * self._scale_range + self._scale_min


# ---------------------------------------------------------------------------
# Main class: QuantumIVSurface
# ---------------------------------------------------------------------------

@dataclass
class QuantumIVSurfaceConfig:
    """Configuration for the quantum IV surface model.

    Parameters
    ----------
    n_qubits : int
        Number of qubits for quantum models (4-8 recommended).
    method : str
        Regression method: 'qsvm', 'vqc', 'sabr', 'svi', or 'all'.
    reps : int
        Feature map repetitions (QSVM only).
    n_layers : int
        Ansatz layers (VQC only).
    C : float
        Regularisation parameter (QSVM only).
    epsilon : float
        SVR epsilon (QSVM only).
    maxiter : int
        Optimizer iterations.
    seed : int or None
        Random seed.
    """

    n_qubits: int = 4
    method: Literal["qsvm", "vqc", "sabr", "svi", "all"] = "qsvm"
    reps: int = 2
    n_layers: int = 3
    C: float = 1.0
    epsilon: float = 0.01
    maxiter: int = 300
    seed: int | None = 42
    sabr_beta: float = 0.5


class QuantumIVSurface:
    """Implied volatility surface modelling via quantum regression.

    Supports QSVM kernel regression, VQC regression, and classical
    baselines (SABR, SVI).  All models are fit on (K/S, T) -> IV.

    Parameters
    ----------
    config : QuantumIVSurfaceConfig
        Model configuration.
    backend : Backend or None
        Quantum backend (required for 'qsvm' and 'vqc' methods).

    Examples
    --------
    >>> from qufin.backends.mock import MockBackend
    >>> cfg = QuantumIVSurfaceConfig(n_qubits=2, method="qsvm")
    >>> model = QuantumIVSurface(cfg, MockBackend(seed=0))
    >>> model.fit(data)
    >>> preds = model.predict(strikes, expiries, spot)
    """

    def __init__(
        self,
        config: QuantumIVSurfaceConfig,
        backend: Backend | None = None,
    ) -> None:
        self.config = config
        self.backend = backend
        self._data: IVSurfaceData | None = None

        # Internal models
        self._qsvm: _QSVMRegressor | None = None
        self._vqc: _VQCRegressor | None = None
        self._sabr: SABRModel | None = None
        self._svi: SVIModel | None = None

        if config.method in ("qsvm", "vqc", "all") and backend is None:
            raise ValueError(
                f"Backend required for method '{config.method}'"
            )

    def fit(
        self,
        data: IVSurfaceData,
        r: float = 0.0,
    ) -> QuantumIVSurface:
        """Fit the IV surface model(s) to observed data.

        Parameters
        ----------
        data : IVSurfaceData
            Observed (strike, expiry, IV) data.
        r : float
            Risk-free rate.

        Returns
        -------
        self
        """
        self._data = data
        X = data.features  # (n, 2): [K/S, T]
        y = data.ivs

        method = self.config.method

        if method in ("qsvm", "all"):
            self._qsvm = _QSVMRegressor(
                n_qubits=self.config.n_qubits,
                backend=self.backend,
                reps=self.config.reps,
                C=self.config.C,
                epsilon=self.config.epsilon,
            )
            # Scale features to [0, 2*pi] for quantum kernel
            X_scaled = self._scale_features(X)
            self._qsvm.fit(X_scaled, y)

        if method in ("vqc", "all"):
            self._vqc = _VQCRegressor(
                n_qubits=self.config.n_qubits,
                backend=self.backend,
                n_layers=self.config.n_layers,
                maxiter=self.config.maxiter,
                seed=self.config.seed,
            )
            self._vqc.fit(X, y)

        if method in ("sabr", "all"):
            self._sabr = SABRModel(beta=self.config.sabr_beta)
            self._sabr.calibrate(data, r=r)

        if method in ("svi", "all"):
            self._svi = SVIModel()
            self._svi.calibrate(data)

        return self

    @staticmethod
    def _scale_features(X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Scale features to [0, 2*pi] range for quantum kernels."""
        mins = X.min(axis=0)
        ranges = X.max(axis=0) - mins
        ranges[ranges < 1e-12] = 1.0
        return (X - mins) / ranges * 2.0 * np.pi

    def predict(
        self,
        strikes: NDArray[np.float64],
        expiries: NDArray[np.float64],
        spot: float,
        method: str | None = None,
        r: float = 0.0,
    ) -> NDArray[np.float64]:
        """Predict implied volatilities at given strikes and expiries.

        Parameters
        ----------
        strikes : array of shape (n,)
        expiries : array of shape (n,)
        spot : float
        method : str or None
            Override the configured method ('qsvm', 'vqc', 'sabr', 'svi').
            If None, uses the configured method (or 'qsvm' when 'all').
        r : float
            Risk-free rate (used by SABR).

        Returns
        -------
        NDArray of shape (n,)
            Predicted implied volatilities.
        """
        strikes = np.asarray(strikes, dtype=np.float64)
        expiries = np.asarray(expiries, dtype=np.float64)
        method = method or self.config.method
        if method == "all":
            method = "qsvm"  # default for 'all'

        if method == "qsvm":
            if self._qsvm is None:
                raise RuntimeError("QSVM model not fitted")
            X = np.column_stack([strikes / spot, expiries])
            X_scaled = self._scale_features(X)
            return self._qsvm.predict(X_scaled)

        if method == "vqc":
            if self._vqc is None:
                raise RuntimeError("VQC model not fitted")
            X = np.column_stack([strikes / spot, expiries])
            return self._vqc.predict(X)

        if method == "sabr":
            if self._sabr is None:
                raise RuntimeError("SABR model not fitted")
            return self._sabr.predict(strikes, expiries, spot, r=r)

        if method == "svi":
            if self._svi is None:
                raise RuntimeError("SVI model not fitted")
            return self._svi.predict(strikes, expiries, spot)

        raise ValueError(f"Unknown method: {method}")

    def evaluate(
        self,
        test_data: IVSurfaceData,
        method: str | None = None,
        r: float = 0.0,
    ) -> SurfaceMetrics:
        """Evaluate model on out-of-sample data.

        Parameters
        ----------
        test_data : IVSurfaceData
        method : str or None
        r : float

        Returns
        -------
        SurfaceMetrics
        """
        preds = self.predict(
            test_data.strikes, test_data.expiries, test_data.spot,
            method=method, r=r,
        )
        return evaluate_surface(test_data.ivs, preds)

    def plot_surface(
        self,
        spot: float = 100.0,
        strike_range: tuple[float, float] = (80.0, 120.0),
        expiry_range: tuple[float, float] = (0.1, 2.0),
        n_grid: int = 30,
        method: str | None = None,
        r: float = 0.0,
        ax: Any = None,
    ) -> Any:
        """Plot the fitted IV surface as a 3D surface.

        Parameters
        ----------
        spot : float
            Spot price.
        strike_range : tuple
            (min_strike, max_strike).
        expiry_range : tuple
            (min_T, max_T).
        n_grid : int
            Grid resolution per axis.
        method : str or None
            Which model to plot.
        r : float
            Risk-free rate.
        ax : matplotlib Axes3D or None
            Existing 3D axes to plot on.

        Returns
        -------
        matplotlib Axes3D
        """
        import matplotlib.pyplot as plt

        K_grid = np.linspace(strike_range[0], strike_range[1], n_grid)
        T_grid = np.linspace(expiry_range[0], expiry_range[1], n_grid)
        KK, TT = np.meshgrid(K_grid, T_grid)

        strikes_flat = KK.ravel()
        expiries_flat = TT.ravel()

        ivs_flat = self.predict(strikes_flat, expiries_flat, spot, method=method, r=r)
        IV_grid = ivs_flat.reshape(KK.shape)

        if ax is None:
            fig = plt.figure(figsize=(10, 7))
            ax = fig.add_subplot(111, projection="3d")

        ax.plot_surface(KK, TT, IV_grid, cmap="viridis", alpha=0.8)
        ax.set_xlabel("Strike")
        ax.set_ylabel("Expiry (years)")
        ax.set_zlabel("Implied Volatility")
        ax.set_title("Implied Volatility Surface")

        # Overlay training data if available
        if self._data is not None:
            ax.scatter(
                self._data.strikes,
                self._data.expiries,
                self._data.ivs,
                color="red",
                s=20,
                label="Training data",
            )
            ax.legend()

        return ax


# ---------------------------------------------------------------------------
# Synthetic data generation (for testing / demos)
# ---------------------------------------------------------------------------

def generate_synthetic_iv_surface(
    spot: float = 100.0,
    n_strikes: int = 10,
    n_expiries: int = 5,
    base_vol: float = 0.20,
    skew: float = -0.10,
    smile: float = 0.05,
    term_slope: float = 0.02,
    seed: int | None = 42,
    noise: float = 0.005,
) -> IVSurfaceData:
    """Generate synthetic IV surface data with skew and smile.

    Creates a realistic-looking IV surface with:
    - Volatility skew (negative correlation with moneyness)
    - Volatility smile (convexity in moneyness)
    - Term structure (increasing vol for longer expiries)

    Parameters
    ----------
    spot : float
        Spot price.
    n_strikes : int
        Number of strike levels.
    n_expiries : int
        Number of expiry levels.
    base_vol : float
        ATM base volatility.
    skew : float
        Skew coefficient (negative = typical equity skew).
    smile : float
        Smile/convexity coefficient.
    term_slope : float
        Term structure slope.
    seed : int or None
        Random seed.
    noise : float
        Gaussian noise standard deviation.

    Returns
    -------
    IVSurfaceData
    """
    rng = np.random.default_rng(seed)

    strikes_1d = np.linspace(0.8 * spot, 1.2 * spot, n_strikes)
    expiries_1d = np.linspace(0.1, 2.0, n_expiries)

    strikes_list = []
    expiries_list = []
    ivs_list = []

    for T in expiries_1d:
        for K in strikes_1d:
            m = np.log(K / spot)  # log-moneyness
            iv = (
                base_vol
                + skew * m
                + smile * m ** 2
                + term_slope * np.sqrt(T)
            )
            iv = max(iv, 0.01)  # floor
            iv += rng.normal(0, noise)
            iv = max(iv, 0.01)
            strikes_list.append(K)
            expiries_list.append(T)
            ivs_list.append(iv)

    return IVSurfaceData(
        strikes=np.array(strikes_list),
        expiries=np.array(expiries_list),
        ivs=np.array(ivs_list),
        spot=spot,
    )
