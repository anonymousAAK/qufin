"""Hybrid Quantum GAN (HQGAN) for synthetic financial market data.

Extends the basic qGAN from ``qufin.ml.qgan`` with finance-specific
capabilities:

* PQC generator (4-8 qubits) with parameterised rotations + entanglement
* Classical 3-layer MLP discriminator
* Wasserstein loss with gradient penalty (WGAN-GP)
* Stylised-fact evaluation of generated return series
* Use-case helpers for privacy-preserving data sharing

References
----------
Zoufal, Lucchi, Woerner, npj Quantum Information 5:103 (2019).
Arjovsky, Chintala, Bottou, "Wasserstein GAN", ICML 2017.
Gulrajani et al., "Improved Training of Wasserstein GANs", NeurIPS 2017.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from qufin.backends.base import Backend

# ---------------------------------------------------------------------------
# Optional torch import
# ---------------------------------------------------------------------------
try:
    import torch
    from torch import nn

    _HAS_TORCH = True
except ImportError:  # pragma: no cover
    _HAS_TORCH = False
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]


def _require_torch() -> None:
    """Raise if torch is not installed."""
    if not _HAS_TORCH:
        raise ImportError(
            "PyTorch is required for HQGAN.  Install with:  pip install torch"
        )


# ===================================================================
# Configuration & result dataclasses
# ===================================================================

@dataclass
class HQGANConfig:
    """Configuration for Hybrid Quantum GAN training."""

    # Generator (PQC) settings
    n_qubits: int = 4
    generator_reps: int = 3
    latent_dim: int = 4

    # Discriminator (classical MLP) settings
    discriminator_hidden: list[int] = field(
        default_factory=lambda: [128, 64, 32]
    )

    # Training hyper-parameters
    n_epochs: int = 200
    batch_size: int = 64
    lr_generator: float = 5e-4
    lr_discriminator: float = 1e-4
    n_critic: int = 5
    gradient_penalty_lambda: float = 10.0
    shots: int = 4096

    # Misc
    seed: int | None = 42
    window_size: int = 20
    convergence_threshold: float = 1e-4
    convergence_window: int = 20


@dataclass
class StylizedFactsResult:
    """Evaluation of stylised facts for a return series."""

    kurtosis: float = 0.0
    fat_tails: bool = False
    arch_lm_stat: float = 0.0
    arch_lm_pvalue: float = 1.0
    volatility_clustering: bool = False
    leverage_corr: float = 0.0
    leverage_effect: bool = False
    abs_return_autocorr: float = 0.0
    autocorrelation_present: bool = False


@dataclass
class HQGANResult:
    """Result from HQGAN training."""

    generator_params: NDArray[np.float64]
    loss_history_g: list[float]
    loss_history_d: list[float]
    wasserstein_estimates: list[float]
    synthetic_data: NDArray[np.float64]
    stylized_facts: StylizedFactsResult
    wall_time_s: float
    converged: bool


# ===================================================================
# Stylised-facts evaluation
# ===================================================================

def evaluate_stylized_facts(returns: NDArray[np.float64]) -> StylizedFactsResult:
    """Evaluate stylised facts of a return series.

    Parameters
    ----------
    returns : 1-D array of log-returns.

    Returns
    -------
    StylizedFactsResult with computed statistics.
    """
    returns = np.asarray(returns, dtype=np.float64).ravel()
    n = len(returns)

    if n < 10:
        return StylizedFactsResult()

    # --- Fat tails (excess kurtosis) ---
    mu = np.mean(returns)
    std = np.std(returns, ddof=1)
    if std < 1e-15:
        kurt = 0.0
    else:
        kurt = float(
            np.mean(((returns - mu) / std) ** 4)
        )
    fat_tails = kurt > 3.0

    # --- Volatility clustering (ARCH LM test) ---
    resid = returns - mu
    resid_sq = resid ** 2
    arch_lm_stat, arch_lm_pvalue = _arch_lm_test(resid_sq, lags=5)
    vol_clustering = arch_lm_pvalue < 0.05

    # --- Leverage effect ---
    vol = np.abs(returns)
    if len(returns) > 1 and np.std(vol) > 1e-15 and np.std(returns) > 1e-15:
        leverage_corr = float(np.corrcoef(returns[:-1], vol[1:])[0, 1])
    else:
        leverage_corr = 0.0
    leverage_effect = leverage_corr < 0.0

    # --- Autocorrelation of absolute returns ---
    abs_ret = np.abs(returns)
    if n > 2 and np.std(abs_ret) > 1e-15:
        abs_return_autocorr = float(
            np.corrcoef(abs_ret[:-1], abs_ret[1:])[0, 1]
        )
    else:
        abs_return_autocorr = 0.0
    autocorrelation_present = abs_return_autocorr > 0.0

    return StylizedFactsResult(
        kurtosis=kurt,
        fat_tails=fat_tails,
        arch_lm_stat=arch_lm_stat,
        arch_lm_pvalue=arch_lm_pvalue,
        volatility_clustering=vol_clustering,
        leverage_corr=leverage_corr,
        leverage_effect=leverage_effect,
        abs_return_autocorr=abs_return_autocorr,
        autocorrelation_present=autocorrelation_present,
    )


def _arch_lm_test(
    resid_sq: NDArray[np.float64], lags: int = 5
) -> tuple[float, float]:
    """Simple ARCH LM test using OLS on squared residuals.

    Returns (test_statistic, p_value).
    """
    n = len(resid_sq)
    if n <= lags + 1:
        return 0.0, 1.0

    # Build lag matrix
    Y = resid_sq[lags:]
    X = np.column_stack(
        [np.ones(n - lags)]
        + [resid_sq[lags - i - 1 : n - i - 1] for i in range(lags)]
    )

    # OLS
    try:
        beta = np.linalg.lstsq(X, Y, rcond=None)[0]
    except np.linalg.LinAlgError:
        return 0.0, 1.0

    Y_hat = X @ beta
    ss_res = np.sum((Y - Y_hat) ** 2)
    ss_tot = np.sum((Y - np.mean(Y)) ** 2)
    if ss_tot < 1e-30:
        return 0.0, 1.0

    r_squared = 1.0 - ss_res / ss_tot
    lm_stat = float(len(Y) * r_squared)

    # Approximate chi-squared p-value (survival function)
    p_value = _chi2_sf(lm_stat, lags)
    return lm_stat, p_value


def _chi2_sf(x: float, k: int) -> float:
    """Survival function of chi-squared distribution (rough approx).

    Uses the Wilson-Hilferty normal approximation to avoid scipy dependency.
    """
    if x <= 0:
        return 1.0
    if k <= 0:
        return 0.0
    # Wilson-Hilferty transform
    z = ((x / k) ** (1.0 / 3.0) - (1.0 - 2.0 / (9.0 * k))) / np.sqrt(
        2.0 / (9.0 * k)
    )
    # Standard normal CDF via error function
    p = 0.5 * (1.0 + _erf(z / np.sqrt(2.0)))
    return float(np.clip(1.0 - p, 0.0, 1.0))


def _erf(x: float | NDArray) -> float | NDArray:
    """Error function approximation (Abramowitz & Stegun 7.1.26)."""
    a = np.asarray(x)
    sign = np.sign(a)
    a = np.abs(a)
    t = 1.0 / (1.0 + 0.3275911 * a)
    poly = t * (
        0.254829592
        + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429)))
    )
    result = sign * (1.0 - poly * np.exp(-a * a))
    return float(result) if np.ndim(x) == 0 else result


# ===================================================================
# PQC Generator circuit builder
# ===================================================================

def build_generator_circuit(
    params: NDArray[np.float64],
    n_qubits: int,
    reps: int,
    latent: NDArray[np.float64] | None = None,
) -> Any:
    """Build a parameterised quantum circuit for the HQGAN generator.

    Parameters
    ----------
    params : 1-D parameter vector.
    n_qubits : Number of qubits (4-8).
    reps : Number of entangling layers.
    latent : Optional latent input vector encoded via RX on each qubit.

    Returns
    -------
    qiskit.circuit.QuantumCircuit
    """
    from qiskit.circuit import QuantumCircuit

    qc = QuantumCircuit(n_qubits, n_qubits)

    # Encode latent noise
    if latent is not None:
        for q in range(min(len(latent), n_qubits)):
            qc.rx(float(latent[q]), q)

    param_idx = 0
    for layer in range(reps + 1):
        # Rotation layer: RY + RZ on each qubit
        for q in range(n_qubits):
            qc.ry(float(params[param_idx]), q)
            param_idx += 1
            qc.rz(float(params[param_idx]), q)
            param_idx += 1

        # Entanglement layer (circular CX)
        if layer < reps:
            for i in range(n_qubits - 1):
                qc.cx(i, i + 1)
            if n_qubits > 2:
                qc.cx(n_qubits - 1, 0)

    qc.measure(range(n_qubits), range(n_qubits))
    return qc


def generator_n_params(n_qubits: int, reps: int) -> int:
    """Number of trainable parameters in the PQC generator."""
    return 2 * n_qubits * (reps + 1)


# ===================================================================
# Classical MLP discriminator (numpy-only fallback)
# ===================================================================

class _NumpyDiscriminator:
    """Minimal 3-layer MLP discriminator using pure numpy.

    Used when torch is not available.
    """

    def __init__(
        self,
        input_dim: int,
        hidden: list[int],
        rng: np.random.Generator,
    ) -> None:
        self.layers: list[tuple[NDArray, NDArray]] = []
        dims = [input_dim, *hidden, 1]
        for i in range(len(dims) - 1):
            w = rng.normal(0, 0.02, (dims[i], dims[i + 1]))
            b = np.zeros(dims[i + 1])
            self.layers.append((w, b))

    def forward(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Forward pass. Returns raw critic score (no sigmoid for WGAN)."""
        h = np.asarray(x, dtype=np.float64)
        if h.ndim == 1:
            h = h.reshape(-1, 1)
        for i, (w, b) in enumerate(self.layers):
            h = h @ w + b
            if i < len(self.layers) - 1:
                h = np.maximum(0.01 * h, h)  # LeakyReLU(0.01)
        return h

    def get_params(self) -> list[NDArray]:
        """Flat list: [w0, b0, w1, b1, ...]."""
        out = []
        for w, b in self.layers:
            out.extend([w, b])
        return out

    def set_params(self, params: list[NDArray]) -> None:
        """Inverse of get_params."""
        for i in range(len(self.layers)):
            self.layers[i] = (params[2 * i], params[2 * i + 1])


# ===================================================================
# Torch discriminator (optional)
# ===================================================================

def _build_torch_discriminator(
    input_dim: int, hidden: list[int]
) -> Any:
    """Build a PyTorch MLP critic for WGAN-GP."""
    _require_torch()
    layers: list[Any] = []
    dims = [input_dim, *hidden, 1]
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(nn.LeakyReLU(0.01))
    return nn.Sequential(*layers)


# ===================================================================
# HQGAN class
# ===================================================================

class HQGAN:
    """Hybrid Quantum GAN for synthetic financial market data.

    Generator: parameterised quantum circuit (PQC) with 4-8 qubits.
    Discriminator: classical 3-layer MLP (Wasserstein critic).
    Loss: Wasserstein distance with gradient penalty.
    """

    def __init__(
        self,
        config: HQGANConfig,
        backend: Backend,
    ) -> None:
        self.config = config
        self.backend = backend
        self._rng = np.random.default_rng(config.seed)
        self._n_params = generator_n_params(config.n_qubits, config.generator_reps)

    # ------------------------------------------------------------------
    # Generator helpers
    # ------------------------------------------------------------------

    def _sample_generator(
        self,
        params: NDArray[np.float64],
        n_samples: int,
    ) -> NDArray[np.float64]:
        """Sample *n_samples* values from the PQC generator.

        Each sample: draw a random latent vector, run the circuit, convert
        the measurement bitstring to a float in [0, 1].
        """
        n_states = 2 ** self.config.n_qubits
        samples = []

        # Batch: run one circuit with many shots, use empirical distribution
        latent = self._rng.uniform(
            0, 2 * np.pi, size=self.config.n_qubits
        )
        circuit = build_generator_circuit(
            params,
            self.config.n_qubits,
            self.config.generator_reps,
            latent=latent,
        )
        result = self.backend.run(circuit, shots=max(self.config.shots, n_samples))

        # Convert bitstrings to normalised floats
        for bitstring, count in result.counts.items():
            val = int(bitstring, 2) / max(n_states - 1, 1)
            samples.extend([val] * count)

        arr = np.array(samples, dtype=np.float64)
        # Subsample to requested size
        if len(arr) > n_samples:
            idx = self._rng.choice(len(arr), size=n_samples, replace=False)
            arr = arr[idx]
        elif len(arr) < n_samples:
            idx = self._rng.choice(len(arr), size=n_samples, replace=True)
            arr = arr[idx]

        return arr

    # ------------------------------------------------------------------
    # Wasserstein loss helpers
    # ------------------------------------------------------------------

    @staticmethod
    def wasserstein_loss_critic(
        real_scores: NDArray[np.float64],
        fake_scores: NDArray[np.float64],
    ) -> float:
        """Wasserstein critic loss: E[D(fake)] - E[D(real)]."""
        return float(np.mean(fake_scores) - np.mean(real_scores))

    @staticmethod
    def wasserstein_loss_generator(fake_scores: NDArray[np.float64]) -> float:
        """Generator loss: -E[D(fake)]."""
        return float(-np.mean(fake_scores))

    @staticmethod
    def gradient_penalty(
        discriminator: _NumpyDiscriminator,
        real: NDArray[np.float64],
        fake: NDArray[np.float64],
        rng: np.random.Generator,
    ) -> float:
        """Compute gradient penalty via finite differences.

        Approximates ||grad D(interpolated)||_2 penalty.
        """
        eps_vec = rng.uniform(0, 1, size=(len(real), 1))
        interpolated = eps_vec * real + (1 - eps_vec) * fake

        delta = 1e-5
        penalties = []
        for i in range(len(interpolated)):
            x = interpolated[i : i + 1]
            grads = np.zeros(x.shape[1])
            for d in range(x.shape[1]):
                x_plus = x.copy()
                x_plus[0, d] += delta
                x_minus = x.copy()
                x_minus[0, d] -= delta
                grads[d] = (
                    discriminator.forward(x_plus)[0, 0]
                    - discriminator.forward(x_minus)[0, 0]
                ) / (2 * delta)
            grad_norm = np.linalg.norm(grads)
            penalties.append((grad_norm - 1.0) ** 2)

        return float(np.mean(penalties))

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        real_returns: NDArray[np.float64],
    ) -> HQGANResult:
        """Train the HQGAN on a real return series.

        Parameters
        ----------
        real_returns : 1-D numpy array of log-returns.

        Returns
        -------
        HQGANResult with trained parameters, losses, and evaluation.
        """
        start = time.perf_counter()
        cfg = self.config

        real_returns = np.asarray(real_returns, dtype=np.float64).ravel()
        # Normalise to [0, 1]
        r_min, r_max = real_returns.min(), real_returns.max()
        spread = r_max - r_min if r_max > r_min else 1.0
        real_norm = (real_returns - r_min) / spread

        # Initialise generator parameters
        g_params = self._rng.uniform(0, 2 * np.pi, self._n_params)

        # Initialise discriminator
        disc = _NumpyDiscriminator(
            input_dim=cfg.window_size,
            hidden=cfg.discriminator_hidden,
            rng=self._rng,
        )

        loss_g_history: list[float] = []
        loss_d_history: list[float] = []
        w_estimates: list[float] = []

        converged = False

        for _epoch in range(cfg.n_epochs):
            # ----- Discriminator (critic) update -----
            for _ in range(cfg.n_critic):
                # Real mini-batch (windowed)
                real_batch = self._make_windows(real_norm, cfg.batch_size, cfg.window_size)

                # Fake mini-batch
                fake_raw = self._sample_generator(g_params, cfg.batch_size * cfg.window_size)
                fake_batch = fake_raw[: cfg.batch_size * cfg.window_size].reshape(
                    cfg.batch_size, cfg.window_size
                )

                real_scores = disc.forward(real_batch)
                fake_scores = disc.forward(fake_batch)

                w_loss = self.wasserstein_loss_critic(real_scores, fake_scores)
                gp = self.gradient_penalty(disc, real_batch, fake_batch, self._rng)
                d_loss = w_loss + cfg.gradient_penalty_lambda * gp

                # SGD update via numerical gradient
                self._update_discriminator(
                    disc, real_batch, fake_batch, cfg.lr_discriminator, cfg.gradient_penalty_lambda
                )

            loss_d_history.append(d_loss)
            w_estimates.append(-w_loss)

            # ----- Generator update (parameter-shift) -----
            g_loss, g_params = self._update_generator(
                g_params, disc, cfg.batch_size, cfg.window_size, cfg.lr_generator
            )
            loss_g_history.append(g_loss)

            # Convergence check
            if len(w_estimates) >= cfg.convergence_window:
                recent = w_estimates[-cfg.convergence_window:]
                if np.std(recent) < cfg.convergence_threshold:
                    converged = True
                    break

        # Generate final synthetic data
        syn_raw = self._sample_generator(g_params, len(real_returns))
        synthetic = syn_raw * spread + r_min

        # Evaluate stylised facts
        facts = evaluate_stylized_facts(synthetic)

        return HQGANResult(
            generator_params=g_params,
            loss_history_g=loss_g_history,
            loss_history_d=loss_d_history,
            wasserstein_estimates=w_estimates,
            synthetic_data=synthetic,
            stylized_facts=facts,
            wall_time_s=time.perf_counter() - start,
            converged=converged,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_windows(
        self,
        data: NDArray[np.float64],
        batch_size: int,
        window_size: int,
    ) -> NDArray[np.float64]:
        """Create sliding windows from data for the discriminator."""
        n = len(data)
        if n < window_size:
            # Pad with repetitions
            data = np.tile(data, (window_size // n) + 2)[:window_size * batch_size]

        max_start = max(n - window_size, 1)
        starts = self._rng.integers(0, max_start, size=batch_size)
        windows = np.array([data[s : s + window_size] for s in starts])
        return windows

    def _update_discriminator(
        self,
        disc: _NumpyDiscriminator,
        real_batch: NDArray,
        fake_batch: NDArray,
        lr: float,
        gp_lambda: float,
    ) -> None:
        """One-step SGD update of discriminator via finite differences."""
        eps = 1e-4
        params = disc.get_params()
        grads = []

        for p_idx, p in enumerate(params):
            g = np.zeros_like(p)
            for idx in range(min(p.size, 50)):  # cap for speed
                flat_idx = idx
                coord = np.unravel_index(flat_idx, p.shape)
                params[p_idx] = p.copy()
                params[p_idx][coord] += eps
                disc.set_params(params)
                loss_plus = self._disc_loss(disc, real_batch, fake_batch, gp_lambda)

                params[p_idx][coord] -= 2 * eps
                disc.set_params(params)
                loss_minus = self._disc_loss(disc, real_batch, fake_batch, gp_lambda)

                params[p_idx][coord] += eps  # restore
                g[coord] = (loss_plus - loss_minus) / (2 * eps)

            grads.append(g)
            params[p_idx] = p  # restore original

        # Apply gradients
        for p_idx in range(len(params)):
            params[p_idx] = params[p_idx] - lr * grads[p_idx]
        disc.set_params(params)

    def _disc_loss(
        self,
        disc: _NumpyDiscriminator,
        real_batch: NDArray,
        fake_batch: NDArray,
        gp_lambda: float,
    ) -> float:
        """Compute total discriminator loss."""
        real_scores = disc.forward(real_batch)
        fake_scores = disc.forward(fake_batch)
        w = self.wasserstein_loss_critic(real_scores, fake_scores)
        gp = self.gradient_penalty(disc, real_batch, fake_batch, self._rng)
        return w + gp_lambda * gp

    def _update_generator(
        self,
        g_params: NDArray[np.float64],
        disc: _NumpyDiscriminator,
        batch_size: int,
        window_size: int,
        lr: float,
    ) -> tuple[float, NDArray[np.float64]]:
        """Update generator parameters via parameter-shift finite differences."""
        shift = np.pi / 4
        grad = np.zeros_like(g_params)

        for i in range(len(g_params)):
            p_plus = g_params.copy()
            p_plus[i] += shift
            fake_plus = self._sample_generator(p_plus, batch_size * window_size)
            fake_plus = fake_plus[: batch_size * window_size].reshape(batch_size, window_size)
            score_plus = float(np.mean(disc.forward(fake_plus)))

            p_minus = g_params.copy()
            p_minus[i] -= shift
            fake_minus = self._sample_generator(p_minus, batch_size * window_size)
            fake_minus = fake_minus[: batch_size * window_size].reshape(batch_size, window_size)
            score_minus = float(np.mean(disc.forward(fake_minus)))

            # Generator wants to maximise critic score -> minimise -score
            grad[i] = -(score_plus - score_minus) / (2 * shift)

        g_params = g_params - lr * grad

        # Compute current generator loss
        fake = self._sample_generator(g_params, batch_size * window_size)
        fake = fake[: batch_size * window_size].reshape(batch_size, window_size)
        g_loss = self.wasserstein_loss_generator(disc.forward(fake))

        return g_loss, g_params

    # ------------------------------------------------------------------
    # Generation & use-case helpers
    # ------------------------------------------------------------------

    def generate(
        self,
        params: NDArray[np.float64],
        n_samples: int,
        real_min: float = 0.0,
        real_max: float = 1.0,
    ) -> NDArray[np.float64]:
        """Generate synthetic return data from trained generator params.

        Parameters
        ----------
        params : Trained generator parameter vector.
        n_samples : Number of samples to produce.
        real_min, real_max : Range of original data for denormalisation.
        """
        raw = self._sample_generator(params, n_samples)
        spread = real_max - real_min if real_max > real_min else 1.0
        return raw * spread + real_min


def privacy_preserving_synthetic(
    real_returns: NDArray[np.float64],
    config: HQGANConfig,
    backend: Backend,
    n_synthetic: int | None = None,
) -> tuple[NDArray[np.float64], HQGANResult]:
    """Train HQGAN and produce privacy-preserving synthetic returns.

    This is a convenience function for the common use case of generating
    synthetic data that preserves statistical properties while not exposing
    original data points.

    Parameters
    ----------
    real_returns : Original (sensitive) return series.
    config : HQGAN configuration.
    backend : Quantum backend.
    n_synthetic : Number of synthetic samples (defaults to len(real_returns)).

    Returns
    -------
    (synthetic_returns, training_result)
    """
    if n_synthetic is None:
        n_synthetic = len(real_returns)

    hqgan = HQGAN(config, backend)
    result = hqgan.train(real_returns)

    r_min, r_max = real_returns.min(), real_returns.max()
    synthetic = hqgan.generate(result.generator_params, n_synthetic, r_min, r_max)

    return synthetic, result


def train_on_synthetic_validate_on_real(
    real_returns: NDArray[np.float64],
    synthetic_returns: NDArray[np.float64],
    train_fraction: float = 0.8,
    seed: int | None = 42,
) -> dict[str, Any]:
    """Train a simple model on synthetic data, validate on real data.

    Uses a basic AR(1) model (no external dependencies). Returns
    metrics comparing in-sample (synthetic) and out-of-sample (real)
    performance.

    Parameters
    ----------
    real_returns : The real return series (used as test set).
    synthetic_returns : HQGAN-generated synthetic returns (training set).
    train_fraction : Fraction of synthetic data used for training.
    seed : Random seed.

    Returns
    -------
    dict with keys: 'train_mse', 'test_mse', 'train_mae', 'test_mae',
                    'synthetic_mean', 'real_mean', 'synthetic_std', 'real_std'.
    """
    syn = np.asarray(synthetic_returns, dtype=np.float64).ravel()
    real = np.asarray(real_returns, dtype=np.float64).ravel()

    # Train AR(1) on synthetic data
    n_train = max(int(len(syn) * train_fraction), 2)
    syn_train = syn[:n_train]

    if len(syn_train) < 3:
        # Not enough data for AR(1)
        return {
            "train_mse": float("nan"),
            "test_mse": float("nan"),
            "train_mae": float("nan"),
            "test_mae": float("nan"),
            "synthetic_mean": float(np.mean(syn)),
            "real_mean": float(np.mean(real)),
            "synthetic_std": float(np.std(syn)),
            "real_std": float(np.std(real)),
        }

    # Fit AR(1): r_t = a + b * r_{t-1}
    X_train = syn_train[:-1].reshape(-1, 1)
    y_train = syn_train[1:]
    X_train_aug = np.column_stack([np.ones(len(X_train)), X_train])
    try:
        beta = np.linalg.lstsq(X_train_aug, y_train, rcond=None)[0]
    except np.linalg.LinAlgError:
        beta = np.array([np.mean(y_train), 0.0])

    # In-sample (synthetic) predictions
    pred_train = X_train_aug @ beta
    train_mse = float(np.mean((y_train - pred_train) ** 2))
    train_mae = float(np.mean(np.abs(y_train - pred_train)))

    # Out-of-sample (real) predictions
    if len(real) < 2:
        test_mse = float("nan")
        test_mae = float("nan")
    else:
        X_test = real[:-1].reshape(-1, 1)
        y_test = real[1:]
        X_test_aug = np.column_stack([np.ones(len(X_test)), X_test])
        pred_test = X_test_aug @ beta
        test_mse = float(np.mean((y_test - pred_test) ** 2))
        test_mae = float(np.mean(np.abs(y_test - pred_test)))

    return {
        "train_mse": train_mse,
        "test_mse": test_mse,
        "train_mae": train_mae,
        "test_mae": test_mae,
        "synthetic_mean": float(np.mean(syn)),
        "real_mean": float(np.mean(real)),
        "synthetic_std": float(np.std(syn)),
        "real_std": float(np.std(real)),
    }
