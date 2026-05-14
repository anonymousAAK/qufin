"""Restricted Quantum Boltzmann Machine for market regime detection.

Implements a Restricted Quantum Boltzmann Machine (RQBM) that uses
quantum sampling for training visible-hidden weight updates, replacing
classical contrastive divergence.  The model classifies market regimes
(risk-on, risk-off, crisis, recovery) from indicators such as VIX,
yield curve slope, and momentum.

References
----------
Amin et al., Phys. Rev. X 8, 021050 (2018).
Kieferova & Wiebe, Phys. Rev. A 96, 062327 (2017).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import IntEnum
from typing import Any

import numpy as np
from numpy.typing import NDArray

from qufin.backends.base import Backend


class MarketRegime(IntEnum):
    """Market regime labels."""

    RISK_ON = 0
    RISK_OFF = 1
    CRISIS = 2
    RECOVERY = 3


@dataclass
class RQBMConfig:
    """Configuration for the Restricted Quantum Boltzmann Machine."""

    n_visible: int = 6
    n_hidden: int = 4
    n_epochs: int = 50
    learning_rate: float = 0.01
    n_gibbs_steps: int = 1
    n_quantum_shots: int = 1024
    temperature: float = 1.0
    momentum: float = 0.0
    weight_decay: float = 0.0
    seed: int | None = 42


@dataclass
class RQBMResult:
    """Result from RQBM training."""

    weights: NDArray[np.float64]
    visible_bias: NDArray[np.float64]
    hidden_bias: NDArray[np.float64]
    loss_history: list[float]
    regime_labels: NDArray[np.int64]
    regime_probabilities: NDArray[np.float64]
    wall_time_s: float


@dataclass
class RegimeBacktestResult:
    """Result from regime-conditional portfolio backtest."""

    portfolio_returns: NDArray[np.float64]
    cumulative_return: float
    sharpe_ratio: float
    max_drawdown: float
    regime_history: NDArray[np.int64]
    regime_allocations: dict[int, NDArray[np.float64]]


class RestrictedQuantumBoltzmannMachine:
    """Restricted Quantum Boltzmann Machine for regime detection.

    Visible units encode market indicators (e.g., VIX level, yield curve
    slope, momentum).  Hidden units are quantum-sampled latent features.
    Training uses quantum-enhanced sampling instead of classical CD-k.

    Parameters
    ----------
    config : RQBMConfig
        Model configuration.
    backend : Backend
        Quantum backend for sampling circuits.
    """

    def __init__(self, config: RQBMConfig, backend: Backend) -> None:
        self.config = config
        self.backend = backend
        self._rng = np.random.default_rng(config.seed)

        # Model parameters
        self.weights = self._rng.normal(
            0, 0.01, (config.n_visible, config.n_hidden)
        )
        self.visible_bias = np.zeros(config.n_visible, dtype=np.float64)
        self.hidden_bias = np.zeros(config.n_hidden, dtype=np.float64)

        # Momentum terms
        self._dw = np.zeros_like(self.weights)
        self._dvb = np.zeros_like(self.visible_bias)
        self._dhb = np.zeros_like(self.hidden_bias)

    def _sigmoid(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Numerically stable sigmoid."""
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    def _build_sampling_circuit(
        self,
        visible: NDArray[np.float64],
    ) -> Any:
        """Build quantum circuit for sampling hidden units.

        Encodes visible-to-hidden activation into rotation angles and
        applies entangling layers for quantum correlations.

        Parameters
        ----------
        visible : array of shape (n_visible,)
            Visible unit activations (binary or continuous in [0, 1]).

        Returns
        -------
        QuantumCircuit
        """
        from qiskit.circuit import QuantumCircuit

        n_h = self.config.n_hidden
        qc = QuantumCircuit(n_h, n_h)

        # Compute pre-activation for hidden units
        pre_activation = visible @ self.weights + self.hidden_bias
        angles = np.pi * self._sigmoid(pre_activation / self.config.temperature)

        # Encode as RY rotations
        for j in range(n_h):
            qc.ry(float(angles[j]), j)

        # Entangling layer for quantum correlations
        for j in range(n_h - 1):
            qc.cx(j, j + 1)
        if n_h > 2:
            qc.cx(n_h - 1, 0)

        # Measure
        qc.measure(range(n_h), range(n_h))
        return qc

    def sample_hidden(
        self, visible: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Sample hidden unit activations using quantum circuit.

        Parameters
        ----------
        visible : array of shape (n_visible,) or (n_samples, n_visible)

        Returns
        -------
        Array of shape (n_hidden,) or (n_samples, n_hidden) with
        marginal probabilities of hidden units being 1.
        """
        if visible.ndim == 1:
            return self._sample_hidden_single(visible)
        return np.array([self._sample_hidden_single(v) for v in visible])

    def _sample_hidden_single(
        self, visible: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Sample hidden units for a single visible vector."""
        circuit = self._build_sampling_circuit(visible)
        result = self.backend.run(circuit, shots=self.config.n_quantum_shots)

        n_h = self.config.n_hidden
        probs = np.zeros(n_h, dtype=np.float64)
        total = 0

        for bitstring, count in result.counts.items():
            total += count
            # Parse bitstring (may be in various formats)
            bits = bitstring.replace(" ", "")
            for j in range(min(len(bits), n_h)):
                if bits[-(j + 1)] == "1":
                    probs[j] += count

        if total > 0:
            probs /= total
        return probs

    def sample_visible(
        self, hidden: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Compute visible unit probabilities given hidden activations.

        Uses classical sigmoid (visible units are not quantum-sampled).

        Parameters
        ----------
        hidden : array of shape (n_hidden,) or (n_samples, n_hidden)

        Returns
        -------
        Array of visible unit probabilities.
        """
        pre_activation = hidden @ self.weights.T + self.visible_bias
        return self._sigmoid(pre_activation / self.config.temperature)

    def _contrastive_divergence_step(
        self, data: NDArray[np.float64]
    ) -> tuple[NDArray, NDArray, NDArray]:
        """One step of quantum-enhanced contrastive divergence.

        Parameters
        ----------
        data : array of shape (n_samples, n_visible)

        Returns
        -------
        Tuple of (dW, dvb, dhb) gradients.
        """
        n_samples = data.shape[0]

        # Positive phase: sample hidden from data
        h_probs_pos = self.sample_hidden(data)

        # Negative phase: Gibbs chain
        v_neg = data.copy()
        for _ in range(self.config.n_gibbs_steps):
            h_sample = (self._rng.random(h_probs_pos.shape) < h_probs_pos).astype(
                np.float64
            )
            v_neg = self.sample_visible(h_sample)
            h_probs_neg = self.sample_hidden(v_neg)

        # Compute gradients
        pos_associations = data.T @ h_probs_pos / n_samples
        neg_associations = v_neg.T @ h_probs_neg / n_samples

        dw = pos_associations - neg_associations
        dvb = np.mean(data - v_neg, axis=0)
        dhb = np.mean(h_probs_pos - h_probs_neg, axis=0)

        return dw, dvb, dhb

    def reconstruction_error(self, data: NDArray[np.float64]) -> float:
        """Compute mean squared reconstruction error.

        Parameters
        ----------
        data : array of shape (n_samples, n_visible)

        Returns
        -------
        float
            Mean squared error between data and reconstruction.
        """
        h_probs = self.sample_hidden(data)
        v_recon = self.sample_visible(h_probs)
        return float(np.mean((data - v_recon) ** 2))

    def fit(self, data: NDArray[np.float64]) -> RQBMResult:
        """Train the RQBM on data.

        Parameters
        ----------
        data : array of shape (n_samples, n_visible)
            Training data (should be normalised to [0, 1]).

        Returns
        -------
        RQBMResult
        """
        start = time.perf_counter()
        loss_history: list[float] = []

        for _epoch in range(self.config.n_epochs):
            dw, dvb, dhb = self._contrastive_divergence_step(data)

            # Momentum
            self._dw = self.config.momentum * self._dw + self.config.learning_rate * dw
            self._dvb = self.config.momentum * self._dvb + self.config.learning_rate * dvb
            self._dhb = self.config.momentum * self._dhb + self.config.learning_rate * dhb

            # Weight decay
            self.weights *= 1.0 - self.config.weight_decay
            self.weights += self._dw
            self.visible_bias += self._dvb
            self.hidden_bias += self._dhb

            loss = self.reconstruction_error(data)
            loss_history.append(loss)

        # Classify regimes
        h_features = self.sample_hidden(data)
        regime_labels, regime_probs = self.classify_regimes(h_features)

        return RQBMResult(
            weights=self.weights.copy(),
            visible_bias=self.visible_bias.copy(),
            hidden_bias=self.hidden_bias.copy(),
            loss_history=loss_history,
            regime_labels=regime_labels,
            regime_probabilities=regime_probs,
            wall_time_s=time.perf_counter() - start,
        )

    def classify_regimes(
        self,
        hidden_features: NDArray[np.float64],
        n_regimes: int = 4,
    ) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
        """Classify samples into market regimes using hidden features.

        Uses simple k-means style clustering on hidden unit activations.

        Parameters
        ----------
        hidden_features : array of shape (n_samples, n_hidden)
        n_regimes : int
            Number of regimes (default 4).

        Returns
        -------
        labels : array of shape (n_samples,)
        probabilities : array of shape (n_samples, n_regimes)
        """
        n_samples = hidden_features.shape[0]
        if n_samples == 0:
            return (
                np.array([], dtype=np.int64),
                np.empty((0, n_regimes), dtype=np.float64),
            )

        # Simple k-means clustering
        centroids = self._init_centroids(hidden_features, n_regimes)

        for _ in range(20):  # max iterations
            distances = self._compute_distances(hidden_features, centroids)
            labels = np.argmin(distances, axis=1)

            new_centroids = np.zeros_like(centroids)
            for k in range(n_regimes):
                mask = labels == k
                if np.any(mask):
                    new_centroids[k] = hidden_features[mask].mean(axis=0)
                else:
                    new_centroids[k] = centroids[k]

            if np.allclose(centroids, new_centroids, atol=1e-8):
                break
            centroids = new_centroids

        # Compute soft probabilities via distances
        distances = self._compute_distances(hidden_features, centroids)
        # Convert distances to probabilities using softmax
        neg_dist = -distances
        neg_dist -= neg_dist.max(axis=1, keepdims=True)
        exp_dist = np.exp(neg_dist)
        probabilities = exp_dist / exp_dist.sum(axis=1, keepdims=True)

        return labels.astype(np.int64), probabilities

    def _init_centroids(
        self,
        data: NDArray[np.float64],
        k: int,
    ) -> NDArray[np.float64]:
        """Initialize centroids using k-means++ style."""
        n = data.shape[0]
        if n <= k:
            centroids = np.zeros((k, data.shape[1]), dtype=np.float64)
            centroids[:n] = data
            return centroids

        indices = [int(self._rng.integers(0, n))]
        for _ in range(1, k):
            dists = np.min(
                self._compute_distances(data, data[indices]),
                axis=1,
            )
            dists_sum = dists.sum()
            if dists_sum == 0:
                idx = int(self._rng.integers(0, n))
            else:
                probs = dists / dists_sum
                idx = int(self._rng.choice(n, p=probs))
            indices.append(idx)

        return data[indices].copy()

    @staticmethod
    def _compute_distances(
        X: NDArray[np.float64],
        centroids: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Compute squared Euclidean distances between X and centroids."""
        # X: (n, d), centroids: (k, d) -> (n, k)
        return np.sum((X[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2, axis=2)

    def extract_features(
        self, data: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Extract hidden features from data.

        Parameters
        ----------
        data : array of shape (n_samples, n_visible)

        Returns
        -------
        array of shape (n_samples, n_hidden)
        """
        return self.sample_hidden(data)


def prepare_market_indicators(
    vix: NDArray[np.float64],
    yield_spread: NDArray[np.float64],
    momentum: NDArray[np.float64],
    *,
    extra_indicators: list[NDArray[np.float64]] | None = None,
) -> NDArray[np.float64]:
    """Prepare and normalise market indicators for RQBM input.

    Parameters
    ----------
    vix : array of shape (n_samples,)
        VIX index values.
    yield_spread : array of shape (n_samples,)
        Yield curve spread (10Y - 2Y).
    momentum : array of shape (n_samples,)
        Market momentum indicator.
    extra_indicators : list of arrays, optional
        Additional indicators to include.

    Returns
    -------
    array of shape (n_samples, n_indicators)
        Normalised indicators in [0, 1].
    """
    indicators = [vix, yield_spread, momentum]
    if extra_indicators:
        indicators.extend(extra_indicators)

    raw = np.column_stack(indicators)
    # Min-max normalisation per column
    mins = raw.min(axis=0)
    maxs = raw.max(axis=0)
    ranges = maxs - mins
    ranges[ranges == 0] = 1.0  # avoid division by zero
    normalised = (raw - mins) / ranges
    return normalised


def regime_conditional_allocation(
    regime: int,
    n_assets: int,
    risk_free_weight: float = 0.1,
) -> NDArray[np.float64]:
    """Return portfolio weights conditioned on market regime.

    Parameters
    ----------
    regime : int
        Market regime (0=risk-on, 1=risk-off, 2=crisis, 3=recovery).
    n_assets : int
        Number of risky assets.
    risk_free_weight : float
        Base risk-free allocation.

    Returns
    -------
    weights : array of shape (n_assets + 1,)
        Portfolio weights [asset_1, ..., asset_n, risk_free].
    """
    weights = np.zeros(n_assets + 1, dtype=np.float64)

    if regime == MarketRegime.RISK_ON:
        # High equity allocation
        risky = 1.0 - risk_free_weight
        weights[:n_assets] = risky / n_assets
        weights[-1] = risk_free_weight
    elif regime == MarketRegime.RISK_OFF:
        # Moderate allocation, more defensive
        risky = 0.5
        weights[:n_assets] = risky / n_assets
        weights[-1] = 1.0 - risky
    elif regime == MarketRegime.CRISIS:
        # Mostly risk-free
        risky = 0.2
        weights[:n_assets] = risky / n_assets
        weights[-1] = 1.0 - risky
    elif regime == MarketRegime.RECOVERY:
        # Balanced growth
        risky = 0.7
        weights[:n_assets] = risky / n_assets
        weights[-1] = 1.0 - risky
    else:
        # Equal weight fallback
        weights[:] = 1.0 / (n_assets + 1)

    return weights


def backtest_regime_strategy(
    regimes: NDArray[np.int64],
    asset_returns: NDArray[np.float64],
    risk_free_rate: float = 0.0,
) -> RegimeBacktestResult:
    """Backtest a regime-conditional portfolio allocation strategy.

    Parameters
    ----------
    regimes : array of shape (n_periods,)
        Regime label for each period.
    asset_returns : array of shape (n_periods, n_assets)
        Asset returns for each period.
    risk_free_rate : float
        Per-period risk-free rate.

    Returns
    -------
    RegimeBacktestResult
    """
    n_periods, n_assets = asset_returns.shape
    portfolio_returns = np.zeros(n_periods, dtype=np.float64)
    regime_allocs: dict[int, NDArray[np.float64]] = {}

    for t in range(n_periods):
        regime = int(regimes[t])
        alloc = regime_conditional_allocation(regime, n_assets)
        regime_allocs[regime] = alloc

        # Portfolio return = weighted sum of asset returns + risk-free
        port_ret = np.dot(alloc[:n_assets], asset_returns[t]) + alloc[-1] * risk_free_rate
        portfolio_returns[t] = port_ret

    # Cumulative return
    cum_return = float(np.prod(1.0 + portfolio_returns) - 1.0)

    # Sharpe ratio (annualised, assuming 252 trading days)
    mean_ret = np.mean(portfolio_returns)
    std_ret = np.std(portfolio_returns, ddof=1) if n_periods > 1 else 1.0
    sharpe = float(mean_ret / std_ret * np.sqrt(252)) if std_ret > 0 else 0.0

    # Max drawdown
    cumulative = np.cumprod(1.0 + portfolio_returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = (running_max - cumulative) / running_max
    max_dd = float(np.max(drawdowns))

    return RegimeBacktestResult(
        portfolio_returns=portfolio_returns,
        cumulative_return=cum_return,
        sharpe_ratio=sharpe,
        max_drawdown=max_dd,
        regime_history=regimes,
        regime_allocations=regime_allocs,
    )


# ---------------------------------------------------------------------------
# Classical baselines
# ---------------------------------------------------------------------------


class ClassicalRBM:
    """Classical Restricted Boltzmann Machine baseline.

    Uses standard contrastive divergence (CD-k) with classical sampling.
    """

    def __init__(
        self,
        n_visible: int,
        n_hidden: int,
        learning_rate: float = 0.01,
        n_gibbs_steps: int = 1,
        seed: int | None = 42,
    ) -> None:
        self._rng = np.random.default_rng(seed)
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.lr = learning_rate
        self.n_gibbs = n_gibbs_steps

        self.weights = self._rng.normal(0, 0.01, (n_visible, n_hidden))
        self.vb = np.zeros(n_visible, dtype=np.float64)
        self.hb = np.zeros(n_hidden, dtype=np.float64)

    @staticmethod
    def _sigmoid(x: NDArray[np.float64]) -> NDArray[np.float64]:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    def sample_hidden(self, v: NDArray[np.float64]) -> NDArray[np.float64]:
        """P(h=1|v) via sigmoid."""
        return self._sigmoid(v @ self.weights + self.hb)

    def sample_visible(self, h: NDArray[np.float64]) -> NDArray[np.float64]:
        """P(v=1|h) via sigmoid."""
        return self._sigmoid(h @ self.weights.T + self.vb)

    def fit(
        self, data: NDArray[np.float64], n_epochs: int = 50
    ) -> list[float]:
        """Train with CD-k. Returns loss history."""
        losses: list[float] = []
        n = data.shape[0]
        for _ in range(n_epochs):
            h_pos = self.sample_hidden(data)
            v_neg = data.copy()
            for __ in range(self.n_gibbs):
                h_sample = (self._rng.random(h_pos.shape) < h_pos).astype(np.float64)
                v_neg = self.sample_visible(h_sample)
                h_pos_neg = self.sample_hidden(v_neg)

            self.weights += self.lr * (data.T @ h_pos - v_neg.T @ h_pos_neg) / n
            self.vb += self.lr * np.mean(data - v_neg, axis=0)
            self.hb += self.lr * np.mean(h_pos - h_pos_neg, axis=0)

            recon = self.sample_visible(self.sample_hidden(data))
            losses.append(float(np.mean((data - recon) ** 2)))
        return losses


class HMMRegimeDetector:
    """Simple Hidden Markov Model baseline for regime detection.

    Uses Gaussian emissions with EM-style parameter estimation.
    This is a lightweight implementation for comparison purposes.
    """

    def __init__(
        self,
        n_regimes: int = 4,
        n_features: int = 3,
        n_iter: int = 20,
        seed: int | None = 42,
    ) -> None:
        self.n_regimes = n_regimes
        self.n_features = n_features
        self.n_iter = n_iter
        self._rng = np.random.default_rng(seed)

        # Parameters
        self.means: NDArray[np.float64] | None = None
        self.covs: list[NDArray[np.float64]] | None = None
        self.transition: NDArray[np.float64] | None = None
        self.initial: NDArray[np.float64] | None = None

    def fit(self, data: NDArray[np.float64]) -> HMMRegimeDetector:
        """Fit HMM parameters using simplified EM.

        Parameters
        ----------
        data : array of shape (n_samples, n_features)

        Returns
        -------
        self
        """
        n = data.shape[0]
        k = self.n_regimes

        # Initialise with random cluster assignments
        labels = self._rng.integers(0, k, size=n)
        self.means = np.zeros((k, self.n_features), dtype=np.float64)
        self.covs = []
        for r in range(k):
            mask = labels == r
            if np.any(mask):
                self.means[r] = data[mask].mean(axis=0)
            else:
                self.means[r] = data[self._rng.integers(0, n)]
            self.covs.append(np.eye(self.n_features) * 0.1)

        self.transition = np.ones((k, k)) / k
        self.initial = np.ones(k) / k

        for _ in range(self.n_iter):
            # E-step: compute responsibilities
            log_likes = np.zeros((n, k))
            for r in range(k):
                diff = data - self.means[r]
                cov_inv = np.linalg.inv(self.covs[r] + 1e-6 * np.eye(self.n_features))
                log_likes[:, r] = -0.5 * np.sum(diff @ cov_inv * diff, axis=1)

            # Normalize responsibilities
            log_likes -= log_likes.max(axis=1, keepdims=True)
            resp = np.exp(log_likes)
            resp /= resp.sum(axis=1, keepdims=True) + 1e-10

            # M-step
            for r in range(k):
                w = resp[:, r]
                w_sum = w.sum() + 1e-10
                self.means[r] = (w[:, np.newaxis] * data).sum(axis=0) / w_sum
                diff = data - self.means[r]
                self.covs[r] = (
                    (w[:, np.newaxis] * diff).T @ diff / w_sum
                    + 1e-4 * np.eye(self.n_features)
                )

        return self

    def predict(self, data: NDArray[np.float64]) -> NDArray[np.int64]:
        """Predict regime labels.

        Parameters
        ----------
        data : array of shape (n_samples, n_features)

        Returns
        -------
        labels : array of shape (n_samples,)
        """
        if self.means is None or self.covs is None:
            raise RuntimeError("Must call fit() first.")

        k = self.n_regimes
        n = data.shape[0]
        log_likes = np.zeros((n, k))

        for r in range(k):
            diff = data - self.means[r]
            cov_inv = np.linalg.inv(self.covs[r] + 1e-6 * np.eye(self.n_features))
            log_likes[:, r] = -0.5 * np.sum(diff @ cov_inv * diff, axis=1)

        return np.argmax(log_likes, axis=1).astype(np.int64)
