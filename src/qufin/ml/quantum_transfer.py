"""Quantum Transfer Learning for trading signal classification.

Implements a hybrid classical-quantum transfer learning pipeline:
1. Classical pre-training: PCA feature extraction from financial data
2. Quantum fine-tuning: VQC head on frozen classical features

The model classifies trading signals (buy/hold/sell) from reduced
feature representations.

References
----------
Mari et al., Quantum 4, 340 (2020).  (Quantum transfer learning)
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import IntEnum
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from qufin.backends.base import Backend


class TradingSignal(IntEnum):
    """Trading signal labels."""

    BUY = 0
    HOLD = 1
    SELL = 2


@dataclass
class TransferConfig:
    """Configuration for quantum transfer learning."""

    n_pca_components: int = 4
    n_qubits: int = 4
    n_layers: int = 3
    optimizer: str = "COBYLA"
    max_iter: int = 200
    learning_rate: float = 0.01
    n_classes: int = 3
    seed: int | None = 42


@dataclass
class TransferResult:
    """Result from quantum transfer learning."""

    pca_components: NDArray[np.float64]
    pca_mean: NDArray[np.float64]
    pca_explained_variance_ratio: NDArray[np.float64]
    vqc_params: NDArray[np.float64]
    loss_history: list[float]
    train_accuracy: float
    wall_time_s: float


@dataclass
class ClassicalTransferResult:
    """Result from classical transfer learning (MLP head)."""

    pca_components: NDArray[np.float64]
    pca_mean: NDArray[np.float64]
    mlp_weights: list[NDArray[np.float64]]
    mlp_biases: list[NDArray[np.float64]]
    loss_history: list[float]
    train_accuracy: float
    wall_time_s: float


class PCAFeatureExtractor:
    """PCA-based classical feature extractor.

    Performs dimensionality reduction on raw financial features.
    This serves as the "frozen classical backbone" in the transfer
    learning pipeline.
    """

    def __init__(self, n_components: int = 4, seed: int | None = 42) -> None:
        self.n_components = n_components
        self._seed = seed
        self.components_: NDArray[np.float64] | None = None
        self.mean_: NDArray[np.float64] | None = None
        self.explained_variance_ratio_: NDArray[np.float64] | None = None

    def fit(self, X: NDArray[np.float64]) -> PCAFeatureExtractor:
        """Fit PCA on training data.

        Parameters
        ----------
        X : array of shape (n_samples, n_features)

        Returns
        -------
        self
        """
        self.mean_ = X.mean(axis=0)
        X_centered = X - self.mean_

        # SVD-based PCA
        n_samples = X_centered.shape[0]
        if n_samples < 2:
            n_feat = X_centered.shape[1]
            k = min(self.n_components, n_feat)
            self.components_ = np.eye(k, n_feat, dtype=np.float64)
            self.explained_variance_ratio_ = np.ones(k) / k
            return self

        _U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        k = min(self.n_components, len(S))

        self.components_ = Vt[:k]
        explained_variance = S[:k] ** 2 / (n_samples - 1)
        total_variance = np.sum(S**2) / (n_samples - 1)
        if total_variance > 0:
            self.explained_variance_ratio_ = explained_variance / total_variance
        else:
            self.explained_variance_ratio_ = np.zeros(k)

        return self

    def transform(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Transform data to PCA space.

        Parameters
        ----------
        X : array of shape (n_samples, n_features)

        Returns
        -------
        array of shape (n_samples, n_components)
        """
        if self.components_ is None or self.mean_ is None:
            raise RuntimeError("Must call fit() first.")
        X_centered = X - self.mean_
        return X_centered @ self.components_.T

    def fit_transform(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Fit and transform in one step."""
        self.fit(X)
        return self.transform(X)


class QuantumTransferLearner:
    """Quantum transfer learning: PCA backbone + VQC classification head.

    The pipeline:
    1. Classical PCA extracts n_pca_components features (frozen)
    2. VQC head maps reduced features to class probabilities
    3. Training optimises only the VQC parameters

    Parameters
    ----------
    config : TransferConfig
        Model configuration.
    backend : Backend
        Quantum backend for circuit simulation.
    """

    def __init__(self, config: TransferConfig, backend: Backend) -> None:
        self.config = config
        self.backend = backend
        self._rng = np.random.default_rng(config.seed)
        self._pca = PCAFeatureExtractor(
            n_components=config.n_pca_components, seed=config.seed
        )
        self._params: NDArray[np.float64] | None = None

    def _n_params(self) -> int:
        """Total number of VQC parameters.

        Each layer: n_qubits * 2 (RY + RZ) + final RY layer.
        """
        return (
            2 * self.config.n_qubits * self.config.n_layers
            + self.config.n_qubits
        )

    def _build_vqc_circuit(
        self,
        x: NDArray[np.float64],
        params: NDArray[np.float64],
    ) -> Any:
        """Build VQC circuit with angle encoding and TwoLocal ansatz.

        Parameters
        ----------
        x : array of shape (n_pca_components,)
            PCA-reduced feature vector.
        params : array
            Variational parameters.

        Returns
        -------
        QuantumCircuit
        """
        from qiskit.circuit import QuantumCircuit

        n = self.config.n_qubits
        qc = QuantumCircuit(n)

        # Angle encoding of PCA features
        for i in range(min(len(x), n)):
            qc.ry(float(x[i]), i)

        # TwoLocal ansatz
        idx = 0
        for _layer in range(self.config.n_layers):
            for i in range(n):
                qc.ry(float(params[idx]), i)
                idx += 1
                qc.rz(float(params[idx]), i)
                idx += 1
            # CNOT ladder
            for i in range(n - 1):
                qc.cx(i, i + 1)
            if n > 2:
                qc.cx(n - 1, 0)

        # Final rotation
        for i in range(n):
            qc.ry(float(params[idx]), i)
            idx += 1

        return qc

    def _circuit_class_probabilities(
        self,
        x: NDArray[np.float64],
        params: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Compute class probabilities from circuit statevector.

        Maps statevector amplitudes to class probabilities by grouping
        basis states.

        Parameters
        ----------
        x : array of shape (n_pca_components,)
        params : array

        Returns
        -------
        array of shape (n_classes,)
        """
        qc = self._build_vqc_circuit(x, params)
        sv = self.backend.statevector(qc)
        n_states = 2 ** self.config.n_qubits
        probs_all = np.abs(sv[:n_states]) ** 2

        n_classes = self.config.n_classes
        class_probs = np.zeros(n_classes, dtype=np.float64)

        # Group basis states into classes
        states_per_class = max(1, n_states // n_classes)
        for c in range(n_classes):
            start = c * states_per_class
            end = start + states_per_class if c < n_classes - 1 else n_states
            class_probs[c] = probs_all[start:end].sum()

        # Normalise
        total = class_probs.sum()
        if total > 0:
            class_probs /= total

        return class_probs

    def _cross_entropy_loss(
        self,
        params: NDArray[np.float64],
        X: NDArray[np.float64],
        y: NDArray[np.int64],
    ) -> float:
        """Multi-class cross-entropy loss."""
        eps = 1e-10
        total = 0.0
        for i in range(X.shape[0]):
            probs = self._circuit_class_probabilities(X[i], params)
            total -= np.log(np.clip(probs[y[i]], eps, 1.0))
        return float(total / X.shape[0])

    def fit(
        self,
        X_raw: NDArray[np.float64],
        y: NDArray[np.int64],
    ) -> TransferResult:
        """Train the quantum transfer learning model.

        Steps:
        1. Fit PCA on raw features (classical pre-training)
        2. Transform data through frozen PCA
        3. Optimise VQC parameters on reduced features

        Parameters
        ----------
        X_raw : array of shape (n_samples, n_raw_features)
            Raw financial features.
        y : array of shape (n_samples,)
            Labels in {0, 1, 2} (buy/hold/sell).

        Returns
        -------
        TransferResult
        """
        start = time.perf_counter()

        # Classical pre-training: PCA
        X_pca = self._pca.fit_transform(X_raw)

        # Scale PCA features to [0, pi] for angle encoding
        X_scaled = self._scale_features(X_pca)

        # Optimise VQC head
        init_params = self._rng.uniform(0, 2 * np.pi, self._n_params())
        loss_history: list[float] = []

        def callback_loss(p: NDArray[np.float64]) -> float:
            loss = self._cross_entropy_loss(p, X_scaled, y)
            loss_history.append(loss)
            return loss

        result = minimize(
            callback_loss,
            init_params,
            method=self.config.optimizer,
            options={"maxiter": self.config.max_iter},
        )
        self._params = result.x

        # Compute train accuracy
        predictions = self._predict_transformed(X_scaled)
        accuracy = float(np.mean(predictions == y))

        return TransferResult(
            pca_components=self._pca.components_.copy(),
            pca_mean=self._pca.mean_.copy(),
            pca_explained_variance_ratio=self._pca.explained_variance_ratio_.copy(),
            vqc_params=self._params.copy(),
            loss_history=loss_history,
            train_accuracy=accuracy,
            wall_time_s=time.perf_counter() - start,
        )

    def predict(self, X_raw: NDArray[np.float64]) -> NDArray[np.int64]:
        """Predict trading signals for raw features.

        Parameters
        ----------
        X_raw : array of shape (n_samples, n_raw_features)

        Returns
        -------
        array of shape (n_samples,) with labels in {0, 1, 2}
        """
        if self._params is None:
            raise RuntimeError("Must call fit() first.")
        X_pca = self._pca.transform(X_raw)
        X_scaled = self._scale_features(X_pca)
        return self._predict_transformed(X_scaled)

    def predict_proba(self, X_raw: NDArray[np.float64]) -> NDArray[np.float64]:
        """Predict class probabilities for raw features.

        Parameters
        ----------
        X_raw : array of shape (n_samples, n_raw_features)

        Returns
        -------
        array of shape (n_samples, n_classes)
        """
        if self._params is None:
            raise RuntimeError("Must call fit() first.")
        X_pca = self._pca.transform(X_raw)
        X_scaled = self._scale_features(X_pca)
        n = X_scaled.shape[0]
        probs = np.zeros((n, self.config.n_classes), dtype=np.float64)
        for i in range(n):
            probs[i] = self._circuit_class_probabilities(X_scaled[i], self._params)
        return probs

    def _predict_transformed(
        self, X: NDArray[np.float64]
    ) -> NDArray[np.int64]:
        """Predict on already-transformed features."""
        assert self._params is not None
        n = X.shape[0]
        predictions = np.zeros(n, dtype=np.int64)
        for i in range(n):
            probs = self._circuit_class_probabilities(X[i], self._params)
            predictions[i] = np.argmax(probs)
        return predictions

    @staticmethod
    def _scale_features(X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Scale features to [0, pi] for angle encoding."""
        mins = X.min(axis=0)
        maxs = X.max(axis=0)
        ranges = maxs - mins
        ranges[ranges == 0] = 1.0
        return (X - mins) / ranges * np.pi


class ClassicalTransferLearner:
    """Classical transfer learning baseline: PCA + MLP head.

    Replaces the VQC head with a simple MLP (multi-layer perceptron)
    for comparison.
    """

    def __init__(
        self,
        n_pca_components: int = 4,
        hidden_sizes: list[int] | None = None,
        n_classes: int = 3,
        learning_rate: float = 0.01,
        n_epochs: int = 200,
        seed: int | None = 42,
    ) -> None:
        self.n_pca = n_pca_components
        self.hidden_sizes = hidden_sizes or [16, 8]
        self.n_classes = n_classes
        self.lr = learning_rate
        self.n_epochs = n_epochs
        self._rng = np.random.default_rng(seed)
        self._pca = PCAFeatureExtractor(n_components=n_pca_components, seed=seed)
        self._weights: list[NDArray[np.float64]] = []
        self._biases: list[NDArray[np.float64]] = []

    def _init_mlp(self, input_dim: int) -> None:
        """Initialise MLP weights."""
        layers = [input_dim, *self.hidden_sizes, self.n_classes]
        self._weights = []
        self._biases = []
        for i in range(len(layers) - 1):
            # Xavier initialisation
            scale = np.sqrt(2.0 / (layers[i] + layers[i + 1]))
            w = self._rng.normal(0, scale, (layers[i], layers[i + 1]))
            b = np.zeros(layers[i + 1], dtype=np.float64)
            self._weights.append(w)
            self._biases.append(b)

    @staticmethod
    def _relu(x: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.maximum(0, x)

    @staticmethod
    def _softmax(x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Row-wise softmax."""
        if x.ndim == 1:
            x = x - x.max()
            e = np.exp(x)
            return e / e.sum()
        x_shifted = x - x.max(axis=1, keepdims=True)
        e = np.exp(x_shifted)
        return e / e.sum(axis=1, keepdims=True)

    def _forward(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Forward pass through MLP. Returns class probabilities."""
        h = X
        for i in range(len(self._weights) - 1):
            h = self._relu(h @ self._weights[i] + self._biases[i])
        logits = h @ self._weights[-1] + self._biases[-1]
        return self._softmax(logits)

    def _loss(
        self, X: NDArray[np.float64], y: NDArray[np.int64]
    ) -> float:
        """Cross-entropy loss."""
        eps = 1e-10
        probs = self._forward(X)
        n = X.shape[0]
        return float(-np.sum(np.log(np.clip(probs[np.arange(n), y], eps, 1.0))) / n)

    def fit(
        self,
        X_raw: NDArray[np.float64],
        y: NDArray[np.int64],
    ) -> ClassicalTransferResult:
        """Train classical transfer model.

        Parameters
        ----------
        X_raw : array of shape (n_samples, n_raw_features)
        y : array of shape (n_samples,) with labels in {0, 1, 2}

        Returns
        -------
        ClassicalTransferResult
        """
        start = time.perf_counter()

        # PCA pre-training
        X_pca = self._pca.fit_transform(X_raw)
        X_scaled = QuantumTransferLearner._scale_features(X_pca)

        # Initialise MLP
        self._init_mlp(X_scaled.shape[1])
        loss_history: list[float] = []

        # Training with numerical gradients
        eps = 1e-4
        for _epoch in range(self.n_epochs):
            loss = self._loss(X_scaled, y)
            loss_history.append(loss)

            # Update weights via finite differences
            for w_idx in range(len(self._weights)):
                grad_w = np.zeros_like(self._weights[w_idx])
                for i in range(min(self._weights[w_idx].size, 100)):
                    idx = np.unravel_index(i, self._weights[w_idx].shape)
                    self._weights[w_idx][idx] += eps
                    l_plus = self._loss(X_scaled, y)
                    self._weights[w_idx][idx] -= 2 * eps
                    l_minus = self._loss(X_scaled, y)
                    self._weights[w_idx][idx] += eps
                    grad_w[idx] = (l_plus - l_minus) / (2 * eps)
                self._weights[w_idx] -= self.lr * grad_w

                grad_b = np.zeros_like(self._biases[w_idx])
                for i in range(self._biases[w_idx].size):
                    self._biases[w_idx][i] += eps
                    l_plus = self._loss(X_scaled, y)
                    self._biases[w_idx][i] -= 2 * eps
                    l_minus = self._loss(X_scaled, y)
                    self._biases[w_idx][i] += eps
                    grad_b[i] = (l_plus - l_minus) / (2 * eps)
                self._biases[w_idx] -= self.lr * grad_b

        # Accuracy
        probs = self._forward(X_scaled)
        preds = np.argmax(probs, axis=1)
        accuracy = float(np.mean(preds == y))

        return ClassicalTransferResult(
            pca_components=self._pca.components_.copy(),
            pca_mean=self._pca.mean_.copy(),
            mlp_weights=[w.copy() for w in self._weights],
            mlp_biases=[b.copy() for b in self._biases],
            loss_history=loss_history,
            train_accuracy=accuracy,
            wall_time_s=time.perf_counter() - start,
        )

    def predict(self, X_raw: NDArray[np.float64]) -> NDArray[np.int64]:
        """Predict trading signals."""
        if not self._weights:
            raise RuntimeError("Must call fit() first.")
        X_pca = self._pca.transform(X_raw)
        X_scaled = QuantumTransferLearner._scale_features(X_pca)
        probs = self._forward(X_scaled)
        return np.argmax(probs, axis=1).astype(np.int64)

    def predict_proba(self, X_raw: NDArray[np.float64]) -> NDArray[np.float64]:
        """Predict class probabilities."""
        if not self._weights:
            raise RuntimeError("Must call fit() first.")
        X_pca = self._pca.transform(X_raw)
        X_scaled = QuantumTransferLearner._scale_features(X_pca)
        return self._forward(X_scaled)


def generate_synthetic_trading_data(
    n_samples: int = 200,
    n_features: int = 10,
    n_classes: int = 3,
    seed: int = 42,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Generate synthetic financial features and trading signals.

    Creates correlated features with class-dependent distributions
    to mimic realistic financial data.

    Parameters
    ----------
    n_samples : int
    n_features : int
    n_classes : int
    seed : int

    Returns
    -------
    X : array of shape (n_samples, n_features)
    y : array of shape (n_samples,) with labels in {0, ..., n_classes-1}
    """
    rng = np.random.default_rng(seed)

    # Generate class labels
    y = rng.integers(0, n_classes, size=n_samples).astype(np.int64)

    # Generate features with class-dependent means
    X = np.zeros((n_samples, n_features), dtype=np.float64)
    for c in range(n_classes):
        mask = y == c
        n_c = mask.sum()
        mean = rng.normal(0, 1, n_features) * (c + 1) * 0.5
        X[mask] = rng.normal(mean, 1.0, (n_c, n_features))

    # Add correlations
    cov_factor = rng.normal(0, 0.3, (n_features, n_features))
    cov_matrix = cov_factor.T @ cov_factor + np.eye(n_features)
    L = np.linalg.cholesky(cov_matrix)
    X = X @ L.T

    return X, y
