"""Quantum autoencoder for anomaly detection in financial time series.

Implements a variational quantum autoencoder (QAE) that compresses
n_qubits into n_latent qubits.  The reconstruction error of unseen
data points serves as an anomaly score: high error signals a regime
or event not captured by the training distribution.

The encoder is a parameterised quantum circuit executed via the Backend
ABC; the decoder mirrors its structure.  Training minimises the
mean-squared reconstruction error on a normalised feature window.

References
----------
Romero, Olson & Aspuru-Guzik, Quantum Sci. Technol. 2, 045001 (2017).
Sakhnenko et al., arXiv:2112.04958 -- QAE for credit fraud detection.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from qufin.backends.base import Backend


@dataclass
class QAutoEncoderConfig:
    """Configuration for the quantum autoencoder.

    Parameters
    ----------
    n_qubits : int
        Number of input qubits (must be >= n_latent).
    n_latent : int
        Number of latent (compressed) qubits.
    n_layers : int
        Number of variational layers in the encoder.
    n_epochs : int
        Training epochs.
    learning_rate : float
        Step size for parameter updates.
    shots : int
        Measurement shots per circuit evaluation.
    anomaly_threshold : float | None
        Reconstruction-error threshold for anomaly flagging.
        If None, set to mean + 2 * std of training errors.
    seed : int | None
        Random seed.
    """

    n_qubits: int = 4
    n_latent: int = 2
    n_layers: int = 3
    n_epochs: int = 30
    learning_rate: float = 0.05
    shots: int = 1024
    anomaly_threshold: float | None = None
    seed: int | None = 42


@dataclass
class QAutoEncoderResult:
    """Result from the quantum autoencoder.

    Parameters
    ----------
    anomaly_scores : NDArray
        Reconstruction error for each sample.
    is_anomaly : NDArray
        Boolean mask of detected anomalies.
    threshold : float
        Anomaly threshold used.
    loss_history : list[float]
        Training loss per epoch.
    params : NDArray
        Optimised circuit parameters.
    wall_time_s : float
        Wall-clock training time.
    """

    anomaly_scores: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(0)
    )
    is_anomaly: NDArray[np.bool_] = field(
        default_factory=lambda: np.zeros(0, dtype=np.bool_)
    )
    threshold: float = 0.0
    loss_history: list[float] = field(default_factory=list)
    params: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(0)
    )
    wall_time_s: float = 0.0


class QuantumAutoEncoder:
    """Variational quantum autoencoder for time-series anomaly detection.

    The encoder circuit maps *n_qubits* input qubits to *n_latent*
    latent qubits using parameterised RY + CNOT layers.  The trash
    qubits (n_qubits - n_latent) are measured; successful compression
    yields the ``|0>`` state on the trash register.  Reconstruction
    error is 1 minus the probability of measuring all-zero on trash.

    Parameters
    ----------
    config : QAutoEncoderConfig
        Autoencoder configuration.
    backend : Backend
        Quantum backend for circuit execution.
    """

    def __init__(self, config: QAutoEncoderConfig, backend: Backend) -> None:
        if config.n_latent >= config.n_qubits:
            raise ValueError("n_latent must be strictly less than n_qubits")
        if config.n_qubits < 2:
            raise ValueError("n_qubits must be at least 2")
        self.config = config
        self.backend = backend
        self._rng = np.random.default_rng(config.seed)
        self._n_trash = config.n_qubits - config.n_latent
        self._n_params = config.n_layers * config.n_qubits * 2
        self.params = self._rng.uniform(
            -np.pi, np.pi, size=self._n_params
        )

    def _build_circuit(
        self,
        input_state: NDArray[np.float64],
        params: NDArray[np.float64],
    ) -> object:
        """Build the encoder circuit with input preparation.

        Parameters
        ----------
        input_state : NDArray, shape (n_qubits,)
            Input features mapped to rotation angles.
        params : NDArray
            Variational parameters.

        Returns
        -------
        QuantumCircuit
        """
        from qiskit.circuit import QuantumCircuit

        nq = self.config.n_qubits
        n_trash = self._n_trash
        qc = QuantumCircuit(nq, n_trash)

        # Input encoding: RY rotations
        for i in range(nq):
            angle = float(np.pi * np.clip(input_state[i], 0, 1))
            qc.ry(angle, i)

        # Variational encoder layers
        idx = 0
        for _layer in range(self.config.n_layers):
            for i in range(nq):
                qc.ry(float(params[idx]), i)
                idx += 1
            for i in range(nq):
                qc.rz(float(params[idx]), i)
                idx += 1
            for i in range(nq - 1):
                qc.cx(i, i + 1)
            if nq > 2:
                qc.cx(nq - 1, 0)

        # Measure trash qubits
        trash_qubits = list(range(self.config.n_latent, nq))
        qc.measure(trash_qubits, list(range(n_trash)))
        return qc

    def _reconstruction_error(
        self,
        input_state: NDArray[np.float64],
        params: NDArray[np.float64],
    ) -> float:
        """Compute reconstruction error for one sample.

        Error = 1 - P(trash = 00...0).  Perfect compression
        yields error = 0.
        """
        circuit = self._build_circuit(input_state, params)
        result = self.backend.run(circuit, shots=self.config.shots)

        target = "0" * self._n_trash
        count_zero = 0
        for bitstring, count in result.counts.items():
            bits = bitstring.replace(" ", "").zfill(self._n_trash)
            if bits[-self._n_trash:] == target:
                count_zero += count

        prob_zero = count_zero / max(result.shots, 1)
        return 1.0 - prob_zero

    def _mean_loss(
        self,
        data: NDArray[np.float64],
        params: NDArray[np.float64],
    ) -> float:
        """Mean reconstruction error over the dataset."""
        errors = [self._reconstruction_error(x, params) for x in data]
        return float(np.mean(errors))

    def fit(
        self,
        data: NDArray[np.float64],
    ) -> QAutoEncoderResult:
        """Train the autoencoder and score the training data.

        Parameters
        ----------
        data : NDArray, shape (n_samples, n_qubits)
            Normalised input features in [0, 1].

        Returns
        -------
        QAutoEncoderResult
        """
        start = time.perf_counter()
        data = np.asarray(data, dtype=np.float64)
        loss_history: list[float] = []

        for _epoch in range(self.config.n_epochs):
            loss = self._mean_loss(data, self.params)
            loss_history.append(loss)

            # Parameter-shift gradient estimation
            grad = np.zeros_like(self.params)
            for i in range(len(self.params)):
                params_plus = self.params.copy()
                params_minus = self.params.copy()
                params_plus[i] += np.pi / 2
                params_minus[i] -= np.pi / 2
                grad[i] = (
                    self._mean_loss(data, params_plus)
                    - self._mean_loss(data, params_minus)
                ) / 2.0

            self.params -= self.config.learning_rate * grad

        # Score all samples
        scores = np.array(
            [self._reconstruction_error(x, self.params) for x in data]
        )

        # Determine threshold
        if self.config.anomaly_threshold is not None:
            threshold = self.config.anomaly_threshold
        else:
            threshold = float(np.mean(scores) + 2.0 * np.std(scores))

        wall_time = time.perf_counter() - start
        return QAutoEncoderResult(
            anomaly_scores=scores,
            is_anomaly=scores > threshold,
            threshold=threshold,
            loss_history=loss_history,
            params=self.params.copy(),
            wall_time_s=wall_time,
        )

    def score(
        self,
        data: NDArray[np.float64],
        threshold: float | None = None,
    ) -> QAutoEncoderResult:
        """Score new data using the trained autoencoder.

        Parameters
        ----------
        data : NDArray, shape (n_samples, n_qubits)
            Input features.
        threshold : float | None
            Anomaly threshold. If None, uses config value or 0.5.

        Returns
        -------
        QAutoEncoderResult
        """
        data = np.asarray(data, dtype=np.float64)
        scores = np.array(
            [self._reconstruction_error(x, self.params) for x in data]
        )
        if threshold is None:
            threshold = self.config.anomaly_threshold or 0.5

        return QAutoEncoderResult(
            anomaly_scores=scores,
            is_anomaly=scores > threshold,
            threshold=threshold,
            loss_history=[],
            params=self.params.copy(),
            wall_time_s=0.0,
        )


# ---------------------------------------------------------------------------
# Classical baseline: PCA reconstruction error
# ---------------------------------------------------------------------------


def classical_pca_anomaly(
    data: NDArray[np.float64],
    n_components: int = 2,
    threshold: float | None = None,
) -> QAutoEncoderResult:
    """Classical PCA-based anomaly detection for comparison.

    Projects data onto the top ``n_components`` principal components
    and measures reconstruction error as an anomaly score.

    Parameters
    ----------
    data : NDArray, shape (n_samples, n_features)
        Input features.
    n_components : int
        Number of principal components to keep.
    threshold : float | None
        Anomaly threshold. If None, mean + 2*std.

    Returns
    -------
    QAutoEncoderResult
    """
    start = time.perf_counter()
    data = np.asarray(data, dtype=np.float64)
    mean = data.mean(axis=0)
    centered = data - mean

    _U, _S, Vt = np.linalg.svd(centered, full_matrices=False)
    components = Vt[:n_components]
    projected = centered @ components.T @ components
    errors = np.sqrt(np.mean((centered - projected) ** 2, axis=1))

    if threshold is None:
        threshold = float(np.mean(errors) + 2.0 * np.std(errors))

    wall_time = time.perf_counter() - start
    return QAutoEncoderResult(
        anomaly_scores=errors,
        is_anomaly=errors > threshold,
        threshold=threshold,
        loss_history=[],
        params=np.zeros(0),
        wall_time_s=wall_time,
    )
