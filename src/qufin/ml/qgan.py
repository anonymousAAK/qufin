"""qGANs for distribution loading (Zoufal et al., npj QI 5:103, 2019).

Implements a quantum Generative Adversarial Network where the generator
is a parameterized quantum circuit and the discriminator is a classical
neural network. The trained generator circuit can then be used to load
arbitrary probability distributions into quantum states.

References
----------
Zoufal, Lucchi, Woerner, npj Quantum Information 5:103 (2019), arXiv:1904.00043.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from qufin.backends.base import Backend
from qufin.options.distributions import DistributionSpec


@dataclass
class QGANConfig:
    """Configuration for qGAN training."""

    n_qubits: int = 3
    generator_reps: int = 3
    discriminator_hidden: list[int] = field(default_factory=lambda: [64, 32])
    n_epochs: int = 200
    batch_size: int = 256
    lr_generator: float = 0.01
    lr_discriminator: float = 0.001
    shots: int = 4096
    seed: int | None = 42


@dataclass
class QGANResult:
    """Result from qGAN training."""

    generator_params: NDArray[np.float64]
    loss_history_g: list[float]
    loss_history_d: list[float]
    trained_distribution: NDArray[np.float64]
    kl_divergence: float
    wall_time_s: float


class QuantumGAN:
    """Quantum GAN for distribution loading.

    The generator is a TwoLocal-style parameterized quantum circuit.
    The discriminator is a simple classical neural network implemented
    with numpy (no torch/tensorflow dependency).
    """

    def __init__(
        self,
        target_distribution: DistributionSpec,
        config: QGANConfig,
        backend: Backend,
    ) -> None:
        self.target = target_distribution
        self.config = config
        self.backend = backend
        self._rng = np.random.default_rng(config.seed)

    def _build_generator_circuit(self, params: NDArray[np.float64]) -> Any:
        """Build the parameterized generator circuit."""
        from qiskit.circuit import QuantumCircuit

        n = self.config.n_qubits
        reps = self.config.generator_reps
        qc = QuantumCircuit(n, n)

        param_idx = 0
        for layer in range(reps + 1):
            for qubit in range(n):
                qc.ry(params[param_idx], qubit)
                param_idx += 1
                qc.rz(params[param_idx], qubit)
                param_idx += 1
            if layer < reps:
                for i in range(n - 1):
                    qc.cx(i, i + 1)
                if n > 2:
                    qc.cx(n - 1, 0)

        qc.measure(range(n), range(n))
        return qc

    def _generator_n_params(self) -> int:
        return 2 * self.config.n_qubits * (self.config.generator_reps + 1)

    def _sample_generator(self, params: NDArray[np.float64]) -> NDArray[np.float64]:
        """Sample from the generator and return empirical distribution."""
        circuit = self._build_generator_circuit(params)
        result = self.backend.run(circuit, shots=self.config.shots)

        n_states = 2**self.config.n_qubits
        dist = np.zeros(n_states)
        for bitstring, count in result.counts.items():
            idx = int(bitstring, 2)
            if idx < n_states:
                dist[idx] = count
        total = dist.sum()
        if total > 0:
            dist = dist / total
        return dist

    def _discriminator_forward(
        self,
        x: NDArray[np.float64],
        weights: list[NDArray[np.float64]],
        biases: list[NDArray[np.float64]],
    ) -> NDArray[np.float64]:
        """Forward pass through the classical discriminator."""
        h = x
        for i in range(len(weights) - 1):
            h = np.maximum(0, h @ weights[i] + biases[i])  # ReLU
        # Output layer: sigmoid
        logits = h @ weights[-1] + biases[-1]
        return 1.0 / (1.0 + np.exp(-np.clip(logits, -20, 20)))

    def _init_discriminator(self) -> tuple[list[NDArray], list[NDArray]]:
        """Initialize discriminator weights."""
        layers = [1, *self.config.discriminator_hidden, 1]
        weights = []
        biases = []
        for i in range(len(layers) - 1):
            w = self._rng.normal(0, 0.1, (layers[i], layers[i + 1]))
            b = np.zeros(layers[i + 1])
            weights.append(w)
            biases.append(b)
        return weights, biases

    def _kl_divergence(self, p: NDArray, q: NDArray) -> float:
        """KL(p || q) with epsilon smoothing."""
        eps = 1e-10
        p_safe = np.clip(p, eps, 1.0)
        q_safe = np.clip(q, eps, 1.0)
        return float(np.sum(p_safe * np.log(p_safe / q_safe)))

    def train(self) -> QGANResult:
        """Train the qGAN.

        Uses finite-difference gradient estimation for the quantum
        generator and numpy-based SGD for the discriminator.
        """
        start = time.perf_counter()

        n_g_params = self._generator_n_params()
        g_params = self._rng.uniform(0, 2 * np.pi, n_g_params)
        d_weights, d_biases = self._init_discriminator()

        target_probs = self.target.probabilities
        n_states = self.target.n_states
        values = np.linspace(0, 1, n_states).reshape(-1, 1)  # normalized input

        loss_g_history: list[float] = []
        loss_d_history: list[float] = []

        for _epoch in range(self.config.n_epochs):
            # Sample from generator
            gen_probs = self._sample_generator(g_params)

            # Train discriminator: classify real (target) vs fake (gen)
            # The discriminator receives the same grid of values but learns
            # to output high scores for states with high target probability
            # and low scores for states with high generator probability.
            for _ in range(3):
                d_out = self._discriminator_forward(values, d_weights, d_biases).flatten()

                loss_d = -float(
                    np.sum(target_probs * np.log(np.clip(d_out, 1e-10, 1)))
                    + np.sum(gen_probs * np.log(np.clip(1 - d_out, 1e-10, 1)))
                )

                def _d_loss(w_list, b_list):
                    out = self._discriminator_forward(values, w_list, b_list).flatten()
                    return -float(
                        np.sum(target_probs * np.log(np.clip(out, 1e-10, 1)))
                        + np.sum(gen_probs * np.log(np.clip(1 - out, 1e-10, 1)))
                    )

                # Numerical gradient for discriminator (all weights)
                lr_d = self.config.lr_discriminator
                eps = 1e-4
                for w_idx in range(len(d_weights)):
                    grad_w = np.zeros_like(d_weights[w_idx])
                    for i in range(d_weights[w_idx].size):
                        idx = np.unravel_index(i, d_weights[w_idx].shape)
                        d_weights[w_idx][idx] += eps
                        l_plus = _d_loss(d_weights, d_biases)
                        d_weights[w_idx][idx] -= 2 * eps
                        l_minus = _d_loss(d_weights, d_biases)
                        d_weights[w_idx][idx] += eps
                        grad_w[idx] = (l_plus - l_minus) / (2 * eps)
                    d_weights[w_idx] -= lr_d * grad_w

            # Train generator: maximize D(G(z))
            # Use parameter-shift-like finite difference
            lr_g = self.config.lr_generator
            grad_g = np.zeros(n_g_params)
            shift = np.pi / 4
            for i in range(n_g_params):
                g_plus = g_params.copy()
                g_plus[i] += shift
                gen_plus = self._sample_generator(g_plus)
                d_plus = self._discriminator_forward(values, d_weights, d_biases).flatten()
                loss_plus = -float(np.sum(gen_plus * np.log(np.clip(d_plus, 1e-10, 1))))

                g_minus = g_params.copy()
                g_minus[i] -= shift
                gen_minus = self._sample_generator(g_minus)
                d_minus = self._discriminator_forward(values, d_weights, d_biases).flatten()
                loss_minus = -float(np.sum(gen_minus * np.log(np.clip(d_minus, 1e-10, 1))))

                grad_g[i] = (loss_plus - loss_minus) / (2 * shift)

            g_params -= lr_g * grad_g

            loss_g = -float(np.sum(gen_probs * np.log(np.clip(
                self._discriminator_forward(values, d_weights, d_biases).flatten(),
                1e-10, 1))))
            loss_g_history.append(loss_g)
            loss_d_history.append(loss_d)

        # Final distribution from trained generator
        trained_dist = self._sample_generator(g_params)
        kl = self._kl_divergence(target_probs, trained_dist)

        return QGANResult(
            generator_params=g_params,
            loss_history_g=loss_g_history,
            loss_history_d=loss_d_history,
            trained_distribution=trained_dist,
            kl_divergence=kl,
            wall_time_s=time.perf_counter() - start,
        )
