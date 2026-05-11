"""Deep hedging (Buhler-Gonon-Teichmann-Wood, Quantitative Finance 2019).

A minimal numpy-only implementation of deep hedging.  A small feedforward
network learns a hedging strategy that minimises P&L variance over
simulated GBM paths.  No PyTorch or TensorFlow dependency is required.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class DeepHedgingConfig:
    """Hyper-parameters for the deep-hedging network.

    Attributes
    ----------
    n_layers : int
        Number of hidden layers.
    hidden_dim : int
        Width of each hidden layer.
    n_epochs : int
        Training epochs.
    lr : float
        Learning rate for vanilla SGD.
    n_paths : int
        Number of GBM paths per training batch.
    n_steps : int
        Number of hedging time-steps.
    """

    n_layers: int = 2
    hidden_dim: int = 32
    n_epochs: int = 200
    lr: float = 1e-3
    n_paths: int = 4096
    n_steps: int = 30


# ---------------------------------------------------------------------------
# Simple numpy MLP utilities
# ---------------------------------------------------------------------------

def _init_params(
    layer_sizes: Sequence[int], rng: np.random.Generator,
) -> list[tuple[NDArray, NDArray]]:
    """Xavier-initialised weights and zero biases."""
    params: list[tuple[NDArray, NDArray]] = []
    for fan_in, fan_out in itertools.pairwise(layer_sizes):
        scale = np.sqrt(2.0 / (fan_in + fan_out))
        w = rng.normal(0, scale, (fan_in, fan_out))
        b = np.zeros(fan_out)
        params.append((w, b))
    return params


def _forward(
    x: NDArray, params: list[tuple[NDArray, NDArray]],
) -> tuple[NDArray, list[NDArray]]:
    """Forward pass returning output and pre-activation caches for back-prop."""
    caches: list[NDArray] = [x]
    for i, (w, b) in enumerate(params):
        z = x @ w + b
        if i < len(params) - 1:
            x = np.maximum(z, 0.0)  # ReLU
        else:
            x = np.tanh(z)  # output in (-1, 1) → hedge ratio
        caches.append(x)
    return x, caches


def _backward(
    grad_out: NDArray,
    params: list[tuple[NDArray, NDArray]],
    caches: list[NDArray],
) -> list[tuple[NDArray, NDArray]]:
    """Back-prop through the MLP; returns gradients w.r.t. each (W, b)."""
    grads: list[tuple[NDArray, NDArray]] = []
    g = grad_out  # (batch,1)
    for i in reversed(range(len(params))):
        a_prev = caches[i]  # activation of previous layer
        a_cur = caches[i + 1]

        if i == len(params) - 1:
            # tanh derivative
            g = g * (1 - a_cur ** 2)
        else:
            # ReLU derivative
            g = g * (a_cur > 0).astype(float)

        dw = a_prev.T @ g / g.shape[0]
        db = g.mean(axis=0)
        grads.append((dw, db))

        if i > 0:
            w, _ = params[i]
            g = g @ w.T
    grads.reverse()
    return grads


# ---------------------------------------------------------------------------
# GBM path generator
# ---------------------------------------------------------------------------

def _simulate_gbm(
    s0: float, r: float, sigma: float, T: float,
    n_steps: int, n_paths: int, rng: np.random.Generator,
) -> NDArray:
    """Return shape ``(n_paths, n_steps + 1)`` GBM spot paths."""
    dt = T / n_steps
    z = rng.standard_normal((n_paths, n_steps))
    increments = (r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z
    log_s = np.zeros((n_paths, n_steps + 1))
    log_s[:, 0] = np.log(s0)
    log_s[:, 1:] = np.log(s0) + np.cumsum(increments, axis=1)
    return np.exp(log_s)


# ---------------------------------------------------------------------------
# DeepHedger
# ---------------------------------------------------------------------------

class DeepHedger:
    """Numpy-only deep hedging agent.

    Parameters
    ----------
    config : DeepHedgingConfig
        Network / training configuration.
    s0 : float
        Initial spot price.
    strike : float
        Option strike price.
    r : float
        Risk-free rate.
    sigma : float
        Volatility.
    T : float
        Time to expiry (years).
    seed : int | None
        Random seed.
    """

    def __init__(
        self,
        config: DeepHedgingConfig | None = None,
        s0: float = 100.0,
        strike: float = 100.0,
        r: float = 0.05,
        sigma: float = 0.2,
        T: float = 1.0,
        seed: int | None = 42,
    ) -> None:
        self.cfg = config or DeepHedgingConfig()
        self.s0 = s0
        self.strike = strike
        self.r = r
        self.sigma = sigma
        self.T = T
        self._rng = np.random.default_rng(seed)

        # Network: input=(spot, tau, position) -> hedge ratio
        sizes = [3] + [self.cfg.hidden_dim] * self.cfg.n_layers + [1]
        self.params = _init_params(sizes, self._rng)

    # ------------------------------------------------------------------
    def train(self) -> list[float]:
        """Train the network to minimise hedging P&L variance.

        Returns
        -------
        list[float]
            Loss (P&L variance) at each epoch.
        """
        cfg = self.cfg
        dt = self.T / cfg.n_steps
        losses: list[float] = []

        for _epoch in range(cfg.n_epochs):
            paths = _simulate_gbm(
                self.s0, self.r, self.sigma, self.T,
                cfg.n_steps, cfg.n_paths, self._rng,
            )

            # Forward: compute hedge ratios and P&L
            position = np.zeros(cfg.n_paths)
            cash = np.zeros(cfg.n_paths)
            all_caches: list[list[NDArray]] = []
            all_ratios: list[NDArray] = []

            for t in range(cfg.n_steps):
                s_t = paths[:, t]
                tau = self.T - t * dt
                x = np.column_stack([
                    s_t / self.s0,  # normalised spot
                    np.full(cfg.n_paths, tau / self.T),
                    position,
                ])
                ratio, caches = _forward(x, self.params)
                ratio = ratio.ravel()
                all_caches.append(caches)
                all_ratios.append(ratio)

                trade = ratio - position
                cash -= trade * s_t
                cash *= np.exp(self.r * dt)
                position = ratio

            # Terminal P&L
            s_T = paths[:, -1]
            payoff = np.maximum(s_T - self.strike, 0.0)
            portfolio = position * s_T + cash
            pnl = portfolio - payoff

            loss = float(np.var(pnl))
            losses.append(loss)

            # Backward: d(Var)/d(pnl) = 2*(pnl - mean)/N
            dpnl = 2.0 * (pnl - pnl.mean()) / cfg.n_paths

            # Accumulate gradients from each time-step (truncated BPTT)
            grad_accum = [(np.zeros_like(w), np.zeros_like(b)) for w, b in self.params]

            for t in reversed(range(cfg.n_steps)):
                s_t = paths[:, t]
                # d(pnl)/d(ratio_t) = s_T  (simplified: ignoring second-order)
                d_ratio = dpnl * s_t
                grad_out = d_ratio.reshape(-1, 1)
                grads = _backward(grad_out, self.params, all_caches[t])

                for j, (dw, db) in enumerate(grads):
                    gw, gb = grad_accum[j]
                    grad_accum[j] = (gw + dw, gb + db)

            # SGD update
            for j, (w, b) in enumerate(self.params):
                gw, gb = grad_accum[j]
                self.params[j] = (w - cfg.lr * gw, b - cfg.lr * gb)

        return losses

    # ------------------------------------------------------------------
    def hedge(self, paths: NDArray) -> NDArray:
        """Compute hedge ratios for given spot paths.

        Parameters
        ----------
        paths : NDArray
            Shape ``(n_paths, n_steps + 1)`` spot price paths.

        Returns
        -------
        NDArray
            Shape ``(n_paths, n_steps)`` hedge ratios at each step.
        """
        n_paths, n_total = paths.shape
        n_steps = n_total - 1
        dt = self.T / n_steps

        ratios = np.empty((n_paths, n_steps))
        position = np.zeros(n_paths)

        for t in range(n_steps):
            tau = self.T - t * dt
            x = np.column_stack([
                paths[:, t] / self.s0,
                np.full(n_paths, tau / self.T),
                position,
            ])
            ratio, _ = _forward(x, self.params)
            ratio = ratio.ravel()
            ratios[:, t] = ratio
            position = ratio

        return ratios
