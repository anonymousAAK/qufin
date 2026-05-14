"""Quantum Reinforcement Learning for Dynamic Hedging.

Implements PPO (Proximal Policy Optimization) with a variational quantum
circuit (VQC) as the policy network.  The quantum policy maps market state
features (delta, gamma, vega, portfolio value) to continuous hedge ratios.

Environments simulate Black-Scholes (GBM) and Heston stochastic-volatility
dynamics with discrete daily rebalancing and transaction costs.

References
----------
Schuld, Sweke, Meyer, "Effect of data encoding on the expressive power
of variational quantum machine learning models", Phys. Rev. A 103, 032430.

Jerbi et al., "Parametrized quantum policies for reinforcement learning",
NeurIPS 2021.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
from numpy.typing import NDArray

# Optional torch dependency -------------------------------------------------
try:
    import torch  # noqa: F401

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


# ---------------------------------------------------------------------------
# Enums & configuration dataclasses
# ---------------------------------------------------------------------------

class RewardType(Enum):
    """Reward function type for the hedging environment."""

    VARIANCE = "variance"
    CVAR = "cvar"


class DynamicsType(Enum):
    """Underlying price dynamics."""

    GBM = "gbm"
    HESTON = "heston"


@dataclass
class QuantumRLHedgingConfig:
    """Configuration for quantum RL hedging.

    Attributes
    ----------
    n_qubits : int
        Number of qubits for the VQC policy.
    n_layers : int
        Number of variational layers.
    entanglement : str
        Entanglement topology: ``"linear"``, ``"full"``, or ``"circular"``.
    dynamics : DynamicsType
        Underlying price dynamics model.
    reward_type : RewardType
        Reward function for the RL agent.
    s0 : float
        Initial spot price.
    strike : float
        Option strike price.
    r : float
        Risk-free rate (annualised).
    sigma : float
        Volatility (annualised, for GBM).
    T : float
        Time to expiry in years.
    n_steps : int
        Number of hedging steps.
    transaction_cost : float
        Proportional transaction cost (e.g. 0.001 = 10 bps).
    cvar_alpha : float
        Tail probability for CVaR reward (lower alpha = deeper tail).
    n_episodes : int
        Number of training episodes.
    n_paths_per_episode : int
        Paths simulated per episode (batch size).
    lr : float
        Learning rate.
    gamma_discount : float
        Discount factor for rewards.
    clip_epsilon : float
        PPO clipping parameter.
    ppo_epochs : int
        Number of PPO optimisation passes per batch.
    seed : int or None
        Random seed.
    """

    n_qubits: int = 4
    n_layers: int = 2
    entanglement: str = "linear"
    dynamics: DynamicsType = DynamicsType.GBM
    reward_type: RewardType = RewardType.VARIANCE
    s0: float = 100.0
    strike: float = 100.0
    r: float = 0.05
    sigma: float = 0.2
    T: float = 1.0
    n_steps: int = 30
    transaction_cost: float = 0.001
    cvar_alpha: float = 0.05
    n_episodes: int = 50
    n_paths_per_episode: int = 256
    lr: float = 0.01
    gamma_discount: float = 1.0
    clip_epsilon: float = 0.2
    ppo_epochs: int = 4
    seed: int | None = 42


@dataclass
class HestonDynamicsConfig:
    """Parameters for Heston stochastic volatility dynamics.

    Attributes
    ----------
    v0 : float
        Initial variance.
    kappa : float
        Mean-reversion speed.
    theta : float
        Long-run variance.
    xi : float
        Vol-of-vol.
    rho : float
        Correlation between spot and variance Brownians.
    """

    v0: float = 0.04
    kappa: float = 2.0
    theta: float = 0.04
    xi: float = 0.3
    rho: float = -0.7


@dataclass
class HedgingEvalResult:
    """Evaluation result for a hedging policy.

    Attributes
    ----------
    pnl_mean : float
        Mean hedging P&L.
    pnl_std : float
        Standard deviation of hedging P&L.
    var_95 : float
        Value-at-Risk at 95% confidence.
    cvar_95 : float
        Conditional VaR (Expected Shortfall) at 95%.
    total_transaction_costs : float
        Mean total transaction costs incurred.
    pnl_distribution : NDArray
        Full P&L sample array.
    hedge_ratios : NDArray
        Hedge ratios over time, shape ``(n_paths, n_steps)``.
    """

    pnl_mean: float
    pnl_std: float
    var_95: float
    cvar_95: float
    total_transaction_costs: float
    pnl_distribution: NDArray[np.float64]
    hedge_ratios: NDArray[np.float64]


@dataclass
class TrainingResult:
    """Result from PPO training loop.

    Attributes
    ----------
    episode_rewards : list[float]
        Mean reward per episode.
    episode_losses : list[float]
        Policy loss per episode.
    final_params : NDArray
        Trained variational parameters.
    """

    episode_rewards: list[float] = field(default_factory=list)
    episode_losses: list[float] = field(default_factory=list)
    final_params: NDArray[np.float64] = field(
        default_factory=lambda: np.array([])
    )


# ---------------------------------------------------------------------------
# Path simulation
# ---------------------------------------------------------------------------

def simulate_gbm_paths(
    s0: float,
    r: float,
    sigma: float,
    T: float,
    n_steps: int,
    n_paths: int,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Simulate GBM spot paths.

    Returns
    -------
    NDArray of shape ``(n_paths, n_steps + 1)``
    """
    dt = T / n_steps
    z = rng.standard_normal((n_paths, n_steps))
    increments = (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
    log_s = np.zeros((n_paths, n_steps + 1))
    log_s[:, 0] = np.log(s0)
    log_s[:, 1:] = np.log(s0) + np.cumsum(increments, axis=1)
    return np.exp(log_s)


def simulate_heston_paths(
    s0: float,
    r: float,
    T: float,
    n_steps: int,
    n_paths: int,
    heston: HestonDynamicsConfig,
    rng: np.random.Generator,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Simulate Heston stochastic-volatility paths (Euler full-truncation).

    Returns
    -------
    spot_paths : NDArray of shape ``(n_paths, n_steps + 1)``
    var_paths  : NDArray of shape ``(n_paths, n_steps + 1)``
    """
    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)

    spot = np.full((n_paths, n_steps + 1), s0)
    var_ = np.full((n_paths, n_steps + 1), heston.v0)

    for t in range(n_steps):
        z1 = rng.standard_normal(n_paths)
        z2 = rng.standard_normal(n_paths)
        # Correlated Brownians
        w1 = z1
        w2 = heston.rho * z1 + np.sqrt(1 - heston.rho**2) * z2

        v_t = np.maximum(var_[:, t], 0.0)
        sqrt_v = np.sqrt(v_t)

        spot[:, t + 1] = spot[:, t] * np.exp(
            (r - 0.5 * v_t) * dt + sqrt_v * sqrt_dt * w1
        )
        var_[:, t + 1] = (
            v_t
            + heston.kappa * (heston.theta - v_t) * dt
            + heston.xi * sqrt_v * sqrt_dt * w2
        )
        var_[:, t + 1] = np.maximum(var_[:, t + 1], 0.0)

    return spot, var_


# ---------------------------------------------------------------------------
# Greeks (Black-Scholes, for state features)
# ---------------------------------------------------------------------------

def _bs_delta(
    spot: NDArray, strike: float, r: float, sigma: float, tau: float,
) -> NDArray:
    """Vectorised Black-Scholes call delta."""
    if tau <= 0:
        return np.where(spot > strike, 1.0, 0.0)
    from scipy.stats import norm

    d1 = (np.log(spot / strike) + (r + 0.5 * sigma**2) * tau) / (
        sigma * np.sqrt(tau)
    )
    return norm.cdf(d1)


def _bs_gamma(
    spot: NDArray, strike: float, r: float, sigma: float, tau: float,
) -> NDArray:
    """Vectorised Black-Scholes call gamma."""
    if tau <= 0:
        return np.zeros_like(spot)
    from scipy.stats import norm

    d1 = (np.log(spot / strike) + (r + 0.5 * sigma**2) * tau) / (
        sigma * np.sqrt(tau)
    )
    return norm.pdf(d1) / (spot * sigma * np.sqrt(tau))


def _bs_vega(
    spot: NDArray, strike: float, r: float, sigma: float, tau: float,
) -> NDArray:
    """Vectorised Black-Scholes call vega."""
    if tau <= 0:
        return np.zeros_like(spot)
    from scipy.stats import norm

    d1 = (np.log(spot / strike) + (r + 0.5 * sigma**2) * tau) / (
        sigma * np.sqrt(tau)
    )
    return spot * norm.pdf(d1) * np.sqrt(tau)


# ---------------------------------------------------------------------------
# Reward computation
# ---------------------------------------------------------------------------

def compute_reward(
    pnl: NDArray[np.float64],
    reward_type: RewardType,
    alpha: float = 0.05,
) -> float:
    """Compute scalar reward from a batch of P&L values.

    Parameters
    ----------
    pnl : NDArray
        1-D array of P&L values.
    reward_type : RewardType
        Which objective to use.
    alpha : float
        Tail fraction for CVaR.

    Returns
    -------
    float
        Negative variance or negative CVaR (higher is better).
    """
    if reward_type == RewardType.VARIANCE:
        return -float(np.var(pnl))

    # CVaR: expected value of worst alpha-fraction
    sorted_pnl = np.sort(pnl)
    n_tail = max(1, int(np.floor(alpha * len(pnl))))
    tail_mean = float(np.mean(sorted_pnl[:n_tail]))
    return tail_mean  # already negative if losses dominate


# ---------------------------------------------------------------------------
# VQC policy circuit
# ---------------------------------------------------------------------------

def build_vqc_policy(
    n_qubits: int, n_layers: int, entanglement: str = "linear",
) -> Any:
    """Build the VQC policy circuit (encoding + ansatz).

    The circuit uses RY angle encoding on the first min(4, n_qubits)
    qubits for market-state features, followed by a hardware-efficient
    TwoLocal ansatz.

    Returns
    -------
    tuple of (encoding_circuit, ansatz_circuit, n_params)
    """
    from qiskit.circuit import QuantumCircuit
    from qiskit.circuit.library import TwoLocal

    # Encoding placeholder (actual angles bound at runtime)
    encoding = QuantumCircuit(n_qubits, name="encoding")
    for i in range(min(4, n_qubits)):
        encoding.ry(0.0, i)  # placeholder

    ansatz = TwoLocal(
        num_qubits=n_qubits,
        rotation_blocks=["ry", "rz"],
        entanglement_blocks="cx",
        entanglement=entanglement,
        reps=n_layers,
        insert_barriers=False,
    )
    n_params = len(ansatz.parameters)

    return encoding, ansatz, n_params


def _encode_market_state(
    n_qubits: int,
    features: NDArray[np.float64],
) -> Any:
    """Amplitude-encode market state features into RY rotations.

    Features are normalised to [0, pi] range for encoding.

    Parameters
    ----------
    n_qubits : int
        Number of qubits.
    features : NDArray
        1-D array of up to 4 values: (delta, gamma, vega, portfolio_value).
    """
    from qiskit.circuit import QuantumCircuit

    qc = QuantumCircuit(n_qubits)
    for i, f in enumerate(features[: min(4, n_qubits)]):
        # Map feature to [0, pi] via sigmoid-like scaling
        angle = float(np.pi * (1.0 / (1.0 + np.exp(-f))))
        qc.ry(angle, i)
    return qc


def evaluate_vqc_policy(
    params: NDArray[np.float64],
    features: NDArray[np.float64],
    n_qubits: int,
    n_layers: int,
    entanglement: str = "linear",
) -> float:
    """Evaluate the VQC policy and return a continuous hedge ratio.

    The hedge ratio is derived from the expectation value of Z on qubit 0,
    mapped to [0, 1] via ``(1 + <Z>) / 2``.

    Parameters
    ----------
    params : NDArray
        Variational parameter vector.
    features : NDArray
        Market state features.
    n_qubits, n_layers, entanglement
        Ansatz specification.

    Returns
    -------
    float
        Hedge ratio in [0, 1].
    """
    from qiskit.circuit import QuantumCircuit
    from qiskit.quantum_info import Statevector

    _, ansatz, _ = build_vqc_policy(n_qubits, n_layers, entanglement)

    # Build full circuit
    encoding = _encode_market_state(n_qubits, features)
    qc = QuantumCircuit(n_qubits)
    qc.compose(encoding, inplace=True)
    qc.compose(ansatz, inplace=True)

    # Bind parameters
    param_symbols = list(ansatz.parameters)
    if len(params) != len(param_symbols):
        raise ValueError(
            f"Expected {len(param_symbols)} params, got {len(params)}"
        )
    bind_dict = dict(zip(param_symbols, params.tolist(), strict=False))
    qc.assign_parameters(bind_dict, inplace=True)

    # Statevector evaluation
    sv = Statevector.from_instruction(qc)
    probs = sv.probabilities()

    # <Z_0> expectation
    n_states = 2**n_qubits
    z_exp = 0.0
    for state_idx in range(n_states):
        bit = (state_idx >> (n_qubits - 1)) & 1  # qubit 0
        z_exp += probs[state_idx] * (1 - 2 * bit)

    # Map <Z> in [-1, 1] to hedge ratio in [0, 1]
    return float((1.0 + z_exp) / 2.0)


# ---------------------------------------------------------------------------
# Hedging environment
# ---------------------------------------------------------------------------

class HedgingEnvironment:
    """Hedging environment with daily rebalancing and transaction costs.

    Parameters
    ----------
    config : QuantumRLHedgingConfig
        Environment configuration.
    heston_config : HestonDynamicsConfig or None
        Heston parameters (only used when ``dynamics == HESTON``).
    """

    def __init__(
        self,
        config: QuantumRLHedgingConfig,
        heston_config: HestonDynamicsConfig | None = None,
    ) -> None:
        self.config = config
        self.heston_config = heston_config or HestonDynamicsConfig()
        self._rng = np.random.default_rng(config.seed)

    def simulate_paths(
        self, n_paths: int | None = None,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64] | None]:
        """Generate price paths according to configured dynamics.

        Returns
        -------
        spot_paths : NDArray of shape ``(n_paths, n_steps + 1)``
        var_paths  : NDArray or None (only for Heston)
        """
        cfg = self.config
        n = n_paths or cfg.n_paths_per_episode

        if cfg.dynamics == DynamicsType.HESTON:
            return simulate_heston_paths(
                cfg.s0, cfg.r, cfg.T, cfg.n_steps, n,
                self.heston_config, self._rng,
            )
        # GBM
        paths = simulate_gbm_paths(
            cfg.s0, cfg.r, cfg.sigma, cfg.T, cfg.n_steps, n, self._rng,
        )
        return paths, None

    def compute_features(
        self,
        spot: NDArray[np.float64],
        tau: float,
        position: NDArray[np.float64],
        portfolio_value: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Compute market-state features for the policy.

        Parameters
        ----------
        spot : NDArray, shape ``(n_paths,)``
            Current spot prices.
        tau : float
            Time to expiry.
        position : NDArray, shape ``(n_paths,)``
            Current hedge position.
        portfolio_value : NDArray, shape ``(n_paths,)``
            Current portfolio value.

        Returns
        -------
        NDArray of shape ``(n_paths, 4)``
            Features: (delta, gamma, vega, normalised_portfolio_value).
        """
        cfg = self.config
        sigma = cfg.sigma

        delta = _bs_delta(spot, cfg.strike, cfg.r, sigma, tau)
        gamma = _bs_gamma(spot, cfg.strike, cfg.r, sigma, tau)
        vega = _bs_vega(spot, cfg.strike, cfg.r, sigma, tau)

        # Normalise portfolio value
        pv_norm = portfolio_value / cfg.s0

        return np.column_stack([delta, gamma, vega, pv_norm])

    def run_episode(
        self,
        policy_fn: Any,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Run one episode of hedging.

        Parameters
        ----------
        policy_fn : callable
            ``policy_fn(features) -> hedge_ratios`` where features has
            shape ``(n_paths, 4)`` and hedge_ratios shape ``(n_paths,)``.

        Returns
        -------
        pnl : NDArray, shape ``(n_paths,)``
            Terminal hedging P&L.
        total_costs : NDArray, shape ``(n_paths,)``
            Total transaction costs incurred.
        all_ratios : NDArray, shape ``(n_paths, n_steps)``
            Hedge ratios at each step.
        """
        cfg = self.config
        dt = cfg.T / cfg.n_steps

        paths, _ = self.simulate_paths()
        n_paths = paths.shape[0]

        position = np.zeros(n_paths)
        cash = np.zeros(n_paths)
        total_costs = np.zeros(n_paths)
        all_ratios = np.zeros((n_paths, cfg.n_steps))

        for t in range(cfg.n_steps):
            s_t = paths[:, t]
            tau = cfg.T - t * dt
            pv = position * s_t + cash

            features = self.compute_features(s_t, tau, position, pv)
            ratios = policy_fn(features)
            ratios = np.clip(ratios, 0.0, 1.0)
            all_ratios[:, t] = ratios

            trade = ratios - position
            cost = cfg.transaction_cost * np.abs(trade) * s_t
            total_costs += cost

            cash -= trade * s_t + cost
            cash *= np.exp(cfg.r * dt)
            position = ratios

        # Terminal P&L
        s_T = paths[:, -1]
        payoff = np.maximum(s_T - cfg.strike, 0.0)
        portfolio = position * s_T + cash
        pnl = portfolio - payoff

        return pnl, total_costs, all_ratios


# ---------------------------------------------------------------------------
# Quantum policy wrapper
# ---------------------------------------------------------------------------

class QuantumHedgingPolicy:
    """VQC-based continuous hedge-ratio policy.

    Parameters
    ----------
    config : QuantumRLHedgingConfig
        Configuration (n_qubits, n_layers, entanglement used).
    """

    def __init__(self, config: QuantumRLHedgingConfig) -> None:
        self.config = config
        _, ansatz, self.n_params = build_vqc_policy(
            config.n_qubits, config.n_layers, config.entanglement,
        )
        self._ansatz = ansatz
        self._rng = np.random.default_rng(config.seed)
        self.params = self._rng.uniform(
            -np.pi, np.pi, size=self.n_params,
        )

    def __call__(self, features: NDArray[np.float64]) -> NDArray[np.float64]:
        """Evaluate policy for a batch of feature vectors.

        Parameters
        ----------
        features : NDArray, shape ``(n_paths, 4)``

        Returns
        -------
        NDArray, shape ``(n_paths,)``
            Hedge ratios in [0, 1].
        """
        n_paths = features.shape[0]
        ratios = np.empty(n_paths)
        for i in range(n_paths):
            ratios[i] = evaluate_vqc_policy(
                self.params,
                features[i],
                self.config.n_qubits,
                self.config.n_layers,
                self.config.entanglement,
            )
        return ratios

    def get_hedge_ratio(
        self, features: NDArray[np.float64],
    ) -> float:
        """Single-sample hedge ratio evaluation.

        Parameters
        ----------
        features : NDArray, shape ``(4,)``

        Returns
        -------
        float
            Hedge ratio in [0, 1].
        """
        return evaluate_vqc_policy(
            self.params,
            features,
            self.config.n_qubits,
            self.config.n_layers,
            self.config.entanglement,
        )


# ---------------------------------------------------------------------------
# Classical baseline policy (deep hedging MLP)
# ---------------------------------------------------------------------------

class ClassicalHedgingPolicy:
    """Simple numpy MLP policy for classical deep hedging baseline.

    Parameters
    ----------
    hidden_dim : int
        Hidden layer width.
    n_layers : int
        Number of hidden layers.
    seed : int or None
        Random seed.
    """

    def __init__(
        self,
        hidden_dim: int = 32,
        n_layers: int = 2,
        seed: int | None = 42,
    ) -> None:
        self.hidden_dim = hidden_dim
        self.n_layers_count = n_layers
        rng = np.random.default_rng(seed)

        # Build layer sizes: 4 -> hidden -> ... -> 1
        sizes = [4] + [hidden_dim] * n_layers + [1]
        self.params: list[tuple[NDArray, NDArray]] = []
        for i in range(len(sizes) - 1):
            fan_in, fan_out = sizes[i], sizes[i + 1]
            scale = np.sqrt(2.0 / (fan_in + fan_out))
            w = rng.normal(0, scale, (fan_in, fan_out))
            b = np.zeros(fan_out)
            self.params.append((w, b))

    def __call__(self, features: NDArray[np.float64]) -> NDArray[np.float64]:
        """Evaluate policy for a batch of feature vectors.

        Parameters
        ----------
        features : NDArray, shape ``(n_paths, 4)``

        Returns
        -------
        NDArray, shape ``(n_paths,)``
            Hedge ratios in [0, 1].
        """
        x = features
        for i, (w, b) in enumerate(self.params):
            z = x @ w + b
            if i < len(self.params) - 1:
                x = np.maximum(z, 0.0)  # ReLU
            else:
                x = 1.0 / (1.0 + np.exp(-z))  # sigmoid -> [0, 1]
        return x.ravel()


# ---------------------------------------------------------------------------
# PPO training (parameter-shift gradient)
# ---------------------------------------------------------------------------

def _parameter_shift_gradient(
    params: NDArray[np.float64],
    features_batch: NDArray[np.float64],
    n_qubits: int,
    n_layers: int,
    entanglement: str,
) -> NDArray[np.float64]:
    """Estimate policy gradient via parameter-shift rule.

    For each parameter, shift by +/- pi/2 and compute the difference
    in mean hedge ratio.

    Parameters
    ----------
    params : NDArray
        Current variational parameters.
    features_batch : NDArray, shape ``(n_samples, 4)``
        Batch of feature vectors (sub-sampled for efficiency).
    n_qubits, n_layers, entanglement
        Ansatz specification.

    Returns
    -------
    NDArray
        Gradient vector of length ``len(params)``.
    """
    n_params = len(params)
    grad = np.zeros(n_params)
    shift = np.pi / 2

    for j in range(n_params):
        params_plus = params.copy()
        params_minus = params.copy()
        params_plus[j] += shift
        params_minus[j] -= shift

        val_plus = 0.0
        val_minus = 0.0
        for feat in features_batch:
            val_plus += evaluate_vqc_policy(
                params_plus, feat, n_qubits, n_layers, entanglement,
            )
            val_minus += evaluate_vqc_policy(
                params_minus, feat, n_qubits, n_layers, entanglement,
            )

        grad[j] = (val_plus - val_minus) / (2 * len(features_batch))

    return grad


def train_quantum_ppo(
    config: QuantumRLHedgingConfig,
    heston_config: HestonDynamicsConfig | None = None,
    max_grad_samples: int = 8,
) -> TrainingResult:
    """Train the quantum hedging policy using PPO.

    Uses parameter-shift gradients for the VQC policy and a simple
    advantage estimation based on batch reward normalisation.

    Parameters
    ----------
    config : QuantumRLHedgingConfig
        Full configuration.
    heston_config : HestonDynamicsConfig or None
        Heston parameters (for Heston dynamics).
    max_grad_samples : int
        Number of feature samples used per gradient estimate
        (sub-sampled from episode for efficiency).

    Returns
    -------
    TrainingResult
    """
    env = HedgingEnvironment(config, heston_config)
    policy = QuantumHedgingPolicy(config)

    result = TrainingResult()
    np.random.default_rng(config.seed)

    for _episode in range(config.n_episodes):
        # Run episode
        pnl, _costs, _ratios = env.run_episode(policy)

        # Compute reward
        reward = compute_reward(pnl, config.reward_type, config.cvar_alpha)
        result.episode_rewards.append(reward)

        # Sub-sample features for gradient estimation
        paths, _ = env.simulate_paths(n_paths=max_grad_samples)
        dt = config.T / config.n_steps
        mid_t = config.n_steps // 2
        tau = config.T - mid_t * dt
        spot_sample = paths[:, mid_t]
        pos_sample = np.zeros(max_grad_samples)
        pv_sample = np.zeros(max_grad_samples)
        features_sample = env.compute_features(
            spot_sample, tau, pos_sample, pv_sample,
        )

        # Parameter-shift gradient
        grad = _parameter_shift_gradient(
            policy.params,
            features_sample,
            config.n_qubits,
            config.n_layers,
            config.entanglement,
        )

        # PPO-style update: use reward as advantage signal
        # (simplified: full PPO would track old/new log-probs)
        advantage = reward
        policy_loss = -advantage

        for _ppo_iter in range(config.ppo_epochs):
            # Gradient ascent on reward (descent on negative reward)
            update = config.lr * advantage * grad
            # Clip update magnitude (PPO-inspired)
            max_update = config.clip_epsilon * np.pi
            update = np.clip(update, -max_update, max_update)
            policy.params += update

        result.episode_losses.append(float(policy_loss))

    result.final_params = policy.params.copy()
    return result


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_policy(
    policy_fn: Any,
    config: QuantumRLHedgingConfig,
    heston_config: HestonDynamicsConfig | None = None,
    n_eval_paths: int = 1000,
) -> HedgingEvalResult:
    """Evaluate a hedging policy and compute performance metrics.

    Parameters
    ----------
    policy_fn : callable
        ``policy_fn(features) -> hedge_ratios``.
    config : QuantumRLHedgingConfig
        Environment configuration.
    heston_config : HestonDynamicsConfig or None
        Heston parameters.
    n_eval_paths : int
        Number of evaluation paths.

    Returns
    -------
    HedgingEvalResult
    """
    env = HedgingEnvironment(config, heston_config)
    pnl, costs, ratios = env.run_episode(policy_fn)

    sorted_pnl = np.sort(pnl)
    n = len(pnl)

    # VaR at 95%: 5th percentile loss
    var_idx = max(1, int(np.floor(0.05 * n)))
    var_95 = -float(sorted_pnl[var_idx - 1])

    # CVaR at 95%: mean of worst 5%
    cvar_95 = -float(np.mean(sorted_pnl[:var_idx]))

    return HedgingEvalResult(
        pnl_mean=float(np.mean(pnl)),
        pnl_std=float(np.std(pnl)),
        var_95=var_95,
        cvar_95=cvar_95,
        total_transaction_costs=float(np.mean(costs)),
        pnl_distribution=pnl,
        hedge_ratios=ratios,
    )


def compare_policies(
    config: QuantumRLHedgingConfig,
    quantum_params: NDArray[np.float64] | None = None,
    heston_config: HestonDynamicsConfig | None = None,
    n_eval_paths: int = 1000,
) -> dict[str, HedgingEvalResult]:
    """Compare quantum RL policy vs classical deep hedging baseline.

    Parameters
    ----------
    config : QuantumRLHedgingConfig
        Shared environment configuration.
    quantum_params : NDArray or None
        Pre-trained VQC parameters.  If ``None``, random params are used.
    heston_config : HestonDynamicsConfig or None
        Heston parameters.
    n_eval_paths : int
        Number of evaluation paths.

    Returns
    -------
    dict mapping ``"quantum"`` and ``"classical"`` to HedgingEvalResult.
    """
    # Quantum policy
    q_policy = QuantumHedgingPolicy(config)
    if quantum_params is not None:
        q_policy.params = quantum_params.copy()

    # Classical policy
    c_policy = ClassicalHedgingPolicy(seed=config.seed)

    results: dict[str, HedgingEvalResult] = {}

    # Use same seed for fair comparison
    eval_config = QuantumRLHedgingConfig(
        **{
            k: v
            for k, v in config.__dict__.items()
            if k != "n_paths_per_episode"
        },
        n_paths_per_episode=n_eval_paths,
    )
    # Fix: rebuild with correct n_paths
    eval_cfg_dict = config.__dict__.copy()
    eval_cfg_dict["n_paths_per_episode"] = n_eval_paths
    eval_config = QuantumRLHedgingConfig(**eval_cfg_dict)

    results["quantum"] = evaluate_policy(
        q_policy, eval_config, heston_config, n_eval_paths,
    )
    results["classical"] = evaluate_policy(
        c_policy, eval_config, heston_config, n_eval_paths,
    )

    return results


# ---------------------------------------------------------------------------
# Convenience: full train + evaluate pipeline
# ---------------------------------------------------------------------------

class QuantumRLHedger:
    """End-to-end quantum RL hedger with PPO training.

    Parameters
    ----------
    config : QuantumRLHedgingConfig
        Full configuration.
    heston_config : HestonDynamicsConfig or None
        Heston dynamics parameters (optional).
    """

    def __init__(
        self,
        config: QuantumRLHedgingConfig | None = None,
        heston_config: HestonDynamicsConfig | None = None,
    ) -> None:
        self.config = config or QuantumRLHedgingConfig()
        self.heston_config = heston_config
        self.policy = QuantumHedgingPolicy(self.config)
        self.training_result: TrainingResult | None = None

    def train(
        self, max_grad_samples: int = 8,
    ) -> TrainingResult:
        """Train the quantum policy via PPO.

        Returns
        -------
        TrainingResult
        """
        result = train_quantum_ppo(
            self.config, self.heston_config, max_grad_samples,
        )
        self.policy.params = result.final_params.copy()
        self.training_result = result
        return result

    def evaluate(
        self, n_eval_paths: int = 1000,
    ) -> HedgingEvalResult:
        """Evaluate the trained policy.

        Returns
        -------
        HedgingEvalResult
        """
        return evaluate_policy(
            self.policy, self.config, self.heston_config, n_eval_paths,
        )

    def compare(
        self, n_eval_paths: int = 1000,
    ) -> dict[str, HedgingEvalResult]:
        """Compare quantum vs classical policy.

        Returns
        -------
        dict mapping policy name to HedgingEvalResult.
        """
        return compare_policies(
            self.config,
            self.policy.params,
            self.heston_config,
            n_eval_paths,
        )

    def hedge(
        self, features: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Compute hedge ratios for given features.

        Parameters
        ----------
        features : NDArray, shape ``(n_paths, 4)`` or ``(4,)``

        Returns
        -------
        NDArray
            Hedge ratios.
        """
        if features.ndim == 1:
            features = features.reshape(1, -1)
        return self.policy(features)
