"""Autocallable / TARF pricing and quantum resource estimates.

Monte Carlo pricer for autocallable structured notes, plus resource
estimation following Chakrabarti et al. (arXiv:2012.03819), Table I.

References
----------
Chakrabarti et al. (2021), *A Threshold for Quantum Advantage in
Derivative Pricing*, arXiv:2012.03819.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import ceil, log2

import numpy as np


@dataclass
class AutocallableSpec:
    """Autocallable structured note specification.

    Parameters
    ----------
    s0 : float
        Initial spot price.
    k : float
        Strike / notional reference level.
    barrier : float
        Autocall barrier level (absolute, not relative).
        If spot >= barrier at any observation date the note is called.
    coupon : float
        Coupon paid (as fraction of notional) upon autocall.
    observation_dates : list[float]
        Observation times in years.
    r : float
        Risk-free rate.
    sigma : float
        Volatility.
    T : float
        Final maturity in years.
    """

    s0: float
    k: float
    barrier: float
    coupon: float
    observation_dates: list[float] = field(default_factory=list)
    r: float = 0.05
    sigma: float = 0.2
    T: float = 1.0


def autocallable_mc(
    spec: AutocallableSpec,
    n_paths: int = 100_000,
    seed: int | None = 42,
) -> dict[str, object]:
    """Price an autocallable note via Monte Carlo.

    At each observation date the spot is checked against the barrier.
    If ``S_t >= barrier`` the note is called and the holder receives
    ``k * (1 + coupon * t_index)``, discounted to t=0.  If the note
    survives to maturity the holder receives ``min(S_T, k)``.

    Parameters
    ----------
    spec : AutocallableSpec
        Note specification.
    n_paths : int
        Number of simulation paths.
    seed : int | None
        Random seed.

    Returns
    -------
    dict
        ``price`` (float), ``std_err`` (float),
        ``autocall_prob`` (float, fraction of paths called early).
    """
    rng = np.random.default_rng(seed)
    obs = sorted(spec.observation_dates)
    if not obs:
        obs = [spec.T]

    # Build time grid from observation dates
    times = np.array([0.0, *obs])
    dt = np.diff(times)

    n_obs = len(obs)
    z = rng.standard_normal((n_paths, n_obs))

    # Simulate spot at each observation date
    log_s = np.log(spec.s0) * np.ones(n_paths)
    spots = np.zeros((n_paths, n_obs))
    for i in range(n_obs):
        drift = (spec.r - 0.5 * spec.sigma ** 2) * dt[i]
        diffusion = spec.sigma * np.sqrt(dt[i]) * z[:, i]
        log_s = log_s + drift + diffusion
        spots[:, i] = np.exp(log_s)

    # Determine payoff for each path
    payoffs = np.zeros(n_paths)
    called = np.zeros(n_paths, dtype=bool)

    for i, t_obs in enumerate(obs):
        newly_called = (~called) & (spots[:, i] >= spec.barrier)
        coupon_payment = spec.k * (1.0 + spec.coupon * (i + 1))
        payoffs[newly_called] = coupon_payment * np.exp(-spec.r * t_obs)
        called |= newly_called

    # Paths surviving to maturity
    survivors = ~called
    s_T = spots[:, -1]
    terminal_payoff = np.minimum(s_T[survivors], spec.k)
    payoffs[survivors] = terminal_payoff * np.exp(-spec.r * spec.T)

    price = float(np.mean(payoffs))
    std_err = float(np.std(payoffs, ddof=1) / np.sqrt(n_paths))
    autocall_prob = float(np.mean(called))

    return {
        "price": price,
        "std_err": std_err,
        "autocall_prob": autocall_prob,
    }


def resource_estimate_chakrabarti(
    n_qubits: int,
    n_timesteps: int,
) -> dict[str, int]:
    """Quantum resource estimate for derivative pricing.

    Produces order-of-magnitude estimates consistent with Table I of
    Chakrabarti et al. (arXiv:2012.03819).

    Parameters
    ----------
    n_qubits : int
        Number of qubits used per asset dimension.
    n_timesteps : int
        Number of discretised time-steps.

    Returns
    -------
    dict
        ``T_count``, ``T_depth``, ``logical_qubits`` — approximate
        gate-level resource requirements.
    """
    # Following Table I scaling: T-count ~ O(n_qubits^2 * n_timesteps)
    t_count = 54 * n_qubits ** 2 * n_timesteps
    # T-depth ~ O(n_qubits * n_timesteps)
    t_depth = 12 * n_qubits * n_timesteps
    # Logical qubits ~ O(n_qubits * n_timesteps)
    logical_qubits = 3 * n_qubits * n_timesteps + ceil(2.0 * log2(n_qubits + 1))

    return {
        "T_count": t_count,
        "T_depth": t_depth,
        "logical_qubits": logical_qubits,
    }
