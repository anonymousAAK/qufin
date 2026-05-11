"""Cox-Ross-Rubinstein binomial tree option pricing.

Supports European and American options (calls and puts).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass
class BinomialResult:
    """Result from binomial tree pricing."""

    price: float
    delta: float
    gamma: float
    early_exercise: bool  # True if American and early exercise is optimal


def crr_tree(
    s: float,
    k: float,
    r: float,
    sigma: float,
    T: float,
    n_steps: int = 200,
    option_type: Literal["call", "put"] = "call",
    exercise: Literal["european", "american"] = "european",
    q: float = 0.0,
) -> BinomialResult:
    """Price an option using the Cox-Ross-Rubinstein binomial tree.

    Parameters
    ----------
    s : float
        Spot price.
    k : float
        Strike price.
    r : float
        Risk-free rate.
    sigma : float
        Volatility.
    T : float
        Time to expiry (years).
    n_steps : int
        Number of time steps in the tree.
    option_type : "call" or "put"
        Option type.
    exercise : "european" or "american"
        Exercise style.
    q : float
        Continuous dividend yield.

    Returns
    -------
    BinomialResult
    """
    dt = T / n_steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    p = (np.exp((r - q) * dt) - d) / (u - d)

    # Asset prices at maturity
    asset_prices = s * u ** np.arange(n_steps, -1, -1) * d ** np.arange(0, n_steps + 1)

    # Option values at maturity
    if option_type == "call":
        option_values = np.maximum(asset_prices - k, 0.0)
    else:
        option_values = np.maximum(k - asset_prices, 0.0)

    early_ex = False

    # Backward induction
    for step in range(n_steps - 1, -1, -1):
        asset_prices_step = s * u ** np.arange(step, -1, -1) * d ** np.arange(0, step + 1)
        option_values = np.exp(-r * dt) * (p * option_values[:-1] + (1 - p) * option_values[1:])

        if exercise == "american":
            if option_type == "call":
                intrinsic = np.maximum(asset_prices_step - k, 0.0)
            else:
                intrinsic = np.maximum(k - asset_prices_step, 0.0)
            exercised = intrinsic > option_values
            if np.any(exercised):
                early_ex = True
            option_values = np.maximum(option_values, intrinsic)

    price = float(option_values[0])

    # Delta and gamma from the full tree via a second backward induction
    # We need option values at steps 1 and 2 for delta/gamma.
    if n_steps >= 2:
        # Recompute option values at step 1 and step 2
        # Step 2 asset prices
        s_uu = s * u * u
        s_ud = s * u * d  # = s (since u*d=1)
        s_dd = s * d * d

        # Run a fresh backward induction on a 2-step sub-tree
        # but use the full-tree pricing for accuracy.
        # Instead, reprice with n_steps for nodes at step 1:
        # f_u = option value at node (step=1, up)
        # f_d = option value at node (step=1, down)
        # We already computed price = option_values[0] from the full tree.
        # Rerun backward induction saving step-1 and step-2 values.
        asset_prices_t = s * u ** np.arange(n_steps, -1, -1) * d ** np.arange(0, n_steps + 1)
        if option_type == "call":
            ov = np.maximum(asset_prices_t - k, 0.0)
        else:
            ov = np.maximum(k - asset_prices_t, 0.0)

        for step in range(n_steps - 1, -1, -1):
            asset_prices_step = s * u ** np.arange(step, -1, -1) * d ** np.arange(0, step + 1)
            ov = np.exp(-r * dt) * (p * ov[:-1] + (1 - p) * ov[1:])
            if exercise == "american":
                if option_type == "call":
                    intrinsic = np.maximum(asset_prices_step - k, 0.0)
                else:
                    intrinsic = np.maximum(k - asset_prices_step, 0.0)
                ov = np.maximum(ov, intrinsic)
            if step == 2:
                f_uu, f_ud, f_dd = float(ov[0]), float(ov[1]), float(ov[2])
            if step == 1:
                f_u, f_d = float(ov[0]), float(ov[1])

        s_u = s * u
        s_d = s * d
        delta_val = (f_u - f_d) / (s_u - s_d) if abs(s_u - s_d) > 1e-12 else 0.0

        # Gamma: second derivative from step-2 values
        delta_u = (f_uu - f_ud) / (s_uu - s_ud) if abs(s_uu - s_ud) > 1e-12 else 0.0
        delta_d = (f_ud - f_dd) / (s_ud - s_dd) if abs(s_ud - s_dd) > 1e-12 else 0.0
        gamma_val = (delta_u - delta_d) / (0.5 * (s_uu - s_dd)) if abs(s_uu - s_dd) > 1e-12 else 0.0
    else:
        delta_val = 0.0
        gamma_val = 0.0

    return BinomialResult(
        price=price,
        delta=delta_val,
        gamma=gamma_val,
        early_exercise=early_ex,
    )
