"""CVaR optimization (Barkoutsos et al., Quantum 4:256, 2020).

Implements CVaR as an objective function for VQE/QAOA, plus
ascending CVaR scheduling (Kolotouros & Wallden, PRR 4.023225, 2022).

In variational quantum optimization, the standard objective is the
expectation value E[C(x)] of the cost function. CVaR replaces this
with the conditional expectation of the alpha-fraction worst outcomes:

    CVaR_alpha = E[C(x) | C(x) >= VaR_alpha]

This biases the optimizer toward exploring low-energy (good) solutions,
improving convergence on combinatorial optimization problems.

References
----------
Barkoutsos, Nannicini, Robert, Tavernelli, Woerner,
"Improving Variational Quantum Optimization using CVaR",
Quantum 4:256 (2020), arXiv:1907.04769.

Kolotouros & Wallden,
"Evolving objective function for improved variational quantum optimization",
Physical Review Research 4:023225 (2022), arXiv:2105.11766.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class CVaRObjective:
    """CVaR objective function for variational optimization.

    Parameters
    ----------
    alpha : float
        CVaR confidence parameter in (0, 1].
        alpha=1.0 recovers standard expectation.
        alpha→0 focuses on the single best sample.
    """

    alpha: float = 0.5

    def __post_init__(self) -> None:
        if not 0 < self.alpha <= 1.0:
            raise ValueError(f"alpha must be in (0, 1], got {self.alpha}")

    def evaluate(
        self,
        costs: NDArray[np.float64],
        counts: NDArray[np.int64] | None = None,
    ) -> float:
        """Compute CVaR_alpha of the given cost samples.

        Parameters
        ----------
        costs : NDArray
            Cost values for each measurement outcome, shape (M,).
        counts : NDArray or None
            Counts for each outcome (if from a histogram).
            If None, each cost is counted once.

        Returns
        -------
        float
            CVaR_alpha value (lower is better for minimization).
        """
        if counts is not None:
            # Expand histogram into individual samples
            expanded = np.repeat(costs, counts.astype(int))
        else:
            expanded = np.asarray(costs)

        if len(expanded) == 0:
            return 0.0

        sorted_costs = np.sort(expanded)
        n = len(sorted_costs)
        k = max(1, int(np.ceil(self.alpha * n)))
        return float(np.mean(sorted_costs[:k]))

    def gradient_weight(
        self,
        costs: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Compute per-sample weights for CVaR gradient estimation.

        Returns weights that are 1/alpha for the alpha-fraction best
        samples and 0 otherwise. Used in parameter-shift gradient
        estimation.
        """
        n = len(costs)
        k = max(1, int(np.ceil(self.alpha * n)))
        threshold = np.sort(costs)[k - 1]
        weights = np.where(costs <= threshold, 1.0 / (self.alpha * n), 0.0)
        return weights


class AscendingCVaR:
    """Ascending CVaR schedule per Kolotouros & Wallden (2105.11766).

    Starts with a small alpha (aggressive focus on tail) and gradually
    increases toward alpha=1 (full expectation) over the optimization.

    This avoids the local minima traps of fixed-alpha CVaR while
    still benefiting from tail-focused exploration early on.

    Parameters
    ----------
    alpha_start : float
        Initial alpha (small = aggressive).
    alpha_end : float
        Final alpha (1.0 = standard expectation).
    n_steps : int
        Total optimization steps.
    schedule : str
        Schedule type: "linear", "cosine", or "exponential".
    """

    def __init__(
        self,
        alpha_start: float = 0.1,
        alpha_end: float = 1.0,
        n_steps: int = 100,
        schedule: str = "linear",
    ) -> None:
        self.alpha_start = alpha_start
        self.alpha_end = alpha_end
        self.n_steps = n_steps
        self.schedule = schedule
        self._step = 0

    def get_alpha(self) -> float:
        """Get current alpha value."""
        t = min(self._step / max(self.n_steps - 1, 1), 1.0)

        if self.schedule == "linear":
            alpha = self.alpha_start + (self.alpha_end - self.alpha_start) * t
        elif self.schedule == "cosine":
            alpha = self.alpha_start + (self.alpha_end - self.alpha_start) * (
                1 - np.cos(np.pi * t)
            ) / 2
        elif self.schedule == "exponential":
            # Exponential interpolation in log-space
            log_start = np.log(self.alpha_start)
            log_end = np.log(self.alpha_end)
            alpha = float(np.exp(log_start + (log_end - log_start) * t))
        else:
            raise ValueError(f"Unknown schedule: {self.schedule}")

        return float(np.clip(alpha, self.alpha_start, self.alpha_end))

    def get_objective(self) -> CVaRObjective:
        """Get CVaR objective with current alpha."""
        return CVaRObjective(alpha=self.get_alpha())

    def step(self) -> None:
        """Advance the schedule by one step."""
        self._step += 1

    def reset(self) -> None:
        """Reset the schedule."""
        self._step = 0


def cvar_from_samples(
    costs: NDArray[np.float64],
    alpha: float = 0.05,
) -> float:
    """Compute CVaR (Expected Shortfall) from cost samples.

    This is the standard risk management CVaR:
    CVaR_alpha = E[X | X >= VaR_alpha]  (for losses)

    For portfolio optimization (minimization), we look at the
    alpha-fraction of LOWEST costs.

    Parameters
    ----------
    costs : NDArray
        Sample costs/losses.
    alpha : float
        Tail fraction (0.05 = 5% worst cases).
    """
    sorted_costs = np.sort(costs)
    k = max(1, int(np.ceil(alpha * len(sorted_costs))))
    return float(np.mean(sorted_costs[:k]))


def portfolio_cvar(
    returns: NDArray[np.float64],
    weights: NDArray[np.float64],
    alpha: float = 0.05,
) -> float:
    """Compute portfolio CVaR (Expected Shortfall).

    Parameters
    ----------
    returns : NDArray
        Asset returns, shape (T, N).
    weights : NDArray
        Portfolio weights, shape (N,).
    alpha : float
        Tail fraction.

    Returns
    -------
    float
        Portfolio CVaR (positive = loss).
    """
    portfolio_returns = np.asarray(returns) @ np.asarray(weights)
    losses = -portfolio_returns
    sorted_losses = np.sort(losses)[::-1]  # descending
    k = max(1, int(np.ceil(alpha * len(sorted_losses))))
    return float(np.mean(sorted_losses[:k]))
