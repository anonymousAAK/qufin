"""Sector rotation with quantum classifiers for regime detection.

Detects market regimes (risk_on, risk_off, crisis) from macro features
using a variational quantum classifier, then rotates sector allocations
based on the predicted regime.

References
----------
Nystrup, Hansen, Madsen, Lindstrom, Journal of Banking & Finance (2017).
Schuld, Bocharov, Svore, Killoran, PRA 101, 032308 (2020).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

import numpy as np
from numpy.typing import NDArray

from qufin.backends.base import Backend
from qufin.ml.classifiers import VariationalQuantumClassifier, VQCConfig

# --------------------------------------------------------------------------
# Regime enum
# --------------------------------------------------------------------------

class Regime(IntEnum):
    """Market regime labels."""

    RISK_ON = 0
    RISK_OFF = 1
    CRISIS = 2


REGIME_NAMES: dict[int, str] = {
    Regime.RISK_ON: "risk_on",
    Regime.RISK_OFF: "risk_off",
    Regime.CRISIS: "crisis",
}

# --------------------------------------------------------------------------
# Default sectors
# --------------------------------------------------------------------------

DEFAULT_SECTORS: list[str] = [
    "Technology",
    "Healthcare",
    "Financials",
    "Energy",
    "Consumer Discretionary",
    "Consumer Staples",
    "Utilities",
    "Industrials",
    "Materials",
    "Real Estate",
    "Communication Services",
]

# --------------------------------------------------------------------------
# Regime detector
# --------------------------------------------------------------------------

FEATURE_NAMES: list[str] = ["vix", "yield_curve_slope", "pmi", "credit_spread"]


@dataclass
class RegimeDetectorConfig:
    """Configuration for the regime detector.

    Parameters
    ----------
    n_qubits : int
        Number of qubits for the VQC (4-8 recommended).
    n_layers : int
        Depth of the variational ansatz.
    n_epochs : int
        Maximum optimizer iterations.
    seed : int | None
        Random seed for reproducibility.
    """

    n_qubits: int = 4
    n_layers: int = 2
    n_epochs: int = 100
    seed: int | None = 42


class RegimeDetector:
    """Detect market regimes from macro features using a VQC ensemble.

    The detector uses one-vs-rest binary VQC classifiers for three-class
    classification (risk_on / risk_off / crisis).  Features are normalised
    to [0, pi] for angle encoding.

    Parameters
    ----------
    config : RegimeDetectorConfig
        Hyperparameters.
    backend : Backend
        Quantum backend for circuit execution.
    """

    def __init__(self, config: RegimeDetectorConfig, backend: Backend) -> None:
        self.config = config
        self.backend = backend
        self._classifiers: dict[int, VariationalQuantumClassifier] = {}
        self._feature_min: NDArray[np.float64] | None = None
        self._feature_max: NDArray[np.float64] | None = None
        self._fitted = False

    # ------------------------------------------------------------------
    # Feature engineering helpers
    # ------------------------------------------------------------------

    @staticmethod
    def build_features(
        vix: NDArray[np.float64],
        yield_curve_slope: NDArray[np.float64],
        pmi: NDArray[np.float64],
        credit_spread: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Stack macro indicators into a feature matrix.

        Parameters
        ----------
        vix : 1-D array
            VIX index values.
        yield_curve_slope : 1-D array
            10Y-2Y yield spread (bps).
        pmi : 1-D array
            Purchasing Managers' Index.
        credit_spread : 1-D array
            Investment-grade credit spread (bps).

        Returns
        -------
        NDArray of shape (n_samples, 4)
        """
        return np.column_stack([vix, yield_curve_slope, pmi, credit_spread])

    @staticmethod
    def label_regimes(
        vix: NDArray[np.float64],
        pmi: NDArray[np.float64],
        *,
        crisis_vix: float = 30.0,
        risk_off_vix: float = 20.0,
        risk_off_pmi: float = 50.0,
    ) -> NDArray[np.int64]:
        """Heuristic labelling based on VIX thresholds and PMI.

        Parameters
        ----------
        vix, pmi : 1-D arrays of the same length.
        crisis_vix : VIX threshold above which we declare *crisis*.
        risk_off_vix : VIX threshold for *risk_off* (when PMI < 50).
        risk_off_pmi : PMI threshold below which conditions are risk-off.

        Returns
        -------
        1-D int array with values in {0, 1, 2} (Regime enum).
        """
        n = len(vix)
        labels = np.full(n, Regime.RISK_ON, dtype=np.int64)
        labels[vix >= crisis_vix] = Regime.CRISIS
        mask_riskoff = (vix >= risk_off_vix) & (vix < crisis_vix) & (pmi < risk_off_pmi)
        labels[mask_riskoff] = Regime.RISK_OFF
        return labels

    # ------------------------------------------------------------------
    # Normalisation
    # ------------------------------------------------------------------

    def _normalise(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Scale features to [0, pi] for angle encoding."""
        assert self._feature_min is not None
        assert self._feature_max is not None
        denom = self._feature_max - self._feature_min
        denom[denom < 1e-12] = 1.0
        return np.pi * (X - self._feature_min) / denom

    # ------------------------------------------------------------------
    # Train / predict
    # ------------------------------------------------------------------

    def fit(
        self, X: NDArray[np.float64], y: NDArray[np.int64]
    ) -> RegimeDetector:
        """Train one-vs-rest VQC classifiers.

        Parameters
        ----------
        X : array of shape (n_samples, n_features)
        y : array of shape (n_samples,) with values in {0, 1, 2}

        Returns
        -------
        self
        """
        self._feature_min = X.min(axis=0).astype(np.float64)
        self._feature_max = X.max(axis=0).astype(np.float64)
        X_norm = self._normalise(X)

        for regime in (Regime.RISK_ON, Regime.RISK_OFF, Regime.CRISIS):
            binary_y = (y == regime).astype(np.int64)
            vqc_cfg = VQCConfig(
                n_qubits=self.config.n_qubits,
                n_layers=self.config.n_layers,
                n_epochs=self.config.n_epochs,
                seed=self.config.seed,
            )
            clf = VariationalQuantumClassifier(vqc_cfg, self.backend)
            clf.fit(X_norm, binary_y)
            self._classifiers[int(regime)] = clf

        self._fitted = True
        return self

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.int64]:
        """Predict regime labels for feature matrix *X*.

        Uses the one-vs-rest classifier with the highest positive-class
        probability.

        Parameters
        ----------
        X : array of shape (n_samples, n_features)

        Returns
        -------
        1-D int array of predicted regime labels.
        """
        if not self._fitted:
            raise RuntimeError("RegimeDetector has not been fitted.")
        X_norm = self._normalise(X)

        scores = np.zeros((X_norm.shape[0], 3), dtype=np.float64)
        for regime in (Regime.RISK_ON, Regime.RISK_OFF, Regime.CRISIS):
            proba = self._classifiers[int(regime)].predict_proba(X_norm)
            scores[:, int(regime)] = proba[:, 1]  # positive-class prob

        return np.argmax(scores, axis=1).astype(np.int64)

    def predict_regime_name(self, X: NDArray[np.float64]) -> list[str]:
        """Predict regime labels as human-readable strings."""
        preds = self.predict(X)
        return [REGIME_NAMES[int(p)] for p in preds]


# --------------------------------------------------------------------------
# Sector rotator
# --------------------------------------------------------------------------

@dataclass
class SectorWeightProfile:
    """Per-regime sector weight overrides.

    Weights that are *not* specified fall back to equal-weight.
    """

    risk_on: dict[str, float] = field(default_factory=dict)
    risk_off: dict[str, float] = field(default_factory=dict)
    crisis: dict[str, float] = field(default_factory=dict)


# Sensible defaults ---------------------------------------------------

_DEFAULT_RISK_ON: dict[str, float] = {
    "Technology": 0.20,
    "Consumer Discretionary": 0.15,
    "Financials": 0.15,
    "Industrials": 0.12,
    "Communication Services": 0.10,
    "Healthcare": 0.08,
    "Energy": 0.06,
    "Materials": 0.05,
    "Real Estate": 0.04,
    "Consumer Staples": 0.03,
    "Utilities": 0.02,
}

_DEFAULT_RISK_OFF: dict[str, float] = {
    "Consumer Staples": 0.18,
    "Healthcare": 0.17,
    "Utilities": 0.16,
    "Communication Services": 0.10,
    "Technology": 0.09,
    "Industrials": 0.08,
    "Financials": 0.07,
    "Real Estate": 0.05,
    "Energy": 0.04,
    "Materials": 0.03,
    "Consumer Discretionary": 0.03,
}

_DEFAULT_CRISIS: dict[str, float] = {
    "Utilities": 0.20,
    "Consumer Staples": 0.20,
    "Healthcare": 0.18,
    "Energy": 0.10,
    "Communication Services": 0.08,
    "Materials": 0.06,
    "Industrials": 0.05,
    "Real Estate": 0.05,
    "Financials": 0.04,
    "Technology": 0.02,
    "Consumer Discretionary": 0.02,
}


def _default_profiles() -> SectorWeightProfile:
    return SectorWeightProfile(
        risk_on=dict(_DEFAULT_RISK_ON),
        risk_off=dict(_DEFAULT_RISK_OFF),
        crisis=dict(_DEFAULT_CRISIS),
    )


class SectorRotator:
    """Map a regime prediction to sector allocation weights.

    Parameters
    ----------
    sectors : list[str] | None
        Sector names.  Defaults to 11 GICS sectors.
    profiles : SectorWeightProfile | None
        Per-regime weight maps.  ``None`` uses sensible defaults.
    """

    def __init__(
        self,
        sectors: list[str] | None = None,
        profiles: SectorWeightProfile | None = None,
    ) -> None:
        self.sectors = list(sectors or DEFAULT_SECTORS)
        self.profiles = profiles or _default_profiles()
        self._regime_map: dict[int, dict[str, float]] = {
            Regime.RISK_ON: self.profiles.risk_on,
            Regime.RISK_OFF: self.profiles.risk_off,
            Regime.CRISIS: self.profiles.crisis,
        }

    def allocate(self, regime: int | Regime) -> dict[str, float]:
        """Return sector weights for the given *regime*.

        Missing sectors are filled so weights sum to 1.0.

        Parameters
        ----------
        regime : int or Regime enum value.

        Returns
        -------
        dict mapping sector name to weight (sums to 1.0).
        """
        profile = self._regime_map.get(int(regime), {})
        weights: dict[str, float] = {}
        total_assigned = 0.0
        unassigned: list[str] = []

        for s in self.sectors:
            if s in profile:
                weights[s] = profile[s]
                total_assigned += profile[s]
            else:
                unassigned.append(s)

        # Distribute remaining weight equally
        remaining = max(0.0, 1.0 - total_assigned)
        if unassigned:
            per_sector = remaining / len(unassigned)
            for s in unassigned:
                weights[s] = per_sector
        elif abs(total_assigned - 1.0) > 1e-9:
            # Re-normalise to sum to 1
            for s in weights:
                weights[s] /= total_assigned

        return weights

    def allocate_array(self, regime: int | Regime) -> NDArray[np.float64]:
        """Return weights as an array aligned with ``self.sectors``."""
        w = self.allocate(regime)
        return np.array([w.get(s, 0.0) for s in self.sectors], dtype=np.float64)

    def allocate_timeseries(
        self, regimes: NDArray[np.int64]
    ) -> NDArray[np.float64]:
        """Return weight matrix of shape ``(T, n_sectors)`` for a regime time series."""
        T = len(regimes)
        n = len(self.sectors)
        weights = np.zeros((T, n), dtype=np.float64)
        for t, r in enumerate(regimes):
            weights[t] = self.allocate_array(r)
        return weights


# --------------------------------------------------------------------------
# Backtest
# --------------------------------------------------------------------------

@dataclass
class BacktestResult:
    """Results from a sector rotation backtest."""

    total_return: float
    annualised_return: float
    annualised_vol: float
    sharpe_ratio: float
    max_drawdown: float
    buy_hold_return: float
    strategy_returns: NDArray[np.float64]
    buy_hold_returns: NDArray[np.float64]
    regime_sequence: NDArray[np.int64]

    def as_dict(self) -> dict[str, Any]:
        """Serialise scalar metrics to a plain dict."""
        return {
            "total_return": self.total_return,
            "annualised_return": self.annualised_return,
            "annualised_vol": self.annualised_vol,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown": self.max_drawdown,
            "buy_hold_return": self.buy_hold_return,
        }


def _max_drawdown(cumulative: NDArray[np.float64]) -> float:
    """Compute maximum drawdown from a cumulative-return series."""
    peak = np.maximum.accumulate(cumulative)
    drawdowns = (peak - cumulative) / np.where(peak > 0, peak, 1.0)
    return float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0


def backtest_sector_rotation(
    sector_returns: NDArray[np.float64],
    regimes: NDArray[np.int64],
    sectors: list[str] | None = None,
    profiles: SectorWeightProfile | None = None,
    *,
    annual_periods: int = 252,
    risk_free_rate: float = 0.0,
) -> BacktestResult:
    """Backtest a sector rotation strategy against equal-weight buy-and-hold.

    Parameters
    ----------
    sector_returns : array of shape (T, n_sectors)
        Daily (or periodic) sector returns.
    regimes : array of shape (T,)
        Regime label for each period.
    sectors : list[str] | None
        Sector names matching columns of *sector_returns*.
    profiles : SectorWeightProfile | None
        Custom weight profiles.  ``None`` uses defaults.
    annual_periods : int
        Number of periods per year (252 for daily).
    risk_free_rate : float
        Annualised risk-free rate for Sharpe calculation.

    Returns
    -------
    BacktestResult
    """
    T, n_sectors = sector_returns.shape
    rotator = SectorRotator(sectors=sectors, profiles=profiles)

    # Strategy returns
    weights = rotator.allocate_timeseries(regimes)
    strat_ret = np.sum(weights * sector_returns, axis=1)

    # Buy-and-hold: equal weight
    bh_weights = np.full(n_sectors, 1.0 / n_sectors)
    bh_ret = sector_returns @ bh_weights

    # Cumulative
    strat_cum = np.cumprod(1.0 + strat_ret)
    bh_cum = np.cumprod(1.0 + bh_ret)

    total_ret = float(strat_cum[-1] - 1.0) if T > 0 else 0.0
    bh_total = float(bh_cum[-1] - 1.0) if T > 0 else 0.0

    ann_ret = float((1.0 + total_ret) ** (annual_periods / max(T, 1)) - 1.0)
    ann_vol = float(np.std(strat_ret) * np.sqrt(annual_periods)) if T > 1 else 0.0
    sharpe = (ann_ret - risk_free_rate) / ann_vol if ann_vol > 1e-12 else 0.0
    mdd = _max_drawdown(strat_cum)

    return BacktestResult(
        total_return=total_ret,
        annualised_return=ann_ret,
        annualised_vol=ann_vol,
        sharpe_ratio=sharpe,
        max_drawdown=mdd,
        buy_hold_return=bh_total,
        strategy_returns=strat_ret,
        buy_hold_returns=bh_ret,
        regime_sequence=regimes,
    )
