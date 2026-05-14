"""Data quality framework: gap detection, outlier flagging, corporate action
adjustment, data lineage tracking, and per-ticker quality scoring.

All computations use only numpy and pandas (no extra dependencies).
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Data lineage
# ---------------------------------------------------------------------------

@dataclass
class TransformationStep:
    """Single transformation applied to a dataset.

    Parameters
    ----------
    name : str
        Short identifier (e.g. ``"split_adjust"``, ``"outlier_flag"``).
    description : str
        Human-readable description.
    timestamp : datetime.datetime
        When the transformation was applied (UTC).
    parameters : dict
        Arbitrary parameters used in the transformation.
    """

    name: str
    description: str
    timestamp: dt.datetime = field(
        default_factory=lambda: dt.datetime.now(dt.timezone.utc)
    )
    parameters: dict = field(default_factory=dict)


@dataclass
class DataLineage:
    """Provenance record for a price series.

    Parameters
    ----------
    ticker : str
        Ticker symbol.
    source : str
        Data provider name.
    fetch_timestamp : datetime.datetime
        When the raw data was obtained (UTC).
    transformations : list[TransformationStep]
        Ordered list of transformations applied.
    """

    ticker: str
    source: str
    fetch_timestamp: dt.datetime = field(
        default_factory=lambda: dt.datetime.now(dt.timezone.utc)
    )
    transformations: list[TransformationStep] = field(default_factory=list)

    def add_step(
        self,
        name: str,
        description: str,
        parameters: dict | None = None,
    ) -> None:
        """Append a transformation step."""
        self.transformations.append(
            TransformationStep(
                name=name,
                description=description,
                parameters=parameters or {},
            )
        )


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class GapReport:
    """Result of gap detection.

    Attributes
    ----------
    missing_dates : list[datetime.date]
        Trading days with no data.
    total_expected : int
        Number of expected trading days.
    total_present : int
        Number of trading days actually present.
    gap_fraction : float
        Fraction of expected days that are missing.
    """

    missing_dates: list[dt.date]
    total_expected: int
    total_present: int
    gap_fraction: float


@dataclass
class OutlierReport:
    """Result of outlier detection.

    Attributes
    ----------
    outlier_dates : list[datetime.date]
        Dates where daily return exceeded the sigma threshold.
    outlier_returns : list[float]
        Corresponding return values.
    sigma_threshold : float
        Threshold used (number of standard deviations).
    mean : float
        Mean of daily returns.
    std : float
        Standard deviation of daily returns.
    """

    outlier_dates: list[dt.date]
    outlier_returns: list[float]
    sigma_threshold: float
    mean: float
    std: float


@dataclass
class QualityScore:
    """Per-ticker quality assessment.

    Each sub-score is in [0, 1]; ``overall`` is their weighted average.

    Attributes
    ----------
    ticker : str
        Ticker symbol.
    completeness : float
        1 - gap_fraction.
    freshness : float
        Decays with age of most recent observation.
    consistency : float
        1 - (outlier fraction).
    overall : float
        Weighted combination.
    """

    ticker: str
    completeness: float
    freshness: float
    consistency: float
    overall: float


# ---------------------------------------------------------------------------
# Gap detection
# ---------------------------------------------------------------------------

def _us_trading_days(start: dt.date, end: dt.date) -> pd.DatetimeIndex:
    """Return US equity trading calendar between *start* and *end* (inclusive).

    Uses ``pandas.bdate_range`` (Mon-Fri) as a reasonable proxy.
    """
    return pd.bdate_range(start, end)


def detect_gaps(
    prices: pd.Series | pd.DataFrame,
    trading_calendar: pd.DatetimeIndex | None = None,
) -> GapReport:
    """Detect missing trading days in a price series.

    Parameters
    ----------
    prices : pd.Series or pd.DataFrame
        Price data with a ``DatetimeIndex``.  If DataFrame, the index is
        used (gaps are detected at the row level).
    trading_calendar : pd.DatetimeIndex, optional
        Expected trading days.  If *None*, ``pandas.bdate_range`` is used.

    Returns
    -------
    GapReport
    """
    if prices.empty:
        return GapReport(
            missing_dates=[],
            total_expected=0,
            total_present=0,
            gap_fraction=0.0,
        )

    idx = prices.index
    start = idx.min().date()
    end = idx.max().date()

    if trading_calendar is None:
        expected = _us_trading_days(start, end)
    else:
        expected = trading_calendar[
            (trading_calendar >= pd.Timestamp(start))
            & (trading_calendar <= pd.Timestamp(end))
        ]

    present = idx.normalize().unique()
    missing = expected.difference(present)

    total_expected = len(expected)
    total_present = len(present)
    gap_frac = len(missing) / total_expected if total_expected > 0 else 0.0

    return GapReport(
        missing_dates=[d.date() for d in missing],
        total_expected=total_expected,
        total_present=total_present,
        gap_fraction=gap_frac,
    )


# ---------------------------------------------------------------------------
# Outlier detection
# ---------------------------------------------------------------------------

def detect_outliers(
    prices: pd.Series,
    sigma_threshold: float = 5.0,
) -> OutlierReport:
    """Flag daily returns that exceed *sigma_threshold* standard deviations.

    Parameters
    ----------
    prices : pd.Series
        Price series with ``DatetimeIndex``.
    sigma_threshold : float
        Number of standard deviations beyond which a return is flagged.

    Returns
    -------
    OutlierReport
    """
    if len(prices) < 2:
        return OutlierReport(
            outlier_dates=[],
            outlier_returns=[],
            sigma_threshold=sigma_threshold,
            mean=0.0,
            std=0.0,
        )

    returns = prices.pct_change().dropna()
    mean = float(returns.mean())
    std = float(returns.std(ddof=1))

    if std == 0.0:
        return OutlierReport(
            outlier_dates=[],
            outlier_returns=[],
            sigma_threshold=sigma_threshold,
            mean=mean,
            std=0.0,
        )

    z_scores = np.abs((returns - mean) / std)
    mask = z_scores > sigma_threshold
    flagged = returns[mask]

    return OutlierReport(
        outlier_dates=[d.date() for d in flagged.index],
        outlier_returns=flagged.tolist(),
        sigma_threshold=sigma_threshold,
        mean=mean,
        std=std,
    )


# ---------------------------------------------------------------------------
# Corporate-action adjustments
# ---------------------------------------------------------------------------

@dataclass
class SplitEvent:
    """Stock split event.

    Parameters
    ----------
    date : datetime.date
        Effective date of the split.
    ratio : float
        Split ratio (e.g. 2.0 for a 2-for-1 split).
    """

    date: dt.date
    ratio: float


@dataclass
class DividendEvent:
    """Cash dividend event.

    Parameters
    ----------
    date : datetime.date
        Ex-dividend date.
    amount : float
        Dividend per share.
    """

    date: dt.date
    amount: float


def adjust_for_splits(
    prices: pd.Series,
    splits: Sequence[SplitEvent],
    lineage: DataLineage | None = None,
) -> pd.Series:
    """Adjust historical prices for stock splits.

    Prices *before* the split date are divided by the cumulative split ratio
    so that the series is continuous.

    Parameters
    ----------
    prices : pd.Series
        Raw price series (``DatetimeIndex``).
    splits : sequence of SplitEvent
        Split events to apply, in any order.
    lineage : DataLineage, optional
        If provided, a transformation step is recorded.

    Returns
    -------
    pd.Series
        Adjusted price series.
    """
    if not splits or prices.empty:
        return prices.copy()

    adjusted = prices.copy().astype(float)

    for split in sorted(splits, key=lambda s: s.date):
        mask = adjusted.index < pd.Timestamp(split.date)
        adjusted.loc[mask] = adjusted.loc[mask] / split.ratio

    if lineage is not None:
        lineage.add_step(
            name="split_adjust",
            description="Adjusted historical prices for stock splits",
            parameters={"splits": [(str(s.date), s.ratio) for s in splits]},
        )

    return adjusted


def adjust_for_dividends(
    prices: pd.Series,
    dividends: Sequence[DividendEvent],
    lineage: DataLineage | None = None,
) -> pd.Series:
    """Adjust historical prices for cash dividends.

    Prices *before* the ex-dividend date are reduced by the ratio
    ``(close - dividend) / close`` on the ex-date, so the series is
    continuous.

    Parameters
    ----------
    prices : pd.Series
        Raw price series (``DatetimeIndex``).
    dividends : sequence of DividendEvent
        Dividend events to apply.
    lineage : DataLineage, optional
        If provided, a transformation step is recorded.

    Returns
    -------
    pd.Series
        Adjusted price series.
    """
    if not dividends or prices.empty:
        return prices.copy()

    adjusted = prices.copy().astype(float)

    for div in sorted(dividends, key=lambda d: d.date, reverse=True):
        ts = pd.Timestamp(div.date)
        mask = adjusted.index < ts
        # Find the closing price on or just after the ex-date
        on_or_after = adjusted.loc[adjusted.index >= ts]
        if on_or_after.empty:
            continue
        close_on_ex = float(on_or_after.iloc[0])
        if close_on_ex == 0.0:
            continue
        factor = (close_on_ex - div.amount) / close_on_ex
        adjusted.loc[mask] = adjusted.loc[mask] * factor

    if lineage is not None:
        lineage.add_step(
            name="dividend_adjust",
            description="Adjusted historical prices for cash dividends",
            parameters={
                "dividends": [(str(d.date), d.amount) for d in dividends]
            },
        )

    return adjusted


# ---------------------------------------------------------------------------
# Quality score
# ---------------------------------------------------------------------------

def compute_quality_score(
    prices: pd.Series,
    ticker: str,
    reference_date: dt.date | None = None,
    freshness_halflife_days: int = 30,
    trading_calendar: pd.DatetimeIndex | None = None,
    sigma_threshold: float = 5.0,
    weights: tuple[float, float, float] = (0.4, 0.3, 0.3),
) -> QualityScore:
    """Compute a composite quality score for a single ticker.

    Parameters
    ----------
    prices : pd.Series
        Price series with ``DatetimeIndex``.
    ticker : str
        Ticker symbol.
    reference_date : datetime.date, optional
        "Today" for freshness calculation.  Defaults to ``date.today()``.
    freshness_halflife_days : int
        Half-life in days for exponential freshness decay.
    trading_calendar : pd.DatetimeIndex, optional
        Expected trading days for gap detection.
    sigma_threshold : float
        Sigma threshold for outlier detection.
    weights : tuple of 3 floats
        Weights for (completeness, freshness, consistency).

    Returns
    -------
    QualityScore
    """
    if reference_date is None:
        reference_date = dt.date.today()

    # --- completeness ---
    gap_report = detect_gaps(prices, trading_calendar=trading_calendar)
    completeness = 1.0 - gap_report.gap_fraction

    # --- freshness ---
    if prices.empty:
        freshness = 0.0
    else:
        last_date = prices.index.max().date()
        age_days = (reference_date - last_date).days
        age_days = max(age_days, 0)
        # exponential decay with given half-life
        freshness = float(np.exp(-np.log(2) * age_days / freshness_halflife_days))

    # --- consistency ---
    outlier_report = detect_outliers(prices, sigma_threshold=sigma_threshold)
    n_returns = max(len(prices) - 1, 0)
    if n_returns > 0:
        outlier_frac = len(outlier_report.outlier_dates) / n_returns
    else:
        outlier_frac = 0.0
    consistency = 1.0 - outlier_frac

    # --- overall ---
    w = np.array(weights, dtype=float)
    w = w / w.sum()  # normalise
    overall = float(
        w[0] * completeness + w[1] * freshness + w[2] * consistency
    )

    return QualityScore(
        ticker=ticker,
        completeness=completeness,
        freshness=freshness,
        consistency=consistency,
        overall=overall,
    )


def compute_quality_scores(
    prices: pd.DataFrame,
    reference_date: dt.date | None = None,
    **kwargs,
) -> dict[str, QualityScore]:
    """Compute quality scores for every column in a DataFrame.

    Parameters
    ----------
    prices : pd.DataFrame
        Columns are tickers; index is ``DatetimeIndex``.
    reference_date : datetime.date, optional
        Reference date for freshness.
    **kwargs
        Forwarded to :func:`compute_quality_score`.

    Returns
    -------
    dict[str, QualityScore]
    """
    scores: dict[str, QualityScore] = {}
    for col in prices.columns:
        series = prices[col].dropna()
        scores[col] = compute_quality_score(
            series,
            ticker=str(col),
            reference_date=reference_date,
            **kwargs,
        )
    return scores
