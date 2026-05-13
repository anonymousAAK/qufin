"""GARCH volatility forecasting for risk management.

Implements GARCH(p,q), EGARCH, and GJR-GARCH models for conditional
volatility estimation and forecasting via the ``arch`` library.

References
----------
Bollerslev, "Generalized Autoregressive Conditional Heteroskedasticity",
    Journal of Econometrics (1986).
Engle & Ng, "Measuring and Testing the Impact of News on Volatility",
    Journal of Finance (1993).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class GARCHResult:
    """Result container for a fitted GARCH-family model.

    Attributes:
        params: Fitted parameters (omega, alpha, beta, etc.).
        conditional_vol: Fitted conditional volatility series.
        forecast: h-step-ahead volatility forecast.
        model_type: Model variant that was fitted (e.g. "garch", "egarch", "gjr").
        loglikelihood: Maximised log-likelihood value.
        aic: Akaike Information Criterion.
        bic: Bayesian Information Criterion.
    """

    params: dict[str, float]
    conditional_vol: NDArray[np.float64]
    forecast: NDArray[np.float64]
    model_type: str
    loglikelihood: float
    aic: float
    bic: float


_MODEL_MAP: dict[str, str] = {
    "garch": "GARCH",
    "egarch": "EGARCH",
    "gjr": "GARCH",
}


def fit_garch(
    returns: NDArray[np.float64],
    model: str = "garch",
    p: int = 1,
    q: int = 1,
    dist: str = "normal",
    horizon: int = 1,
) -> GARCHResult:
    """Fit a GARCH-family model and produce volatility forecasts.

    Args:
        returns: Array of asset returns (e.g. log-returns).
        model: Model type — ``"garch"``, ``"egarch"``, or ``"gjr"``
            (GJR-GARCH with leverage term).
        p: Lag order of the GARCH component (variance lags).
        q: Lag order of the ARCH component (squared-return lags).
        dist: Error distribution — ``"normal"``, ``"t"``, or ``"skewt"``.
        horizon: Number of steps ahead to forecast.

    Returns:
        A :class:`GARCHResult` with fitted parameters, conditional
        volatility series, and *h*-step-ahead forecast.

    Raises:
        ValueError: If *model* is not one of the supported types.
    """
    from arch import arch_model  # type: ignore[import-untyped]

    model_lower = model.lower()
    if model_lower not in _MODEL_MAP:
        raise ValueError(
            f"Unknown model '{model}'. Choose from: {sorted(_MODEL_MAP)}"
        )

    vol_model = _MODEL_MAP[model_lower]
    o = 1 if model_lower == "gjr" else 0

    am = arch_model(
        returns,
        vol=vol_model,
        p=p,
        o=o,
        q=q,
        dist=dist,
        mean="Constant",
    )
    res = am.fit(disp="off")

    # Extract conditional volatility (annualised is left to the caller)
    cond_vol: NDArray[np.float64] = np.asarray(
        res.conditional_volatility, dtype=np.float64
    )

    # h-step-ahead variance forecast
    fcast = res.forecast(horizon=horizon)
    # fcast.variance is a DataFrame; last row contains the forecast
    fcast_var: NDArray[np.float64] = np.asarray(
        fcast.variance.iloc[-1].values, dtype=np.float64
    )
    fcast_vol: NDArray[np.float64] = np.sqrt(fcast_var)

    # Information criteria
    n_params = len(res.params)
    n_obs = len(returns)
    ll = float(res.loglikelihood)
    aic = -2.0 * ll + 2.0 * n_params
    bic = -2.0 * ll + n_params * np.log(n_obs)

    return GARCHResult(
        params=dict(res.params),
        conditional_vol=cond_vol,
        forecast=fcast_vol,
        model_type=model_lower,
        loglikelihood=ll,
        aic=float(aic),
        bic=float(bic),
    )


def rolling_garch_forecast(
    returns: NDArray[np.float64],
    model: str = "garch",
    window: int = 252,
    horizon: int = 1,
    step: int = 1,
) -> NDArray[np.float64]:
    """Rolling-window GARCH fit with out-of-sample volatility forecasts.

    At each step the model is re-estimated on the most recent *window*
    observations and a 1-step-ahead volatility forecast is produced.

    Args:
        returns: Full array of asset returns.
        model: GARCH variant (see :func:`fit_garch`).
        window: Number of observations in each estimation window.
        horizon: Forecast horizon (only the first step is stored).
        step: Step size between successive re-estimations.

    Returns:
        1-D array of out-of-sample volatility forecasts, one per step.
    """
    n = len(returns)
    if window >= n:
        raise ValueError(
            f"Window ({window}) must be smaller than the number of "
            f"observations ({n})."
        )

    forecasts: list[float] = []
    for start in range(0, n - window, step):
        end = start + window
        sub = returns[start:end]
        result = fit_garch(sub, model=model, horizon=horizon)
        # First step of the forecast
        forecasts.append(float(result.forecast[0]))

    return np.asarray(forecasts, dtype=np.float64)


def forecast_evaluate(
    actual_vol: NDArray[np.float64],
    predicted_vol: NDArray[np.float64],
) -> dict[str, float]:
    """Evaluate volatility forecast accuracy.

    Args:
        actual_vol: Realised volatility proxy (e.g. squared returns or
            rolling standard deviation).
        predicted_vol: Model-predicted volatility values, aligned with
            *actual_vol*.

    Returns:
        Dict with keys ``"mae"``, ``"rmse"``, ``"qlike"``, ``"mz_r2"``.

    Notes:
        * **QLIKE** is the quasi-likelihood loss:
          ``mean(actual / predicted^2 - log(actual / predicted^2) - 1)``.
        * **Mincer-Zarnowitz R-squared** is the R-squared from regressing
          realised volatility on a constant and the forecast.
    """
    actual = np.asarray(actual_vol, dtype=np.float64).ravel()
    predicted = np.asarray(predicted_vol, dtype=np.float64).ravel()

    if len(actual) != len(predicted):
        raise ValueError(
            f"Length mismatch: actual ({len(actual)}) vs "
            f"predicted ({len(predicted)})."
        )

    # MAE
    mae = float(np.mean(np.abs(actual - predicted)))

    # RMSE
    rmse = float(np.sqrt(np.mean((actual - predicted) ** 2)))

    # QLIKE loss (using variance proxy = actual^2 and forecast variance =
    # predicted^2 when both inputs are volatilities)
    actual_var = actual ** 2
    pred_var = predicted ** 2
    # Guard against zero / negative predicted variance
    pred_var_safe = np.where(pred_var > 0, pred_var, 1e-12)
    ratio = actual_var / pred_var_safe
    qlike = float(np.mean(ratio - np.log(ratio) - 1.0))

    # Mincer-Zarnowitz regression: actual = a + b * predicted + eps
    # R^2 = 1 - SS_res / SS_tot
    x = np.column_stack([np.ones_like(predicted), predicted])
    # OLS via normal equations
    beta, *_ = np.linalg.lstsq(x, actual, rcond=None)
    fitted = x @ beta
    ss_res = float(np.sum((actual - fitted) ** 2))
    ss_tot = float(np.sum((actual - np.mean(actual)) ** 2))
    mz_r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return {
        "mae": mae,
        "rmse": rmse,
        "qlike": qlike,
        "mz_r2": mz_r2,
    }
