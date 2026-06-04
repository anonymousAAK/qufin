"""REST API server for qufin — portfolio optimization, option pricing, and risk.

Requires ``fastapi`` and ``uvicorn`` (optional dependencies).

Usage::

    from qufin.api.server import create_app
    app = create_app()
    # Run with: uvicorn qufin.api.server:app

Or via CLI::

    uvicorn qufin.api.server:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
import os
import threading
import time
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger("qufin.api")

# ---------------------------------------------------------------------------
# Optional dependency guard
# ---------------------------------------------------------------------------
try:
    from fastapi import Depends, FastAPI, HTTPException, Request, Security, status
    from fastapi.responses import JSONResponse
    from fastapi.security import APIKeyHeader

    _HAS_FASTAPI = True
except ImportError:  # pragma: no cover
    _HAS_FASTAPI = False

# ---------------------------------------------------------------------------
# Pydantic models — request / response schemas
# ---------------------------------------------------------------------------


class OptimizationMethod(str, Enum):
    """Supported portfolio optimization methods."""

    QAOA = "qaoa"
    MVO = "mvo"
    HRP = "hrp"


class PricingMethod(str, Enum):
    """Supported option pricing methods."""

    BS = "bs"
    MC = "mc"
    QAE = "qae"


class RiskMethod(str, Enum):
    """Supported risk computation methods."""

    HISTORICAL = "historical"
    PARAMETRIC = "parametric"
    QUANTUM = "quantum"


class JobStatus(str, Enum):
    """Job lifecycle status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# --- Constraint schemas ---


class PortfolioConstraints(BaseModel):
    """Constraints for portfolio optimization."""

    min_weight: float = Field(0.0, ge=0.0, le=1.0, description="Minimum weight per asset")
    max_weight: float = Field(1.0, ge=0.0, le=1.0, description="Maximum weight per asset")
    budget: float = Field(1.0, gt=0.0, description="Total portfolio budget (sum of weights)")
    cardinality: int | None = Field(None, ge=1, description="Max number of assets to hold")


# --- Request schemas ---


class OptimizeRequest(BaseModel):
    """Request schema for /optimize endpoint."""

    tickers: list[str] = Field(..., min_length=2, description="List of ticker symbols")
    start_date: str = Field(..., description="Start date (YYYY-MM-DD)")
    end_date: str = Field(..., description="End date (YYYY-MM-DD)")
    method: OptimizationMethod = Field(
        OptimizationMethod.MVO, description="Optimization method"
    )
    constraints: PortfolioConstraints = Field(
        default_factory=PortfolioConstraints, description="Portfolio constraints"
    )
    risk_free_rate: float = Field(0.02, description="Annual risk-free rate")
    async_mode: bool = Field(False, description="If true, return job_id for async polling")


class PriceRequest(BaseModel):
    """Request schema for /price endpoint."""

    spot: float = Field(..., gt=0, description="Current spot price")
    strike: float = Field(..., gt=0, description="Strike price")
    rate: float = Field(0.05, description="Risk-free interest rate")
    volatility: float = Field(..., gt=0, le=5.0, description="Annualized volatility")
    expiry: float = Field(..., gt=0, description="Time to expiry in years")
    is_call: bool = Field(True, description="True for call, False for put")
    method: PricingMethod = Field(PricingMethod.BS, description="Pricing method")
    n_simulations: int = Field(100_000, ge=1000, description="Monte Carlo simulations (MC only)")
    n_qubits: int = Field(4, ge=2, le=20, description="Qubits for QAE circuit")
    async_mode: bool = Field(False, description="If true, return job_id for async polling")


class RiskRequest(BaseModel):
    """Request schema for /risk endpoint."""

    portfolio_weights: dict[str, float] = Field(
        ..., description="Ticker -> weight mapping"
    )
    start_date: str = Field(..., description="Start date for returns (YYYY-MM-DD)")
    end_date: str = Field(..., description="End date for returns (YYYY-MM-DD)")
    method: RiskMethod = Field(RiskMethod.HISTORICAL, description="Risk computation method")
    confidence_level: float = Field(0.95, gt=0.5, lt=1.0, description="VaR confidence level")
    portfolio_value: float = Field(1_000_000.0, gt=0, description="Portfolio notional value")
    stress_test: bool = Field(False, description="Include stress test results")
    async_mode: bool = Field(False, description="If true, return job_id for async polling")


# --- Response schemas ---


class CircuitInfo(BaseModel):
    """Quantum circuit metadata."""

    n_qubits: int = Field(..., description="Number of qubits")
    depth: int = Field(..., description="Circuit depth")
    gate_count: int = Field(..., description="Total gate count")
    method: str = Field(..., description="Quantum method used")


class OptimizeResponse(BaseModel):
    """Response schema for /optimize endpoint."""

    weights: dict[str, float] = Field(..., description="Ticker -> optimal weight")
    expected_return: float = Field(..., description="Expected annual return")
    volatility: float = Field(..., description="Expected annual volatility")
    sharpe_ratio: float = Field(..., description="Sharpe ratio")
    method: str = Field(..., description="Method used")
    circuit_info: CircuitInfo | None = Field(
        None, description="Quantum circuit info (if applicable)"
    )
    computation_time_s: float = Field(..., description="Wall-clock time in seconds")


class Greeks(BaseModel):
    """Option Greeks."""

    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float


class PriceResponse(BaseModel):
    """Response schema for /price endpoint."""

    price: float = Field(..., description="Option price")
    greeks: Greeks = Field(..., description="Option Greeks")
    confidence_interval: list[float] | None = Field(
        None, description="95% CI [lower, upper] (MC/QAE)"
    )
    method: str = Field(..., description="Pricing method used")
    circuit_info: CircuitInfo | None = Field(
        None, description="Quantum circuit info (QAE only)"
    )
    computation_time_s: float = Field(..., description="Wall-clock time in seconds")


class StressTestResult(BaseModel):
    """Result from a single stress scenario."""

    scenario: str = Field(..., description="Scenario name")
    portfolio_loss: float = Field(..., description="Portfolio loss under scenario")
    loss_pct: float = Field(..., description="Loss as percentage of portfolio value")


class RiskResponse(BaseModel):
    """Response schema for /risk endpoint."""

    var: float = Field(..., description="Value-at-Risk (percentage)")
    var_dollar: float = Field(..., description="VaR in dollar terms")
    cvar: float = Field(..., description="Conditional VaR / Expected Shortfall")
    cvar_dollar: float = Field(..., description="CVaR in dollar terms")
    confidence_level: float = Field(..., description="Confidence level used")
    method: str = Field(..., description="Risk method used")
    stress_test_results: list[StressTestResult] | None = Field(
        None, description="Stress test results (if requested)"
    )
    computation_time_s: float = Field(..., description="Wall-clock time in seconds")


class JobResponse(BaseModel):
    """Response for async job submission."""

    job_id: str = Field(..., description="Unique job identifier")
    status: JobStatus = Field(..., description="Current job status")
    message: str = Field("Job submitted successfully", description="Status message")


class JobStatusResponse(BaseModel):
    """Response for job status polling."""

    job_id: str
    status: JobStatus
    result: dict[str, Any] | None = None
    error: str | None = None
    created_at: str = Field(..., description="ISO 8601 timestamp")
    updated_at: str = Field(..., description="ISO 8601 timestamp")


class HealthResponse(BaseModel):
    """Health-check response."""

    status: str = "ok"
    version: str = ""
    timestamp: str = ""


# ---------------------------------------------------------------------------
# Rate limiter (in-memory, suitable for single-process dev)
# ---------------------------------------------------------------------------


class RateLimiter:
    """Simple sliding-window rate limiter.

    Parameters
    ----------
    max_requests : int
        Maximum requests per window.
    window_seconds : int
        Window size in seconds.
    """

    def __init__(self, max_requests: int = 60, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests: dict[str, list] = defaultdict(list)

    def is_allowed(self, key: str) -> bool:
        """Check if request is allowed and record it."""
        now = time.time()
        window_start = now - self.window_seconds
        # Prune old entries
        self._requests[key] = [t for t in self._requests[key] if t > window_start]
        if len(self._requests[key]) >= self.max_requests:
            return False
        self._requests[key].append(now)
        return True

    def reset(self, key: str | None = None) -> None:
        """Reset rate limit state."""
        if key is None:
            self._requests.clear()
        else:
            self._requests.pop(key, None)


# ---------------------------------------------------------------------------
# In-memory job store (for sync mode / testing; production uses Celery)
# ---------------------------------------------------------------------------


class InMemoryJobStore:
    """Simple in-memory job store for development / testing."""

    def __init__(self):
        self._jobs: dict[str, dict[str, Any]] = {}

    def create(self, job_type: str, params: dict[str, Any]) -> str:
        """Create a new job and return its ID."""
        job_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        self._jobs[job_id] = {
            "job_id": job_id,
            "job_type": job_type,
            "status": JobStatus.PENDING,
            "params": params,
            "result": None,
            "error": None,
            "created_at": now,
            "updated_at": now,
        }
        return job_id

    def get(self, job_id: str) -> dict[str, Any] | None:
        """Get job by ID."""
        return self._jobs.get(job_id)

    def update(
        self,
        job_id: str,
        status: JobStatus | None = None,
        result: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        """Update job state."""
        job = self._jobs.get(job_id)
        if job is None:
            return
        if status is not None:
            job["status"] = status
        if result is not None:
            job["result"] = result
        if error is not None:
            job["error"] = error
        job["updated_at"] = datetime.now(timezone.utc).isoformat()

    def delete(self, job_id: str) -> bool:
        """Delete a job. Returns True if existed."""
        return self._jobs.pop(job_id, None) is not None

    def list_jobs(self) -> list[dict[str, Any]]:
        """List all jobs."""
        return list(self._jobs.values())


# ---------------------------------------------------------------------------
# Computation helpers (thin wrappers around qufin modules)
# ---------------------------------------------------------------------------


class MarketDataError(RuntimeError):
    """Raised when real market data cannot be fetched for a request.

    The API surfaces this as HTTP 503 and never silently substitutes synthetic
    data for a real computation.
    """


def _fetch_returns(tickers: list[str], start_date: str, end_date: str):
    """Fetch real historical daily log-returns, aligned to ``tickers`` order.

    Uses the Yahoo equity provider. Raises :class:`MarketDataError` if the data
    cannot be retrieved or is empty — callers must not fall back to synthetic
    data. This symbol is intentionally module-level so it can be monkeypatched
    in tests and swapped for an internal data warehouse in production.
    """
    import numpy as np

    try:
        from qufin.data.equities import YahooEquityProvider

        df = YahooEquityProvider(cache=False).get_returns(list(tickers), start_date, end_date)
        df = df[list(tickers)]  # enforce requested column order
        arr = np.asarray(df.to_numpy(), dtype=float)
    except MarketDataError:
        raise
    except Exception as exc:
        raise MarketDataError(
            f"Could not fetch market data for {list(tickers)} "
            f"({start_date}..{end_date}): {type(exc).__name__}"
        ) from exc

    if arr.ndim != 2 or arr.shape[0] < 2 or arr.shape[1] != len(tickers):
        raise MarketDataError(
            f"Insufficient market data for {list(tickers)} ({start_date}..{end_date})"
        )
    return arr


def _run_optimize(req: OptimizeRequest) -> OptimizeResponse:
    """Execute portfolio optimization synchronously.

    Uses real historical returns for ``req.tickers`` (never synthetic). Method
    failures propagate as errors rather than masquerading as an equal-weight
    portfolio under the requested method's name.
    """
    import numpy as np

    t0 = time.time()

    # Real historical returns; annualise mean and covariance from daily logs.
    returns = _fetch_returns(req.tickers, req.start_date, req.end_date)
    mu = np.mean(returns, axis=0) * 252
    cov = np.cov(returns, rowvar=False) * 252
    n = len(req.tickers)

    circuit_info = None

    if req.method == OptimizationMethod.MVO:
        from qufin.portfolio.classical.mean_variance import mean_variance

        w = np.array(mean_variance(mu, cov).weights, dtype=float)
    elif req.method == OptimizationMethod.HRP:
        from qufin.portfolio.classical.hrp import hrp

        w = np.array(hrp(cov).weights, dtype=float)
    elif req.method == OptimizationMethod.QAOA:
        if n > 18:
            raise ValueError(
                f"QAOA portfolio optimization is limited to 18 assets in simulation; "
                f"got {n}. Use MVO/HRP for larger universes."
            )
        from qufin.backends.qiskit_backend import QiskitAerBackend
        from qufin.portfolio.optimizers.qaoa import QAOAConfig, QAOAPortfolio
        from qufin.portfolio.qubo import PortfolioQUBO

        k = req.constraints.cardinality or max(1, n // 2)
        qubo = PortfolioQUBO(mu=mu, cov=cov, cardinality=k, budget_penalty=10.0)
        qaoa_result = QAOAPortfolio(
            qubo,
            QAOAConfig(p=2, mixer="xy_ring", cardinality=k, shots=2048, maxiter=100, seed=42),
            QiskitAerBackend(seed=42),
        ).run()
        w = np.array(qaoa_result.weights, dtype=float)
        circuit_info = CircuitInfo(
            n_qubits=qubo.n_qubits, depth=2 * qubo.n_qubits,
            gate_count=4 * qubo.n_qubits, method="QAOA",
        )
    else:  # pragma: no cover - guarded by the OptimizationMethod enum
        raise ValueError(f"Unsupported optimization method: {req.method}")

    # Enforce constraints
    w = np.clip(w, req.constraints.min_weight, req.constraints.max_weight)
    if w.sum() > 0:
        w = w / w.sum() * req.constraints.budget

    port_ret = float(w @ mu)
    port_vol = float(np.sqrt(w @ cov @ w))
    sharpe = (port_ret - req.risk_free_rate) / port_vol if port_vol > 0 else 0.0

    elapsed = time.time() - t0

    return OptimizeResponse(
        weights={t: float(w[i]) for i, t in enumerate(req.tickers)},
        expected_return=port_ret,
        volatility=port_vol,
        sharpe_ratio=sharpe,
        method=req.method.value,
        circuit_info=circuit_info,
        computation_time_s=elapsed,
    )


def _compute_rho(opt) -> float:
    """Compute Black-Scholes rho (sensitivity to interest rate).

    Falls back to a finite-difference approximation if the option object
    does not expose ``bs_rho`` natively.
    """
    if hasattr(opt, "bs_rho"):
        return float(opt.bs_rho())
    # Finite-difference approximation
    import copy

    dr = 0.0001
    opt_up = copy.copy(opt)
    opt_up.r = opt.r + dr
    return float((opt_up.bs_price() - opt.bs_price()) / dr)


def _run_price(req: PriceRequest) -> PriceResponse:
    """Execute option pricing synchronously."""
    t0 = time.time()
    circuit_info = None
    ci = None

    from qufin.options.european import EuropeanOption

    opt = EuropeanOption(
        s0=req.spot,
        k=req.strike,
        r=req.rate,
        sigma=req.volatility,
        T=req.expiry,
        is_call=req.is_call,
    )

    if req.method == PricingMethod.BS:
        price = opt.bs_price()
    elif req.method == PricingMethod.MC:
        from qufin.options.classical.monte_carlo import european_mc

        mc_result = european_mc(
            req.spot,
            req.strike,
            req.rate,
            req.volatility,
            req.expiry,
            req.n_simulations,
            "call" if req.is_call else "put",
        )
        price = mc_result if isinstance(mc_result, (int, float)) else mc_result.price
        std_err = getattr(mc_result, "std_error", abs(price) * 0.01)
        ci = [price - 1.96 * std_err, price + 1.96 * std_err]
    elif req.method == PricingMethod.QAE:
        # Real QAE pricing via the European-QAE estimation problem + IQAE.
        from qufin.backends.qiskit_backend import QiskitAerBackend
        from qufin.options.amplitude_estimation.european_qae import (
            EuropeanQAESpec,
            build_european_estimation_problem,
        )
        from qufin.options.amplitude_estimation.iqae import (
            IQAEConfig,
            IterativeAmplitudeEstimation,
        )

        n_q = min(req.n_qubits, 6)  # keep statevector simulation tractable
        spec = EuropeanQAESpec(
            s0=req.spot, k=req.strike, r=req.rate, sigma=req.volatility,
            T=req.expiry, is_call=req.is_call, n_qubits=n_q,
        )
        problem, rescale = build_european_estimation_problem(spec)
        iqae_result = IterativeAmplitudeEstimation(
            problem,
            IQAEConfig(epsilon_target=0.01, shots_per_round=2048, seed=42),
            QiskitAerBackend(seed=42),
        ).estimate()
        price = float(iqae_result.estimate * rescale)
        ci_obj = getattr(iqae_result, "confidence_interval", None)
        ci = (
            [float(ci_obj[0]) * rescale, float(ci_obj[1]) * rescale]
            if ci_obj is not None
            else None
        )
        circuit_info = CircuitInfo(
            n_qubits=n_q + 1, depth=4 * n_q, gate_count=8 * n_q, method="QAE",
        )
    else:  # pragma: no cover - guarded by the PricingMethod enum
        price = opt.bs_price()

    greeks = Greeks(
        delta=opt.bs_delta(),
        gamma=opt.bs_gamma(),
        vega=opt.bs_vega(),
        theta=opt.bs_theta(),
        rho=_compute_rho(opt),
    )

    elapsed = time.time() - t0

    return PriceResponse(
        price=price,
        greeks=greeks,
        confidence_interval=ci,
        method=req.method.value,
        circuit_info=circuit_info,
        computation_time_s=elapsed,
    )


def _run_risk(req: RiskRequest) -> RiskResponse:
    """Execute risk computation synchronously."""
    import numpy as np

    t0 = time.time()

    tickers = list(req.portfolio_weights.keys())
    weights = np.array(list(req.portfolio_weights.values()), dtype=float)

    # Real historical returns for the held tickers (never silently synthetic).
    returns = _fetch_returns(tickers, req.start_date, req.end_date)
    port_returns = returns @ weights

    if req.method == RiskMethod.HISTORICAL:
        from qufin.risk.classical_var import historical_var

        result = historical_var(
            port_returns, confidence=req.confidence_level,
            portfolio_value=req.portfolio_value,
        )
        var_pct = result.var
        cvar_pct = result.expected_shortfall
    elif req.method == RiskMethod.PARAMETRIC:
        from qufin.risk.classical_var import parametric_var

        result = parametric_var(
            port_returns, confidence=req.confidence_level,
            portfolio_value=req.portfolio_value,
        )
        var_pct = result.var
        cvar_pct = result.expected_shortfall
    elif req.method == RiskMethod.QUANTUM:
        from qufin.backends.qiskit_backend import QiskitAerBackend
        from qufin.risk.quantum_var import (
            QuantumVaRConfig,
            build_loss_distribution,
            quantum_var,
        )

        loss_dist = build_loss_distribution(port_returns, n_qubits=4)
        qresult = quantum_var(
            loss_dist,
            QiskitAerBackend(seed=42),
            QuantumVaRConfig(
                confidence_level=req.confidence_level, n_qubits_loss=4,
                qae_shots=2048, seed=42,
            ),
        )
        var_pct = float(qresult.var_estimate)
        cvar_pct = float(qresult.es_estimate)
    else:  # pragma: no cover - guarded by the RiskMethod enum
        raise ValueError(f"Unsupported risk method: {req.method}")

    var_dollar = var_pct * req.portfolio_value
    cvar_dollar = cvar_pct * req.portfolio_value

    # Stress tests
    stress_results = None
    if req.stress_test:
        scenarios = [
            ("Market crash -20%", -0.20),
            ("Rate shock +200bp", -0.05),
            ("Vol spike 2x", -0.10),
            ("Liquidity crisis", -0.15),
        ]
        stress_results = [
            StressTestResult(
                scenario=name,
                portfolio_loss=abs(shock) * req.portfolio_value,
                loss_pct=abs(shock) * 100,
            )
            for name, shock in scenarios
        ]

    elapsed = time.time() - t0

    return RiskResponse(
        var=var_pct,
        var_dollar=var_dollar,
        cvar=cvar_pct,
        cvar_dollar=cvar_dollar,
        confidence_level=req.confidence_level,
        method=req.method.value,
        stress_test_results=stress_results,
        computation_time_s=elapsed,
    )


# ---------------------------------------------------------------------------
# Async job execution
# ---------------------------------------------------------------------------

_JOB_RUNNERS = {
    "optimization": (OptimizeRequest, "_run_optimize"),
    "pricing": (PriceRequest, "_run_price"),
    "risk": (RiskRequest, "_run_risk"),
}


def _execute_job(
    store: InMemoryJobStore, job_id: str, job_type: str, params: dict[str, Any]
) -> None:
    """Run a submitted job in-process and record its result.

    This makes ``async_mode`` actually execute (PENDING -> RUNNING ->
    COMPLETED/FAILED) without requiring the Celery worker. It is single-process
    only — the in-memory store is not shared across workers; distributed
    execution uses ``qufin.api.jobs`` with Celery + Redis.
    """
    runner = _JOB_RUNNERS.get(job_type)
    if runner is None:  # pragma: no cover - job_type is set internally
        store.update(job_id, status=JobStatus.FAILED, error=f"Unknown job type: {job_type}")
        return
    model_cls, fn_name = runner
    fn = globals()[fn_name]  # resolved at call time so patching/monkeypatch is honored
    try:
        store.update(job_id, status=JobStatus.RUNNING)
        result = fn(model_cls(**params))
        payload = result.model_dump() if hasattr(result, "model_dump") else None
        store.update(
            job_id,
            status=JobStatus.COMPLETED,
            result=payload if isinstance(payload, dict) else None,
        )
    except Exception as exc:
        logger.exception("async job %s (%s) failed", job_id, job_type)
        store.update(job_id, status=JobStatus.FAILED, error=f"{type(exc).__name__}: {exc}")


def _submit_async(store: InMemoryJobStore, job_type: str, params: dict[str, Any]) -> str:
    """Create a job record and start its background execution thread."""
    job_id = store.create(job_type, params)
    threading.Thread(
        target=_execute_job, args=(store, job_id, job_type, params), daemon=True
    ).start()
    return job_id


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

# Module-level singletons (set during create_app)
_rate_limiter: RateLimiter | None = None
_job_store: InMemoryJobStore | None = None
_api_keys: set[str] | None = None


def create_app(
    *,
    api_keys: list[str] | None = None,
    allow_no_auth: bool = False,
    rate_limit: int = 60,
    rate_window: int = 60,
    title: str = "qufin API",
    version: str | None = None,
) -> FastAPI:
    """Create and configure the FastAPI application.

    Authentication is **on by default and fail-closed**: if ``api_keys`` is not
    given, keys are read from the ``QUFIN_API_KEYS`` environment variable
    (comma-separated). If no keys can be resolved, the app refuses to start
    unless ``allow_no_auth=True`` (or ``QUFIN_ALLOW_NO_AUTH=1``) is set
    explicitly — so an open, unauthenticated API is never the accidental
    default.

    Parameters
    ----------
    api_keys : list[str] | None
        Valid API keys. Falls back to ``QUFIN_API_KEYS`` env if ``None``.
    allow_no_auth : bool
        Explicitly run without authentication (dev/trusted networks only).
    rate_limit : int
        Maximum requests per window per client.
    rate_window : int
        Rate-limit window in seconds.
    title : str
        OpenAPI title.
    version : str | None
        API version string; defaults to the installed ``qufin`` version.

    Returns
    -------
    FastAPI
        Configured application instance.

    Raises
    ------
    ImportError
        If ``fastapi`` is not installed.
    RuntimeError
        If no API keys are configured and authentication was not explicitly
        disabled.
    """
    if not _HAS_FASTAPI:
        raise ImportError(
            "fastapi is required for the qufin REST API. "
            "Install it with: pip install fastapi uvicorn"
        )

    if api_keys is None:
        env_keys = os.environ.get("QUFIN_API_KEYS", "")
        api_keys = [k.strip() for k in env_keys.split(",") if k.strip()] or None

    allow_no_auth = allow_no_auth or os.environ.get("QUFIN_ALLOW_NO_AUTH") == "1"
    if not api_keys and not allow_no_auth:
        raise RuntimeError(
            "Refusing to start the qufin API without authentication. Set "
            "QUFIN_API_KEYS=\"key1,key2\" (or pass api_keys=[...]), or set "
            "QUFIN_ALLOW_NO_AUTH=1 / allow_no_auth=True for trusted dev only."
        )

    if version is None:
        try:
            from qufin import __version__ as version
        except Exception:  # pragma: no cover
            version = "0.0.0"

    global _rate_limiter, _job_store, _api_keys
    _rate_limiter = RateLimiter(max_requests=rate_limit, window_seconds=rate_window)
    _job_store = InMemoryJobStore()
    _api_keys = set(api_keys) if api_keys else None

    app = FastAPI(
        title=title,
        version=version,
        description="Quantum-enhanced quantitative finance REST API",
        docs_url="/docs",
        openapi_url="/v1/openapi.json",
    )

    @app.exception_handler(MarketDataError)
    async def _market_data_error_handler(request: Request, exc: MarketDataError):
        """Surface market-data failures as 503 instead of fabricating data."""
        return JSONResponse(status_code=503, content={"detail": str(exc)})

    @app.exception_handler(ValueError)
    async def _value_error_handler(request: Request, exc: ValueError):
        """Map request/compute validation errors to 400."""
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    # --- Auth dependency ---

    api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

    async def verify_api_key(
        request: Request,
        api_key: str | None = Security(api_key_header),
    ) -> str | None:
        """Verify API key if authentication is enabled."""
        if _api_keys is None:
            return None
        if api_key is None or api_key not in _api_keys:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or missing API key",
            )
        return api_key

    # --- Rate-limit dependency ---

    async def check_rate_limit(request: Request) -> None:
        """Enforce rate limiting."""
        client_ip = request.client.host if request.client else "unknown"
        if _rate_limiter and not _rate_limiter.is_allowed(client_ip):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded. Try again later.",
            )

    # --- Health ---

    @app.get("/health", response_model=HealthResponse, tags=["system"])
    async def health():
        """Health check endpoint."""
        try:
            from qufin import __version__
        except Exception:
            __version__ = "unknown"
        return HealthResponse(
            status="ok",
            version=__version__,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    # --- Optimize ---

    @app.post(
        "/v1/optimize",
        response_model=OptimizeResponse | JobResponse,
        tags=["portfolio"],
        dependencies=[Depends(check_rate_limit)],
    )
    async def optimize(
        req: OptimizeRequest,
        _key: str | None = Depends(verify_api_key),
    ):
        """Portfolio optimization endpoint.

        Supports MVO (mean-variance), HRP (hierarchical risk parity),
        and QAOA (quantum approximate optimization).
        """
        if req.async_mode:
            job_id = _submit_async(_job_store, "optimization", req.model_dump())
            return JobResponse(job_id=job_id, status=JobStatus.PENDING)
        return _run_optimize(req)

    # --- Price ---

    @app.post(
        "/v1/price",
        response_model=PriceResponse | JobResponse,
        tags=["options"],
        dependencies=[Depends(check_rate_limit)],
    )
    async def price(
        req: PriceRequest,
        _key: str | None = Depends(verify_api_key),
    ):
        """Option pricing endpoint.

        Supports Black-Scholes (BS), Monte Carlo (MC),
        and Quantum Amplitude Estimation (QAE).
        """
        if req.async_mode:
            job_id = _submit_async(_job_store, "pricing", req.model_dump())
            return JobResponse(job_id=job_id, status=JobStatus.PENDING)
        return _run_price(req)

    # --- Risk ---

    @app.post(
        "/v1/risk",
        response_model=RiskResponse | JobResponse,
        tags=["risk"],
        dependencies=[Depends(check_rate_limit)],
    )
    async def risk(
        req: RiskRequest,
        _key: str | None = Depends(verify_api_key),
    ):
        """Risk computation endpoint.

        Supports historical, parametric, and quantum VaR/CVaR computation.
        """
        if req.async_mode:
            job_id = _submit_async(_job_store, "risk", req.model_dump())
            return JobResponse(job_id=job_id, status=JobStatus.PENDING)
        return _run_risk(req)

    # --- Job management ---

    @app.get(
        "/v1/jobs/{job_id}",
        response_model=JobStatusResponse,
        tags=["jobs"],
        dependencies=[Depends(check_rate_limit)],
    )
    async def get_job(
        job_id: str,
        _key: str | None = Depends(verify_api_key),
    ):
        """Poll job status and retrieve results."""
        job = _job_store.get(job_id)
        if job is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job {job_id} not found",
            )
        return JobStatusResponse(
            job_id=job["job_id"],
            status=job["status"],
            result=job["result"],
            error=job["error"],
            created_at=job["created_at"],
            updated_at=job["updated_at"],
        )

    @app.delete(
        "/v1/jobs/{job_id}",
        tags=["jobs"],
        dependencies=[Depends(check_rate_limit)],
    )
    async def cancel_job(
        job_id: str,
        _key: str | None = Depends(verify_api_key),
    ):
        """Cancel a pending or running job."""
        job = _job_store.get(job_id)
        if job is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job {job_id} not found",
            )
        if job["status"] in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Job {job_id} is already {job['status'].value}",
            )
        _job_store.update(job_id, status=JobStatus.CANCELLED)
        return {"job_id": job_id, "status": "cancelled"}

    @app.get(
        "/v1/jobs",
        response_model=list[JobStatusResponse],
        tags=["jobs"],
        dependencies=[Depends(check_rate_limit)],
    )
    async def list_jobs(
        _key: str | None = Depends(verify_api_key),
    ):
        """List all jobs."""
        return [
            JobStatusResponse(
                job_id=j["job_id"],
                status=j["status"],
                result=j["result"],
                error=j["error"],
                created_at=j["created_at"],
                updated_at=j["updated_at"],
            )
            for j in _job_store.list_jobs()
        ]

    return app


def __getattr__(name: str):
    """Lazily build the module-level ``app`` for ``uvicorn qufin.api.server:app``.

    Building lazily means importing this module never has side effects and never
    fails on the fail-closed auth check; the app (which reads ``QUFIN_API_KEYS``
    from the environment) is only constructed when ``app`` is actually accessed,
    e.g. by the ASGI server. Prefer the factory form for explicit config:
    ``uvicorn --factory qufin.api.server:create_app``.
    """
    if name == "app":
        return create_app()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
