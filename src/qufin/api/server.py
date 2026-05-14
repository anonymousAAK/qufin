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

import time
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Optional dependency guard
# ---------------------------------------------------------------------------
try:
    from fastapi import Depends, FastAPI, HTTPException, Request, Security, status
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


def _run_optimize(req: OptimizeRequest) -> OptimizeResponse:
    """Execute portfolio optimization synchronously."""
    import numpy as np

    t0 = time.time()

    # Synthetic returns (qufin.data.market not available)
    rng = np.random.default_rng(42)
    n_days = 252
    returns = rng.normal(0.0005, 0.02, size=(n_days, len(req.tickers)))

    mu = np.mean(returns, axis=0) * 252
    cov = np.cov(returns, rowvar=False) * 252
    n = len(req.tickers)

    circuit_info = None

    if req.method == OptimizationMethod.MVO:
        try:
            from qufin.portfolio.classical.mean_variance import mean_variance

            result = mean_variance(mu, cov)
            w = np.array(result.weights)
        except Exception:
            # Simple equal-weight fallback
            w = np.ones(n) / n
    elif req.method == OptimizationMethod.HRP:
        try:
            from qufin.portfolio.classical.hrp import hrp

            result = hrp(cov)
            w = np.array(result.weights)
        except Exception:
            w = np.ones(n) / n
    elif req.method == OptimizationMethod.QAOA:
        try:
            from qufin.portfolio.qubo import PortfolioQUBO

            qubo = PortfolioQUBO(mu, cov, budget_penalty=10.0)
            result = qubo.solve_qaoa(p=1)
            w = np.array(result.weights)
            circuit_info = CircuitInfo(
                n_qubits=n,
                depth=result.circuit_depth if hasattr(result, "circuit_depth") else 2 * n,
                gate_count=result.gate_count if hasattr(result, "gate_count") else 4 * n,
                method="QAOA",
            )
        except Exception:
            w = np.ones(n) / n
    else:
        w = np.ones(n) / n

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
        try:
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
        except Exception:
            price = opt.bs_price()
            ci = [price * 0.98, price * 1.02]
    elif req.method == PricingMethod.QAE:
        try:
            from qufin.options.amplitude_estimation.canonical_qae import canonical_qae_price

            qae_result = canonical_qae_price(opt, n_qubits=req.n_qubits)
            price = qae_result.price if hasattr(qae_result, "price") else qae_result
            ci_hw = getattr(qae_result, "confidence_interval", None)
            ci = list(ci_hw) if ci_hw is not None else [price * 0.95, price * 1.05]
            circuit_info = CircuitInfo(
                n_qubits=req.n_qubits,
                depth=getattr(qae_result, "circuit_depth", 4 * req.n_qubits),
                gate_count=getattr(qae_result, "gate_count", 8 * req.n_qubits),
                method="QAE",
            )
        except Exception:
            price = opt.bs_price()
            ci = [price * 0.95, price * 1.05]
    else:
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
    weights = np.array(list(req.portfolio_weights.values()))

    # Synthetic returns (qufin.data.market not available)
    rng = np.random.default_rng(42)
    returns = rng.normal(0.0003, 0.015, size=(252, len(tickers)))

    port_returns = returns @ weights

    if req.method == RiskMethod.HISTORICAL:
        try:
            from qufin.risk.classical_var import historical_var

            result = historical_var(
                port_returns,
                confidence=req.confidence_level,
                portfolio_value=req.portfolio_value,
            )
            var_pct = result.var
            cvar_pct = result.expected_shortfall
        except Exception:
            alpha = 1 - req.confidence_level
            var_pct = float(-np.percentile(port_returns, alpha * 100))
            tail = port_returns[port_returns <= -var_pct]
            cvar_pct = float(-np.mean(tail)) if len(tail) > 0 else var_pct * 1.2

    elif req.method == RiskMethod.PARAMETRIC:
        try:
            from qufin.risk.classical_var import parametric_var

            result = parametric_var(
                port_returns,
                confidence=req.confidence_level,
                portfolio_value=req.portfolio_value,
            )
            var_pct = result.var
            cvar_pct = result.expected_shortfall
        except Exception:
            from scipy.stats import norm as _norm

            mu = float(np.mean(port_returns))
            sigma = float(np.std(port_returns))
            z = _norm.ppf(req.confidence_level)
            var_pct = -(mu - z * sigma)
            cvar_pct = var_pct * 1.2

    elif req.method == RiskMethod.QUANTUM:
        try:
            from qufin.risk.quantum_var import quantum_var

            result = quantum_var(port_returns, confidence=req.confidence_level)
            var_pct = result.var if hasattr(result, "var") else float(result)
            cvar_pct = (
                result.expected_shortfall
                if hasattr(result, "expected_shortfall")
                else var_pct * 1.2
            )
        except Exception:
            alpha = 1 - req.confidence_level
            var_pct = float(-np.percentile(port_returns, alpha * 100))
            cvar_pct = var_pct * 1.2
    else:
        var_pct = 0.0
        cvar_pct = 0.0

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
# App factory
# ---------------------------------------------------------------------------

# Module-level singletons (set during create_app)
_rate_limiter: RateLimiter | None = None
_job_store: InMemoryJobStore | None = None
_api_keys: set[str] | None = None


def create_app(
    *,
    api_keys: list[str] | None = None,
    rate_limit: int = 60,
    rate_window: int = 60,
    title: str = "qufin API",
    version: str = "1.0.0",
) -> FastAPI:
    """Create and configure the FastAPI application.

    Parameters
    ----------
    api_keys : list[str] | None
        Valid API keys. If ``None``, authentication is disabled.
    rate_limit : int
        Maximum requests per window per client.
    rate_window : int
        Rate-limit window in seconds.
    title : str
        OpenAPI title.
    version : str
        API version string.

    Returns
    -------
    FastAPI
        Configured application instance.

    Raises
    ------
    ImportError
        If ``fastapi`` is not installed.
    """
    if not _HAS_FASTAPI:
        raise ImportError(
            "fastapi is required for the qufin REST API. "
            "Install it with: pip install fastapi uvicorn"
        )

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
            job_id = _job_store.create("optimization", req.model_dump())
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
            job_id = _job_store.create("pricing", req.model_dump())
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
            job_id = _job_store.create("risk", req.model_dump())
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


# Module-level app for ``uvicorn qufin.api.server:app``
try:
    app = create_app()
except ImportError:
    app = None  # type: ignore[assignment]
