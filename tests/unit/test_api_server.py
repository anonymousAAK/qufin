"""Tests for the qufin REST API server.

Uses FastAPI TestClient with mocked computations to avoid
requiring actual quantum backends or market data.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from qufin.api.server import (
    CircuitInfo,
    Greeks,
    HealthResponse,
    InMemoryJobStore,
    JobStatus,
    OptimizationMethod,
    OptimizeRequest,
    OptimizeResponse,
    PriceRequest,
    PriceResponse,
    PricingMethod,
    RateLimiter,
    RiskMethod,
    RiskRequest,
    RiskResponse,
    StressTestResult,
    _run_optimize,
    _run_price,
    _run_risk,
    create_app,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient


@pytest.fixture
def app():
    """Create a test app with no auth."""
    return create_app(api_keys=None, rate_limit=1000, rate_window=60)


@pytest.fixture
def client(app):
    """TestClient for the app."""
    return TestClient(app)


@pytest.fixture
def auth_app():
    """Create a test app with API key auth."""
    return create_app(api_keys=["test-key-123"], rate_limit=1000)


@pytest.fixture
def auth_client(auth_app):
    """TestClient with auth enabled."""
    return TestClient(auth_app)


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    """Tests for /health endpoint."""

    def test_health_returns_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "version" in data
        assert "timestamp" in data

    def test_health_response_model(self, client):
        resp = client.get("/health")
        health = HealthResponse(**resp.json())
        assert health.status == "ok"


# ---------------------------------------------------------------------------
# Optimize endpoint
# ---------------------------------------------------------------------------


class TestOptimizeEndpoint:
    """Tests for /v1/optimize endpoint."""

    @patch("qufin.api.server._run_optimize")
    def test_optimize_mvo(self, mock_opt, client):
        mock_opt.return_value = OptimizeResponse(
            weights={"AAPL": 0.4, "MSFT": 0.6},
            expected_return=0.12,
            volatility=0.18,
            sharpe_ratio=0.55,
            method="mvo",
            computation_time_s=0.01,
        )
        resp = client.post(
            "/v1/optimize",
            json={
                "tickers": ["AAPL", "MSFT"],
                "start_date": "2023-01-01",
                "end_date": "2024-01-01",
                "method": "mvo",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "weights" in data
        assert data["method"] == "mvo"
        mock_opt.assert_called_once()

    @patch("qufin.api.server._run_optimize")
    def test_optimize_qaoa_has_circuit_info(self, mock_opt, client):
        mock_opt.return_value = OptimizeResponse(
            weights={"AAPL": 0.5, "GOOG": 0.5},
            expected_return=0.10,
            volatility=0.15,
            sharpe_ratio=0.53,
            method="qaoa",
            circuit_info=CircuitInfo(n_qubits=2, depth=4, gate_count=8, method="QAOA"),
            computation_time_s=0.5,
        )
        resp = client.post(
            "/v1/optimize",
            json={
                "tickers": ["AAPL", "GOOG"],
                "start_date": "2023-01-01",
                "end_date": "2024-01-01",
                "method": "qaoa",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["circuit_info"] is not None
        assert data["circuit_info"]["method"] == "QAOA"

    @patch("qufin.api.server._run_optimize")
    def test_optimize_async_returns_job(self, mock_opt, client):
        resp = client.post(
            "/v1/optimize",
            json={
                "tickers": ["AAPL", "MSFT"],
                "start_date": "2023-01-01",
                "end_date": "2024-01-01",
                "method": "mvo",
                "async_mode": True,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "job_id" in data
        assert data["status"] == "pending"
        mock_opt.assert_not_called()

    def test_optimize_bad_tickers(self, client):
        """Tickers list too short should fail validation."""
        resp = client.post(
            "/v1/optimize",
            json={
                "tickers": ["AAPL"],
                "start_date": "2023-01-01",
                "end_date": "2024-01-01",
            },
        )
        assert resp.status_code == 422

    def test_optimize_invalid_method(self, client):
        resp = client.post(
            "/v1/optimize",
            json={
                "tickers": ["AAPL", "MSFT"],
                "start_date": "2023-01-01",
                "end_date": "2024-01-01",
                "method": "invalid_method",
            },
        )
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Price endpoint
# ---------------------------------------------------------------------------


class TestPriceEndpoint:
    """Tests for /v1/price endpoint."""

    @patch("qufin.api.server._run_price")
    def test_price_bs(self, mock_price, client):
        mock_price.return_value = PriceResponse(
            price=10.45,
            greeks=Greeks(delta=0.6, gamma=0.03, vega=15.0, theta=-0.05, rho=8.0),
            method="bs",
            computation_time_s=0.001,
        )
        resp = client.post(
            "/v1/price",
            json={
                "spot": 100.0,
                "strike": 105.0,
                "rate": 0.05,
                "volatility": 0.2,
                "expiry": 1.0,
                "is_call": True,
                "method": "bs",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["price"] == pytest.approx(10.45)
        assert "greeks" in data

    @patch("qufin.api.server._run_price")
    def test_price_mc_has_ci(self, mock_price, client):
        mock_price.return_value = PriceResponse(
            price=10.5,
            greeks=Greeks(delta=0.6, gamma=0.03, vega=15.0, theta=-0.05, rho=8.0),
            confidence_interval=[10.2, 10.8],
            method="mc",
            computation_time_s=0.5,
        )
        resp = client.post(
            "/v1/price",
            json={
                "spot": 100.0,
                "strike": 105.0,
                "volatility": 0.2,
                "expiry": 1.0,
                "method": "mc",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["confidence_interval"] is not None
        assert len(data["confidence_interval"]) == 2

    @patch("qufin.api.server._run_price")
    def test_price_async(self, mock_price, client):
        resp = client.post(
            "/v1/price",
            json={
                "spot": 100.0,
                "strike": 105.0,
                "volatility": 0.2,
                "expiry": 1.0,
                "method": "bs",
                "async_mode": True,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "job_id" in data
        mock_price.assert_not_called()

    def test_price_invalid_spot(self, client):
        resp = client.post(
            "/v1/price",
            json={
                "spot": -1.0,
                "strike": 105.0,
                "volatility": 0.2,
                "expiry": 1.0,
            },
        )
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Risk endpoint
# ---------------------------------------------------------------------------


class TestRiskEndpoint:
    """Tests for /v1/risk endpoint."""

    @patch("qufin.api.server._run_risk")
    def test_risk_historical(self, mock_risk, client):
        mock_risk.return_value = RiskResponse(
            var=0.025,
            var_dollar=25000.0,
            cvar=0.035,
            cvar_dollar=35000.0,
            confidence_level=0.95,
            method="historical",
            computation_time_s=0.1,
        )
        resp = client.post(
            "/v1/risk",
            json={
                "portfolio_weights": {"AAPL": 0.5, "MSFT": 0.5},
                "start_date": "2023-01-01",
                "end_date": "2024-01-01",
                "method": "historical",
                "confidence_level": 0.95,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "var" in data
        assert "cvar" in data
        assert data["confidence_level"] == 0.95

    @patch("qufin.api.server._run_risk")
    def test_risk_with_stress_test(self, mock_risk, client):
        mock_risk.return_value = RiskResponse(
            var=0.03,
            var_dollar=30000.0,
            cvar=0.04,
            cvar_dollar=40000.0,
            confidence_level=0.95,
            method="historical",
            stress_test_results=[
                StressTestResult(
                    scenario="Market crash -20%",
                    portfolio_loss=200000.0,
                    loss_pct=20.0,
                ),
            ],
            computation_time_s=0.2,
        )
        resp = client.post(
            "/v1/risk",
            json={
                "portfolio_weights": {"AAPL": 0.5, "MSFT": 0.5},
                "start_date": "2023-01-01",
                "end_date": "2024-01-01",
                "stress_test": True,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["stress_test_results"] is not None
        assert len(data["stress_test_results"]) >= 1

    @patch("qufin.api.server._run_risk")
    def test_risk_async(self, mock_risk, client):
        resp = client.post(
            "/v1/risk",
            json={
                "portfolio_weights": {"AAPL": 0.5, "MSFT": 0.5},
                "start_date": "2023-01-01",
                "end_date": "2024-01-01",
                "async_mode": True,
            },
        )
        assert resp.status_code == 200
        assert "job_id" in resp.json()
        mock_risk.assert_not_called()


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------


class TestAuthentication:
    """Tests for API key authentication."""

    def test_auth_required_no_key(self, auth_client):
        resp = auth_client.post(
            "/v1/optimize",
            json={
                "tickers": ["AAPL", "MSFT"],
                "start_date": "2023-01-01",
                "end_date": "2024-01-01",
            },
        )
        assert resp.status_code == 401

    @patch("qufin.api.server._run_optimize")
    def test_auth_valid_key(self, mock_opt, auth_client):
        mock_opt.return_value = OptimizeResponse(
            weights={"AAPL": 0.5, "MSFT": 0.5},
            expected_return=0.1,
            volatility=0.15,
            sharpe_ratio=0.5,
            method="mvo",
            computation_time_s=0.01,
        )
        resp = auth_client.post(
            "/v1/optimize",
            json={
                "tickers": ["AAPL", "MSFT"],
                "start_date": "2023-01-01",
                "end_date": "2024-01-01",
            },
            headers={"X-API-Key": "test-key-123"},
        )
        assert resp.status_code == 200

    def test_auth_invalid_key(self, auth_client):
        resp = auth_client.post(
            "/v1/optimize",
            json={
                "tickers": ["AAPL", "MSFT"],
                "start_date": "2023-01-01",
                "end_date": "2024-01-01",
            },
            headers={"X-API-Key": "wrong-key"},
        )
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------


class TestRateLimiting:
    """Tests for rate limiter."""

    def test_rate_limiter_allows_within_limit(self):
        limiter = RateLimiter(max_requests=5, window_seconds=60)
        for _ in range(5):
            assert limiter.is_allowed("client1")

    def test_rate_limiter_blocks_over_limit(self):
        limiter = RateLimiter(max_requests=3, window_seconds=60)
        for _ in range(3):
            assert limiter.is_allowed("client1")
        assert not limiter.is_allowed("client1")

    def test_rate_limiter_per_client(self):
        limiter = RateLimiter(max_requests=2, window_seconds=60)
        assert limiter.is_allowed("client1")
        assert limiter.is_allowed("client1")
        assert not limiter.is_allowed("client1")
        # Different client should still be allowed
        assert limiter.is_allowed("client2")

    def test_rate_limiter_reset(self):
        limiter = RateLimiter(max_requests=1, window_seconds=60)
        assert limiter.is_allowed("client1")
        assert not limiter.is_allowed("client1")
        limiter.reset("client1")
        assert limiter.is_allowed("client1")

    def test_rate_limiter_reset_all(self):
        limiter = RateLimiter(max_requests=1, window_seconds=60)
        limiter.is_allowed("client1")
        limiter.is_allowed("client2")
        limiter.reset()
        assert limiter.is_allowed("client1")
        assert limiter.is_allowed("client2")


# ---------------------------------------------------------------------------
# Job management endpoints
# ---------------------------------------------------------------------------


class TestJobEndpoints:
    """Tests for /v1/jobs endpoints."""

    @patch("qufin.api.server._run_optimize")
    def test_create_and_get_job(self, mock_opt, client):
        # Submit async job
        resp = client.post(
            "/v1/optimize",
            json={
                "tickers": ["AAPL", "MSFT"],
                "start_date": "2023-01-01",
                "end_date": "2024-01-01",
                "async_mode": True,
            },
        )
        job_id = resp.json()["job_id"]

        # Poll for status
        resp = client.get(f"/v1/jobs/{job_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["job_id"] == job_id
        assert data["status"] == "pending"

    def test_get_nonexistent_job(self, client):
        resp = client.get("/v1/jobs/nonexistent-id")
        assert resp.status_code == 404

    def test_cancel_job(self, client):
        # Create a job
        resp = client.post(
            "/v1/optimize",
            json={
                "tickers": ["AAPL", "MSFT"],
                "start_date": "2023-01-01",
                "end_date": "2024-01-01",
                "async_mode": True,
            },
        )
        job_id = resp.json()["job_id"]

        # Cancel it
        resp = client.delete(f"/v1/jobs/{job_id}")
        assert resp.status_code == 200
        assert resp.json()["status"] == "cancelled"

    def test_cancel_nonexistent_job(self, client):
        resp = client.delete("/v1/jobs/nonexistent-id")
        assert resp.status_code == 404

    def test_list_jobs(self, client):
        # Create two jobs
        for _ in range(2):
            client.post(
                "/v1/price",
                json={
                    "spot": 100.0,
                    "strike": 105.0,
                    "volatility": 0.2,
                    "expiry": 1.0,
                    "async_mode": True,
                },
            )
        resp = client.get("/v1/jobs")
        assert resp.status_code == 200
        jobs = resp.json()
        assert len(jobs) >= 2


# ---------------------------------------------------------------------------
# InMemoryJobStore
# ---------------------------------------------------------------------------


class TestInMemoryJobStore:
    """Tests for InMemoryJobStore."""

    def test_create_and_get(self):
        store = InMemoryJobStore()
        job_id = store.create("optimization", {"tickers": ["AAPL"]})
        job = store.get(job_id)
        assert job is not None
        assert job["job_type"] == "optimization"
        assert job["status"] == JobStatus.PENDING

    def test_update(self):
        store = InMemoryJobStore()
        job_id = store.create("pricing", {})
        store.update(job_id, status=JobStatus.COMPLETED, result={"price": 10.0})
        job = store.get(job_id)
        assert job["status"] == JobStatus.COMPLETED
        assert job["result"]["price"] == 10.0

    def test_delete(self):
        store = InMemoryJobStore()
        job_id = store.create("risk", {})
        assert store.delete(job_id)
        assert store.get(job_id) is None
        assert not store.delete(job_id)

    def test_list_jobs(self):
        store = InMemoryJobStore()
        store.create("optimization", {})
        store.create("pricing", {})
        jobs = store.list_jobs()
        assert len(jobs) == 2

    def test_get_nonexistent(self):
        store = InMemoryJobStore()
        assert store.get("nonexistent") is None

    def test_update_nonexistent(self):
        store = InMemoryJobStore()
        # Should not raise
        store.update("nonexistent", status=JobStatus.COMPLETED)


# ---------------------------------------------------------------------------
# Computation helpers (integration-like, with mocked internals)
# ---------------------------------------------------------------------------


class TestComputationHelpers:
    """Tests for _run_optimize, _run_price, _run_risk with mocked data."""

    def test_run_optimize_fallback(self):
        """Optimize falls back to synthetic data when fetch fails."""
        req = OptimizeRequest(
            tickers=["AAPL", "MSFT", "GOOG"],
            start_date="2023-01-01",
            end_date="2024-01-01",
            method=OptimizationMethod.MVO,
        )
        # The function handles the import error internally (synthetic data)
        result = _run_optimize(req)
        assert isinstance(result, OptimizeResponse)
        assert len(result.weights) == 3
        assert sum(result.weights.values()) == pytest.approx(1.0, abs=0.01)

    def test_run_price_bs(self):
        """BS pricing uses EuropeanOption."""
        req = PriceRequest(
            spot=100.0,
            strike=105.0,
            rate=0.05,
            volatility=0.2,
            expiry=1.0,
            is_call=True,
            method=PricingMethod.BS,
        )
        result = _run_price(req)
        assert isinstance(result, PriceResponse)
        assert result.price > 0
        assert result.greeks.delta > 0

    def test_run_risk_historical_fallback(self):
        """Risk computation with synthetic data fallback."""
        req = RiskRequest(
            portfolio_weights={"AAPL": 0.5, "MSFT": 0.5},
            start_date="2023-01-01",
            end_date="2024-01-01",
            method=RiskMethod.HISTORICAL,
            confidence_level=0.95,
            portfolio_value=1_000_000,
        )
        result = _run_risk(req)
        assert isinstance(result, RiskResponse)
        assert result.var > 0
        assert result.cvar > 0
        assert result.confidence_level == 0.95

    def test_run_risk_with_stress_test(self):
        """Risk computation with stress test."""
        req = RiskRequest(
            portfolio_weights={"AAPL": 0.5, "MSFT": 0.5},
            start_date="2023-01-01",
            end_date="2024-01-01",
            method=RiskMethod.HISTORICAL,
            stress_test=True,
        )
        result = _run_risk(req)
        assert result.stress_test_results is not None
        assert len(result.stress_test_results) == 4


# ---------------------------------------------------------------------------
# OpenAPI schema
# ---------------------------------------------------------------------------


class TestOpenAPI:
    """Tests for OpenAPI schema generation."""

    def test_openapi_schema_accessible(self, client):
        resp = client.get("/v1/openapi.json")
        assert resp.status_code == 200
        schema = resp.json()
        assert "openapi" in schema
        assert schema["info"]["title"] == "qufin API"

    def test_openapi_has_all_endpoints(self, client):
        resp = client.get("/v1/openapi.json")
        paths = resp.json()["paths"]
        assert "/v1/optimize" in paths
        assert "/v1/price" in paths
        assert "/v1/risk" in paths
        assert "/v1/jobs/{job_id}" in paths
