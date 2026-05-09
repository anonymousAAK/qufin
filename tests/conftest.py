"""Shared fixtures for qufin tests."""

from __future__ import annotations

import numpy as np
import pytest

from qufin.backends.mock import MockBackend


@pytest.fixture
def rng() -> np.random.Generator:
    """Seeded random generator for reproducible tests."""
    return np.random.default_rng(42)


@pytest.fixture
def mock_backend() -> MockBackend:
    """Mock backend for unit tests."""
    return MockBackend(seed=42)


@pytest.fixture
def sample_returns(rng: np.random.Generator) -> np.ndarray:
    """Sample return matrix: 252 days x 5 assets."""
    return rng.normal(0.0005, 0.02, (252, 5))


@pytest.fixture
def sample_cov(sample_returns: np.ndarray) -> np.ndarray:
    """Sample covariance matrix from sample_returns."""
    return np.cov(sample_returns, rowvar=False)


@pytest.fixture
def sample_mu(sample_returns: np.ndarray) -> np.ndarray:
    """Sample expected returns from sample_returns."""
    return sample_returns.mean(axis=0)
