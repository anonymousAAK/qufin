"""Tests for the global settings singleton."""

from __future__ import annotations

import pytest

from qufin.utils.settings import Settings, get_settings, reset_settings


@pytest.fixture(autouse=True)
def _clean_settings():
    """Ensure each test starts and ends with a fresh settings singleton."""
    reset_settings()
    yield
    reset_settings()


def test_default_backend_matches_backend_layer():
    """The default backend name must be the canonical underscore form used by
    qufin.backends.auto_select (qiskit_aer), not the hyphenated 'qiskit-aer'."""
    assert get_settings().default_backend == "qiskit_aer"


def test_singleton_identity():
    assert get_settings() is get_settings()


def test_overrides_do_not_mutate_global():
    """A per-call override must not corrupt the shared singleton."""
    base = get_settings()
    assert base.seed == 42

    overridden = get_settings(seed=999, default_shots=4096)
    assert overridden.seed == 999
    assert overridden.default_shots == 4096
    assert overridden is not base

    # The global is untouched by the override.
    assert get_settings().seed == 42
    assert get_settings() is base


def test_overrides_are_validated():
    """Overrides go through pydantic validation (default_shots >= 1)."""
    with pytest.raises(Exception):  # noqa: B017 - pydantic ValidationError
        get_settings(default_shots=0)


def test_reset_rebuilds_singleton():
    first = get_settings()
    reset_settings()
    assert get_settings() is not first


def test_settings_is_constructible_directly():
    s = Settings()
    assert s.default_shots >= 1
    assert s.default_backend == "qiskit_aer"
