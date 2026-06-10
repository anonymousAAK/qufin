"""Global settings via pydantic, read from ~/.qufin/config.yaml, env vars, or code."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """qufin configuration.

    Resolution order: code overrides > QUFIN_* env vars > ~/.qufin/config.yaml.
    """

    model_config = {"env_prefix": "QUFIN_"}

    # Canonical backend name as used by qufin.backends.auto_select
    # (underscore form, e.g. "qiskit_aer" / "cudaq" / "mock").
    default_backend: str = Field(default="qiskit_aer", description="Default backend name")
    default_shots: int = Field(default=1024, ge=1)
    seed: int | None = Field(default=42, description="Global RNG seed")
    cache_dir: Path = Field(
        default=Path.home() / ".cache" / "qufin",
        description="Directory for data caches",
    )
    log_level: str = Field(default="WARNING", description="Logging level")
    fred_api_key: str | None = Field(default=None, description="FRED API key")


# Process-wide singleton, guarded for thread-safe lazy initialisation.
_settings: Settings | None = None
_lock = threading.Lock()


def get_settings(**overrides: Any) -> Settings:
    """Return the qufin settings.

    With no overrides, returns the process-wide singleton, created once and
    thread-safely. With overrides, returns a **new** validated ``Settings``
    derived from the singleton's values with the given fields replaced —
    without mutating the global. This prevents one caller's per-call override
    from silently corrupting configuration for every other caller (the previous
    behaviour rebuilt and replaced the global singleton on any override).
    """
    if overrides:
        base = get_settings()
        return Settings.model_validate({**base.model_dump(), **overrides})

    global _settings
    if _settings is None:
        with _lock:
            if _settings is None:
                _settings = Settings()
    return _settings


def reset_settings() -> None:
    """Clear the cached singleton so the next :func:`get_settings` rebuilds it.

    Useful for tests and for picking up environment changes at runtime.
    """
    global _settings
    with _lock:
        _settings = None
