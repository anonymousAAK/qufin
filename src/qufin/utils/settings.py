"""Global settings via pydantic, read from ~/.qufin/config.yaml, env vars, or code."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """qufin configuration.

    Resolution order: code overrides > QUFIN_* env vars > ~/.qufin/config.yaml.
    """

    model_config = {"env_prefix": "QUFIN_"}

    default_backend: str = Field(default="qiskit-aer", description="Default backend ID")
    default_shots: int = Field(default=1024, ge=1)
    seed: int | None = Field(default=42, description="Global RNG seed")
    cache_dir: Path = Field(
        default=Path.home() / ".cache" / "qufin",
        description="Directory for data caches",
    )
    log_level: str = Field(default="WARNING", description="Logging level")
    fred_api_key: str | None = Field(default=None, description="FRED API key")


# Singleton
_settings: Settings | None = None


def get_settings(**overrides: Any) -> Settings:
    """Get or create the global settings instance."""
    global _settings
    if _settings is None or overrides:
        _settings = Settings(**overrides)
    return _settings
