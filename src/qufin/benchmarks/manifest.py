"""Reproducibility manifest: RNG seeds, dependency versions, git commit.

Captures all information needed to reproduce a benchmark run.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class Manifest:
    """Reproducibility manifest for a benchmark run."""

    timestamp: str = ""
    python_version: str = ""
    platform_info: str = ""
    git_commit: str = ""
    git_dirty: bool = False
    package_versions: dict[str, str] = field(default_factory=dict)
    seeds: list[int] = field(default_factory=list)
    problem_ids: list[str] = field(default_factory=list)
    solver_names: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.to_json())


def _get_git_info() -> tuple[str, bool]:
    """Get current git commit hash and dirty status."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"],
            stderr=subprocess.DEVNULL,
        ).decode().strip())
        return commit, dirty
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "", False


def _get_package_versions() -> dict[str, str]:
    """Get versions of key dependencies."""
    packages = [
        "qufin", "qiskit", "qiskit-aer", "qiskit-ibm-runtime",
        "numpy", "scipy", "cvxpy", "pandas",
    ]
    versions = {}
    for pkg in packages:
        try:
            from importlib.metadata import version
            versions[pkg] = version(pkg)
        except Exception:
            pass
    return versions


def build_manifest(
    problem_ids: list[str] | None = None,
    solver_names: list[str] | None = None,
    seeds: list[int] | None = None,
) -> Manifest:
    """Build a reproducibility manifest for the current environment."""
    git_commit, git_dirty = _get_git_info()
    return Manifest(
        timestamp=datetime.now(timezone.utc).isoformat(),
        python_version=sys.version,
        platform_info=platform.platform(),
        git_commit=git_commit,
        git_dirty=git_dirty,
        package_versions=_get_package_versions(),
        seeds=seeds or [],
        problem_ids=problem_ids or [],
        solver_names=solver_names or [],
    )
