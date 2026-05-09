"""Standardized benchmark problem definitions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class ProblemType(str, Enum):
    PORTFOLIO = "portfolio"
    OPTION = "option"
    CREDIT = "credit"


@dataclass
class Problem:
    """A benchmark problem specification."""

    problem_id: str
    problem_type: ProblemType
    description: str
    params: dict[str, Any]
    reference_value: float | None = None
    reference_source: str = ""
