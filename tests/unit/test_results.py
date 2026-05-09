"""Tests for Result dataclass."""

from __future__ import annotations

import json

import numpy as np

from qufin.utils.results import Result


class TestResult:
    def test_to_dict(self) -> None:
        r = Result(value=1.5, std_err=0.1, n_shots=1024)
        d = r.to_dict()
        assert d["value"] == 1.5
        assert d["n_shots"] == 1024

    def test_to_json(self) -> None:
        r = Result(value=2.0, backend_id="mock")
        j = r.to_json()
        parsed = json.loads(j)
        assert parsed["value"] == 2.0
        assert parsed["backend_id"] == "mock"

    def test_numpy_serialization(self) -> None:
        r = Result(value=float(np.float64(3.14)))
        j = r.to_json()
        parsed = json.loads(j)
        assert abs(parsed["value"] - 3.14) < 1e-10
