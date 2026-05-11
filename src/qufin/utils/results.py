"""Serializable Result dataclass for all algorithm outputs."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np


class _NumpyEncoder(json.JSONEncoder):
    def default(self, o: Any) -> Any:
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        return super().default(o)


@dataclass
class Result:
    """Base result from any qufin algorithm."""

    value: float = 0.0
    std_err: float = 0.0
    n_shots: int = 0
    circuit_depth: int = 0
    wall_time_s: float = 0.0
    backend_id: str = ""
    seed: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), cls=_NumpyEncoder, indent=indent)
