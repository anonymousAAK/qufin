"""Shared type aliases for qufin."""

from __future__ import annotations

from typing import NewType, TypeAlias

import numpy as np
from numpy.typing import NDArray

AssetReturns: TypeAlias = NDArray[np.float64]  # shape (T, N)
CovMatrix: TypeAlias = NDArray[np.float64]  # shape (N, N)
Weights: TypeAlias = NDArray[np.float64]  # shape (N,)
Bitstring = NewType("Bitstring", str)
Shots = NewType("Shots", int)
