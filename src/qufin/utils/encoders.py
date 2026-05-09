"""Bitstring / integer / binary encoding utilities."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def bitstring_to_array(bs: str) -> NDArray[np.int8]:
    """Convert a bitstring like '01101' to a numpy array [0, 1, 1, 0, 1]."""
    return np.array([int(c) for c in bs], dtype=np.int8)


def array_to_bitstring(arr: NDArray[np.int8]) -> str:
    """Convert a binary array back to a bitstring."""
    return "".join(str(int(x)) for x in arr)


def binary_to_weights(
    bitstring: str, n_assets: int, bits_per_asset: int, w_max: float = 1.0
) -> NDArray[np.float64]:
    """Decode a binary-encoded bitstring into portfolio weights.

    Each asset uses `bits_per_asset` bits for its weight level.
    Weight for asset i = (decoded integer) / (2^bits_per_asset - 1) * w_max.
    """
    n_levels = 2**bits_per_asset - 1
    weights = np.zeros(n_assets, dtype=np.float64)
    for i in range(n_assets):
        start = i * bits_per_asset
        end = start + bits_per_asset
        bits = bitstring[start:end]
        level = int(bits, 2)
        weights[i] = level / n_levels * w_max
    return weights


def onehot_to_selection(bitstring: str) -> NDArray[np.bool_]:
    """Convert a one-hot bitstring to a boolean selection mask."""
    return np.array([c == "1" for c in bitstring], dtype=np.bool_)
