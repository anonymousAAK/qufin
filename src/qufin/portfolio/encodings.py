"""Qubit encoding schemes for portfolio variables.

Supports one-hot (binary inclusion), binary (integer weights),
and unary encoding. Documents qubit cost for each.

One-hot: N qubits for N assets (1 qubit per asset, binary in/out).
Binary:  N * ceil(log2(K)) qubits for N assets with K weight levels.
Unary:   N * K qubits (thermometer encoding, simple but expensive).

References
----------
Hodson et al., arXiv:1911.05296 — QAOA portfolio rebalancing with
  various encodings.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class EncodingInfo:
    """Metadata about a qubit encoding scheme."""

    name: str
    n_assets: int
    n_qubits: int
    n_levels: int  # number of weight discretization levels
    bits_per_asset: int


def one_hot_encoding(n_assets: int) -> EncodingInfo:
    """One-hot (binary inclusion) encoding: 1 qubit per asset.

    Each qubit indicates whether the asset is selected (1) or not (0).
    All selected assets receive equal weight = 1/K where K is the
    number of selected assets.
    """
    return EncodingInfo(
        name="one_hot",
        n_assets=n_assets,
        n_qubits=n_assets,
        n_levels=2,
        bits_per_asset=1,
    )


def binary_encoding(n_assets: int, bits_per_asset: int = 3) -> EncodingInfo:
    """Binary (integer weight) encoding.

    Each asset uses `bits_per_asset` qubits. The decoded weight for
    asset i is: w_i = (sum_b 2^b * x_{i,b}) / (2^bits - 1).

    Parameters
    ----------
    n_assets : int
        Number of assets.
    bits_per_asset : int
        Bits per asset. 3 bits -> 8 levels (0, 1/7, 2/7, ..., 1).
    """
    n_levels = 2**bits_per_asset
    return EncodingInfo(
        name="binary",
        n_assets=n_assets,
        n_qubits=n_assets * bits_per_asset,
        n_levels=n_levels,
        bits_per_asset=bits_per_asset,
    )


def unary_encoding(n_assets: int, n_levels: int = 4) -> EncodingInfo:
    """Unary (thermometer) encoding.

    Each asset uses `n_levels` qubits. The weight is proportional to
    the number of qubits set to 1 (thermometer code).
    Simple but qubit-expensive.
    """
    return EncodingInfo(
        name="unary",
        n_assets=n_assets,
        n_qubits=n_assets * n_levels,
        n_levels=n_levels,
        bits_per_asset=n_levels,
    )


def decode_one_hot(bitstring: str) -> NDArray[np.float64]:
    """Decode a one-hot bitstring to equal-weight portfolio.

    Returns weights: 1/K for selected assets, 0 for unselected.
    """
    selection = np.array([int(c) for c in bitstring], dtype=np.float64)
    total = selection.sum()
    if total > 0:
        return selection / total
    return selection


def decode_binary(
    bitstring: str, n_assets: int, bits_per_asset: int
) -> NDArray[np.float64]:
    """Decode a binary-encoded bitstring to portfolio weights.

    Weights are normalized to sum to 1.
    """
    max_level = 2**bits_per_asset - 1
    weights = np.zeros(n_assets, dtype=np.float64)
    for i in range(n_assets):
        start = i * bits_per_asset
        end = start + bits_per_asset
        bits = bitstring[start:end]
        level = int(bits, 2)
        weights[i] = level / max_level if max_level > 0 else 0.0
    total = weights.sum()
    if total > 0:
        weights = weights / total
    return weights


def decode_unary(
    bitstring: str, n_assets: int, n_levels: int
) -> NDArray[np.float64]:
    """Decode a unary (thermometer) encoded bitstring to weights.

    Weight_i = (number of 1s in asset i's qubits) / n_levels.
    Normalized to sum to 1.
    """
    weights = np.zeros(n_assets, dtype=np.float64)
    for i in range(n_assets):
        start = i * n_levels
        end = start + n_levels
        bits = bitstring[start:end]
        weights[i] = sum(int(c) for c in bits) / n_levels
    total = weights.sum()
    if total > 0:
        weights = weights / total
    return weights


def qubit_cost_table(n_assets_list: list[int]) -> list[dict[str, int | str]]:
    """Generate a qubit cost comparison table for different encodings.

    Parameters
    ----------
    n_assets_list : list[int]
        List of asset counts to compare.

    Returns
    -------
    List of dicts with columns: n_assets, one_hot, binary_3b, binary_5b, unary_4.
    """
    rows = []
    for n in n_assets_list:
        rows.append({
            "n_assets": n,
            "one_hot": one_hot_encoding(n).n_qubits,
            "binary_3b": binary_encoding(n, 3).n_qubits,
            "binary_5b": binary_encoding(n, 5).n_qubits,
            "unary_4": unary_encoding(n, 4).n_qubits,
        })
    return rows
