"""Tests for encoding utilities."""

from __future__ import annotations

import numpy as np

from qufin.utils.encoders import (
    array_to_bitstring,
    binary_to_weights,
    bitstring_to_array,
    onehot_to_selection,
)


class TestBitstringRoundtrip:
    def test_roundtrip(self) -> None:
        bs = "01101"
        arr = bitstring_to_array(bs)
        assert array_to_bitstring(arr) == bs

    def test_all_zeros(self) -> None:
        arr = bitstring_to_array("0000")
        np.testing.assert_array_equal(arr, [0, 0, 0, 0])

    def test_all_ones(self) -> None:
        arr = bitstring_to_array("1111")
        np.testing.assert_array_equal(arr, [1, 1, 1, 1])


class TestBinaryWeights:
    def test_uniform(self) -> None:
        # All bits set -> max weight
        bs = "111111"  # 2 assets, 3 bits each
        w = binary_to_weights(bs, n_assets=2, bits_per_asset=3, w_max=1.0)
        np.testing.assert_allclose(w, [1.0, 1.0])

    def test_zero(self) -> None:
        bs = "000000"
        w = binary_to_weights(bs, n_assets=2, bits_per_asset=3, w_max=1.0)
        np.testing.assert_allclose(w, [0.0, 0.0])


class TestOneHot:
    def test_selection(self) -> None:
        mask = onehot_to_selection("10110")
        np.testing.assert_array_equal(mask, [True, False, True, True, False])
