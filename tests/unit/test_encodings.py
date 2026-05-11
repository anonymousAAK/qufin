"""Tests for portfolio qubit encoding schemes."""

from __future__ import annotations

import numpy as np
import pytest

from qufin.portfolio.encodings import (
    binary_encoding,
    decode_binary,
    decode_one_hot,
    decode_unary,
    one_hot_encoding,
    qubit_cost_table,
    unary_encoding,
)


class TestOneHotEncoding:
    def test_qubit_count(self) -> None:
        info = one_hot_encoding(15)
        assert info.n_qubits == 15
        assert info.bits_per_asset == 1

    def test_decode_equal_weight(self) -> None:
        w = decode_one_hot("11000")
        assert abs(w.sum() - 1.0) < 1e-10
        assert w[0] == pytest.approx(0.5)
        assert w[1] == pytest.approx(0.5)

    def test_decode_single_asset(self) -> None:
        w = decode_one_hot("00100")
        assert abs(w.sum() - 1.0) < 1e-10
        assert w[2] == pytest.approx(1.0)

    def test_decode_all_zeros(self) -> None:
        w = decode_one_hot("00000")
        np.testing.assert_array_equal(w, 0.0)


class TestBinaryEncoding:
    def test_qubit_count(self) -> None:
        info = binary_encoding(10, bits_per_asset=3)
        assert info.n_qubits == 30
        assert info.bits_per_asset == 3
        assert info.n_levels == 8

    def test_decode_all_max(self) -> None:
        # 3 assets, 2 bits each, all max = "111111"
        w = decode_binary("111111", n_assets=3, bits_per_asset=2)
        assert abs(w.sum() - 1.0) < 1e-10
        np.testing.assert_allclose(w, 1 / 3, atol=1e-10)

    def test_decode_mixed(self) -> None:
        # 2 assets, 3 bits: "111" = 7/7 = 1.0, "011" = 3/7 ≈ 0.4286
        w = decode_binary("111011", n_assets=2, bits_per_asset=3)
        assert abs(w.sum() - 1.0) < 1e-10
        assert w[0] > w[1]  # first asset has higher raw weight

    def test_decode_all_zeros(self) -> None:
        w = decode_binary("000000", n_assets=2, bits_per_asset=3)
        np.testing.assert_array_equal(w, 0.0)


class TestUnaryEncoding:
    def test_qubit_count(self) -> None:
        info = unary_encoding(5, n_levels=4)
        assert info.n_qubits == 20
        assert info.bits_per_asset == 4

    def test_decode_thermometer(self) -> None:
        # 2 assets, 3 levels: "110" = 2/3, "100" = 1/3
        w = decode_unary("110100", n_assets=2, n_levels=3)
        assert abs(w.sum() - 1.0) < 1e-10
        assert w[0] > w[1]


class TestQubitCostTable:
    def test_table_structure(self) -> None:
        table = qubit_cost_table([15, 25, 50, 100])
        assert len(table) == 4
        assert table[0]["n_assets"] == 15
        assert table[0]["one_hot"] == 15
        assert table[0]["binary_3b"] == 45
        assert table[0]["binary_5b"] == 75
        assert table[0]["unary_4"] == 60

    def test_scaling(self) -> None:
        table = qubit_cost_table([10, 20])
        assert table[1]["one_hot"] == 2 * table[0]["one_hot"]
