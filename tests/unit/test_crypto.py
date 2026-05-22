"""Unit tests for cryptocurrency data connector."""

from __future__ import annotations

import time as _time
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from qufin.data import crypto as crypto_mod
from qufin.data.crypto import (
    fetch_crypto_prices,
    fetch_crypto_returns,
)


def _mock_coingecko_response(n_days: int = 10) -> dict:
    """Create a mock CoinGecko market_chart response."""
    base_ts = int(_time.time() * 1000) - n_days * 86400_000
    rng = np.random.default_rng(42)
    prices = [
        [base_ts + i * 86400_000, 30000 + i * 100 + rng.normal(0, 50)]
        for i in range(n_days)
    ]
    return {"prices": prices}


@pytest.fixture
def mock_requests():
    mock_req = MagicMock()
    resp = MagicMock()
    resp.json.return_value = _mock_coingecko_response(10)
    resp.raise_for_status = MagicMock()
    mock_req.get.return_value = resp
    original = crypto_mod.requests
    crypto_mod.requests = mock_req
    yield mock_req
    crypto_mod.requests = original


class TestFetchCryptoPrices:
    def test_returns_dataframe(self, mock_requests) -> None:
        df = fetch_crypto_prices(["bitcoin"], days=10)
        assert isinstance(df, pd.DataFrame)

    def test_columns_match_coins(self, mock_requests) -> None:
        df = fetch_crypto_prices(["bitcoin"], days=10)
        assert "bitcoin" in df.columns

    def test_index_is_datetime(self, mock_requests) -> None:
        df = fetch_crypto_prices(["bitcoin"], days=10)
        assert pd.api.types.is_datetime64_any_dtype(df.index)

    def test_multiple_coins(self, mock_requests) -> None:
        with patch.object(crypto_mod, "time") as mock_time:
            mock_time.sleep = MagicMock()
            fetch_crypto_prices(["bitcoin", "ethereum"], days=10)
            assert mock_time.sleep.called

    def test_calls_correct_url(self, mock_requests) -> None:
        fetch_crypto_prices(["bitcoin"], vs_currency="eur", days=30)
        call_args = mock_requests.get.call_args
        assert "bitcoin" in call_args[0][0]
        assert call_args[1]["params"]["vs_currency"] == "eur"
        assert call_args[1]["params"]["days"] == 30

    def test_raise_for_status_called(self, mock_requests) -> None:
        resp = mock_requests.get.return_value
        fetch_crypto_prices(["bitcoin"], days=10)
        resp.raise_for_status.assert_called_once()

    def test_non_empty_dataframe(self, mock_requests) -> None:
        df = fetch_crypto_prices(["bitcoin"], days=10)
        assert len(df) > 0


class TestFetchCryptoReturns:
    def test_returns_dataframe(self, mock_requests) -> None:
        df = fetch_crypto_returns(["bitcoin"], days=10)
        assert isinstance(df, pd.DataFrame)

    def test_returns_are_pct_change(self, mock_requests) -> None:
        df = fetch_crypto_returns(["bitcoin"], days=10)
        assert len(df) > 0

    def test_one_less_row_than_prices(self, mock_requests) -> None:
        prices = fetch_crypto_prices(["bitcoin"], days=10)
        returns = fetch_crypto_returns(["bitcoin"], days=10)
        assert len(returns) <= len(prices)

    def test_columns_match(self, mock_requests) -> None:
        df = fetch_crypto_returns(["bitcoin"], days=10)
        assert "bitcoin" in df.columns


class TestCryptoEdgeCases:
    def test_empty_prices_list(self, mock_requests) -> None:
        resp = MagicMock()
        resp.json.return_value = {"prices": []}
        resp.raise_for_status = MagicMock()
        mock_requests.get.return_value = resp
        df = fetch_crypto_prices(["bitcoin"], days=0)
        assert isinstance(df, pd.DataFrame)

    def test_vs_currency_default(self, mock_requests) -> None:
        fetch_crypto_prices(["bitcoin"], days=10)
        call_args = mock_requests.get.call_args
        assert call_args[1]["params"]["vs_currency"] == "usd"

    def test_interval_daily(self, mock_requests) -> None:
        fetch_crypto_prices(["bitcoin"], days=10)
        call_args = mock_requests.get.call_args
        assert call_args[1]["params"]["interval"] == "daily"
