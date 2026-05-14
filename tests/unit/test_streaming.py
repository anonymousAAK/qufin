"""Tests for qufin.data.streaming — real-time WebSocket streaming."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pandas as pd
import pytest

from qufin.data.streaming import (
    LatencyMonitor,
    PortfolioTracker,
    PriceBuffer,
    PriceStream,
    Provider,
    RebalanceConfig,
    RebalanceTrigger,
    StreamConfig,
    _auth_message,
    _parse_alpaca,
    _parse_iex,
    _parse_polygon,
    _subscribe_message,
)

# ---------------------------------------------------------------------------
# StreamConfig
# ---------------------------------------------------------------------------


class TestStreamConfig:
    def test_default_url_alpaca(self):
        cfg = StreamConfig(provider=Provider.ALPACA)
        assert "alpaca" in cfg.ws_url

    def test_default_url_polygon(self):
        cfg = StreamConfig(provider=Provider.POLYGON)
        assert "polygon" in cfg.ws_url

    def test_default_url_iex(self):
        cfg = StreamConfig(provider=Provider.IEX)
        assert "iex" in cfg.ws_url

    def test_url_override(self):
        cfg = StreamConfig(url_override="ws://localhost:9999")
        assert cfg.ws_url == "ws://localhost:9999"


# ---------------------------------------------------------------------------
# PriceBuffer
# ---------------------------------------------------------------------------


class TestPriceBuffer:
    def test_append_and_get_latest(self):
        buf = PriceBuffer(max_size=5)
        buf.append("AAPL", {"price": 150.0, "timestamp": "t1"})
        buf.append("AAPL", {"price": 151.0, "timestamp": "t2"})
        latest = buf.get_latest("AAPL")
        assert latest is not None
        assert latest["price"] == 151.0

    def test_get_latest_missing_ticker(self):
        buf = PriceBuffer()
        assert buf.get_latest("MISSING") is None

    def test_max_size_eviction(self):
        buf = PriceBuffer(max_size=3)
        for i in range(5):
            buf.append("X", {"price": float(i)})
        ticks = buf.get_all("X")
        assert len(ticks) == 3
        assert ticks[0]["price"] == 2.0  # oldest surviving

    def test_to_dataframe(self):
        buf = PriceBuffer()
        buf.append("AAPL", {"timestamp": "t1", "price": 100.0, "volume": 10})
        buf.append("AAPL", {"timestamp": "t2", "price": 101.0, "volume": 20})
        df = buf.to_dataframe("AAPL")
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2

    def test_to_dataframe_empty(self):
        buf = PriceBuffer()
        df = buf.to_dataframe("NONE")
        assert df.empty

    def test_tickers_property(self):
        buf = PriceBuffer()
        buf.append("AAPL", {"price": 1.0})
        buf.append("GOOG", {"price": 2.0})
        assert set(buf.tickers) == {"AAPL", "GOOG"}

    def test_clear_specific_ticker(self):
        buf = PriceBuffer()
        buf.append("AAPL", {"price": 1.0})
        buf.append("GOOG", {"price": 2.0})
        buf.clear("AAPL")
        assert "AAPL" not in buf.tickers
        assert "GOOG" in buf.tickers

    def test_clear_all(self):
        buf = PriceBuffer()
        buf.append("AAPL", {"price": 1.0})
        buf.clear()
        assert len(buf) == 0

    def test_len(self):
        buf = PriceBuffer()
        buf.append("A", {"p": 1})
        buf.append("A", {"p": 2})
        buf.append("B", {"p": 3})
        assert len(buf) == 3


# ---------------------------------------------------------------------------
# LatencyMonitor
# ---------------------------------------------------------------------------


class TestLatencyMonitor:
    def test_record_and_avg(self):
        mon = LatencyMonitor()
        mon.record_latency(10.0)
        mon.record_latency(20.0)
        assert mon.avg_latency_ms == 15.0
        assert mon.max_latency_ms == 20.0
        assert mon.count == 2

    def test_empty_latency(self):
        mon = LatencyMonitor()
        assert mon.avg_latency_ms == 0.0
        assert mon.max_latency_ms == 0.0

    def test_heartbeat_healthy(self):
        mon = LatencyMonitor()
        mon.record_heartbeat()
        assert mon.is_healthy(30.0) is True

    def test_heartbeat_no_heartbeat_yet(self):
        mon = LatencyMonitor()
        assert mon.is_healthy(30.0) is True


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------


class TestParsers:
    def test_parse_alpaca_trade(self):
        msg = {"T": "t", "S": "AAPL", "p": 150.5, "s": 100, "t": "2024-01-01"}
        tick = _parse_alpaca(msg)
        assert tick is not None
        assert tick["ticker"] == "AAPL"
        assert tick["price"] == 150.5

    def test_parse_alpaca_non_trade(self):
        assert _parse_alpaca({"T": "success"}) is None

    def test_parse_polygon_trade(self):
        msg = {"ev": "T", "sym": "GOOG", "p": 2800.0, "s": 50, "t": 123}
        tick = _parse_polygon(msg)
        assert tick is not None
        assert tick["ticker"] == "GOOG"

    def test_parse_polygon_non_trade(self):
        assert _parse_polygon({"ev": "status"}) is None

    def test_parse_iex(self):
        msg = {"symbol": "MSFT", "latestPrice": 300.0, "latestVolume": 1000}
        tick = _parse_iex(msg)
        assert tick is not None
        assert tick["ticker"] == "MSFT"
        assert tick["price"] == 300.0

    def test_parse_iex_missing_symbol(self):
        assert _parse_iex({"latestPrice": 100.0}) is None


# ---------------------------------------------------------------------------
# Auth / Subscribe messages
# ---------------------------------------------------------------------------


class TestMessages:
    def test_auth_alpaca(self):
        cfg = StreamConfig(provider=Provider.ALPACA, api_key="key123")
        msg = _auth_message(cfg)
        assert msg["action"] == "auth"
        assert msg["key"] == "key123"

    def test_auth_polygon(self):
        cfg = StreamConfig(provider=Provider.POLYGON, api_key="pk")
        msg = _auth_message(cfg)
        assert msg["params"] == "pk"

    def test_auth_iex(self):
        cfg = StreamConfig(provider=Provider.IEX, api_key="tok")
        msg = _auth_message(cfg)
        assert msg["token"] == "tok"

    def test_subscribe_alpaca(self):
        cfg = StreamConfig(provider=Provider.ALPACA, tickers=["AAPL", "GOOG"])
        msg = _subscribe_message(cfg)
        assert msg["trades"] == ["AAPL", "GOOG"]

    def test_subscribe_polygon(self):
        cfg = StreamConfig(provider=Provider.POLYGON, tickers=["AAPL"])
        msg = _subscribe_message(cfg)
        assert "T.AAPL" in msg["params"]


# ---------------------------------------------------------------------------
# RebalanceTrigger
# ---------------------------------------------------------------------------


class TestRebalanceTrigger:
    def test_no_drift(self):
        cfg = RebalanceConfig(drift_threshold=0.05)
        trigger = RebalanceTrigger(cfg, {"AAPL": 0.5, "GOOG": 0.5})
        assert trigger.check_drift({"AAPL": 0.52, "GOOG": 0.48}) is False

    def test_drift_detected(self):
        cfg = RebalanceConfig(drift_threshold=0.05)
        trigger = RebalanceTrigger(cfg, {"AAPL": 0.5, "GOOG": 0.5})
        assert trigger.check_drift({"AAPL": 0.60, "GOOG": 0.40}) is True

    def test_signal_callback(self):
        cfg = RebalanceConfig(signal_callback=lambda: True)
        trigger = RebalanceTrigger(cfg)
        assert trigger.check_signal() is True

    def test_signal_callback_none(self):
        cfg = RebalanceConfig()
        trigger = RebalanceTrigger(cfg)
        assert trigger.check_signal() is False

    def test_signal_callback_exception(self):
        def bad_signal():
            raise RuntimeError("boom")

        cfg = RebalanceConfig(signal_callback=bad_signal)
        trigger = RebalanceTrigger(cfg)
        assert trigger.check_signal() is False

    def test_should_rebalance_with_cooldown(self):
        cfg = RebalanceConfig(drift_threshold=0.01, cooldown=1000.0)
        trigger = RebalanceTrigger(cfg, {"A": 0.5})
        # First rebalance should fire
        assert trigger.should_rebalance({"A": 0.7}) is True
        # Second should be blocked by cooldown
        assert trigger.should_rebalance({"A": 0.7}) is False
        assert trigger.rebalance_count == 1

    def test_no_target_weights(self):
        cfg = RebalanceConfig(drift_threshold=0.05)
        trigger = RebalanceTrigger(cfg, {})
        assert trigger.check_drift({"A": 0.5}) is False


# ---------------------------------------------------------------------------
# PortfolioTracker
# ---------------------------------------------------------------------------


class TestPortfolioTracker:
    def test_portfolio_value(self):
        tracker = PortfolioTracker({"AAPL": 10, "GOOG": 5})
        tracker.update_price("AAPL", 150.0)
        tracker.update_price("GOOG", 2800.0)
        assert tracker.portfolio_value == 10 * 150.0 + 5 * 2800.0

    def test_current_weights(self):
        tracker = PortfolioTracker({"A": 100, "B": 100})
        tracker.update_price("A", 1.0)
        tracker.update_price("B", 1.0)
        weights = tracker.current_weights
        assert abs(weights["A"] - 0.5) < 1e-10
        assert abs(weights["B"] - 0.5) < 1e-10

    def test_zero_value(self):
        tracker = PortfolioTracker({"A": 0})
        weights = tracker.current_weights
        assert weights["A"] == 0.0


# ---------------------------------------------------------------------------
# PriceStream (async, mocked WebSocket)
# ---------------------------------------------------------------------------


class TestPriceStream:
    @pytest.fixture
    def config(self):
        return StreamConfig(
            provider=Provider.ALPACA,
            api_key="test",
            tickers=["AAPL"],
            url_override="ws://localhost:9999",
            buffer_size=100,
            heartbeat_interval=60.0,
        )

    def test_init(self, config):
        stream = PriceStream(config)
        assert stream.is_running is False

    def test_stop(self, config):
        stream = PriceStream(config)
        stream._running = True
        stream.stop()
        assert stream.is_running is False

    @pytest.mark.asyncio
    async def test_process_message_trade(self, config):
        ticks_received = []
        stream = PriceStream(config, on_tick=ticks_received.append)

        trade_msg = json.dumps(
            {"T": "t", "S": "AAPL", "p": 155.0, "s": 200, "t": "2024-01-01"}
        )
        await stream._process_message(trade_msg)

        assert len(ticks_received) == 1
        assert ticks_received[0]["ticker"] == "AAPL"
        assert stream.buffer.get_latest("AAPL")["price"] == 155.0

    @pytest.mark.asyncio
    async def test_process_message_non_json(self, config):
        stream = PriceStream(config)
        await stream._process_message("not json at all")
        assert len(stream.buffer) == 0

    @pytest.mark.asyncio
    async def test_process_message_array(self, config):
        stream = PriceStream(config)
        msgs = json.dumps([
            {"T": "t", "S": "AAPL", "p": 100.0, "s": 10, "t": "t1"},
            {"T": "t", "S": "AAPL", "p": 101.0, "s": 20, "t": "t2"},
        ])
        await stream._process_message(msgs)
        assert len(stream.buffer) == 2

    @pytest.mark.asyncio
    async def test_rebalance_trigger_fires(self, config):
        rebalance_calls = []
        rebal_cfg = RebalanceConfig(drift_threshold=0.01, cooldown=0.0)
        stream = PriceStream(
            config,
            rebalance_config=rebal_cfg,
            target_weights={"AAPL": 0.5},
            holdings={"AAPL": 100},
            on_rebalance=rebalance_calls.append,
        )
        trade_msg = json.dumps(
            {"T": "t", "S": "AAPL", "p": 155.0, "s": 200, "t": "now"}
        )
        await stream._process_message(trade_msg)
        # With only AAPL, weight is 1.0, target is 0.5, drift > 0.01
        assert len(rebalance_calls) == 1

    @pytest.mark.asyncio
    async def test_connect_with_mock_ws(self, config):
        """Test the full connect flow with a mocked websocket."""
        trade_data = json.dumps(
            {"T": "t", "S": "AAPL", "p": 160.0, "s": 50, "t": "ts1"}
        )

        async def _aiter_messages():
            yield trade_data

        mock_ws = AsyncMock()
        mock_ws.__aiter__ = MagicMock(return_value=_aiter_messages())
        mock_ws.ping = AsyncMock(return_value=asyncio.Future())
        mock_ws.ping.return_value.set_result(None)

        mock_connect_ctx = AsyncMock()
        mock_connect_ctx.__aenter__ = AsyncMock(return_value=mock_ws)
        mock_connect_ctx.__aexit__ = AsyncMock(return_value=False)

        import qufin.data.streaming as streaming_mod
        original_ws = streaming_mod.websockets
        try:
            mock_mod = MagicMock()
            mock_mod.connect = MagicMock(return_value=mock_connect_ctx)
            mock_mod.ConnectionClosed = Exception
            streaming_mod.websockets = mock_mod

            stream = PriceStream(config)
            await stream.connect(max_messages=1)

            assert stream.buffer.get_latest("AAPL") is not None
        finally:
            streaming_mod.websockets = original_ws
