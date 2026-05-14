"""Real-time WebSocket streaming for live market data.

Supports Alpaca, Polygon, and IEX providers. Provides event-driven
rebalancing triggers (threshold drift, signal-based regime change)
and latency monitoring with heartbeat.

Requires ``websockets`` (optional dependency).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

try:
    import websockets

    _HAS_WEBSOCKETS = True
except ImportError:  # pragma: no cover
    _HAS_WEBSOCKETS = False


def _require_websockets() -> None:
    """Raise if websockets is not installed."""
    if not _HAS_WEBSOCKETS:
        raise ImportError(
            "websockets is required for streaming. "
            "Install it with: pip install websockets"
        )


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------


class Provider(str, Enum):
    """Supported WebSocket data providers."""

    ALPACA = "alpaca"
    POLYGON = "polygon"
    IEX = "iex"


_PROVIDER_URLS: dict[Provider, str] = {
    Provider.ALPACA: "wss://stream.data.alpaca.markets/v2/iex",
    Provider.POLYGON: "wss://socket.polygon.io/stocks",
    Provider.IEX: "wss://cloud-sse.iexapis.com/stable/stocksUS",
}


@dataclass
class StreamConfig:
    """Configuration for a price stream.

    Parameters
    ----------
    provider : Provider
        Which WebSocket data provider to use.
    api_key : str
        API key / token for the provider.
    tickers : list[str]
        Symbols to subscribe to.
    buffer_size : int
        Maximum number of price ticks to buffer per ticker.
    heartbeat_interval : float
        Seconds between heartbeat pings.
    url_override : str | None
        Override the default WebSocket URL (useful for testing).
    """

    provider: Provider = Provider.ALPACA
    api_key: str = ""
    tickers: list[str] = field(default_factory=list)
    buffer_size: int = 1000
    heartbeat_interval: float = 30.0
    url_override: str | None = None

    @property
    def ws_url(self) -> str:
        if self.url_override:
            return self.url_override
        return _PROVIDER_URLS[self.provider]


@dataclass
class RebalanceConfig:
    """Configuration for rebalance triggers.

    Parameters
    ----------
    drift_threshold : float
        Trigger rebalance when any asset drifts more than this fraction
        from its target weight (e.g. 0.05 = 5%).
    signal_callback : Callable | None
        Optional callback that returns True when a regime-change signal
        fires (e.g. from a VQC classifier).
    cooldown : float
        Minimum seconds between consecutive rebalances.
    """

    drift_threshold: float = 0.05
    signal_callback: Callable[[], bool] | None = None
    cooldown: float = 300.0


# ---------------------------------------------------------------------------
# Latency monitor
# ---------------------------------------------------------------------------


@dataclass
class LatencyMonitor:
    """Tracks message latency and heartbeat health."""

    _latencies: deque[float] = field(default_factory=lambda: deque(maxlen=500))
    last_heartbeat: float = 0.0
    last_message: float = 0.0

    def record_latency(self, latency_ms: float) -> None:
        self._latencies.append(latency_ms)
        self.last_message = time.monotonic()

    def record_heartbeat(self) -> None:
        self.last_heartbeat = time.monotonic()

    @property
    def avg_latency_ms(self) -> float:
        if not self._latencies:
            return 0.0
        return sum(self._latencies) / len(self._latencies)

    @property
    def max_latency_ms(self) -> float:
        if not self._latencies:
            return 0.0
        return max(self._latencies)

    @property
    def count(self) -> int:
        return len(self._latencies)

    def is_healthy(self, heartbeat_interval: float) -> bool:
        """True if we received a heartbeat within 2x the expected interval."""
        if self.last_heartbeat == 0.0:
            return True  # no heartbeat expected yet
        elapsed = time.monotonic() - self.last_heartbeat
        return elapsed < heartbeat_interval * 2


# ---------------------------------------------------------------------------
# Price buffer
# ---------------------------------------------------------------------------


class PriceBuffer:
    """Thread-safe (asyncio-safe) ring buffer for price ticks per ticker."""

    def __init__(self, max_size: int = 1000) -> None:
        self._max_size = max_size
        self._buffers: dict[str, deque[dict[str, Any]]] = {}

    def append(self, ticker: str, tick: dict[str, Any]) -> None:
        if ticker not in self._buffers:
            self._buffers[ticker] = deque(maxlen=self._max_size)
        self._buffers[ticker].append(tick)

    def get_latest(self, ticker: str) -> dict[str, Any] | None:
        buf = self._buffers.get(ticker)
        if not buf:
            return None
        return buf[-1]

    def get_all(self, ticker: str) -> list[dict[str, Any]]:
        return list(self._buffers.get(ticker, []))

    def to_dataframe(self, ticker: str) -> pd.DataFrame:
        ticks = self.get_all(ticker)
        if not ticks:
            return pd.DataFrame(columns=["timestamp", "price", "volume"])
        return pd.DataFrame(ticks)

    @property
    def tickers(self) -> list[str]:
        return list(self._buffers.keys())

    def clear(self, ticker: str | None = None) -> None:
        if ticker:
            self._buffers.pop(ticker, None)
        else:
            self._buffers.clear()

    def __len__(self) -> int:
        return sum(len(b) for b in self._buffers.values())


# ---------------------------------------------------------------------------
# Rebalance trigger engine
# ---------------------------------------------------------------------------


class RebalanceTrigger:
    """Evaluates whether a rebalance should fire.

    Parameters
    ----------
    config : RebalanceConfig
        Threshold and signal configuration.
    target_weights : dict[str, float]
        Target portfolio weights keyed by ticker.
    """

    def __init__(
        self,
        config: RebalanceConfig,
        target_weights: dict[str, float] | None = None,
    ) -> None:
        self.config = config
        self.target_weights = target_weights or {}
        self._last_rebalance: float = 0.0
        self._rebalance_count: int = 0

    def _in_cooldown(self) -> bool:
        if self._last_rebalance == 0.0:
            return False
        return (time.monotonic() - self._last_rebalance) < self.config.cooldown

    def check_drift(self, current_weights: dict[str, float]) -> bool:
        """Return True if any asset has drifted beyond the threshold."""
        if not self.target_weights:
            return False
        for ticker, target in self.target_weights.items():
            current = current_weights.get(ticker, 0.0)
            if abs(current - target) > self.config.drift_threshold:
                return True
        return False

    def check_signal(self) -> bool:
        """Return True if the signal callback fires."""
        if self.config.signal_callback is None:
            return False
        try:
            return bool(self.config.signal_callback())
        except Exception:
            logger.warning("Signal callback raised an exception", exc_info=True)
            return False

    def should_rebalance(self, current_weights: dict[str, float]) -> bool:
        """Evaluate all triggers; return True if rebalance is warranted."""
        if self._in_cooldown():
            return False
        if self.check_drift(current_weights) or self.check_signal():
            self._last_rebalance = time.monotonic()
            self._rebalance_count += 1
            return True
        return False

    @property
    def rebalance_count(self) -> int:
        return self._rebalance_count


# ---------------------------------------------------------------------------
# Message parsers (provider-specific)
# ---------------------------------------------------------------------------


def _parse_alpaca(msg: dict[str, Any]) -> dict[str, Any] | None:
    """Parse an Alpaca trade/quote message into a normalised tick."""
    if msg.get("T") not in ("t", "q"):
        return None
    return {
        "ticker": msg.get("S", ""),
        "price": float(msg.get("p", msg.get("bp", 0))),
        "volume": int(msg.get("s", msg.get("bs", 0))),
        "timestamp": msg.get("t", ""),
    }


def _parse_polygon(msg: dict[str, Any]) -> dict[str, Any] | None:
    """Parse a Polygon trade message."""
    ev = msg.get("ev")
    if ev not in ("T", "Q"):
        return None
    return {
        "ticker": msg.get("sym", ""),
        "price": float(msg.get("p", msg.get("bp", 0))),
        "volume": int(msg.get("s", msg.get("bs", 0))),
        "timestamp": msg.get("t", ""),
    }


def _parse_iex(msg: dict[str, Any]) -> dict[str, Any] | None:
    """Parse an IEX message."""
    if "symbol" not in msg:
        return None
    return {
        "ticker": msg["symbol"],
        "price": float(msg.get("latestPrice", msg.get("iexRealtimePrice", 0))),
        "volume": int(msg.get("latestVolume", 0)),
        "timestamp": msg.get("latestUpdate", ""),
    }


_PARSERS: dict[Provider, Callable[[dict[str, Any]], dict[str, Any] | None]] = {
    Provider.ALPACA: _parse_alpaca,
    Provider.POLYGON: _parse_polygon,
    Provider.IEX: _parse_iex,
}


# ---------------------------------------------------------------------------
# Auth message builders
# ---------------------------------------------------------------------------


def _auth_message(config: StreamConfig) -> dict[str, Any]:
    """Build the provider-specific authentication message."""
    if config.provider == Provider.ALPACA:
        return {"action": "auth", "key": config.api_key, "secret": ""}
    elif config.provider == Provider.POLYGON:
        return {"action": "auth", "params": config.api_key}
    else:  # IEX
        return {"token": config.api_key}


def _subscribe_message(config: StreamConfig) -> dict[str, Any]:
    """Build the provider-specific subscription message."""
    if config.provider == Provider.ALPACA:
        return {"action": "subscribe", "trades": config.tickers}
    elif config.provider == Provider.POLYGON:
        channels = ",".join(f"T.{t}" for t in config.tickers)
        return {"action": "subscribe", "params": channels}
    else:  # IEX
        return {"symbols": config.tickers, "channels": ["tops"]}


# ---------------------------------------------------------------------------
# Portfolio value tracker
# ---------------------------------------------------------------------------


class PortfolioTracker:
    """Track live portfolio value from streaming prices.

    Parameters
    ----------
    holdings : dict[str, float]
        Number of shares held per ticker.
    """

    def __init__(self, holdings: dict[str, float]) -> None:
        self.holdings = dict(holdings)
        self._prices: dict[str, float] = {}

    def update_price(self, ticker: str, price: float) -> None:
        self._prices[ticker] = price

    @property
    def portfolio_value(self) -> float:
        total = 0.0
        for ticker, shares in self.holdings.items():
            price = self._prices.get(ticker, 0.0)
            total += shares * price
        return total

    @property
    def current_weights(self) -> dict[str, float]:
        total = self.portfolio_value
        if total == 0.0:
            return dict.fromkeys(self.holdings, 0.0)
        return {
            t: (self.holdings[t] * self._prices.get(t, 0.0)) / total
            for t in self.holdings
        }


# ---------------------------------------------------------------------------
# Main streaming class
# ---------------------------------------------------------------------------


class PriceStream:
    """Async WebSocket price stream with buffering, rebalancing, and monitoring.

    Parameters
    ----------
    config : StreamConfig
        Connection and subscription settings.
    rebalance_config : RebalanceConfig | None
        If provided, enables automatic rebalance triggers.
    target_weights : dict[str, float] | None
        Target portfolio weights for drift detection.
    holdings : dict[str, float] | None
        Current share holdings for portfolio tracking.
    on_tick : Callable | None
        Callback invoked on each parsed tick: ``on_tick(tick_dict)``.
    on_rebalance : Callable | None
        Callback invoked when a rebalance fires: ``on_rebalance(current_weights)``.
    """

    def __init__(
        self,
        config: StreamConfig,
        rebalance_config: RebalanceConfig | None = None,
        target_weights: dict[str, float] | None = None,
        holdings: dict[str, float] | None = None,
        on_tick: Callable[[dict[str, Any]], None] | None = None,
        on_rebalance: Callable[[dict[str, float]], None] | None = None,
    ) -> None:
        _require_websockets()
        self.config = config
        self.buffer = PriceBuffer(max_size=config.buffer_size)
        self.latency = LatencyMonitor()
        self._parser = _PARSERS[config.provider]
        self._on_tick = on_tick
        self._on_rebalance = on_rebalance
        self._running = False
        self._ws: Any = None

        # Portfolio tracking
        self.tracker = PortfolioTracker(holdings or {})

        # Rebalance trigger
        self.trigger: RebalanceTrigger | None = None
        if rebalance_config:
            self.trigger = RebalanceTrigger(rebalance_config, target_weights)

    async def _heartbeat_loop(self) -> None:
        """Send periodic pings to keep the connection alive."""
        while self._running and self._ws:
            try:
                pong = await self._ws.ping()
                await asyncio.wait_for(pong, timeout=10)
                self.latency.record_heartbeat()
            except (asyncio.TimeoutError, websockets.ConnectionClosed):
                logger.warning("Heartbeat failed")
                break
            await asyncio.sleep(self.config.heartbeat_interval)

    async def _process_message(self, raw: str) -> None:
        """Parse and buffer a single message."""
        import json

        recv_time = time.monotonic()

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            logger.debug("Non-JSON message: %s", raw[:100])
            return

        # Some providers send arrays of messages
        messages = data if isinstance(data, list) else [data]

        for msg in messages:
            tick = self._parser(msg)
            if tick is None:
                continue

            # Latency estimation (if timestamp available)
            latency_ms = (time.monotonic() - recv_time) * 1000
            self.latency.record_latency(latency_ms)

            ticker = tick["ticker"]
            self.buffer.append(ticker, tick)
            self.tracker.update_price(ticker, tick["price"])

            if self._on_tick:
                self._on_tick(tick)

            # Check rebalance trigger
            if self.trigger:
                weights = self.tracker.current_weights
                if self.trigger.should_rebalance(weights):
                    logger.info("Rebalance triggered, weights: %s", weights)
                    if self._on_rebalance:
                        self._on_rebalance(weights)

    async def connect(self, max_messages: int | None = None) -> None:
        """Connect to the WebSocket and begin streaming.

        Parameters
        ----------
        max_messages : int | None
            If set, disconnect after this many messages (useful for testing).
        """
        import json

        self._running = True
        msg_count = 0

        async with websockets.connect(self.config.ws_url) as ws:
            self._ws = ws

            # Authenticate
            await ws.send(json.dumps(_auth_message(self.config)))

            # Subscribe
            await ws.send(json.dumps(_subscribe_message(self.config)))

            # Start heartbeat
            heartbeat_task = asyncio.create_task(self._heartbeat_loop())

            try:
                async for message in ws:
                    if not self._running:
                        break
                    await self._process_message(message)
                    msg_count += 1
                    if max_messages and msg_count >= max_messages:
                        break
            finally:
                self._running = False
                heartbeat_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await heartbeat_task
                self._ws = None

    def stop(self) -> None:
        """Signal the stream to stop."""
        self._running = False

    @property
    def is_running(self) -> bool:
        return self._running
