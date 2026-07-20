"""
Real-Time Market Data Feeds — CONCEPT:AU-KG.research.research-pipeline-runner

Finance-specific adapter on top of the universal StreamBus (KG-2.6)
for real-time market data streaming with OHLCV aggregation and
tick-level processing.

Source: FinceptTerminal WebSocket Architecture (independent implementation)
"""

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from .streaming import CallbackSubscriber, StreamBus, StreamMessage

logger = logging.getLogger(__name__)


@dataclass
class Tick:
    """A single market tick (price update)."""

    symbol: str
    price: float
    volume: float = 0.0
    bid: float = 0.0
    ask: float = 0.0
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(UTC).isoformat()


@dataclass
class LiveBar:
    """A live OHLCV bar being aggregated from ticks."""

    symbol: str
    open: float = 0.0
    high: float = 0.0
    low: float = float("inf")
    close: float = 0.0
    volume: float = 0.0
    tick_count: int = 0
    bar_start: str = ""
    bar_end: str = ""

    def update(self, tick: Tick) -> None:
        """Update the bar with a new tick."""
        if self.tick_count == 0:
            self.open = tick.price
            self.bar_start = tick.timestamp

        self.high = max(self.high, tick.price)
        self.low = min(self.low, tick.price)
        self.close = tick.price
        self.volume += tick.volume
        self.tick_count += 1
        self.bar_end = tick.timestamp

    @property
    def is_empty(self) -> bool:
        return self.tick_count == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "tick_count": self.tick_count,
        }


@dataclass
class FeedSubscription:
    """A subscription to a market data feed."""

    symbol: str
    feed_type: str = "tick"  # "tick", "bar_1m", "bar_5m", "bar_1h"
    callback: Any = None


class TickAggregator:
    """
    Aggregates raw ticks into OHLCV bars at configurable intervals.

    Usage:
        agg = TickAggregator(bar_size=60)  # 60-second bars
        agg.on_tick(tick)
        completed_bars = agg.get_completed_bars()
    """

    def __init__(self, symbol: str, bar_size_seconds: int = 60):
        self.symbol = symbol
        self.bar_size = bar_size_seconds
        self._current_bar = LiveBar(symbol=symbol)
        self._completed_bars: list[LiveBar] = []
        self._tick_count = 0
        self._bar_start_time: float | None = None

    def on_tick(self, tick: Tick) -> LiveBar | None:
        """
        Process a tick. Returns a completed bar if the current bar
        has been closed by the time boundary.
        """
        now = datetime.now(UTC).timestamp()

        if self._bar_start_time is None:
            self._bar_start_time = now

        # Check if current bar should be closed
        elapsed = now - self._bar_start_time
        completed = None

        if elapsed >= self.bar_size and not self._current_bar.is_empty:
            completed = self._current_bar
            self._completed_bars.append(completed)
            self._current_bar = LiveBar(symbol=self.symbol)
            self._bar_start_time = now

        self._current_bar.update(tick)
        self._tick_count += 1
        return completed

    @property
    def current_bar(self) -> LiveBar:
        return self._current_bar

    @property
    def completed_bars(self) -> list[LiveBar]:
        return list(self._completed_bars)

    @property
    def total_ticks(self) -> int:
        return self._tick_count


class MarketFeedBus:
    """
    Finance-specific adapter on top of the universal StreamBus.

    Handles:
    - Symbol-specific topic routing (market.{symbol}.ticks)
    - Tick-to-bar aggregation
    - Multi-symbol subscription management
    - Price alert triggers

    Usage:
        bus = MarketFeedBus()
        bus.subscribe_ticks("AAPL", my_callback)
        await bus.publish_tick(Tick(symbol="AAPL", price=150.0))
    """

    def __init__(self, bus: StreamBus | None = None):
        self.bus = bus or StreamBus()
        self._aggregators: dict[str, TickAggregator] = {}
        self._alerts: list[dict[str, Any]] = []
        self._tick_count = 0

    def subscribe_ticks(self, symbol: str, callback) -> None:
        """Subscribe to real-time ticks for a symbol."""
        topic = f"market.{symbol}.ticks"
        sub = CallbackSubscriber(on_message_fn=callback)
        self.bus.subscribe(topic, sub)

    def subscribe_bars(self, symbol: str, bar_size: int, callback) -> None:
        """Subscribe to aggregated bars for a symbol."""
        topic = f"market.{symbol}.bars.{bar_size}s"
        sub = CallbackSubscriber(on_message_fn=callback)
        self.bus.subscribe(topic, sub)

        if symbol not in self._aggregators:
            self._aggregators[symbol] = TickAggregator(symbol, bar_size)

    def add_price_alert(
        self, symbol: str, price: float, direction: str = "above", callback=None
    ) -> None:
        """Set a price alert that triggers when price crosses a level."""
        self._alerts.append(
            {
                "symbol": symbol,
                "price": price,
                "direction": direction,
                "callback": callback,
                "triggered": False,
            }
        )

    async def publish_tick(self, tick: Tick) -> int:
        """
        Publish a tick to the bus and process aggregations/alerts.
        """
        self._tick_count += 1

        # Publish raw tick
        msg = StreamMessage(
            topic=f"market.{tick.symbol}.ticks",
            data={
                "price": tick.price,
                "volume": tick.volume,
                "bid": tick.bid,
                "ask": tick.ask,
            },
            source="market_feed",
        )
        delivered = await self.bus.publish(msg)

        # Process aggregation
        if tick.symbol in self._aggregators:
            agg = self._aggregators[tick.symbol]
            completed = agg.on_tick(tick)
            if completed:
                bar_msg = StreamMessage(
                    topic=f"market.{tick.symbol}.bars.{agg.bar_size}s",
                    data=completed.to_dict(),
                    source="market_feed_aggregator",
                )
                delivered += await self.bus.publish(bar_msg)

        # Check price alerts
        for alert in self._alerts:
            if alert["triggered"] or alert["symbol"] != tick.symbol:
                continue
            if alert["direction"] == "above" and tick.price >= alert["price"]:
                alert["triggered"] = True
                if alert["callback"]:
                    alert["callback"](tick)
            elif alert["direction"] == "below" and tick.price <= alert["price"]:
                alert["triggered"] = True
                if alert["callback"]:
                    alert["callback"](tick)

        return delivered

    @property
    def total_ticks(self) -> int:
        return self._tick_count

    @property
    def active_symbols(self) -> list[str]:
        return list(self._aggregators.keys())

    @property
    def pending_alerts(self) -> int:
        return sum(1 for a in self._alerts if not a["triggered"])
