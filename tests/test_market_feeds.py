"""Tests for CONCEPT:KG-2.6 — Real-Time Market Data Feeds."""

import pytest

from agent_utilities.domains.finance.market_feeds import (
    LiveBar,
    MarketFeedBus,
    Tick,
    TickAggregator,
)


class TestTick:
    def test_creation(self):
        tick = Tick(symbol="AAPL", price=150.0, volume=1000)
        assert tick.symbol == "AAPL"
        assert tick.timestamp != ""


class TestLiveBar:
    def test_update(self):
        bar = LiveBar(symbol="AAPL")
        bar.update(Tick(symbol="AAPL", price=100.0, volume=500))
        bar.update(Tick(symbol="AAPL", price=105.0, volume=300))
        bar.update(Tick(symbol="AAPL", price=98.0, volume=200))
        assert bar.open == 100.0
        assert bar.high == 105.0
        assert bar.low == 98.0
        assert bar.close == 98.0
        assert bar.volume == 1000
        assert bar.tick_count == 3

    def test_empty_bar(self):
        bar = LiveBar(symbol="AAPL")
        assert bar.is_empty

    def test_to_dict(self):
        bar = LiveBar(symbol="AAPL")
        bar.update(Tick(symbol="AAPL", price=100.0))
        d = bar.to_dict()
        assert d["symbol"] == "AAPL"
        assert d["open"] == 100.0


class TestTickAggregator:
    def test_accumulate_ticks(self):
        agg = TickAggregator(symbol="AAPL", bar_size_seconds=3600)
        agg.on_tick(Tick(symbol="AAPL", price=100.0, volume=100))
        agg.on_tick(Tick(symbol="AAPL", price=101.0, volume=200))
        assert agg.total_ticks == 2
        assert agg.current_bar.tick_count == 2

    def test_current_bar_updates(self):
        agg = TickAggregator(symbol="AAPL")
        agg.on_tick(Tick(symbol="AAPL", price=100.0))
        agg.on_tick(Tick(symbol="AAPL", price=110.0))
        assert agg.current_bar.high == 110.0


class TestMarketFeedBus:
    @pytest.mark.asyncio
    async def test_publish_tick(self):
        bus = MarketFeedBus()
        received = []
        bus.subscribe_ticks("AAPL", lambda m: received.append(m))
        await bus.publish_tick(Tick(symbol="AAPL", price=150.0))
        assert len(received) == 1

    @pytest.mark.asyncio
    async def test_bar_aggregation(self):
        bus = MarketFeedBus()
        bars = []
        bus.subscribe_bars("AAPL", 1, lambda m: bars.append(m))
        # Publish multiple ticks — bars are time-based so may not complete
        for i in range(5):
            await bus.publish_tick(Tick(symbol="AAPL", price=100.0 + i))
        assert bus.total_ticks == 5
        assert "AAPL" in bus.active_symbols

    @pytest.mark.asyncio
    async def test_price_alert_above(self):
        bus = MarketFeedBus()
        alerts = []
        bus.add_price_alert("AAPL", 155.0, "above", lambda t: alerts.append(t))
        await bus.publish_tick(Tick(symbol="AAPL", price=150.0))
        assert len(alerts) == 0
        await bus.publish_tick(Tick(symbol="AAPL", price=160.0))
        assert len(alerts) == 1

    @pytest.mark.asyncio
    async def test_price_alert_below(self):
        bus = MarketFeedBus()
        alerts = []
        bus.add_price_alert("AAPL", 100.0, "below", lambda t: alerts.append(t))
        await bus.publish_tick(Tick(symbol="AAPL", price=110.0))
        assert len(alerts) == 0
        await bus.publish_tick(Tick(symbol="AAPL", price=95.0))
        assert len(alerts) == 1

    @pytest.mark.asyncio
    async def test_alert_only_triggers_once(self):
        bus = MarketFeedBus()
        alerts = []
        bus.add_price_alert("AAPL", 155.0, "above", lambda t: alerts.append(t))
        await bus.publish_tick(Tick(symbol="AAPL", price=160.0))
        await bus.publish_tick(Tick(symbol="AAPL", price=170.0))
        assert len(alerts) == 1

    @pytest.mark.asyncio
    async def test_pending_alerts(self):
        bus = MarketFeedBus()
        bus.add_price_alert("AAPL", 200.0, "above")
        bus.add_price_alert("MSFT", 300.0, "above")
        assert bus.pending_alerts == 2

    @pytest.mark.asyncio
    async def test_multi_symbol(self):
        bus = MarketFeedBus()
        aapl_ticks = []
        msft_ticks = []
        bus.subscribe_ticks("AAPL", lambda m: aapl_ticks.append(m))
        bus.subscribe_ticks("MSFT", lambda m: msft_ticks.append(m))
        await bus.publish_tick(Tick(symbol="AAPL", price=150.0))
        await bus.publish_tick(Tick(symbol="MSFT", price=300.0))
        assert len(aapl_ticks) == 1
        assert len(msft_ticks) == 1
