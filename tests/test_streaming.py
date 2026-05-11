"""Tests for CONCEPT:KG-2.6 — Universal Real-Time Streaming."""

import asyncio
import json

import pytest

from agent_utilities.domains.finance.streaming import (
    CallbackSubscriber,
    StreamBus,
    StreamMessage,
)


class TestStreamMessage:
    def test_creation(self):
        msg = StreamMessage(topic="market.AAPL.trades", data={"price": 150.0})
        assert msg.topic == "market.AAPL.trades"
        assert msg.data["price"] == 150.0
        assert msg.timestamp != ""

    def test_serialization(self):
        msg = StreamMessage(topic="test", data={"key": "value"}, source="unit_test")
        json_str = msg.to_json()
        restored = StreamMessage.from_json(json_str)
        assert restored.topic == "test"
        assert restored.data["key"] == "value"
        assert restored.source == "unit_test"


class TestCallbackSubscriber:
    @pytest.mark.asyncio
    async def test_on_message_callback(self):
        received = []
        sub = CallbackSubscriber(on_message_fn=lambda m: received.append(m))
        msg = StreamMessage(topic="test", data={"x": 1})
        await sub.on_message(msg)
        assert len(received) == 1

    @pytest.mark.asyncio
    async def test_on_error_callback(self):
        errors = []
        sub = CallbackSubscriber(on_error_fn=lambda e: errors.append(str(e)))
        await sub.on_error(ValueError("test error"))
        assert "test error" in errors[0]

    @pytest.mark.asyncio
    async def test_no_callback(self):
        sub = CallbackSubscriber()
        await sub.on_message(StreamMessage(topic="test"))  # Should not raise


class TestStreamBus:
    @pytest.mark.asyncio
    async def test_publish_subscribe(self):
        bus = StreamBus()
        received = []
        sub = CallbackSubscriber(on_message_fn=lambda m: received.append(m))
        bus.subscribe("market.AAPL", sub)

        msg = StreamMessage(topic="market.AAPL", data={"price": 150.0})
        delivered = await bus.publish(msg)

        assert delivered == 1
        assert len(received) == 1
        assert received[0].data["price"] == 150.0

    @pytest.mark.asyncio
    async def test_wildcard_subscribe(self):
        bus = StreamBus()
        received = []
        sub = CallbackSubscriber(on_message_fn=lambda m: received.append(m))
        bus.subscribe("market.*", sub)

        await bus.publish(StreamMessage(topic="market.AAPL", data={}))
        await bus.publish(StreamMessage(topic="market.MSFT", data={}))
        await bus.publish(StreamMessage(topic="system.health", data={}))

        assert len(received) == 2  # Only market.* topics

    @pytest.mark.asyncio
    async def test_global_wildcard(self):
        bus = StreamBus()
        received = []
        sub = CallbackSubscriber(on_message_fn=lambda m: received.append(m))
        bus.subscribe("*", sub)

        await bus.publish(StreamMessage(topic="anything", data={}))
        await bus.publish(StreamMessage(topic="else", data={}))

        assert len(received) == 2

    @pytest.mark.asyncio
    async def test_sequence_numbering(self):
        bus = StreamBus()
        received = []
        sub = CallbackSubscriber(on_message_fn=lambda m: received.append(m))
        bus.subscribe("test", sub)

        await bus.publish(StreamMessage(topic="test"))
        await bus.publish(StreamMessage(topic="test"))

        assert received[0].sequence == 1
        assert received[1].sequence == 2

    @pytest.mark.asyncio
    async def test_unsubscribe(self):
        bus = StreamBus()
        received = []
        sub = CallbackSubscriber(on_message_fn=lambda m: received.append(m))
        bus.subscribe("test", sub)
        bus.unsubscribe("test", sub)

        await bus.publish(StreamMessage(topic="test"))
        assert len(received) == 0

    @pytest.mark.asyncio
    async def test_message_history(self):
        bus = StreamBus()
        await bus.publish(StreamMessage(topic="a", data={"x": 1}))
        await bus.publish(StreamMessage(topic="b", data={"x": 2}))

        all_history = bus.get_history()
        assert len(all_history) == 2

        a_history = bus.get_history("a")
        assert len(a_history) == 1

    @pytest.mark.asyncio
    async def test_topic_and_subscriber_counts(self):
        bus = StreamBus()
        sub1 = CallbackSubscriber()
        sub2 = CallbackSubscriber()
        bus.subscribe("a", sub1)
        bus.subscribe("b", sub2)
        bus.subscribe("b", sub1)

        assert bus.topic_count == 2
        assert bus.subscriber_count == 3

    @pytest.mark.asyncio
    async def test_error_handling(self):
        bus = StreamBus()
        errors = []

        def error_handler(e):
            errors.append(e)

        def bad_handler(m):
            raise ValueError("intentional error")

        sub = CallbackSubscriber(on_message_fn=bad_handler, on_error_fn=error_handler)
        bus.subscribe("test", sub)

        await bus.publish(StreamMessage(topic="test"))
        assert len(errors) == 1

    @pytest.mark.asyncio
    async def test_no_match_delivers_zero(self):
        bus = StreamBus()
        sub = CallbackSubscriber()
        bus.subscribe("specific_topic", sub)

        delivered = await bus.publish(StreamMessage(topic="other_topic"))
        assert delivered == 0
