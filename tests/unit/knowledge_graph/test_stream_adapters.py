"""Concrete stream adapter tests (CONCEPT:KG-2.6).

Fully offline: a fake aiokafka consumer / NATS subscription is injected so no
broker or optional dependency is required.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

from agent_utilities.knowledge_graph.streams import make_stream_adapter
from agent_utilities.knowledge_graph.streams.kafka_adapter import KafkaStreamAdapter
from agent_utilities.knowledge_graph.streams.nats_adapter import NatsStreamAdapter


class FakeKafkaConsumer:
    def __init__(self, records):
        self._records = records

    async def getmany(self, timeout_ms=0, max_records=100):
        return {"topic-0": self._records}


class FakeNatsMsg:
    def __init__(self, data):
        self.data = data
        self.acked = False

    async def ack(self):
        self.acked = True


class FakeNatsSub:
    def __init__(self, msgs):
        self._msgs = msgs

    async def fetch(self, n, timeout=1):
        return self._msgs


def _cfg(**kw):
    base = dict(
        stream_id="s1",
        name="company-brain",
        source_type="kafka",
        endpoint="localhost:9092",
    )
    base.update(kw)
    return SimpleNamespace(**base)


def test_kafka_adapter_consumes_injected_records():
    recs = [
        SimpleNamespace(
            value=json.dumps({"event_id": "e1", "payload": {"x": 1}}).encode(), offset=0
        ),
        SimpleNamespace(
            value=json.dumps(
                {"event_id": "e2", "event_type": "ticket", "payload": {"y": 2}}
            ).encode(),
            offset=1,
        ),
    ]
    adapter = KafkaStreamAdapter(_cfg(), consumer=FakeKafkaConsumer(recs))
    batch = asyncio.run(adapter.consume_batch(batch_size=10))
    assert [e["event_id"] for e in batch.events] == ["e1", "e2"]
    assert batch.events[1]["event_type"] == "ticket"
    assert batch.events[0]["payload"] == {"x": 1}


def test_kafka_adapter_requires_connection():
    adapter = KafkaStreamAdapter(_cfg())  # no consumer, not connected
    try:
        asyncio.run(adapter.consume_batch())
        raise AssertionError("expected RuntimeError")
    except RuntimeError:
        pass


def test_nats_adapter_consumes_and_acks():
    msgs = [FakeNatsMsg(json.dumps({"event_id": "n1", "payload": {"a": 1}}).encode())]
    adapter = NatsStreamAdapter(
        _cfg(source_type="nats"), subscription=FakeNatsSub(msgs)
    )
    batch = asyncio.run(adapter.consume_batch(batch_size=5))
    assert batch.events[0]["event_id"] == "n1"
    assert msgs[0].acked  # message was acknowledged


def test_make_stream_adapter_selects_by_source_type():
    assert isinstance(
        make_stream_adapter(_cfg(source_type="kafka")), KafkaStreamAdapter
    )
    assert isinstance(make_stream_adapter(_cfg(source_type="nats")), NatsStreamAdapter)


def test_decode_tolerates_non_json():
    adapter = KafkaStreamAdapter(
        _cfg(),
        consumer=FakeKafkaConsumer([SimpleNamespace(value=b"not json", offset=0)]),
    )
    batch = asyncio.run(adapter.consume_batch())
    assert batch.events[0]["payload"] == {"raw": "not json"}
