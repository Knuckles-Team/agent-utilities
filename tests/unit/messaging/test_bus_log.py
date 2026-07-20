"""Current bounded AgentBus partition-log contract."""

from __future__ import annotations

import inspect
import json
import time
from collections import defaultdict
from types import SimpleNamespace

import pytest

from agent_utilities.messaging.bus_log import (
    BUS_LOG_BACKENDS,
    BusLogUnavailable,
    EngineBrokerBusLog,
    bus_partition_key,
    resolve_bus_log_backend,
)


class FakeBroker:
    def __init__(self) -> None:
        self.exchanges: dict[str, dict[str, list[str]]] = {}
        self.queues: dict[str, list[bytes]] = defaultdict(list)
        self.inflight: dict[int, tuple[str, bytes]] = {}
        self.tag = 0

    def declare_exchange(self, exchange, kind="direct"):
        del kind
        self.exchanges.setdefault(exchange, defaultdict(list))

    def declare_queue(self, queue, **_policy):
        self.queues.setdefault(queue, [])

    def bind_queue(self, exchange, queue, routing_key):
        if queue not in self.exchanges[exchange][routing_key]:
            self.exchanges[exchange][routing_key].append(queue)

    def publish(self, exchange, routing_key, payload):
        delivered = 0
        for queue in self.exchanges[exchange][routing_key]:
            self.queues[queue].append(payload)
            delivered += 1
        return delivered

    def consume(
        self,
        queue,
        *,
        group,
        consumer,
        now_ms,
        lease_ms=0,
        prefetch=0,
    ):
        del group, consumer, now_ms, lease_ms
        if prefetch and len(self.inflight) >= prefetch:
            return None
        if not self.queues[queue]:
            return None
        payload = self.queues[queue].pop(0)
        self.tag += 1
        self.inflight[self.tag] = (queue, payload)
        return f"message:{self.tag}", {
            "payload": payload.hex(),
            "delivery_tag": self.tag,
        }

    def ack_tag(self, delivery_tag):
        return self.inflight.pop(delivery_tag, None) is not None

    def nack_tag(self, delivery_tag, requeue=True, now_ms=None):
        del now_ms
        item = self.inflight.pop(delivery_tag, None)
        if item is None:
            return "absent"
        queue, payload = item
        if requeue:
            self.queues[queue].insert(0, payload)
        return "requeued" if requeue else "discarded"


def backend(partitions: int = 4) -> tuple[EngineBrokerBusLog, FakeBroker]:
    broker = FakeBroker()
    return (
        EngineBrokerBusLog(
            SimpleNamespace(broker=broker),
            partitions=partitions,
            delivery_lease_ms=120_000,
        ),
        broker,
    )


def test_only_current_required_backends_exist() -> None:
    assert BUS_LOG_BACKENDS == ("engine", "kafka")


def test_engine_log_is_pinned_to_the_current_broker_client_contract() -> None:
    from epistemic_graph.client import BrokerClient

    expected = {
        "declare_exchange": ("self", "exchange", "kind"),
        "declare_queue": (
            "self",
            "queue",
            "dl_exchange",
            "dl_routing_key",
            "max_delivery_count",
            "message_ttl_ms",
            "queue_expiry_ms",
            "max_priority",
        ),
        "bind_queue": ("self", "exchange", "queue", "routing_key"),
        "publish": ("self", "exchange", "routing_key", "payload"),
        "consume": (
            "self",
            "queue",
            "group",
            "consumer",
            "now_ms",
            "lease_ms",
            "prefetch",
        ),
        "ack_tag": ("self", "delivery_tag"),
        "nack_tag": ("self", "delivery_tag", "requeue", "now_ms"),
    }
    observed = {
        method: tuple(inspect.signature(getattr(BrokerClient, method)).parameters)
        for method in expected
    }
    assert observed == expected


def test_partition_key_is_opaque_and_tenant_qualified() -> None:
    key_a = bus_partition_key("tenant-a", "recipient")
    key_b = bus_partition_key("tenant-b", "recipient")
    assert key_a != key_b
    assert "tenant-a" not in key_a and "recipient" not in key_a


def test_engine_log_ack_occurs_after_explicit_commit_boundary() -> None:
    log, broker = backend()
    assert log.publish_direct(
        tenant="tenant",
        group="group",
        sender="sender",
        to="recipient",
        payload="payload",
        meta_json="{}",
        created=time.time(),
    )
    messages = log.receive(
        tenant="tenant", agent_id="ignored", topics=[], max_messages=10
    )
    assert [message["payload"] for message in messages] == ["payload"]
    assert broker.inflight
    assert log.ack(messages[0])
    assert not broker.inflight


def test_nack_requeues_for_idempotent_replay() -> None:
    log, _broker = backend()
    log.publish_topic(
        tenant="tenant",
        group="group",
        sender="sender",
        topic="topic",
        payload="payload",
        meta_json=json.dumps({"priority": 1}),
        created=time.time(),
    )
    first = log.receive(tenant="tenant", agent_id="", topics=[], max_messages=10)
    assert log.nack(first[0], requeue=True)
    replay = log.receive(tenant="tenant", agent_id="", topics=[], max_messages=10)
    assert replay[0]["msg_group"] == first[0]["msg_group"]


def test_engine_log_poison_payload_is_reference_only_in_dlq() -> None:
    log, broker = backend(partitions=1)
    _exchange, queue = log._partition_queue("tenant", 0)
    broker.queues[queue].append(b"not-json")

    assert log.receive(tenant="tenant", agent_id="", topics=[], max_messages=1) == []
    assert not broker.inflight
    records = log.read_dlq(tenant="tenant", max_messages=1)
    assert len(records) == 1
    assert set(records[0]) == {
        "error",
        "raw_bytes",
        "raw_sha256",
        "source_queue",
        "ts",
    }


def test_engine_log_fails_on_unacknowledgeable_delivery() -> None:
    log, broker = backend(partitions=1)
    assert log.publish_direct(
        tenant="tenant",
        group="group",
        sender="sender",
        to="recipient",
        payload="payload",
        meta_json="{}",
        created=time.time(),
    )
    consume = broker.consume

    def without_delivery_tag(*args, **kwargs):
        node_id, properties = consume(*args, **kwargs)
        properties.pop("delivery_tag")
        return node_id, properties

    broker.consume = without_delivery_tag  # type: ignore[method-assign]
    with pytest.raises(BusLogUnavailable, match="unacknowledgeable"):
        log.receive(tenant="tenant", agent_id="", topics=[], max_messages=1)


def test_engine_log_rejects_nonfinite_wire_values() -> None:
    log, broker = backend(partitions=1)
    assert not log.publish_direct(
        tenant="tenant",
        group="group",
        sender="sender",
        to="recipient",
        payload="payload",
        meta_json="{}",
        created=float("nan"),
    )
    assert not any(broker.queues.values())


def test_queue_count_is_fixed_by_partitions_not_agents() -> None:
    log, broker = backend(partitions=3)
    log.receive(tenant="tenant", agent_id="a", topics=[], max_messages=3)
    log.receive(tenant="tenant", agent_id="b", topics=["x"], max_messages=3)
    partition_queues = [name for name in broker.queues if "bus.partition." in name]
    assert len(partition_queues) == 3


def test_resolver_rejects_removed_graph_backend() -> None:
    config = SimpleNamespace(agent_bus_log_backend="graph")
    with pytest.raises(ValueError, match="AGENT_BUS_LOG_BACKEND"):
        resolve_bus_log_backend(config=config)


def test_resolver_uses_configured_visibility_lease() -> None:
    broker = FakeBroker()
    config = SimpleNamespace(
        agent_bus_log_backend="engine",
        agent_bus_partitions=2,
        agent_bus_delivery_lease_seconds=90,
    )
    log = resolve_bus_log_backend(engine=SimpleNamespace(broker=broker), config=config)
    assert isinstance(log, EngineBrokerBusLog)
    assert log.delivery_lease_ms == 90_000
