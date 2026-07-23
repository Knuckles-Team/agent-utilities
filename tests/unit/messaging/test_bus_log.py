"""Current bounded AgentBus partition-log contract.

Exercises the two log backends (engine-native broker, Kafka) directly against
small in-memory fakes — no live broker / no ``confluent_kafka`` install needed
— plus the ``resolve_bus_log_backend`` selection logic that picks between
them (and the tenant-scoped keying both backends share).
"""

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
    KafkaBusLog,
    bus_partition_key,
    current_bus_tenant,
    resolve_bus_log_backend,
)

# ── tenant-qualified keying ───────────────────────────────────────────────────


def test_current_bus_tenant_defaults_to_default(monkeypatch):
    """Outside any bound actor, the bus falls back to the literal tenant
    "default". This suite's autouse ``isolate_graph_compute_engine`` fixture
    (tests/conftest.py) binds an ambient test actor for every test, so
    "no actor bound" must be simulated explicitly here rather than relied on
    ambiently."""
    import agent_utilities.security.brain_context as brain_context

    def _no_actor() -> None:
        raise brain_context.IdentityRequiredError("no actor bound")

    monkeypatch.setattr(brain_context, "current_actor", _no_actor)
    assert current_bus_tenant() == "default"


def test_current_bus_tenant_scoped_to_actor():
    from agent_utilities.models.company_brain import ActorType
    from agent_utilities.security.brain_context import ActorContext, use_actor

    actor = ActorContext("u1", ActorType.HUMAN, tenant_id="acme")
    with use_actor(actor):
        assert current_bus_tenant() == "acme"


# ══════════════════════════════════════════════════════════════════════════
# Engine-native broker (lease-based delivery: consume/ack_tag/nack_tag)
# ══════════════════════════════════════════════════════════════════════════
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

    def ack_tag(self, delivery_tag, *, consumer=None):
        del consumer
        return self.inflight.pop(delivery_tag, None) is not None

    def nack_tag(self, delivery_tag, *, consumer=None, requeue=True, now_ms=None):
        del consumer, now_ms
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
        "ack_tag": ("self", "delivery_tag", "consumer"),
        "nack_tag": ("self", "delivery_tag", "consumer", "requeue", "now_ms"),
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


# ══════════════════════════════════════════════════════════════════════════
# Kafka fallback
# ══════════════════════════════════════════════════════════════════════════
class _FakeFuture:
    def result(self, timeout=None):
        return None


class _FakeAdmin:
    """Pre-seeded so ``_ensure_topics`` never hits the real confluent_kafka.admin import."""

    def __init__(self, topics: dict[str, int]):
        self.topics = dict(topics)

    def list_topics(self, topic=None, timeout=None):
        metas = {
            n: SimpleNamespace(partitions={i: object() for i in range(p)})
            for n, p in self.topics.items()
        }
        if topic is not None:
            metas = {k: v for k, v in metas.items() if k == topic}
        return SimpleNamespace(topics=metas)

    def create_topics(self, new_topics):
        return {nt.topic: _FakeFuture() for nt in new_topics}

    def create_partitions(self, new_partitions):
        return {np_.topic: _FakeFuture() for np_ in new_partitions}


def _fully_provisioned_admin() -> _FakeAdmin:
    from agent_utilities.messaging.bus_log import DIRECT_TOPIC, DLQ_TOPIC, TOPIC_TOPIC

    return _FakeAdmin({DIRECT_TOPIC: 6, TOPIC_TOPIC: 6, DLQ_TOPIC: 1})


class _FakeProducer:
    def __init__(self, cluster: dict[str, list[tuple[bytes, bytes]]]):
        self.cluster = cluster

    def produce(self, topic, value=None, key=None):
        self.cluster.setdefault(topic, []).append((key, value))

    def flush(self, timeout=None):
        return 0


class _FakeKafkaMsg:
    def __init__(self, key, value):
        self._key, self._value = key, value

    def value(self):
        return self._value

    def key(self):
        return self._key

    def error(self):
        return None


class _FakeKafkaConsumer:
    """One independent read-position over a shared in-memory topic (one consumer group)."""

    def __init__(self, cluster, topic, *, start_pos: int):
        self.cluster = cluster
        self.topic = topic
        self.pos = start_pos
        self.commits = 0
        self.closed = False

    def poll(self, timeout=None):
        records = self.cluster.get(self.topic, [])
        if self.pos >= len(records):
            return None
        key, value = records[self.pos]
        self.pos += 1
        return _FakeKafkaMsg(key, value)

    def commit(self, message=None, asynchronous=False):
        self.commits += 1

    def close(self):
        self.closed = True


def _kafka_backend(cluster: dict[str, list[tuple[bytes, bytes]]], **kw) -> KafkaBusLog:
    def factory(*, topic, group, seed_ts=None, default_offset="latest"):
        records = cluster.setdefault(topic, [])
        if seed_ts is not None or default_offset == "earliest":
            start_pos = 0
        else:
            start_pos = len(records)  # "latest": only messages produced from now on
        return _FakeKafkaConsumer(cluster, topic, start_pos=start_pos)

    return KafkaBusLog(
        bootstrap_servers="broker.test:9092",
        producer=_FakeProducer(cluster),
        admin_client=_fully_provisioned_admin(),
        consumer_factory=factory,
        **kw,
    )


def test_kafka_bus_log_provisions_topics_idempotently():
    cluster: dict = {}
    _kafka_backend(cluster)  # constructing must not raise with pre-seeded topics


def test_kafka_bus_direct_delivery_via_offsets_multiple_messages():
    cluster: dict = {}
    backend_ = _kafka_backend(cluster)
    now = time.time()
    assert backend_.publish_direct(
        tenant="acme",
        group="g1",
        sender="a",
        to="b",
        payload="hi",
        meta_json="{}",
        created=now,
    )
    assert backend_.publish_direct(
        tenant="acme",
        group="g2",
        sender="a",
        to="b",
        payload="again",
        meta_json="{}",
        created=now + 1,
    )
    got = backend_.receive(tenant="acme", agent_id="b", topics=[], max_messages=10)
    assert [m["payload"] for m in got] == ["hi", "again"]
    # Delivered via the consumer's own committed offset — a second receive is empty.
    assert (
        backend_.receive(tenant="acme", agent_id="b", topics=[], max_messages=10) == []
    )


def test_kafka_bus_topic_multiple_subscribers_each_own_committed_offset():
    """N subscribers each get their OWN consumer/offset — one publish, every subscriber
    reads the full message exactly once via its own group, no shared cursor."""
    cluster: dict = {}
    backend_ = _kafka_backend(cluster)
    backend_.bind_subscriber(tenant="acme", agent_id="sub1", topic="news")
    backend_.bind_subscriber(tenant="acme", agent_id="sub2", topic="news")
    backend_.publish_topic(
        tenant="acme",
        group="g",
        sender="pub",
        topic="news",
        payload="breaking",
        meta_json="{}",
        created=time.time(),
    )
    got1 = backend_.receive(
        tenant="acme", agent_id="sub1", topics=["news"], max_messages=10
    )
    got2 = backend_.receive(
        tenant="acme", agent_id="sub2", topics=["news"], max_messages=10
    )
    assert [m["payload"] for m in got1] == ["breaking"]
    assert [m["payload"] for m in got2] == ["breaking"]
    # Each subscriber's own offset has advanced — a second receive is empty for both.
    assert (
        backend_.receive(
            tenant="acme", agent_id="sub1", topics=["news"], max_messages=10
        )
        == []
    )
    assert (
        backend_.receive(
            tenant="acme", agent_id="sub2", topics=["news"], max_messages=10
        )
        == []
    )


def test_kafka_bus_new_topic_subscriber_default_gets_only_future_messages():
    """No history dump by default (mirrors the graph model): a message published BEFORE a
    subscriber binds is not replayed unless ``from_ts`` seeds a recent window."""
    cluster: dict = {}
    backend_ = _kafka_backend(cluster)
    backend_.publish_topic(
        tenant="acme",
        group="g",
        sender="pub",
        topic="news",
        payload="before",
        meta_json="{}",
        created=time.time(),
    )
    backend_.bind_subscriber(tenant="acme", agent_id="late", topic="news")
    assert (
        backend_.receive(
            tenant="acme", agent_id="late", topics=["news"], max_messages=10
        )
        == []
    )
    backend_.publish_topic(
        tenant="acme",
        group="g2",
        sender="pub",
        topic="news",
        payload="after",
        meta_json="{}",
        created=time.time(),
    )
    got = backend_.receive(
        tenant="acme", agent_id="late", topics=["news"], max_messages=10
    )
    assert [m["payload"] for m in got] == ["after"]


def test_kafka_bus_late_subscriber_replay_recent_via_seek():
    """``replay_recent`` (``from_ts`` in the past) backfills messages already in the log —
    the log-backed equivalent of the graph model's cursor baseline."""
    cluster: dict = {}
    backend_ = _kafka_backend(cluster)
    backend_.publish_topic(
        tenant="acme",
        group="g",
        sender="pub",
        topic="news",
        payload="missed-it",
        meta_json="{}",
        created=time.time(),
    )
    backend_.bind_subscriber(
        tenant="acme", agent_id="late", topic="news", from_ts=time.time() - 3600.0
    )
    got = backend_.receive(
        tenant="acme", agent_id="late", topics=["news"], max_messages=10
    )
    assert [m["payload"] for m in got] == ["missed-it"]


def test_kafka_bus_dlq_on_poison_message():
    cluster: dict = {}
    backend_ = _kafka_backend(cluster)
    from agent_utilities.messaging.bus_log import DIRECT_TOPIC

    # Directly inject a poison (non-JSON) record, bypassing publish_direct.
    cluster.setdefault(DIRECT_TOPIC, []).append((b"acme:vic", b"not-json{{{"))
    got = backend_.receive(tenant="acme", agent_id="vic", topics=[], max_messages=10)
    assert got == []
    dlq = backend_.read_dlq(tenant="acme")
    assert len(dlq) == 1
    assert dlq[0]["error"] == "decode_error"


def test_kafka_bus_unreachable_admin_fails_loud_when_explicit():
    class _DeadAdmin:
        def list_topics(self, topic=None, timeout=None):
            raise ConnectionError("broker unreachable")

    with pytest.raises(BusLogUnavailable, match="broker.test:9092"):
        KafkaBusLog(
            bootstrap_servers="broker.test:9092",
            admin_client=_DeadAdmin(),
            fail_loud=True,
        )


# ══════════════════════════════════════════════════════════════════════════
# resolve_bus_log_backend — auto never probes the network unless signaled
# ══════════════════════════════════════════════════════════════════════════
def _cfg(**overrides):
    base = {
        "agent_bus_log_backend": None,
        "engine_endpoint": None,
        "task_queue_backend": None,
        "kafka_bootstrap_servers": None,
        "agent_bus_partitions": 6,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def test_resolve_auto_with_nothing_configured_returns_none() -> None:
    assert resolve_bus_log_backend(config=_cfg()) is None


def test_resolver_rejects_removed_graph_backend() -> None:
    config = SimpleNamespace(agent_bus_log_backend="graph")
    with pytest.raises(ValueError, match="AGENT_BUS_LOG_BACKEND"):
        resolve_bus_log_backend(config=config)


def test_resolve_rejects_unknown_value():
    with pytest.raises(ValueError, match="AGENT_BUS_LOG_BACKEND"):
        resolve_bus_log_backend(config=_cfg(agent_bus_log_backend="rabbitmq"))


def test_resolve_auto_prefers_kafka_when_bootstrap_configured():
    """Auto mode never raises: an unreachable Kafka broker degrades to no
    configured backend (``None``), same contract as ``TASK_QUEUE_BACKEND``'s
    auto mode."""
    result = resolve_bus_log_backend(
        engine=SimpleNamespace(),
        config=_cfg(kafka_bootstrap_servers="nowhere.invalid:9092"),
    )
    assert result is None or result.name == "kafka"


def test_resolve_auto_kafka_construction_succeeds_uses_kafka(monkeypatch):
    """When Kafka construction succeeds (broker reachable / provisioned), auto mode
    picks it over no backend at all."""
    from agent_utilities.messaging import bus_log as bus_log_mod

    sentinel = object()
    monkeypatch.setattr(bus_log_mod, "KafkaBusLog", lambda **kw: sentinel)
    result = resolve_bus_log_backend(
        config=_cfg(kafka_bootstrap_servers="broker.test:9092")
    )
    assert result is sentinel


def test_resolve_explicit_kafka_unreachable_raises():
    with pytest.raises(BusLogUnavailable):
        resolve_bus_log_backend(
            config=_cfg(
                agent_bus_log_backend="kafka",
                kafka_bootstrap_servers="nowhere.invalid:9092",
            )
        )


def test_resolve_engine_broker_present_on_bound_engine_wins():
    """The direct test seam: an engine object carrying ``.broker`` is used without any
    separate MCP-tool client connection."""
    fake_engine = SimpleNamespace(broker=FakeBroker())
    result = resolve_bus_log_backend(
        engine=fake_engine, config=_cfg(agent_bus_log_backend="engine")
    )
    assert isinstance(result, EngineBrokerBusLog)
    assert result.name == "engine"


def test_resolve_auto_with_engine_endpoint_signal_uses_bound_engine_broker():
    fake_engine = SimpleNamespace(broker=FakeBroker())
    result = resolve_bus_log_backend(
        engine=fake_engine, config=_cfg(engine_endpoint="tcp://engine:9999")
    )
    assert isinstance(result, EngineBrokerBusLog)


def test_resolve_explicit_engine_unreachable_raises(monkeypatch):
    """An explicit ``engine`` selection is a hard contract — an unreachable engine
    client raises, never a silent degrade (mirrors ``TASK_QUEUE_BACKEND=kafka``)."""
    from agent_utilities.mcp.tools import engine_tools

    def _boom(graph):
        raise ConnectionError("engine down")

    monkeypatch.setattr(engine_tools, "_client_for", _boom)
    with pytest.raises(BusLogUnavailable):
        resolve_bus_log_backend(
            engine=SimpleNamespace(),  # no .broker attribute of its own
            config=_cfg(agent_bus_log_backend="engine"),
        )


def test_resolve_explicit_engine_client_without_broker_surface_raises(monkeypatch):
    """The connected engine build has no broker surface at all — also a hard failure
    for an EXPLICIT ``engine`` selection."""
    from agent_utilities.mcp.tools import engine_tools

    monkeypatch.setattr(
        engine_tools,
        "_client_for",
        lambda graph: SimpleNamespace(),  # no .broker
    )
    with pytest.raises(BusLogUnavailable):
        resolve_bus_log_backend(
            engine=SimpleNamespace(),
            config=_cfg(agent_bus_log_backend="engine"),
        )


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
