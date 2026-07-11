"""Partitioned-log AgentBus delivery/wakeup plane (CONCEPT:AU-ECO.bus.partitioned-log-delivery, AU-P1-2).

Exercises the two log backends (engine-native broker, Kafka) directly against small
in-memory fakes — no live broker / no ``confluent_kafka`` install needed, same DI
pattern as ``kafka_queue_backend.py``'s own tests — plus the ``resolve_bus_log_backend``
selection logic and the ``AgentBus`` wiring that makes the log the hot path while the
semantic registry (roster/subscriptions) stays in the graph.
"""

from __future__ import annotations

import time
from collections import defaultdict
from types import SimpleNamespace

import pytest

from agent_utilities.messaging.bus import AgentBus
from agent_utilities.messaging.bus_log import (
    BUS_LOG_BACKENDS,
    BusLogUnavailable,
    EngineBrokerBusLog,
    KafkaBusLog,
    bus_partition_key,
    current_bus_tenant,
    resolve_bus_log_backend,
)
from tests.unit.messaging.test_bus import _FakeGraph

# ── tenant-qualified keying ───────────────────────────────────────────────────


def test_current_bus_tenant_defaults_to_default():
    assert current_bus_tenant() == "default"


def test_current_bus_tenant_scoped_to_actor():
    from agent_utilities.models.company_brain import ActorType
    from agent_utilities.security.brain_context import ActorContext, use_actor

    actor = ActorContext("u1", ActorType.HUMAN, tenant_id="acme")
    with use_actor(actor):
        assert current_bus_tenant() == "acme"


def test_bus_partition_key_is_tenant_qualified():
    assert bus_partition_key("acme", "bob") == "acme:bob"
    assert bus_partition_key("", "bob") == "default:bob"


# ══════════════════════════════════════════════════════════════════════════
# Engine-native broker (AMQP-style exchanges/queues)
# ══════════════════════════════════════════════════════════════════════════
class _FakeAmqpBroker:
    """Minimal in-memory AMQP-ish broker: exchanges route to bound queues.

    Real enough to exercise ``EngineBrokerBusLog``'s exchange/queue/bind/publish/consume
    calls end-to-end: a ``direct`` exchange routes by routing_key; a ``fanout`` exchange
    reaches every bound queue from ONE publish call (the broker owns fan-out, not the
    caller).
    """

    def __init__(self) -> None:
        self.exchanges: dict[str, dict] = {}
        self.queues: dict[str, list[str]] = defaultdict(list)
        self.publish_calls: list[tuple[str, str]] = []

    def declare_exchange(self, *, exchange, exchange_type="direct"):
        self.exchanges.setdefault(
            exchange, {"type": exchange_type, "bindings": defaultdict(list)}
        )

    def declare_queue(self, *, queue, durable=True):
        self.queues.setdefault(queue, [])

    def bind(self, *, queue, exchange, routing_key=""):
        exch = self.exchanges[exchange]
        key = routing_key if exch["type"] == "direct" else "*"
        if queue not in exch["bindings"][key]:
            exch["bindings"][key].append(queue)

    def publish(self, *, exchange, routing_key="", payload=""):
        self.publish_calls.append((exchange, routing_key))
        if exchange == "":
            # Default-exchange semantics (used by the DLQ path): routing_key IS the queue.
            self.queues.setdefault(routing_key, []).append(payload)
            return
        exch = self.exchanges.get(exchange)
        if exch is None:
            raise RuntimeError(f"no such exchange {exchange!r}")
        if exch["type"] == "fanout":
            targets: set[str] = set()
            for qs in exch["bindings"].values():
                targets.update(qs)
        else:
            targets = set(exch["bindings"].get(routing_key, []))
        for q in targets:
            self.queues.setdefault(q, []).append(payload)

    def consume(self, *, queue, max_messages=200, ack=True):
        msgs = self.queues.get(queue, [])
        taken = msgs[:max_messages]
        del msgs[: len(taken)]
        return taken

    def stats(self):
        return {"queues": {q: len(v) for q, v in self.queues.items()}}


def _engine_backend() -> tuple[EngineBrokerBusLog, _FakeAmqpBroker]:
    broker = _FakeAmqpBroker()
    client = SimpleNamespace(broker=broker)
    return EngineBrokerBusLog(client), broker


def test_engine_broker_direct_delivery_via_offsets_not_graph_scan():
    """Two direct messages are delivered once each; a second receive is empty — the
    broker's own consumed queue position is the cursor, never a graph MATCH."""
    backend, broker = _engine_backend()
    now = time.time()
    assert backend.publish_direct(
        tenant="acme",
        group="g1",
        sender="a",
        to="b",
        payload="hi",
        meta_json="{}",
        created=now,
    )
    assert backend.publish_direct(
        tenant="acme",
        group="g2",
        sender="a",
        to="b",
        payload="again",
        meta_json="{}",
        created=now + 1,
    )
    got = backend.receive(tenant="acme", agent_id="b", topics=[], max_messages=10)
    assert [m["payload"] for m in got] == ["hi", "again"]
    # Delivered via consumed offsets: the queue is drained, a second receive is empty.
    assert (
        backend.receive(tenant="acme", agent_id="b", topics=[], max_messages=10) == []
    )
    # Tenant-qualified naming, not a bare agent id.
    assert "bus.direct.acme" in broker.exchanges
    assert "bus.inbox.acme.b" in broker.queues


def test_engine_broker_direct_is_tenant_isolated():
    backend, broker = _engine_backend()
    now = time.time()
    backend.publish_direct(
        tenant="acme",
        group="g1",
        sender="a",
        to="b",
        payload="acme-msg",
        meta_json="{}",
        created=now,
    )
    backend.publish_direct(
        tenant="globex",
        group="g2",
        sender="a",
        to="b",
        payload="globex-msg",
        meta_json="{}",
        created=now,
    )
    got_acme = backend.receive(tenant="acme", agent_id="b", topics=[], max_messages=10)
    got_globex = backend.receive(
        tenant="globex", agent_id="b", topics=[], max_messages=10
    )
    assert [m["payload"] for m in got_acme] == ["acme-msg"]
    assert [m["payload"] for m in got_globex] == ["globex-msg"]


def test_engine_broker_topic_fanout_multiple_subscribers():
    """ONE publish reaches every bound subscriber queue — the broker fans out, never a
    per-recipient application write."""
    backend, broker = _engine_backend()
    backend.bind_subscriber(tenant="acme", agent_id="sub1", topic="news")
    backend.bind_subscriber(tenant="acme", agent_id="sub2", topic="news")
    publishes_before = len(broker.publish_calls)
    ok = backend.publish_topic(
        tenant="acme",
        group="g",
        sender="pub",
        topic="news",
        payload="breaking",
        meta_json="{}",
        created=time.time(),
    )
    assert ok
    # ONE publish call for N subscribers — not O(subscribers) writes.
    assert len(broker.publish_calls) == publishes_before + 1

    got1 = backend.receive(
        tenant="acme", agent_id="sub1", topics=["news"], max_messages=10
    )
    got2 = backend.receive(
        tenant="acme", agent_id="sub2", topics=["news"], max_messages=10
    )
    assert [m["payload"] for m in got1] == ["breaking"]
    assert [m["payload"] for m in got2] == ["breaking"]
    # Each subscriber's queue is now drained (its own offset/position advanced).
    assert (
        backend.receive(
            tenant="acme", agent_id="sub1", topics=["news"], max_messages=10
        )
        == []
    )


def test_engine_broker_publisher_excluded_from_own_topic_broadcast():
    backend, _ = _engine_backend()
    backend.bind_subscriber(tenant="acme", agent_id="pub", topic="t")
    backend.publish_topic(
        tenant="acme",
        group="g",
        sender="pub",
        topic="t",
        payload="x",
        meta_json="{}",
        created=time.time(),
    )
    assert (
        backend.receive(tenant="acme", agent_id="pub", topics=["t"], max_messages=10)
        == []
    )


def test_engine_broker_dlq_on_poison_message():
    backend, broker = _engine_backend()
    _, inbox = backend._direct_inbox("acme", "vic")
    broker.queues[inbox].append("not-json{{{")  # poison: not decodable as an envelope
    got = backend.receive(tenant="acme", agent_id="vic", topics=[], max_messages=10)
    assert got == []
    dlq = backend.read_dlq(tenant="acme")
    assert len(dlq) == 1
    assert dlq[0]["error"] == "decode_error"
    assert "not-json" in dlq[0]["raw"]


def test_engine_broker_stats_reports_backend_name():
    backend, _ = _engine_backend()
    st = backend.stats()
    assert st["backend"] == "engine"


def test_engine_broker_publish_failure_returns_false_not_raise():
    """A broker that cannot publish degrades to ``False``, never a crash on the hot path."""

    class _Boom:
        def declare_exchange(self, **kw):
            pass

        def declare_queue(self, **kw):
            pass

        def bind(self, **kw):
            pass

        def publish(self, **kw):
            raise ConnectionError("broker down")

    backend = EngineBrokerBusLog(SimpleNamespace(broker=_Boom()))
    ok = backend.publish_direct(
        tenant="acme",
        group="g",
        sender="a",
        to="b",
        payload="x",
        meta_json="{}",
        created=time.time(),
    )
    assert ok is False


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
    backend = _kafka_backend(cluster)
    now = time.time()
    assert backend.publish_direct(
        tenant="acme",
        group="g1",
        sender="a",
        to="b",
        payload="hi",
        meta_json="{}",
        created=now,
    )
    assert backend.publish_direct(
        tenant="acme",
        group="g2",
        sender="a",
        to="b",
        payload="again",
        meta_json="{}",
        created=now + 1,
    )
    got = backend.receive(tenant="acme", agent_id="b", topics=[], max_messages=10)
    assert [m["payload"] for m in got] == ["hi", "again"]
    # Delivered via the consumer's own committed offset — a second receive is empty.
    assert (
        backend.receive(tenant="acme", agent_id="b", topics=[], max_messages=10) == []
    )


def test_kafka_bus_tenant_qualified_partition_key():
    cluster: dict = {}
    backend = _kafka_backend(cluster)
    backend.publish_direct(
        tenant="acme",
        group="g",
        sender="a",
        to="bob",
        payload="x",
        meta_json="{}",
        created=time.time(),
    )
    from agent_utilities.messaging.bus_log import DIRECT_TOPIC

    key, _value = cluster[DIRECT_TOPIC][0]
    assert key == b"acme:bob"


def test_kafka_bus_topic_multiple_subscribers_each_own_committed_offset():
    """N subscribers each get their OWN consumer/offset — one publish, every subscriber
    reads the full message exactly once via its own group, no shared cursor."""
    cluster: dict = {}
    backend = _kafka_backend(cluster)
    backend.bind_subscriber(tenant="acme", agent_id="sub1", topic="news")
    backend.bind_subscriber(tenant="acme", agent_id="sub2", topic="news")
    backend.publish_topic(
        tenant="acme",
        group="g",
        sender="pub",
        topic="news",
        payload="breaking",
        meta_json="{}",
        created=time.time(),
    )
    got1 = backend.receive(
        tenant="acme", agent_id="sub1", topics=["news"], max_messages=10
    )
    got2 = backend.receive(
        tenant="acme", agent_id="sub2", topics=["news"], max_messages=10
    )
    assert [m["payload"] for m in got1] == ["breaking"]
    assert [m["payload"] for m in got2] == ["breaking"]
    # Each subscriber's own offset has advanced — a second receive is empty for both.
    assert (
        backend.receive(
            tenant="acme", agent_id="sub1", topics=["news"], max_messages=10
        )
        == []
    )
    assert (
        backend.receive(
            tenant="acme", agent_id="sub2", topics=["news"], max_messages=10
        )
        == []
    )


def test_kafka_bus_new_topic_subscriber_default_gets_only_future_messages():
    """No history dump by default (mirrors the graph model): a message published BEFORE a
    subscriber binds is not replayed unless ``from_ts`` seeds a recent window."""
    cluster: dict = {}
    backend = _kafka_backend(cluster)
    backend.publish_topic(
        tenant="acme",
        group="g",
        sender="pub",
        topic="news",
        payload="before",
        meta_json="{}",
        created=time.time(),
    )
    backend.bind_subscriber(tenant="acme", agent_id="late", topic="news")
    assert (
        backend.receive(
            tenant="acme", agent_id="late", topics=["news"], max_messages=10
        )
        == []
    )
    backend.publish_topic(
        tenant="acme",
        group="g2",
        sender="pub",
        topic="news",
        payload="after",
        meta_json="{}",
        created=time.time(),
    )
    got = backend.receive(
        tenant="acme", agent_id="late", topics=["news"], max_messages=10
    )
    assert [m["payload"] for m in got] == ["after"]


def test_kafka_bus_late_subscriber_replay_recent_via_seek():
    """``replay_recent`` (``from_ts`` in the past) backfills messages already in the log —
    the log-backed equivalent of the graph model's cursor baseline."""
    cluster: dict = {}
    backend = _kafka_backend(cluster)
    backend.publish_topic(
        tenant="acme",
        group="g",
        sender="pub",
        topic="news",
        payload="missed-it",
        meta_json="{}",
        created=time.time(),
    )
    backend.bind_subscriber(
        tenant="acme", agent_id="late", topic="news", from_ts=time.time() - 3600.0
    )
    got = backend.receive(
        tenant="acme", agent_id="late", topics=["news"], max_messages=10
    )
    assert [m["payload"] for m in got] == ["missed-it"]


def test_kafka_bus_dlq_on_poison_message():
    cluster: dict = {}
    backend = _kafka_backend(cluster)
    from agent_utilities.messaging.bus_log import DIRECT_TOPIC

    # Directly inject a poison (non-JSON) record, bypassing publish_direct.
    cluster.setdefault(DIRECT_TOPIC, []).append((b"acme:vic", b"not-json{{{"))
    got = backend.receive(tenant="acme", agent_id="vic", topics=[], max_messages=10)
    assert got == []
    dlq = backend.read_dlq(tenant="acme")
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


def test_resolve_auto_with_nothing_configured_returns_none_graph_fallback():
    assert resolve_bus_log_backend(config=_cfg()) is None


def test_resolve_explicit_graph_returns_none():
    assert resolve_bus_log_backend(config=_cfg(agent_bus_log_backend="graph")) is None


def test_resolve_rejects_unknown_value():
    with pytest.raises(ValueError, match="AGENT_BUS_LOG_BACKEND"):
        resolve_bus_log_backend(config=_cfg(agent_bus_log_backend="rabbitmq"))


def test_resolve_auto_prefers_kafka_when_bootstrap_configured():
    """Auto mode never raises: an unreachable Kafka broker degrades to the graph
    fallback (``None``), same contract as ``TASK_QUEUE_BACKEND``'s auto mode."""
    backend = resolve_bus_log_backend(
        engine=SimpleNamespace(),
        config=_cfg(kafka_bootstrap_servers="nowhere.invalid:9092"),
    )
    assert backend is None or backend.name == "kafka"


def test_resolve_auto_kafka_construction_succeeds_uses_kafka(monkeypatch):
    """When Kafka construction succeeds (broker reachable / provisioned), auto mode
    picks it over the graph fallback."""
    from agent_utilities.messaging import bus_log as bus_log_mod

    sentinel = object()
    monkeypatch.setattr(bus_log_mod, "KafkaBusLog", lambda **kw: sentinel)
    backend = resolve_bus_log_backend(
        config=_cfg(kafka_bootstrap_servers="broker.test:9092")
    )
    assert backend is sentinel


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
    fake_engine = SimpleNamespace(broker=_FakeAmqpBroker())
    backend = resolve_bus_log_backend(
        engine=fake_engine, config=_cfg(agent_bus_log_backend="engine")
    )
    assert isinstance(backend, EngineBrokerBusLog)
    assert backend.name == "engine"


def test_resolve_auto_with_engine_endpoint_signal_uses_bound_engine_broker():
    fake_engine = SimpleNamespace(broker=_FakeAmqpBroker())
    backend = resolve_bus_log_backend(
        engine=fake_engine, config=_cfg(engine_endpoint="tcp://engine:9999")
    )
    assert isinstance(backend, EngineBrokerBusLog)


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


def test_bus_log_backends_tuple_is_stable():
    assert BUS_LOG_BACKENDS == ("engine", "kafka", "graph")


# ══════════════════════════════════════════════════════════════════════════
# AgentBus wiring: the log is the hot path, no per-recipient graph node
# ══════════════════════════════════════════════════════════════════════════
@pytest.fixture()
def log_backed_bus():
    AgentBus._instance = None
    fake_graph = _FakeGraph()
    bus = AgentBus(engine=fake_graph)
    engine_backend, broker = _engine_backend()
    bus._log_backend_cache = engine_backend
    return bus, fake_graph, broker


def test_agentbus_send_direct_uses_log_no_graph_message_node(log_backed_bus):
    bus, fake_graph, broker = log_backed_bus
    bus.register("a")
    bus.register("b")
    out = bus.send(sender="a", to="b", payload="hello")
    assert out["ok"] is True and out["delivered"] == ["b"]
    # NO :BusMessage node was written to the graph — the log carried the body.
    assert not any(n.get("type") == "BusMessage" for n in fake_graph.nodes.values())
    got = bus.receive("b")
    assert [m["payload"] for m in got["messages"]] == ["hello"]


def test_agentbus_send_topic_fanout_via_log_no_per_recipient_graph_write(
    log_backed_bus,
):
    bus, fake_graph, broker = log_backed_bus
    for a in ("pub", "sub1", "sub2"):
        bus.register(a)
    bus.subscribe("sub1", "research")
    bus.subscribe("sub2", "research")
    out = bus.send(sender="pub", topic="research", payload="paper dropped")
    assert out["ok"] is True
    assert set(out["delivered"]) == {"sub1", "sub2"}
    assert not any(n.get("type") == "BusMessage" for n in fake_graph.nodes.values())
    assert [m["payload"] for m in bus.receive("sub1")["messages"]] == ["paper dropped"]
    assert [m["payload"] for m in bus.receive("sub2")["messages"]] == ["paper dropped"]


def test_agentbus_ack_is_best_effort_success_when_log_backed(log_backed_bus):
    bus, _fake_graph, _broker = log_backed_bus
    bus.register("a")
    bus.register("b")
    bus.send(sender="a", to="b", payload="hi")
    mid = bus.receive("b")["messages"][0]["id"]
    assert bus.ack("b", mid) is True


def test_agentbus_status_reports_log_backend_name(log_backed_bus):
    bus, _fake_graph, _broker = log_backed_bus
    assert bus.status()["log_backend"] == "engine"


def test_agentbus_status_reports_graph_when_no_backend_configured():
    AgentBus._instance = None
    bus = AgentBus(engine=_FakeGraph())
    bus._log_backend_cache = None
    assert bus.status()["log_backend"] == "graph"


def test_agentbus_late_subscriber_replay_recent_via_kafka_log():
    """Late-subscriber replay via the log's cursor (Kafka: real timestamp seek).

    The engine-broker fixture above cannot exercise this: AMQP-style fanout only reaches
    queues bound at PUBLISH time (there is no time-indexed replay without a queue already
    existing) — a documented follow-up. Kafka's ``offsets_for_times`` seek gives the
    log-backed equivalent of the graph model's per-(agent,topic) cursor baseline.
    """
    AgentBus._instance = None
    bus = AgentBus(engine=_FakeGraph())
    cluster: dict = {}
    bus._log_backend_cache = _kafka_backend(cluster)

    bus.register("pub")
    bus.send(sender="pub", topic="news", payload="breaking")
    bus.register("late")
    bus.subscribe("late", "news", replay_recent=True)
    got = bus.receive("late")
    assert [m["payload"] for m in got["messages"]] == ["breaking"]
