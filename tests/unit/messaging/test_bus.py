"""Transactional AgentBus inbox/outbox and privacy contracts."""

from __future__ import annotations

import inspect
from collections import defaultdict
from types import SimpleNamespace

from agent_utilities.knowledge_graph.backends.epistemic_graph_backend import (
    EpistemicGraphBackend,
)
from agent_utilities.messaging import bus_inbox
from agent_utilities.messaging.bus import AgentBus
from agent_utilities.messaging.bus_privacy import bus_reference


class _FakeBusBroker:
    """Minimal in-memory broker satisfying ``EngineBrokerBusLog``'s contract
    (declare_exchange/declare_queue/bind_queue/publish/consume/ack_tag/nack_tag),
    mirroring ``tests/unit/messaging/test_bus_log.py``'s ``FakeBroker``.
    """

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

    def consume(self, queue, *, group, consumer, now_ms, lease_ms=0, prefetch=0):
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


class _FakeGraph(EpistemicGraphBackend):
    """Real in-memory native-Cypher engine adapted to AgentBus's calling
    conventions — used by tests that drive the full AgentBus/BusFederationRelay
    surface (register/roster/subscribe/send/receive/federation), unlike the
    narrower ``FakeEngine`` below which only covers the inbox/outbox helpers.

    Adds the two seams the live code requires beyond plain add_node/query_cypher:

    - ``.broker`` — makes ``resolve_bus_log_backend`` pick the engine-native
      partitioned-log backend (CONCEPT:AU-ECO.bus.partitioned-log-delivery)
      instead of requiring a live broker connection or Kafka.
    - ``.txn`` — the engine-native transaction surface ``bus_inbox.py``'s
      outbox/inbox commits require (begin/add_node/commit), wired onto the
      same node store the inherited native Cypher engine reads.
    """

    def __init__(self) -> None:
        super().__init__()
        self.broker = _FakeBusBroker()
        self.txn = SimpleNamespace(
            begin=lambda *, graph: {"graph": graph, "nodes": []},
            add_node=self._txn_add_node,
            commit=self._txn_commit,
        )

    def add_node(self, node_id, node_type="", properties=None, **extra):
        merged = dict(properties or {})
        merged.update(extra)
        super().add_node(node_id, node_type, **merged)

    def query_cypher(self, query, params=None):
        return self.execute(query, params or {})

    def _txn_add_node(self, txn, node_id, props):
        txn["nodes"].append((node_id, dict(props)))

    def _txn_commit(self, txn) -> bool:
        for node_id, props in txn["nodes"]:
            self.add_node(node_id, "", **props)
        return True


class FakeTxn:
    def __init__(self, engine: FakeEngine) -> None:
        self.engine = engine

    def begin(self, *, graph):
        return {"graph": graph, "nodes": []}

    def add_node(self, txn, node_id, props):
        txn["nodes"].append((node_id, dict(props)))

    def commit(self, txn):
        for node_id, props in txn["nodes"]:
            self.engine.nodes[node_id] = {"id": node_id, **props}
        self.engine.commits.append([node_id for node_id, _ in txn["nodes"]])
        return True


class FakeEngine:
    def __init__(self) -> None:
        self.nodes: dict[str, dict] = {}
        self.commits: list[list[str]] = []
        self.txn = FakeTxn(self)

    def query_cypher(self, query, params=None):
        params = params or {}
        node_id = params.get("id")
        if "MATCH (o:BusOutbox" in query and node_id:
            node = self.nodes.get(node_id)
            return [{"status": node.get("status")}] if node else []
        if "MATCH (w:WorkItem" in query and node_id:
            node = self.nodes.get(node_id)
            return [dict(node)] if node and node.get("node_type") == "WorkItem" else []
        return []


def message() -> dict:
    return {
        "id": "message-ref",
        "msg_group": "group-ref",
        "sender": "sender-ref",
        "recipient": "recipient-ref",
        "topic": "",
        "payload": "execute the requested work",
        "meta": {"priority": 1},
        "created": 100.0,
    }


def test_inbox_workitem_outcome_and_mutation_outbox_commit_atomically(
    monkeypatch,
) -> None:
    engine = FakeEngine()
    monkeypatch.setattr(
        bus_inbox,
        "_session_graph_and_tenant",
        lambda tenant: ("tenant_graph", tenant),
    )
    committed = bus_inbox.commit_message_to_work_item(
        engine, message(), tenant="tenant", recipient="recipient-ref", now=200.0
    )
    assert len(engine.commits) == 1
    assert set(engine.commits[0]) == {
        committed.inbox_id,
        committed.work_item_id,
        committed.outcome_id,
        committed.outbox_id,
    }
    assert engine.nodes[committed.work_item_id]["status"] == "ready"
    assert engine.nodes[committed.work_item_id]["node_type"] == "WorkItem"


def test_duplicate_delivery_reuses_deterministic_work_item(monkeypatch) -> None:
    engine = FakeEngine()
    monkeypatch.setattr(
        bus_inbox,
        "_session_graph_and_tenant",
        lambda tenant: ("tenant_graph", tenant),
    )
    first = bus_inbox.commit_message_to_work_item(
        engine, message(), tenant="tenant", recipient="recipient-ref"
    )
    second = bus_inbox.commit_message_to_work_item(
        engine, message(), tenant="tenant", recipient="recipient-ref"
    )
    assert first.work_item_id == second.work_item_id
    assert second.replay
    assert len(engine.commits) == 1


def test_send_outbox_has_explicit_pending_published_delivered_states(
    monkeypatch,
) -> None:
    engine = FakeEngine()
    monkeypatch.setattr(
        bus_inbox,
        "_session_graph_and_tenant",
        lambda tenant: ("tenant_graph", tenant),
    )
    pending = bus_inbox.commit_message_outbox(engine, message(), tenant="tenant")
    assert engine.nodes[pending.outbox_id]["status"] == "pending"
    bus_inbox.mark_message_outbox_published(engine, message(), tenant="tenant")
    assert engine.nodes[pending.outbox_id]["status"] == "published"
    bus_inbox.mark_message_outbox_delivered(engine, message(), tenant="tenant")
    assert engine.nodes[pending.outbox_id]["status"] == "delivered"


def test_outbox_authority_read_failure_fails_closed(monkeypatch) -> None:
    engine = FakeEngine()
    monkeypatch.setattr(
        bus_inbox,
        "_session_graph_and_tenant",
        lambda tenant: ("tenant_graph", tenant),
    )

    def _failed_query(*_args, **_kwargs):
        raise ConnectionError("private endpoint")

    monkeypatch.setattr(engine, "query_cypher", _failed_query)
    try:
        bus_inbox.commit_message_outbox(engine, message(), tenant="tenant")
    except RuntimeError as exc:
        assert "authority read failed" in str(exc)
        assert "private endpoint" not in str(exc)
    else:
        raise AssertionError("outbox authority read failure must fail closed")


def test_identifiers_are_opaque_and_idempotent() -> None:
    reference = bus_reference("agent", "person@example.invalid", tenant="tenant")
    assert reference == bus_reference("agent", reference, tenant="tenant")
    assert "person" not in reference and "tenant" not in reference


def test_engine_queue_stats_sum_to_backpressure_depth() -> None:
    assert (
        AgentBus._depth_from_stats(
            {"backend": "engine", "queues": {"opaque-a": 2, "opaque-b": 3}}
        )
        == 5
    )


def test_removed_manual_ack_and_late_history_api_are_absent() -> None:
    assert "replay" not in inspect.signature(AgentBus.subscribe).parameters
    assert not hasattr(AgentBus, "ack")
    assert not hasattr(AgentBus, "prune_topic_log")
