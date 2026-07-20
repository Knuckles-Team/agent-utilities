"""Transactional AgentBus inbox/outbox and privacy contracts."""

from __future__ import annotations

import inspect

from agent_utilities.messaging import bus_inbox
from agent_utilities.messaging.bus import AgentBus
from agent_utilities.messaging.bus_privacy import bus_reference


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
    assert AgentBus._depth_from_stats(
        {"backend": "engine", "queues": {"opaque-a": 2, "opaque-b": 3}}
    ) == 5


def test_removed_manual_ack_and_late_history_api_are_absent() -> None:
    assert "replay" not in inspect.signature(AgentBus.subscribe).parameters
    assert not hasattr(AgentBus, "ack")
    assert not hasattr(AgentBus, "prune_topic_log")
