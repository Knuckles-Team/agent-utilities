from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from agent_utilities.messaging import bus_inbox


@pytest.fixture(autouse=True)
def _verified_session_scope(monkeypatch):
    monkeypatch.setattr(
        bus_inbox,
        "_session_graph_and_tenant",
        lambda tenant: ("tenant_graph", tenant),
    )


class _TxnSurface:
    def __init__(self, *, commit=True):
        self.commit_result = commit
        self.begun: list[str] = []
        self.nodes: list[tuple[str, str, dict]] = []

    def begin(self, *, graph):
        self.begun.append(graph)
        return "txn-1"

    def add_node(self, txn, node_id, properties):
        assert txn == "txn-1"
        self.nodes.append((txn, node_id, dict(properties)))

    def commit(self, txn):
        assert txn == "txn-1"
        return self.commit_result


def _message(**overrides):
    value = {
        "id": "opaque-message-id",
        "msg_group": "opaque-message-group",
        "sender": "person@example.test",
        "recipient": "worker-a",
        "payload": "review /home/local-account/private.txt",
        "meta": {"token": "secret-value", "priority": 1},
        "created": 123.0,
    }
    value.update(overrides)
    return value


def test_native_transaction_commits_inbox_work_outcome_and_outbox(monkeypatch):
    txn = _TxnSurface()
    engine = SimpleNamespace(_client=SimpleNamespace(txn=txn))
    monkeypatch.setattr(bus_inbox, "_already_committed", lambda *_args: False)

    result = bus_inbox.commit_message_to_work_item(
        engine, _message(), tenant="tenant-a", recipient="worker-a", now=200.0
    )

    assert len(txn.begun) == 1
    assert len(txn.nodes) == 4
    assert {properties.get("node_type") for _, _, properties in txn.nodes} == {
        "BusInbox",
        "BusDeliveryOutcome",
        "MutationOutbox",
        "WorkItem",
    }
    serialized = json.dumps(txn.nodes, default=str)
    assert "person@example.test" not in serialized
    assert "/home/local-account" not in serialized
    assert "secret-value" not in serialized
    tenant_authority = [
        properties
        for _, _, properties in txn.nodes
        if properties.get("tenant") == "tenant-a"
    ]
    assert [properties["node_type"] for properties in tenant_authority] == [
        "WorkItem"
    ]
    assert result.work_item_id.startswith("workitem:bus:")


def test_native_transaction_failure_is_not_acknowledgeable(monkeypatch):
    txn = _TxnSurface(commit=False)
    engine = SimpleNamespace(_client=SimpleNamespace(txn=txn))
    monkeypatch.setattr(bus_inbox, "_already_committed", lambda *_args: False)

    with pytest.raises(RuntimeError, match="did not commit"):
        bus_inbox.commit_message_to_work_item(
            engine, _message(), tenant="tenant-a", recipient="worker-a"
        )


def test_replay_returns_same_ids_without_second_transaction(monkeypatch):
    txn = _TxnSurface()
    engine = SimpleNamespace(_client=SimpleNamespace(txn=txn))
    monkeypatch.setattr(bus_inbox, "_already_committed", lambda *_args: True)

    result = bus_inbox.commit_message_to_work_item(
        engine, _message(), tenant="tenant-a", recipient="worker-a", now=200.0
    )

    assert result.replay is True
    assert txn.nodes == []


def test_idempotency_authority_read_failure_fails_closed(monkeypatch):
    txn = _TxnSurface()
    engine = SimpleNamespace(_client=SimpleNamespace(txn=txn))

    def _failed_read(*_args):
        raise ConnectionError("endpoint details must not escape")

    monkeypatch.setattr(
        "agent_utilities.orchestration.work_item.get_work_item", _failed_read
    )
    with pytest.raises(RuntimeError, match="idempotency read failed") as exc_info:
        bus_inbox.commit_message_to_work_item(
            engine, _message(), tenant="tenant-a", recipient="worker-a"
        )
    assert "endpoint details" not in str(exc_info.value)
    assert txn.nodes == []
