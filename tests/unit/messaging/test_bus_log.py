"""AgentBus delivery through EG streams and durable inbox cursor ordering."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent_utilities.messaging.bus_log import (
    BUS_LOG_BACKENDS,
    BusLogUnavailable,
    EngineStreamBusLog,
    bus_partition_key,
    current_bus_tenant,
    resolve_bus_log_backend,
)
from tests.unit.messaging.test_bus import _FakeBusBroker


def _backend() -> tuple[EngineStreamBusLog, _FakeBusBroker]:
    broker = _FakeBusBroker()
    return EngineStreamBusLog(SimpleNamespace(broker=broker), partitions=4), broker


def _publish(backend: EngineStreamBusLog, tenant: str, group: str) -> None:
    assert backend.publish_direct(
        tenant=tenant,
        group=group,
        sender="sender",
        to="recipient",
        payload="hello",
        meta_json="{}",
        created=1.0,
    )


def test_single_engine_writer_and_tenant_partition_are_opaque() -> None:
    assert BUS_LOG_BACKENDS == ("engine",)
    assert bus_partition_key("tenant-a", "route") != bus_partition_key(
        "tenant-b", "route"
    )
    assert "tenant-a" not in bus_partition_key("tenant-a", "route")


def test_requires_verified_session(monkeypatch) -> None:
    from agent_utilities.knowledge_graph.core import session

    monkeypatch.setattr(session, "current_session", lambda: None)
    with pytest.raises(PermissionError, match="verified tenant"):
        current_bus_tenant()


def test_stream_cursor_advances_only_after_ack() -> None:
    tenant = current_bus_tenant()
    backend, broker = _backend()
    _publish(backend, tenant, "group-1")
    first = backend.receive(tenant=tenant, agent_id="recipient", topics=[])
    assert len(first) == 1
    assert backend.receive(tenant=tenant, agent_id="recipient", topics=[]) == first
    assert broker.cursors == {}
    assert backend.ack(first[0])
    assert backend.receive(tenant=tenant, agent_id="recipient", topics=[]) == []


def test_failed_inbox_commit_replays_same_record() -> None:
    tenant = current_bus_tenant()
    backend, _broker = _backend()
    _publish(backend, tenant, "group-1")
    first = backend.receive(tenant=tenant, agent_id="recipient", topics=[])
    assert backend.nack(first[0], requeue=True)
    assert backend.receive(tenant=tenant, agent_id="recipient", topics=[]) == first


def test_poison_is_digest_only_in_dlq_before_cursor_commit() -> None:
    tenant = current_bus_tenant()
    backend, broker = _backend()
    backend._log.append(tenant, "route", b"secret invalid bytes", now_ms=1)
    assert backend.receive(tenant=tenant, agent_id="recipient", topics=[]) == []
    rows = backend.read_dlq(tenant=tenant)
    assert len(rows) == 1
    assert rows[0]["reason"] == "decode_error"
    assert "secret" not in str(rows)
    assert broker.cursors


def test_reject_cross_tenant_read_and_second_writer() -> None:
    tenant = current_bus_tenant()
    backend, _broker = _backend()
    with pytest.raises(PermissionError, match="tenant differs"):
        backend.receive(tenant=tenant + "-other", agent_id="recipient", topics=[])
    with pytest.raises(BusLogUnavailable, match="requires"):
        resolve_bus_log_backend(
            engine=SimpleNamespace(broker=_FakeBusBroker()),
            config=SimpleNamespace(agent_bus_log_backend="kafka"),
        )


def test_resolver_rejects_broker_without_stream_port() -> None:
    with pytest.raises(BusLogUnavailable, match="stream broker"):
        resolve_bus_log_backend(
            engine=SimpleNamespace(broker=SimpleNamespace()),
            config=SimpleNamespace(
                agent_bus_log_backend="engine", agent_bus_partitions=4
            ),
        )
