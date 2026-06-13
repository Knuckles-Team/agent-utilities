"""Queryable cross-agent correlation / blast-radius (CONCEPT:OS-5.11).

Correlation ids were emitted into outbound headers but never persisted on the
durable effect nodes, so "which agents touched resource X" was answerable only
from Langfuse / ad-hoc Cypher. These tests cover the two halves of the fix:
``persist_event`` now stamps correlation/actor onto the ``FleetEvent`` node, and
the ``/api/fleet/trace`` + ``/api/fleet/touched`` handlers read it back.
"""

from __future__ import annotations

import json

import pytest

from agent_utilities.gateway import fleet, fleet_events
from agent_utilities.observability import correlation


class _CaptureEngine:
    """Minimal engine double: records add_node and answers the two queries."""

    def __init__(self) -> None:
        self.nodes: list[dict] = []

    def add_node(self, node_id, node_type, properties=None):
        self.nodes.append({"id": node_id, "type": node_type, **(properties or {})})

    def query_cypher(self, cypher, params=None):
        params = params or {}
        if "correlation_id = $cid" in cypher:
            return [
                {"n": n}
                for n in self.nodes
                if n.get("correlation_id") == params.get("cid")
            ]
        if "e.subject = $res" in cypher:
            return [
                {"e": n}
                for n in self.nodes
                if n.get("type") == "FleetEvent" and n.get("subject") == params.get("res")
            ]
        return []


class _Req:
    def __init__(self, query=None):
        self.query_params = query or {}


async def _payload(resp):
    return json.loads(resp.body)


def _event(subject="svc-a", source="prometheus"):
    return fleet_events.FleetEvent(
        source=source,
        severity="critical",
        subject=subject,
        status="firing",
        summary="down",
        raw={},
        received_at=1.0,
    )


def test_persist_event_stamps_correlation():
    engine = _CaptureEngine()
    with correlation.bind_carrier({correlation.CORRELATION_HEADER: "cid-123"}):
        fleet_events.persist_event(engine, _event())
    assert engine.nodes[0]["correlation_id"] == "cid-123"


@pytest.mark.asyncio
async def test_fleet_trace_returns_stamped_nodes(monkeypatch):
    engine = _CaptureEngine()
    with correlation.bind_carrier({correlation.CORRELATION_HEADER: "cid-xyz"}):
        fleet_events.persist_event(engine, _event(subject="svc-a"))
        fleet_events.persist_event(engine, _event(subject="svc-b"))
    # An unrelated event under a different correlation id must not leak in.
    with correlation.bind_carrier({correlation.CORRELATION_HEADER: "other"}):
        fleet_events.persist_event(engine, _event(subject="svc-c"))

    monkeypatch.setattr(
        "agent_utilities.mcp.kg_server._get_engine", lambda: engine
    )
    resp = await fleet.fleet_trace(_Req(query={"correlation_id": "cid-xyz"}))
    data = await _payload(resp)
    assert data["status"] == "success"
    subjects = {n["subject"] for n in data["nodes"]}
    assert subjects == {"svc-a", "svc-b"}


@pytest.mark.asyncio
async def test_fleet_touched_resolves_resource(monkeypatch):
    engine = _CaptureEngine()
    with correlation.bind_carrier({correlation.CORRELATION_HEADER: "cid-1"}):
        fleet_events.persist_event(engine, _event(subject="db-primary"))

    monkeypatch.setattr(
        "agent_utilities.mcp.kg_server._get_engine", lambda: engine
    )
    resp = await fleet.fleet_touched(_Req(query={"resource": "db-primary"}))
    data = await _payload(resp)
    assert data["status"] == "success"
    assert len(data["events"]) == 1
    assert data["events"][0]["correlation_id"] == "cid-1"


@pytest.mark.asyncio
async def test_fleet_trace_requires_correlation_id():
    resp = await fleet.fleet_trace(_Req(query={}))
    assert resp.status_code == 400
