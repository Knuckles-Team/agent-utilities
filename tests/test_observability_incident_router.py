"""Regression tests for incident → ticket routing
(``agent_utilities.observability.incident_router``) — the ticketing half of
Phase D/E (``reports/unified-infra-intelligence-plan.md``). Verifies the
pluggable-adapter interface, the default graph-only (no-SoR) path, and the
fail-closed/dry-run-first gating shared with ``kg_writeback``.
"""

from __future__ import annotations

from typing import Any

import agent_utilities.knowledge_graph.memory.native_ingest as native_ingest
from agent_utilities.observability import incident_router as router


class _Capture:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(self, entities, relationships=None, *, source, domain, **kw):
        self.calls.append(
            {
                "entities": entities,
                "relationships": relationships or [],
                "source": source,
                "domain": domain,
            }
        )
        return {"nodes": len(entities), "edges": len(relationships or [])}


_INCIDENT = {
    "id": "health:incident:r510:abc",
    "entity": "systems:host:r510",
    "layers": ["hardware", "os"],
    "signals": ["cpu_temp_c", "load1"],
    "root_cause_layer": "hardware",
    "severity": "critical",
    "opened_at": "2026-07-11T00:00:00Z",
    "summary": "r510: hardware/cpu_temp_c + os/load1 — correlated within 2 anomalies",
}


def test_get_adapter_defaults_to_graph_only(monkeypatch):
    monkeypatch.delenv("INCIDENT_TICKET_BACKEND", raising=False)
    assert isinstance(router.get_adapter(), router.GraphOnlyAdapter)


def test_get_adapter_resolves_configured_backend(monkeypatch):
    monkeypatch.setenv("INCIDENT_TICKET_BACKEND", "jira")
    assert isinstance(router.get_adapter(), router.JiraAdapter)
    monkeypatch.setenv("INCIDENT_TICKET_BACKEND", "unknown-backend")
    assert isinstance(router.get_adapter(), router.GraphOnlyAdapter)


def test_route_incident_graph_only_records_intended_ticket_and_has_ticket_edge(
    monkeypatch,
):
    cap = _Capture()
    monkeypatch.setattr(native_ingest, "ingest_entities", cap)

    out = router.route_incident(_INCIDENT, adapter=router.GraphOnlyAdapter())

    assert out["backend"] == "none"
    assert out["ticket_status"] == "proposed"
    assert out["ticket_id"] == f"proposed:{_INCIDENT['id']}"
    assert len(cap.calls) == 1
    ticket_node = cap.calls[0]["entities"][0]
    assert ticket_node["type"] == "Ticket"
    assert ticket_node["ticketStatus"] == "proposed"
    rel = cap.calls[0]["relationships"][0]
    assert rel == {
        "source": _INCIDENT["id"],
        "target": f"health:ticket:{_INCIDENT['id']}",
        "type": "hasTicket",
    }


def test_route_incident_adapter_failure_degrades_to_failed_ticket(monkeypatch):
    monkeypatch.setattr(native_ingest, "ingest_entities", _Capture())

    class _Boom:
        name = "boom"

        def create_ticket(self, incident):
            raise RuntimeError("adapter exploded")

    out = router.route_incident(_INCIDENT, adapter=_Boom())
    assert out["ticket_status"] == "failed"


def test_jira_adapter_dry_run_by_default(monkeypatch):
    monkeypatch.delenv("INCIDENT_TICKET_ENABLE", raising=False)
    captured: dict[str, Any] = {}

    def fake_run_writeback(target, *, dry_run, **ops):
        captured["target"] = target
        captured["dry_run"] = dry_run
        captured["creations"] = ops.get("creations")
        return {
            "status": "completed",
            "created": 0,
            "proposals": [{"op": "create_issue"}],
        }

    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.enrichment.writeback.core.run_writeback",
        fake_run_writeback,
    )

    adapter = router.JiraAdapter()
    out = adapter.create_ticket(_INCIDENT)

    assert captured["target"] == "jira"
    assert captured["dry_run"] is True
    assert captured["creations"][0]["title"] == f"[incident] {_INCIDENT['summary']}"
    assert out["status"] == "proposed"


def test_jira_adapter_live_when_enabled_and_created(monkeypatch):
    monkeypatch.setenv("INCIDENT_TICKET_ENABLE", "true")

    def fake_run_writeback(target, *, dry_run, **ops):
        assert dry_run is False
        return {"status": "completed", "created": 1}

    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.enrichment.writeback.core.run_writeback",
        fake_run_writeback,
    )

    out = router.JiraAdapter().create_ticket(_INCIDENT)
    assert out["status"] == "created"
    assert out["ticket_id"] == f"jira:{_INCIDENT['id']}"


def test_servicenow_adapter_dry_run_never_calls_client(monkeypatch):
    monkeypatch.delenv("INCIDENT_TICKET_ENABLE", raising=False)

    def fail_if_called():
        raise AssertionError("servicenow client must not be constructed in dry-run")

    adapter = router.ServiceNowAdapter()
    monkeypatch.setattr(adapter, "_client", fail_if_called)
    out = adapter.create_ticket(_INCIDENT)
    assert out["status"] == "proposed"


def test_servicenow_adapter_live_creates_incident(monkeypatch):
    monkeypatch.setenv("INCIDENT_TICKET_ENABLE", "true")

    class _Result:
        number = "INC0012345"
        sys_id = "abc123"

    class _Response:
        result = _Result()

    class _FakeClient:
        def create_incident(self, **kwargs):
            return _Response()

    adapter = router.ServiceNowAdapter()
    monkeypatch.setattr(adapter, "_client", lambda: _FakeClient())
    out = adapter.create_ticket(_INCIDENT)
    assert out == {"ticket_id": "INC0012345", "url": "", "status": "created"}


def test_servicenow_adapter_update_uses_generic_table_patch(monkeypatch):
    monkeypatch.setenv("INCIDENT_TICKET_ENABLE", "true")
    calls: list[dict[str, Any]] = []

    class _FakeClient:
        def patch_table_record(self, **kwargs):
            calls.append(kwargs)

    adapter = router.ServiceNowAdapter()
    monkeypatch.setattr(adapter, "_client", lambda: _FakeClient())
    out = adapter.update_ticket("abc123", "resolved")
    assert out == {"ticket_id": "abc123", "status": "resolved"}
    assert calls == [
        {"table": "incident", "table_record_sys_id": "abc123", "data": {"state": "6"}}
    ]


def test_erpnext_adapter_dry_run_never_calls_client(monkeypatch):
    monkeypatch.delenv("INCIDENT_TICKET_ENABLE", raising=False)
    adapter = router.ErpNextAdapter()
    monkeypatch.setattr(
        adapter,
        "_client",
        lambda: (_ for _ in ()).throw(AssertionError("must not be called")),
    )
    out = adapter.create_ticket(_INCIDENT)
    assert out["status"] == "proposed"


def test_erpnext_adapter_live_creates_issue(monkeypatch):
    monkeypatch.setenv("INCIDENT_TICKET_ENABLE", "true")

    class _FakeClient:
        def create_document(self, doctype, data):
            assert doctype == "Issue"
            return {"name": "ISS-0001"}

    adapter = router.ErpNextAdapter()
    monkeypatch.setattr(adapter, "_client", lambda: _FakeClient())
    out = adapter.create_ticket(_INCIDENT)
    assert out == {"ticket_id": "ISS-0001", "url": "", "status": "created"}


def test_close_ticket_updates_status_and_ticket_node(monkeypatch):
    cap = _Capture()
    monkeypatch.setattr(native_ingest, "ingest_entities", cap)

    out = router.close_ticket(
        _INCIDENT,
        "proposed:health:incident:r510:abc",
        adapter=router.GraphOnlyAdapter(),
    )
    assert out["ticket_status"] == "resolved"
    node = cap.calls[0]["entities"][0]
    assert node["ticketStatus"] == "resolved"
