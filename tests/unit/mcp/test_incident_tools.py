"""Tests for the graph_incident MCP tool (CONCEPT:AU-KG.enrichment.cross-layer-incident-correlation).

Mirrors the ``_CollectingMCP`` pattern of ``test_audit_tools.py``. `list`/`get`
monkeypatch ``observability.health_ingest._engine`` (the SAME accessor
``observability.incidents`` itself uses for its own ``:Incident`` reads);
`correlate` monkeypatches ``observability.incidents.correlate_incidents``
directly (proving the tool is a thin wrapper, not a reimplementation).
"""

from __future__ import annotations

import json

from agent_utilities.mcp import kg_server
from agent_utilities.mcp.tools.incident_tools import register_incident_tools
from agent_utilities.observability import health_ingest


class _CollectingMCP:
    def __init__(self) -> None:
        self.tools: dict[str, object] = {}

    def tool(self, *, name, description="", tags=None):  # noqa: ANN001
        def _deco(fn):
            self.tools[name] = fn
            return fn

        return _deco


class _FakeEngine:
    def __init__(
        self,
        incidents: list[tuple[str, dict]],
        anomalies: list[tuple[str, dict]] | None = None,
        neighbors: dict[str, list[str]] | None = None,
    ):
        self._incidents = incidents
        self._anomalies = anomalies or []
        self._neighbors = neighbors or {}

    def get_nodes_by_label(self, label, limit=0):
        assert label in ("Incident", "HealthAnomaly")
        return (
            list(self._anomalies) if label == "HealthAnomaly" else list(self._incidents)
        )

    def get_neighbors(self, node_id):
        return list(self._neighbors.get(node_id, []))


def _register(monkeypatch, engine=None):
    mcp = _CollectingMCP()
    register_incident_tools(mcp)
    monkeypatch.setattr(health_ingest, "_engine", lambda: engine)
    return mcp.tools["graph_incident"]


def test_registered_on_graphos_tool_table():
    mcp = _CollectingMCP()
    register_incident_tools(mcp)
    assert "graph_incident" in mcp.tools
    assert kg_server.REGISTERED_TOOLS.get("graph_incident") is not None
    assert kg_server.ACTION_TOOL_ROUTES.get("graph_incident") == "/incident"


def test_correlate_action_delegates_to_incidents_module(monkeypatch):
    from agent_utilities.observability import incidents

    calls: list[tuple[int, int]] = []

    def _fake_correlate(*, window_s, days):
        calls.append((window_s, days))
        return [{"id": "health:incident:storage-node-a:abc", "status": "open"}]

    monkeypatch.setattr(incidents, "correlate_incidents", _fake_correlate)
    tool = _register(monkeypatch)
    out = json.loads(tool(action="correlate", window_s=600, days=2))
    assert out["surface"] == "incident"
    assert out["action"] == "correlate"
    assert out["count"] == 1
    assert calls == [(600, 2)]


def test_list_returns_open_incidents_newest_first(monkeypatch):
    engine = _FakeEngine(
        [
            (
                "health:incident:storage-node-a:1",
                {"status": "open", "observedAt": "2026-07-01T00:00:00Z"},
            ),
            (
                "health:incident:analysis-node-a:2",
                {"status": "closed", "observedAt": "2026-07-05T00:00:00Z"},
            ),
            (
                "health:incident:compute-node-b:3",
                {"status": "open", "observedAt": "2026-07-10T00:00:00Z"},
            ),
        ]
    )
    tool = _register(monkeypatch, engine)
    out = json.loads(tool(action="list", status="open", severity="", limit=50))
    assert out["surface"] == "incident"
    assert out["count"] == 2
    ids = [i["id"] for i in out["incidents"]]
    # newest opened_at first
    assert ids == [
        "health:incident:compute-node-b:3",
        "health:incident:storage-node-a:1",
    ]


def test_list_no_status_filter_returns_all(monkeypatch):
    engine = _FakeEngine(
        [
            ("i1", {"status": "open", "observedAt": "2026-07-01T00:00:00Z"}),
            ("i2", {"status": "closed", "opened_at": "2026-07-02T00:00:00Z"}),
        ]
    )
    tool = _register(monkeypatch, engine)
    out = json.loads(tool(action="list", status="", severity="", limit=50))
    assert out["count"] == 2


def test_list_no_reachable_engine(monkeypatch):
    tool = _register(monkeypatch, None)
    out = json.loads(tool(action="list", status="", limit=50))
    assert "error" in out
    assert out["incidents"] == []


def test_get_returns_matching_incident(monkeypatch):
    engine = _FakeEngine(
        [
            (
                "health:incident:storage-node-a:1",
                {"status": "open", "summary": "disk full"},
            )
        ]
    )
    tool = _register(monkeypatch, engine)
    out = json.loads(tool(action="get", incident_id="health:incident:storage-node-a:1"))
    assert out["incident"]["id"] == "health:incident:storage-node-a:1"
    assert out["incident"]["summary"] == "disk full"


def test_get_not_found(monkeypatch):
    engine = _FakeEngine([("other:id", {"status": "open"})])
    tool = _register(monkeypatch, engine)
    out = json.loads(tool(action="get", incident_id="missing:id"))
    assert "error" in out


def test_get_requires_incident_id(monkeypatch):
    tool = _register(monkeypatch, _FakeEngine([]))
    out = json.loads(tool(action="get", incident_id=""))
    assert "error" in out


def test_unknown_action(monkeypatch):
    tool = _register(monkeypatch, _FakeEngine([]))
    out = json.loads(tool(action="bogus"))
    assert "error" in out


def test_list_filters_by_severity(monkeypatch):
    engine = _FakeEngine(
        [
            (
                "i1",
                {
                    "status": "open",
                    "severity": "critical",
                    "observedAt": "2026-07-01T00:00:00Z",
                },
            ),
            (
                "i2",
                {
                    "status": "open",
                    "severity": "warning",
                    "opened_at": "2026-07-02T00:00:00Z",
                },
            ),
        ]
    )
    tool = _register(monkeypatch, engine)
    out = json.loads(tool(action="list", status="", severity="critical", limit=50))
    assert [i["id"] for i in out["incidents"]] == ["i1"]


def test_get_includes_correlated_anomaly_group_and_affected_entities(monkeypatch):
    incident_id = "health:incident:storage-node-a:sig1"
    engine = _FakeEngine(
        [(incident_id, {"status": "open", "summary": "storage-node-a stress"})],
        anomalies=[
            (
                "health:anomaly:storage-node-a:cpu:t2",
                {
                    "entity": "cm:node:storage-node-a",
                    "signal": "cpu",
                    "kind": "above-baseline",
                    "observedAt": "2026-07-02T00:00:00Z",
                },
            ),
            (
                "health:anomaly:storage-node-a:temp:t1",
                {
                    "entity": "fan:host:storage-node-a",
                    "signal": "temp",
                    "kind": "above-baseline",
                    "observedAt": "2026-07-01T00:00:00Z",
                },
            ),
        ],
        neighbors={
            incident_id: [
                "health:anomaly:storage-node-a:cpu:t2",
                "health:anomaly:storage-node-a:temp:t1",
                "cm:node:storage-node-a",
                "fan:host:storage-node-a",
            ]
        },
    )
    tool = _register(monkeypatch, engine)
    out = json.loads(tool(action="get", incident_id=incident_id))
    assert out["incident"]["id"] == incident_id
    assert [a["id"] for a in out["anomalies"]] == [
        "health:anomaly:storage-node-a:temp:t1",
        "health:anomaly:storage-node-a:cpu:t2",
    ]
    assert sorted(out["entities"]) == [
        "cm:node:storage-node-a",
        "fan:host:storage-node-a",
    ]


def test_timeline_orders_lifecycle_and_anomaly_events(monkeypatch):
    incident_id = "health:incident:storage-node-a:sig1"
    engine = _FakeEngine(
        [
            (
                incident_id,
                {
                    "status": "resolved",
                    "summary": "storage-node-a stress",
                    "observedAt": "2026-07-01T00:00:00Z",
                    "ackedAt": "2026-07-01T00:05:00Z",
                    "ackedBy": "op1",
                    "resolvedAt": "2026-07-01T00:10:00Z",
                    "resolvedBy": "op1",
                },
            )
        ],
        anomalies=[
            (
                "health:anomaly:storage-node-a:temp:t1",
                {
                    "entity": "fan:host:storage-node-a",
                    "signal": "temp",
                    "kind": "above-baseline",
                    "observedAt": "2026-07-01T00:00:30Z",
                },
            )
        ],
        neighbors={incident_id: ["health:anomaly:storage-node-a:temp:t1"]},
    )
    tool = _register(monkeypatch, engine)
    out = json.loads(tool(action="timeline", incident_id=incident_id))
    assert [e["kind"] for e in out["timeline"]] == [
        "incident_opened",
        "anomaly",
        "incident_acknowledged",
        "incident_resolved",
    ]


def test_timeline_requires_incident_id(monkeypatch):
    tool = _register(monkeypatch, _FakeEngine([]))
    out = json.loads(tool(action="timeline", incident_id=""))
    assert "error" in out


def test_timeline_not_found(monkeypatch):
    engine = _FakeEngine([("other:id", {"status": "open"})])
    tool = _register(monkeypatch, engine)
    out = json.loads(tool(action="timeline", incident_id="missing:id"))
    assert "error" in out


def test_ack_requires_incident_id(monkeypatch):
    tool = _register(monkeypatch, _FakeEngine([]))
    out = json.loads(tool(action="ack", incident_id=""))
    assert "error" in out


def test_resolve_requires_incident_id(monkeypatch):
    tool = _register(monkeypatch, _FakeEngine([]))
    out = json.loads(tool(action="resolve", incident_id=""))
    assert "error" in out


def _allow_gate(monkeypatch):
    """Force the incident-state-mutation ActionPolicy gate to allow, for tests
    of the dispatch itself (the denial path gets its own dedicated tests
    below) — mirrors ``test_claim_tools.py``'s ``allow_policy`` fixture."""
    import agent_utilities.mcp.tools.incident_tools as it

    monkeypatch.setattr(
        it,
        "_gate",
        lambda kind, target, reason, actor_id: (
            True,
            {"decision": "allow", "tier": "auto", "reason": "test"},
        ),
    )


def test_ack_allowed_dispatches_to_set_incident_status(monkeypatch):
    from agent_utilities.observability import incidents as inc

    _allow_gate(monkeypatch)
    calls: list[tuple[str, str, str]] = []

    def _fake_set_status(incident_id, status, *, actor="", engine=None):
        calls.append((incident_id, status, actor))
        return {"nodes": 1, "edges": 0}

    monkeypatch.setattr(inc, "set_incident_status", _fake_set_status)
    incident_id = "health:incident:storage-node-a:sig1"
    engine = _FakeEngine([(incident_id, {"status": "open", "summary": "x"})])
    tool = _register(monkeypatch, engine)
    out = json.loads(
        tool(action="ack", incident_id=incident_id, actor_id="op1", reason="triage")
    )
    assert "error" not in out
    assert calls == [(incident_id, inc.STATUS_ACKNOWLEDGED, "op1")]
    assert out["status"] == inc.STATUS_ACKNOWLEDGED
    assert out["policy"]["decision"] == "allow"
    assert out["incident"]["id"] == incident_id


def test_resolve_allowed_dispatches_to_set_incident_status(monkeypatch):
    from agent_utilities.observability import incidents as inc

    _allow_gate(monkeypatch)
    calls: list[tuple[str, str, str]] = []

    def _fake_set_status(incident_id, status, *, actor="", engine=None):
        calls.append((incident_id, status, actor))
        return {"nodes": 1, "edges": 0}

    monkeypatch.setattr(inc, "set_incident_status", _fake_set_status)
    incident_id = "health:incident:storage-node-a:sig1"
    engine = _FakeEngine([(incident_id, {"status": "open", "summary": "x"})])
    tool = _register(monkeypatch, engine)
    out = json.loads(tool(action="resolve", incident_id=incident_id, actor_id="op1"))
    assert "error" not in out
    assert calls == [(incident_id, inc.STATUS_RESOLVED, "op1")]
    assert out["status"] == inc.STATUS_RESOLVED


def test_ack_mutation_not_found_reports_error(monkeypatch):
    from agent_utilities.observability import incidents as inc

    _allow_gate(monkeypatch)
    monkeypatch.setattr(inc, "set_incident_status", lambda *a, **k: None)
    tool = _register(monkeypatch, _FakeEngine([]))
    out = json.loads(tool(action="ack", incident_id="missing:id"))
    assert out["error"]
    assert out["policy"]["decision"] == "allow"


def test_ack_denied_by_real_default_policy_never_mutates(monkeypatch):
    """No ``_gate`` stub here: the REAL fail-closed ActionPolicy is consulted.
    ``incident.*`` has no explicit rule in the shipped default policy (mirrors
    ``claim.*`` in ``test_claim_tools.py``), so it falls to the conservative
    default tier (approval_required) — not an allowing decision.
    ``kg_server._get_engine`` is stubbed to ``None`` — ``ActionPolicy(engine=
    None)`` degrades gracefully throughout, exactly like
    ``incidents.actuate_remediation``'s own no-engine default-held behavior."""
    from agent_utilities.observability import incidents as inc

    monkeypatch.setattr(kg_server, "_get_engine", lambda: None)
    mutated: list = []
    monkeypatch.setattr(
        inc, "set_incident_status", lambda *a, **k: mutated.append((a, k))
    )
    incident_id = "health:incident:storage-node-a:sig1"
    engine = _FakeEngine([(incident_id, {"status": "open", "summary": "x"})])
    tool = _register(monkeypatch, engine)
    out = json.loads(
        tool(action="ack", incident_id=incident_id, reason="", actor_id="")
    )
    assert out["error"] == "policy_denied"
    assert out["policy"]["decision"] in ("queue_approval", "deny")
    assert mutated == []  # the mutation never ran


def test_resolve_denied_by_real_default_policy_never_mutates(monkeypatch):
    from agent_utilities.observability import incidents as inc

    monkeypatch.setattr(kg_server, "_get_engine", lambda: None)
    mutated: list = []
    monkeypatch.setattr(
        inc, "set_incident_status", lambda *a, **k: mutated.append((a, k))
    )
    incident_id = "health:incident:storage-node-a:sig1"
    engine = _FakeEngine([(incident_id, {"status": "open", "summary": "x"})])
    tool = _register(monkeypatch, engine)
    out = json.loads(
        tool(action="resolve", incident_id=incident_id, reason="", actor_id="")
    )
    assert out["error"] == "policy_denied"
    assert out["policy"]["decision"] in ("queue_approval", "deny")
    assert mutated == []
