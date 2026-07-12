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
    def __init__(self, incidents: list[tuple[str, dict]]):
        self._incidents = incidents

    def get_nodes_by_label(self, label, limit=0):
        assert label == "Incident"
        return list(self._incidents)


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
        return [{"id": "health:incident:r510:abc", "status": "open"}]

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
                "health:incident:r510:1",
                {"status": "open", "opened_at": "2026-07-01T00:00:00Z"},
            ),
            (
                "health:incident:r820:2",
                {"status": "closed", "opened_at": "2026-07-05T00:00:00Z"},
            ),
            (
                "health:incident:r710:3",
                {"status": "open", "opened_at": "2026-07-10T00:00:00Z"},
            ),
        ]
    )
    tool = _register(monkeypatch, engine)
    out = json.loads(tool(action="list", status="open", limit=50))
    assert out["surface"] == "incident"
    assert out["count"] == 2
    ids = [i["id"] for i in out["incidents"]]
    # newest opened_at first
    assert ids == ["health:incident:r710:3", "health:incident:r510:1"]


def test_list_no_status_filter_returns_all(monkeypatch):
    engine = _FakeEngine(
        [
            ("i1", {"status": "open", "opened_at": "2026-07-01T00:00:00Z"}),
            ("i2", {"status": "closed", "opened_at": "2026-07-02T00:00:00Z"}),
        ]
    )
    tool = _register(monkeypatch, engine)
    out = json.loads(tool(action="list", status="", limit=50))
    assert out["count"] == 2


def test_list_no_reachable_engine(monkeypatch):
    tool = _register(monkeypatch, None)
    out = json.loads(tool(action="list", status="", limit=50))
    assert "error" in out
    assert out["incidents"] == []


def test_get_returns_matching_incident(monkeypatch):
    engine = _FakeEngine(
        [("health:incident:r510:1", {"status": "open", "summary": "disk full"})]
    )
    tool = _register(monkeypatch, engine)
    out = json.loads(tool(action="get", incident_id="health:incident:r510:1"))
    assert out["incident"]["id"] == "health:incident:r510:1"
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
