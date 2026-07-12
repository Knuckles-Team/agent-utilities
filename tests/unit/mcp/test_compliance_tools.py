"""Tests for the graph_compliance MCP tool (CONCEPT:AU-KG.audit.compliance-posture-rollup).

Mirrors the ``_CollectingMCP`` pattern of ``test_audit_tools.py``. ``posture``
monkeypatches ``kg_server._get_engine`` + ``audit_tools._verify`` (proving the
rollup REUSES the existing audit-ledger primitive rather than reimplementing
it); ``export`` monkeypatches ``engine_tools._client_for`` (proving bulk
export reuses the SAME ``explain_belief`` dispatch ``graph_epistemic``'s
``why`` action uses).
"""

from __future__ import annotations

import json

from agent_utilities.mcp import kg_server
from agent_utilities.mcp.tools import audit_tools, engine_tools
from agent_utilities.mcp.tools.compliance_tools import register_compliance_tools


class _CollectingMCP:
    def __init__(self) -> None:
        self.tools: dict[str, object] = {}

    def tool(self, *, name, description="", tags=None):  # noqa: ANN001
        def _deco(fn):
            self.tools[name] = fn
            return fn

        return _deco


class _FakeComputeEngine:
    def __init__(self, nodes_by_label: dict[str, list[tuple[str, dict]]]):
        self._nodes_by_label = nodes_by_label

    def get_nodes_by_label(self, label, limit=0):
        return list(self._nodes_by_label.get(label, []))


class _FakeEngine:
    def __init__(self, graph, cypher_rows=None):
        self.graph = graph
        self._cypher_rows = cypher_rows or []

    def query_cypher(self, cypher, as_of=None):
        return list(self._cypher_rows)


def _register(monkeypatch):
    mcp = _CollectingMCP()
    register_compliance_tools(mcp)
    return mcp.tools["graph_compliance"]


def test_registered_on_graphos_tool_table():
    mcp = _CollectingMCP()
    register_compliance_tools(mcp)
    assert "graph_compliance" in mcp.tools
    assert kg_server.REGISTERED_TOOLS.get("graph_compliance") is not None
    assert kg_server.ACTION_TOOL_ROUTES.get("graph_compliance") == "/compliance"


def test_posture_joins_audit_verify_and_node_counts(monkeypatch):
    tool = _register(monkeypatch)
    fake_graph = _FakeComputeEngine(
        {
            "Control": [
                ("c1", {"status": "satisfied"}),
                ("c2", {"status": "gap"}),
            ],
            "Incident": [("i1", {"status": "open"})],
        }
    )
    engine = _FakeEngine(fake_graph)
    monkeypatch.setattr(kg_server, "_get_engine", lambda: engine)
    monkeypatch.setattr(
        audit_tools,
        "_verify",
        lambda: {"surface": "audit", "action": "verify", "ok": True, "entries": 4},
    )

    out = json.loads(
        tool(
            action="posture",
            cypher="",
            node_ids="[]",
            disclosure_level="Full",
            as_of="",
            limit=200,
        )
    )
    assert out["surface"] == "compliance"
    assert out["action"] == "posture"
    assert out["audit_ledger"]["ok"] is True
    assert out["node_counts"]["Control"] == 2
    assert out["node_counts"]["Incident"] == 1
    assert out["status_breakdown"]["Control"] == {"satisfied": 1, "gap": 1}


def test_posture_no_active_engine(monkeypatch):
    tool = _register(monkeypatch)
    monkeypatch.setattr(kg_server, "_get_engine", lambda: None)
    out = json.loads(
        tool(
            action="posture",
            cypher="",
            node_ids="[]",
            disclosure_level="Full",
            as_of="",
            limit=200,
        )
    )
    assert "error" in out


def test_export_by_explicit_node_ids_reuses_explain_belief(monkeypatch):
    tool = _register(monkeypatch)
    engine = _FakeEngine(_FakeComputeEngine({}))
    monkeypatch.setattr(kg_server, "_get_engine", lambda: engine)

    calls: list[tuple[str, dict]] = []

    class _Query:
        def __getattr__(self, name):
            def _call(**kwargs):
                calls.append((name, kwargs))
                return {"root": {"claim": kwargs.get("node_id"), "rule": "Asserted"}}

            return _call

    class _Client:
        query = _Query()

    monkeypatch.setattr(engine_tools, "_client_for", lambda graph: _Client())

    out = json.loads(
        tool(
            action="export",
            cypher="",
            node_ids=json.dumps(["control:1", "control:2"]),
            disclosure_level="Skeleton",
            as_of="",
            limit=200,
        )
    )
    assert out["surface"] == "compliance"
    assert out["action"] == "export"
    assert out["exported"] == 2
    assert out["disclosure_level"] == "Skeleton"
    assert {c[0] for c in calls} == {"explain_belief"}
    assert all(c[1]["disclosure_level"] == "Skeleton" for c in calls)
    assert {e["node_id"] for e in out["entries"]} == {"control:1", "control:2"}


def test_export_by_cypher_selection(monkeypatch):
    tool = _register(monkeypatch)
    engine = _FakeEngine(
        _FakeComputeEngine({}),
        cypher_rows=[{"id": "control:1"}, {"id": "control:2"}],
    )
    monkeypatch.setattr(kg_server, "_get_engine", lambda: engine)

    class _Query:
        def __getattr__(self, name):
            def _call(**kwargs):
                return {"root": {}}

            return _call

    class _Client:
        query = _Query()

    monkeypatch.setattr(engine_tools, "_client_for", lambda graph: _Client())

    out = json.loads(
        tool(
            action="export",
            cypher="MATCH (n:Control) RETURN n.id AS id",
            node_ids="[]",
            disclosure_level="Full",
            as_of="",
            limit=200,
        )
    )
    assert out["requested"] == 2
    assert out["exported"] == 2


def test_export_requires_ids_or_cypher(monkeypatch):
    tool = _register(monkeypatch)
    engine = _FakeEngine(_FakeComputeEngine({}))
    monkeypatch.setattr(kg_server, "_get_engine", lambda: engine)
    out = json.loads(
        tool(
            action="export",
            cypher="",
            node_ids="[]",
            disclosure_level="Full",
            as_of="",
            limit=200,
        )
    )
    assert "error" in out


def test_export_respects_limit_and_reports_truncation(monkeypatch):
    tool = _register(monkeypatch)
    engine = _FakeEngine(_FakeComputeEngine({}))
    monkeypatch.setattr(kg_server, "_get_engine", lambda: engine)

    class _Query:
        def __getattr__(self, name):
            def _call(**kwargs):
                return {"root": {}}

            return _call

    class _Client:
        query = _Query()

    monkeypatch.setattr(engine_tools, "_client_for", lambda graph: _Client())

    out = json.loads(
        tool(
            action="export",
            cypher="",
            node_ids=json.dumps(["a", "b", "c"]),
            disclosure_level="Full",
            as_of="",
            limit=2,
        )
    )
    assert out["requested"] == 3
    assert out["exported"] == 2
    assert out["truncated"] is True


def test_unknown_action(monkeypatch):
    tool = _register(monkeypatch)
    out = json.loads(
        tool(
            action="bogus",
            cypher="",
            node_ids="[]",
            disclosure_level="Full",
            as_of="",
            limit=200,
        )
    )
    assert "error" in out
