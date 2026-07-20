"""Tests for the graph_ops_causal MCP tool (Codex X-2).

CONCEPT:AU-KG.enrichment.ops-causal-graph

Mirrors the ``_CollectingMCP`` + ``kg_server._get_engine`` monkeypatch pattern
used across the other MCP tool-surface tests (e.g.
``tests/unit/test_engine_surface_tools.py``) — no live engine required.
"""

from __future__ import annotations

import json

import pytest

from agent_utilities.mcp import kg_server
from agent_utilities.mcp.tools.ops_causal_tools import register_ops_causal_tools
from tests.kg_recording_backend import RecordingGraphBackend


class _CollectingMCP:
    """Minimal FastMCP stand-in that captures ``@mcp.tool``-registered functions."""

    def __init__(self) -> None:
        self.tools: dict[str, object] = {}

    def tool(self, *, name, description="", tags=None):  # noqa: ANN001
        def _deco(fn):
            self.tools[name] = fn
            return fn

        return _deco


@pytest.fixture
def tool(monkeypatch):
    mcp = _CollectingMCP()
    register_ops_causal_tools(mcp)
    monkeypatch.setattr(kg_server, "_get_engine", lambda: None)
    return mcp.tools["graph_ops_causal"]


_LINKS = json.dumps(
    [
        {"source": "commit:bad123", "target": "svc:checkout", "rel_type": "affects"},
        {
            "source": "commit:bad123",
            "target": "incident:INC001",
            "rel_type": "caused_incident",
        },
        {"source": "svc:checkout", "target": "agent:checkout", "rel_type": "part_of"},
        {"source": "agent:checkout", "target": "trace:1", "rel_type": "executed_by"},
        {"source": "policy:pci", "target": "svc:checkout", "rel_type": "governs"},
        {"source": "policy:pci", "target": "evidence:ev1", "rel_type": "has_evidence"},
    ]
)


def _call(tool_fn, **overrides):
    """Invoke the tool function with EVERY parameter explicit.

    Calling a ``@mcp.tool``-decorated function directly (bypassing the real
    FastMCP/pydantic validation layer) leaves any omitted parameter as its raw
    ``pydantic.Field(...)`` sentinel rather than the resolved default — the
    same caveat ``test_engine_surface_tools.py`` documents, so every call here
    supplies the full parameter set explicitly.
    """
    defaults = dict(
        action="root_cause",
        node_id="",
        links_json="[]",
        depth=6,
        max_results=10,
        incident_history_json="[]",
        now=0.0,
    )
    defaults.update(overrides)
    return tool_fn(**defaults)


def test_registered_on_graphos_tool_table():
    mcp = _CollectingMCP()
    register_ops_causal_tools(mcp)
    assert "graph_ops_causal" in mcp.tools
    assert kg_server.REGISTERED_TOOLS.get("graph_ops_causal") is not None


def test_root_cause_action(tool):
    out = json.loads(_call(tool, action="root_cause", node_id="trace:1", links_json=_LINKS))
    assert out["surface"] == "ops_causal"
    assert out["action"] == "root_cause"
    result = out["result"]
    assert result[0]["node_id"] == "commit:bad123"
    assert result[0]["is_root"] is True


def test_blast_radius_action(tool):
    out = json.loads(
        _call(tool, action="blast_radius", node_id="commit:bad123", links_json=_LINKS)
    )
    ids = {r["node_id"] for r in out["result"]}
    assert "svc:checkout" in ids
    assert "trace:1" in ids
    assert "incident:INC001" in ids


def test_change_risk_action(tool):
    history = json.dumps([{"node_id": "incident:INC001", "severity": 0.9}])
    out = json.loads(
        _call(
            tool,
            action="change_risk",
            node_id="commit:bad123",
            links_json=_LINKS,
            incident_history_json=history,
        )
    )
    result = out["result"]
    assert result["node_id"] == "commit:bad123"
    assert result["historical_severity"] == pytest.approx(0.9)
    assert len(result["contributing_incidents"]) == 1


def test_control_evidence_action(tool):
    out = json.loads(
        _call(tool, action="control_evidence", node_id="policy:pci", links_json=_LINKS)
    )
    result = out["result"]
    assert "svc:checkout" in result["governs"]
    assert result["is_consistent"] is True


def test_missing_node_id_returns_error(tool):
    out = json.loads(_call(tool, action="root_cause", node_id="", links_json=_LINKS))
    assert "error" in out


def test_invalid_links_json_returns_error(tool):
    out = json.loads(
        _call(tool, action="root_cause", node_id="trace:1", links_json="not json")
    )
    assert "error" in out


def test_unknown_action_returns_error(tool):
    out = json.loads(
        _call(tool, action="not_a_real_action", node_id="trace:1", links_json=_LINKS)
    )
    assert "error" in out


def test_join_action_materializes_edges_via_engine_backend(monkeypatch):
    mcp = _CollectingMCP()
    register_ops_causal_tools(mcp)
    backend = RecordingGraphBackend()

    class _FakeEngine:
        pass

    engine = _FakeEngine()
    engine.backend = backend
    monkeypatch.setattr(kg_server, "_get_engine", lambda: engine)

    tool_fn = mcp.tools["graph_ops_causal"]
    out = json.loads(_call(tool_fn, action="join", links_json=_LINKS))
    assert out["result"]["nodes_written"] == 0
    assert out["result"]["edges_written"] == 6
    assert ("commit:bad123", "svc:checkout", "affects") in backend.edges


def test_join_action_without_engine_backend_errors(tool):
    out = json.loads(_call(tool, action="join", links_json=_LINKS))
    assert "error" in out
