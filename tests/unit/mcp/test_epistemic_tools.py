"""Tests for the graph_epistemic MCP tool (CONCEPT:AU-KB-CURRENCY).

Mirrors the ``_CollectingMCP`` + monkeypatched-client pattern of
``test_audit_tools.py`` / ``test_engine_tools_scope_policy.py`` — no live
engine required. ``graph_epistemic`` dispatches through
``engine_tools._dispatch``, so the client is faked the same way
``test_engine_tools_scope_policy.py`` does (monkeypatch ``engine_tools.
_client_for``).
"""

from __future__ import annotations

import json

from agent_utilities.mcp import kg_server
from agent_utilities.mcp.tools import engine_tools
from agent_utilities.mcp.tools.epistemic_tools import register_epistemic_tools


class _CollectingMCP:
    def __init__(self) -> None:
        self.tools: dict[str, object] = {}

    def tool(self, *, name, description="", tags=None):  # noqa: ANN001
        def _deco(fn):
            self.tools[name] = fn
            return fn

        return _deco


def _fake_query_client(responses: dict[str, object]):
    calls: list[tuple[str, dict]] = []

    class _Query:
        def __getattr__(self, name):
            def _call(**kwargs):
                calls.append((name, kwargs))
                if name in responses:
                    result = responses[name]
                    if isinstance(result, Exception):
                        raise result
                    return result
                raise AttributeError(name)

            return _call

    class _Client:
        query = _Query()

    return _Client(), calls


def _register(monkeypatch, responses):
    mcp = _CollectingMCP()
    register_epistemic_tools(mcp)
    client, calls = _fake_query_client(responses)
    monkeypatch.setattr(engine_tools, "_client_for", lambda graph: client)
    return mcp.tools["graph_epistemic"], calls


def test_registered_on_graphos_tool_table():
    mcp = _CollectingMCP()
    register_epistemic_tools(mcp)
    assert "graph_epistemic" in mcp.tools
    assert kg_server.REGISTERED_TOOLS.get("graph_epistemic") is not None
    assert kg_server.ACTION_TOOL_ROUTES.get("graph_epistemic") == "/epistemic"


def test_status_action_dispatches_epistemic_status(monkeypatch):
    tool, calls = _register(
        monkeypatch,
        {"epistemic_status": {"believed": True, "since": 100, "evidence": ["ev:1"]}},
    )
    out = json.loads(tool(action="status", node_id="claim:1"))
    assert out["surface"] == "epistemic"
    assert out["action"] == "status"
    assert out["engine_method"] == "epistemic_status"
    assert out["result"]["believed"] is True
    assert calls == [("epistemic_status", {"node_id": "claim:1"})]


def test_why_action_dispatches_explain_belief_with_disclosure_level(monkeypatch):
    tool, calls = _register(
        monkeypatch,
        {"explain_belief": {"root": {"claim": "claim:1", "rule": "Asserted"}}},
    )
    out = json.loads(tool(action="why", node_id="claim:1", disclosure_level="Skeleton"))
    assert out["result"]["root"]["rule"] == "Asserted"
    assert calls == [
        ("explain_belief", {"node_id": "claim:1", "disclosure_level": "Skeleton"})
    ]


def test_why_action_without_disclosure_level_omits_it(monkeypatch):
    tool, calls = _register(monkeypatch, {"explain_belief": {"root": {}}})
    tool(action="why", node_id="claim:1", disclosure_level="")
    assert calls == [("explain_belief", {"node_id": "claim:1"})]


def test_why_requires_node_id(monkeypatch):
    tool, calls = _register(monkeypatch, {})
    out = json.loads(tool(action="why", node_id=""))
    assert "error" in out
    assert calls == []


def test_what_changed_action(monkeypatch):
    tool, calls = _register(
        monkeypatch, {"what_changed": {"added": [], "removed": [], "modified": []}}
    )
    out = json.loads(tool(action="what_changed", tx_from=100, tx_to=200))
    assert out["engine_method"] == "what_changed"
    assert calls == [("what_changed", {"tx_from": 100, "tx_to": 200})]


def test_resolve_conflict_action(monkeypatch):
    tool, calls = _register(
        monkeypatch, {"resolve_conflict": {"accepted": ["claim:1"], "rejected": []}}
    )
    out = json.loads(
        tool(
            action="resolve_conflict",
            node_ids=json.dumps(["claim:1", "claim:2"]),
            semantics="grounded",
        )
    )
    assert out["result"]["accepted"] == ["claim:1"]
    assert calls == [
        (
            "resolve_conflict",
            {"node_ids": ["claim:1", "claim:2"], "semantics": "grounded"},
        )
    ]


def test_resolve_conflict_requires_non_empty_node_ids(monkeypatch):
    tool, calls = _register(monkeypatch, {})
    out = json.loads(tool(action="resolve_conflict", node_ids="[]"))
    assert "error" in out
    assert calls == []


def test_degrades_cleanly_when_engine_lacks_epistemic_tms(monkeypatch):
    tool, _calls = _register(
        monkeypatch, {"epistemic_status": RuntimeError("epistemic-tms not built")}
    )
    out = json.loads(tool(action="status", node_id="claim:1"))
    assert "error" in out["result"]


def test_unknown_action(monkeypatch):
    tool, calls = _register(monkeypatch, {})
    out = json.loads(tool(action="bogus"))
    assert "error" in out
    assert calls == []
