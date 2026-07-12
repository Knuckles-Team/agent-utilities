"""Tests for the graph_audit MCP tool (G23, audit-trail closure).

CONCEPT:AU-KG.audit.hash-chain-verify

Mirrors the ``_CollectingMCP`` + ``kg_server._get_engine`` monkeypatch pattern
used across the other MCP tool-surface tests (e.g. ``test_ops_causal_tools.py``)
— no live engine required.
"""

from __future__ import annotations

import json

import pytest

from agent_utilities.mcp import kg_server
from agent_utilities.mcp.tools.audit_tools import register_audit_tools


class _CollectingMCP:
    """Minimal FastMCP stand-in that captures ``@mcp.tool``-registered functions."""

    def __init__(self) -> None:
        self.tools: dict[str, object] = {}

    def tool(self, *, name, description="", tags=None):  # noqa: ANN001
        def _deco(fn):
            self.tools[name] = fn
            return fn

        return _deco


class _FakeGraphClient:
    def __init__(self, report=None, error=None):
        self._report = report
        self._error = error

    def audit_verify(self):
        if self._error is not None:
            raise self._error
        return dict(self._report)


class _FakeBackend:
    def __init__(self, rows):
        self._rows = rows

    def execute(self, query, params=None):
        return list(self._rows)


class _FakeEngine:
    def __init__(self, graph=None, backend=None):
        self.graph = graph
        self.backend = backend


def _register(monkeypatch, engine):
    mcp = _CollectingMCP()
    register_audit_tools(mcp)
    monkeypatch.setattr(kg_server, "_get_engine", lambda: engine)
    return mcp.tools["graph_audit"]


def test_registered_on_graphos_tool_table():
    mcp = _CollectingMCP()
    register_audit_tools(mcp)
    assert "graph_audit" in mcp.tools
    assert kg_server.REGISTERED_TOOLS.get("graph_audit") is not None
    assert kg_server.ACTION_TOOL_ROUTES.get("graph_audit") == "/audit"


def test_verify_returns_clean_chain_report(monkeypatch):
    report = {
        "graph": "__commons__",
        "ok": True,
        "entries": 5,
        "first_broken_seq": None,
        "detail": "chain verified",
    }
    engine = _FakeEngine(graph=_FakeGraphClient(report=report))
    tool = _register(monkeypatch, engine)
    out = json.loads(tool(action="verify", target_id=""))
    assert out["surface"] == "audit"
    assert out["action"] == "verify"
    assert out["available"] is True
    assert out["ok"] is True
    assert out["entries"] == 5


def test_verify_surfaces_tampered_chain(monkeypatch):
    report = {
        "graph": "g",
        "ok": False,
        "entries": 2,
        "first_broken_seq": 2,
        "detail": "hash-chain break at seq 2",
    }
    engine = _FakeEngine(graph=_FakeGraphClient(report=report))
    tool = _register(monkeypatch, engine)
    out = json.loads(tool(action="verify", target_id=""))
    assert out["ok"] is False
    assert out["first_broken_seq"] == 2


def test_verify_degrades_cleanly_when_engine_build_lacks_support(monkeypatch):
    """The engine build/config doesn't expose AuditVerify (no 'security'
    feature, no durable redb dir) ⇒ a clean, informative degrade payload, not
    a 500 / unhandled exception."""
    engine = _FakeEngine(graph=_FakeGraphClient(error=RuntimeError("no such method")))
    tool = _register(monkeypatch, engine)
    out = json.loads(tool(action="verify", target_id=""))
    assert out["available"] is False
    assert "epistemic-graph" in out["error"]


def test_verify_no_active_engine(monkeypatch):
    tool = _register(monkeypatch, None)
    out = json.loads(tool(action="verify", target_id=""))
    assert "error" in out
    assert out["action"] == "verify"


def test_for_target_requires_target_id(monkeypatch):
    tool = _register(monkeypatch, _FakeEngine())
    out = json.loads(tool(action="for_target", target_id=""))
    assert "error" in out


def test_for_target_reverse_index_reconstructs_history(monkeypatch):
    rows = [
        {
            "id": "toolcall:1:0",
            "run_id": "run:1",
            "agent_name": "agent-x",
            "server": "server-y",
            "tool_name": "engine_nodes",
            "args": '{"incident_id": "incident:INC1"}',
            "result_preview": "ok",
            "error": "",
            "status": "ok",
            "sequence": 0,
            "timestamp": "2026-01-01T00:00:00Z",
        }
    ]
    report = {"graph": "g", "ok": True, "entries": 1, "first_broken_seq": None, "detail": ""}
    engine = _FakeEngine(
        graph=_FakeGraphClient(report=report), backend=_FakeBackend(rows)
    )
    tool = _register(monkeypatch, engine)
    out = json.loads(tool(action="for_target", target_id="incident:INC1"))
    assert out["surface"] == "audit"
    assert out["action"] == "for_target"
    assert out["target_id"] == "incident:INC1"
    assert out["tool_call_count"] == 1
    assert out["tool_calls"][0]["id"] == "toolcall:1:0"
    assert out["audit"]["ok"] is True


def test_unknown_action(monkeypatch):
    tool = _register(monkeypatch, _FakeEngine())
    out = json.loads(tool(action="bogus", target_id=""))
    assert "error" in out
