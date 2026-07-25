"""Tests for the graph_governance MCP tool's new actions:

- ``ownership_report`` / ``ownership_apply`` (CONCEPT:AU-KG.audit.graph-ownership-disposition,
  W2.8/HG-4) — the per-graph RBAC-ownership disposition pass, wired onto the
  existing governance surface.
- ``policy_status`` (CONCEPT:AU-OS.identity.identity-policy-check /
  AU-KG.audit.kg-native-audit-sink, PA-R1) — read-only PermissionPolicy/
  context_policy + KG-audit-sink status.

Mirrors the ``_CollectingMCP`` + monkeypatch pattern used across the other MCP
tool-surface tests (e.g. ``test_audit_tools.py``) — no live engine required.
"""

from __future__ import annotations

import json

from agent_utilities.knowledge_graph.maintenance import graph_ownership as go
from agent_utilities.mcp import kg_server
from agent_utilities.mcp.tools.governance_tools import register_governance_tools


class _CollectingMCP:
    def __init__(self) -> None:
        self.tools: dict[str, object] = {}

    def tool(self, *, name, description="", tags=None):  # noqa: ANN001
        def _deco(fn):
            self.tools[name] = fn
            return fn

        return _deco


class _FakeEngine:
    """Just enough of the engine surface for the pre-action `engine is None` guard."""


def _register(monkeypatch, *, engine: object | None = _FakeEngine()) -> object:
    mcp = _CollectingMCP()
    register_governance_tools(mcp)
    monkeypatch.setattr(kg_server, "_get_engine", lambda: engine)
    return mcp.tools["graph_governance"]


def _small_fixture_client() -> go.FixtureCatalogClient:
    graphs = [
        {"name": "__commons__", "type": "Commons", "valid": True},
        {"name": "code_agent-utilities", "type": "Agent", "valid": True},
        {"name": "tenant__homelab__default", "type": "Global", "valid": True},
    ]
    grants = [
        {
            "role": "owner:tenant-homelab",
            "resource": {"Graph": "tenant__homelab__default"},
            "action": "Read",
            "effect": "Allow",
        },
        {
            "role": "owner:tenant-homelab",
            "resource": {"Graph": "tenant__homelab__default"},
            "action": "Write",
            "effect": "Allow",
        },
    ]
    return go.FixtureCatalogClient(
        graphs,
        grants=grants,
        node_samples={
            "code_agent-utilities": [{"source_system": "code:agent-utilities"}] * 5
        },
    )


def test_registered_on_graphos_tool_table():
    mcp = _CollectingMCP()
    register_governance_tools(mcp)
    assert "graph_governance" in mcp.tools
    assert kg_server.REGISTERED_TOOLS.get("graph_governance") is not None
    assert kg_server.ACTION_TOOL_ROUTES.get("graph_governance") == "/graph/governance"


# ---------------------------------------------------------------------------
# ownership_report / ownership_apply
# ---------------------------------------------------------------------------


def test_ownership_report_runs_the_live_disposition_pass(monkeypatch):
    monkeypatch.setattr(
        go, "resolve_catalog_client", lambda config=None: _small_fixture_client()
    )
    tool = _register(monkeypatch)

    out = json.loads(tool(action="ownership_report"))
    assert out["action"] == "ownership_report"
    assert out["mode"] == "live"
    assert out["counts"]["total"] == 3
    # code_agent-utilities: name convention + corroborating sample -> UNAMBIGUOUS.
    assert out["counts"]["unambiguous"] >= 1
    assert "code_agent-utilities" in out["markdown"]


def test_ownership_report_no_active_engine(monkeypatch):
    tool = _register(monkeypatch, engine=None)
    out = tool(action="ownership_report")
    assert "not active" in out


def test_ownership_apply_is_always_a_dry_run_preview(monkeypatch):
    monkeypatch.setattr(
        go, "resolve_catalog_client", lambda config=None: _small_fixture_client()
    )
    tool = _register(monkeypatch)

    out = json.loads(tool(action="ownership_apply"))
    assert out["action"] == "ownership_apply"
    assert out["mode"] == "dry_run"
    # code_agent-utilities is UNAMBIGUOUS + uncovered -> exactly one planned grant pair
    # (Read + Write) under the auto-apply-UNAMBIGUOUS / hold-ambiguous program decision;
    # __commons__/tenant__homelab__default are public/already-covered -> no plan entries.
    graphs_in_plan = {entry["graph"] for entry in out["plan"]}
    assert graphs_in_plan == {"code_agent-utilities"}
    assert len(out["preview"]) == len(out["plan"])
    assert all("DRY-RUN" in p["grant_result"] for p in out["preview"])
    assert len(out["rollback"]) == len(out["plan"])
    assert all(op["op"] == "remove_grant" for op in out["rollback"])
    assert "HG-4" in out["hint"]


def test_ownership_apply_engine_unreachable_surfaces_structured_error(monkeypatch):
    def _raise(config=None):
        raise go.EngineUnreachableError("no engine configured")

    monkeypatch.setattr(go, "resolve_catalog_client", _raise)
    tool = _register(monkeypatch)

    out = json.loads(tool(action="ownership_apply"))
    assert "error" in out


# ---------------------------------------------------------------------------
# policy_status
# ---------------------------------------------------------------------------


def test_policy_status_needs_no_engine(monkeypatch):
    """policy_status is pure config introspection — must work with no active
    engine at all (unlike every other graph_governance action)."""
    tool = _register(monkeypatch, engine=None)
    out = json.loads(tool(action="policy_status"))
    assert out["action"] == "policy_status"
    assert out["tool_guard_mode"] in ("on", "strict")
    assert out["sensitive_tool_pattern_count"] > 0
    assert out["permission_policy"]["default_verdict"] == "deny"
    assert out["kg_audit_sink"]["enabled_by_default"] is True


def test_unknown_action(monkeypatch):
    tool = _register(monkeypatch)
    out = json.loads(tool(action="verify_action"))  # missing required `kind`
    assert "error" in out
    bogus = tool(action="bogus-action")
    assert "Unknown graph_governance action" in bogus
