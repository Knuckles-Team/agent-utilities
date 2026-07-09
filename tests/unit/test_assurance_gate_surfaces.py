"""Two-surfaces test for the assurance-gate payload check (CONCEPT:AU-OS.governance.assurance-state-machine-verifier).

Exercises the MCP ``graph_orchestrate action=verify_action`` tool and its REST
twin ``POST /fleet/actions/verify`` end-to-end through the REAL
``_execute_tool`` dispatch core (not a mock of the action), proving both
surfaces reach the same :meth:`ActionPolicy.evaluate` verdict.
"""

from __future__ import annotations

import json

import pytest

from agent_utilities.gateway import fleet

pytestmark = pytest.mark.concept("AU-OS.governance.assurance-state-machine-verifier")


class _Req:
    def __init__(self, body=None):
        self._body = body or {}

    async def json(self):
        return self._body


async def _payload(resp):
    return json.loads(resp.body)


class _FakeMCP:
    """Captures the tool coroutines ``register_analysis_tools`` registers.

    Mirrors the pattern in ``tests/unit/test_memory_weights_distillation.py`` —
    a minimal FastMCP double so we exercise the REAL registered tool coroutine
    (via ``kg_server._execute_tool``) without booting the whole MCP server.
    """

    def __init__(self) -> None:
        self.tools: dict = {}

    def tool(self, *, name: str, description: str = "", tags=None):
        def _decorator(fn):
            self.tools[name] = fn
            return fn

        return _decorator


@pytest.mark.asyncio
async def test_mcp_verify_action_allows_a_valid_payload(monkeypatch):
    import agent_utilities.mcp.kg_server as kg
    from agent_utilities.mcp.tools.analysis_tools import register_analysis_tools
    from tests.unit.fleet_autonomy_fakes import FakeEngine

    register_analysis_tools(_FakeMCP())
    monkeypatch.setattr(kg, "_get_engine", lambda: FakeEngine())

    res = await kg._execute_tool(
        "graph_orchestrate",
        action="verify_action",
        task="diagnose",
        dependencies=json.dumps({"target": "anything"}),
    )
    data = json.loads(res)
    assert data["allowed"] is True
    assert data["decision"] == "allow"
    assert data["invariant"] == ""


@pytest.mark.asyncio
async def test_mcp_verify_action_denies_out_of_role_kind(monkeypatch):
    import agent_utilities.mcp.kg_server as kg
    from agent_utilities.mcp.tools.analysis_tools import register_analysis_tools
    from tests.unit.fleet_autonomy_fakes import FakeEngine

    register_analysis_tools(_FakeMCP())
    monkeypatch.setattr(kg, "_get_engine", lambda: FakeEngine())

    res = await kg._execute_tool(
        "graph_orchestrate",
        action="verify_action",
        task="secret.delete",
        dependencies=json.dumps(
            {"target": "db-creds", "params": {"path": "apps/x"}, "source": "reconciler"}
        ),
    )
    data = json.loads(res)
    assert data["allowed"] is False
    assert data["decision"] == "deny"
    assert data["invariant"] == "role"


@pytest.mark.asyncio
async def test_rest_twin_verify_action_matches_mcp(monkeypatch):
    """The REST route dispatches into the SAME core as the MCP tool — same verdict."""
    import agent_utilities.mcp.kg_server as kg
    from agent_utilities.mcp.tools.analysis_tools import register_analysis_tools
    from tests.unit.fleet_autonomy_fakes import FakeEngine

    register_analysis_tools(_FakeMCP())
    monkeypatch.setattr(kg, "_get_engine", lambda: FakeEngine())

    resp = await fleet.fleet_verify_action(
        _Req(
            {
                "kind": "scale_service",
                "target": "caddy-mcp",
                "source": "reconciler",
            }
        )
    )
    data = await _payload(resp)
    assert data["status"] == "success"
    result = data["result"]
    assert result["allowed"] is False
    assert result["invariant"] == "schema"
    assert "replicas" in result["reason"]


@pytest.mark.asyncio
async def test_rest_twin_requires_kind():
    resp = await fleet.fleet_verify_action(_Req({}))
    assert resp.status_code == 400
