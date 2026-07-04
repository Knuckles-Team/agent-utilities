"""CONCEPT:AU-ORCH.sandbox.graph-sandbox-surface — graph_sandbox operator surface: status/warm/reap on both surfaces."""

from __future__ import annotations

import json

import pytest

from agent_utilities.mcp import kg_server
from agent_utilities.mcp.tools.state_tools import register_state_tools
from agent_utilities.runtime.warm_registry import WarmParentRegistry


class _FakeMCP:
    """Pass-through registrar so register_state_tools populates REGISTERED_TOOLS for a unit test."""

    def tool(self, *a, **k):
        return lambda fn: fn

    def resource(self, *a, **k):
        return lambda fn: fn

    def prompt(self, *a, **k):
        return lambda fn: fn


@pytest.fixture
def sandbox_tool():
    WarmParentRegistry._instance = None  # noqa: SLF001 - test isolation
    register_state_tools(_FakeMCP())
    yield kg_server.REGISTERED_TOOLS["graph_sandbox"]
    WarmParentRegistry.drain_active()
    WarmParentRegistry._instance = None  # noqa: SLF001


def test_registered_on_both_surfaces():
    # REST twin auto-generated from the route table (CONCEPT:AU-ORCH.sandbox.graph-sandbox-surface).
    assert kg_server.ACTION_TOOL_ROUTES.get("graph_sandbox") == "/graph/sandbox"


async def test_status_reports_rungs(sandbox_tool):
    out = json.loads(await sandbox_tool(action="status", rung=""))
    assert out["status"] in ("ok", "warn")
    assert "forkserver" in out["rungs"]
    assert isinstance(out["warm_rungs"], list)
    assert "rewards" in out and "pool" in out


async def test_warm_then_pool_reflects_it(sandbox_tool):
    out = json.loads(await sandbox_tool(action="warm", rung="forkserver"))
    assert out["action"] == "warm" and out["rung"] == "forkserver"
    assert out["already_warm"] is False
    assert out["pool"]["by_kind"].get("forkserver") == 1
    # Warming again is idempotent — reuses the pooled parent.
    again = json.loads(await sandbox_tool(action="warm", rung="forkserver"))
    assert again["already_warm"] is True


async def test_warm_rejects_non_forkable_rung(sandbox_tool):
    out = json.loads(await sandbox_tool(action="warm", rung="local"))
    assert "error" in out and "not a warm-fork rung" in out["error"]


async def test_reap_returns_structure(sandbox_tool):
    out = json.loads(await sandbox_tool(action="reap", rung=""))
    assert out["action"] == "reap"
    assert "reaped_parents" in out and "pool" in out


async def test_unknown_action_errors(sandbox_tool):
    out = json.loads(await sandbox_tool(action="bogus", rung=""))
    assert out["error"] == "unknown action 'bogus'"
