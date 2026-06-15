"""E2E live-path: a ``create_agent`` pydantic-ai agent is wired to a REAL
``agent-packages/agents/*`` MCP server toolset, and that toolset's tool both
lists and executes.

CONCEPT:OS-5.0 (Agent Creation) — Wire-First proof that:

1. a toolset built from a real fleet MCP server (``repository-manager``) is
   **assigned to the agent** by the ``create_agent`` factory (``agent.toolsets``
   carries the ``FastMCPToolset``), and
2. a tool **designated from that agents/* server** (``rm_workspace``) is
   **executed** — listed and called *through the agent's own attached toolset
   client*, not a side channel — returning a real result.

Fully in-process: the repository-manager FastMCP server is built via
``get_mcp_instance()`` (import is side-effect-free; ``mcp.run`` only fires under
``__main__``) and handed to the agent as a pre-built toolset, so no subprocess
is spawned and no network transport is used. This is the same in-memory FastMCP
client pattern the connector live-path test (``test_mcp_tool_connector_live_path``)
uses.
"""

from __future__ import annotations

import sys

import pytest

# This live-path test deliberately drives a REAL agents/* MCP server. Skip
# cleanly when the fleet package isn't installed (keeps agent-utilities CI
# self-contained while still proving the contract when the fleet is present).
pytest.importorskip(
    "repository_manager",
    reason="needs the repository-manager agents/* package installed",
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.asyncio,
    pytest.mark.concept("OS-5.0"),
]


def _build_repository_manager_server():
    """The real repository-manager FastMCP server, built in-process.

    ``create_mcp_server`` parses argv via argparse, so callers must hand it a
    clean ``sys.argv`` (done by the fixture below).
    """
    from repository_manager.mcp_server import get_mcp_instance

    mcp, _args, _middlewares, _tags = get_mcp_instance()
    return mcp


def _find_fastmcp_toolset(toolsets):
    """Return the FastMCPToolset among ``toolsets`` (the rm wrapper), or None."""
    for ts in toolsets:
        if type(ts).__name__ == "FastMCPToolset" and getattr(ts, "client", None):
            return ts
    return None


@pytest.fixture
def clean_argv(monkeypatch):
    # repository-manager's create_mcp_server reads argv; keep pytest's args out.
    monkeypatch.setattr(sys, "argv", ["repository-manager"])
    return None


async def test_repository_manager_toolset_assigned_and_tool_executes(
    clean_argv, monkeypatch
):
    rm_server = _build_repository_manager_server()

    # The hermetic test env sets AGENT_UTILITIES_TESTING=true, which flips the
    # factory into VALIDATION_MODE (skips every live MCP attachment). Force it off
    # so the real toolset-wiring path under test actually runs. The factory reads
    # this module-level constant directly, so patch it there.
    from agent_utilities.agent import factory as factory_mod

    monkeypatch.setattr(factory_mod, "DEFAULT_VALIDATION_MODE", False)

    # --- 1) Build the agent and let the factory assign the MCP toolset. --------
    create_agent = factory_mod.create_agent

    agent, initialized_toolsets = create_agent(
        name="e2e-repo-agent",
        mcp_url=None,
        mcp_config=None,
        mcp_toolsets=[rm_server],
        enable_skills=False,
        enable_universal_tools=False,
        auto_graph_trace=False,
    )

    # The factory wrapped the FastMCP server in a FastMCPToolset and returned it.
    assert initialized_toolsets, "factory returned no initialized toolsets"
    factory_ts = _find_fastmcp_toolset(initialized_toolsets)
    assert factory_ts is not None, (
        "repository-manager server was not wrapped as a FastMCPToolset: "
        f"{[type(t).__name__ for t in initialized_toolsets]}"
    )

    # ASSIGNED: the Agent object itself carries that toolset.
    agent_toolsets = list(getattr(agent, "toolsets", []) or [])
    assert factory_ts in agent_toolsets, (
        "the repository-manager toolset is not attached to agent.toolsets: "
        f"{[type(t).__name__ for t in agent_toolsets]}"
    )

    # --- 2) DESIGNATED + EXECUTED through the agent's OWN attached toolset. -----
    # factory_ts.client is the fastmcp Client wrapping the rm server.
    async with factory_ts.client as client:
        tool_names = {t.name for t in await client.list_tools()}
        assert "rm_workspace" in tool_names, (
            f"repository-manager tools not designated to the toolset: {sorted(tool_names)}"
        )

        # Execute a read-only tool and assert a real result comes back.
        result = await client.call_tool("rm_workspace", {"action": "list"})

    assert result is not None
    assert getattr(result, "is_error", False) is False, (
        f"rm_workspace execution errored: {getattr(result, 'content', result)}"
    )
    # A structured/text payload proves the tool actually ran and returned data.
    payload = getattr(result, "data", None)
    if payload is None:
        payload = getattr(result, "structured_content", None)
    if payload is None:
        payload = getattr(result, "content", None)
    assert payload not in (None, "", [], {}), "rm_workspace returned an empty result"
