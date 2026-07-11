"""End-to-end: building the REAL graph-os server under MCP_TOOL_MODE=intent
(CONCEPT:AU-ECO.mcp.intent-surface-condensed-collapse) registers the six intent verbs AND keeps
every granular condensed tool fully registered (REGISTERED_TOOLS/REST are the
SAME backing surface — only the FastMCP tool list default view shrinks).

``bootstrap=False`` skips the engine/daemon startup thread (as
``ensure_tools_registered`` already does for the API gateway), so this needs no
live engine — it only exercises tool *registration*, not execution.
"""

from __future__ import annotations

from agent_utilities.mcp import kg_server
from agent_utilities.mcp.verbose_tools import gated_tool_names


def test_intent_mode_registers_verbs_and_keeps_the_granular_surface(monkeypatch):
    monkeypatch.setenv("MCP_TOOL_MODE", "intent")

    _args, mcp, _middlewares = kg_server._build_server(bootstrap=False)

    intent_verbs = {"ask", "find", "write", "act", "manage", "why"}
    assert intent_verbs <= set(kg_server.REGISTERED_TOOLS)
    # Nothing lost: the granular condensed surface is STILL fully registered.
    assert "graph_query" in kg_server.REGISTERED_TOOLS
    assert "graph_write" in kg_server.REGISTERED_TOOLS
    assert "nl_query" in kg_server.REGISTERED_TOOLS

    # REST twins wired for every intent verb (generic ACTION_TOOL_ROUTES loop —
    # see test_gateway_mcp_parity.py for the full contract).
    for verb in intent_verbs:
        assert kg_server.ACTION_TOOL_ROUTES[verb] == f"/intent/{verb}"

    # The condensed tools are gated (held back from the default session view);
    # the intent verbs themselves are never gated.
    gated = gated_tool_names(mcp)
    assert "graph_query" in gated
    assert "graph_write" in gated
    assert not (intent_verbs & gated)

    # No verbose 1:1 surface in intent mode (that richness lives in the
    # resolver/engine, not a bigger tool list).
    tool_names = (
        {t.name for t in mcp._tool_manager._tools.values()}
        if hasattr(mcp, "_tool_manager")
        else set()
    )
    assert not any(
        n.startswith("graph_write_") and n != "graph_write" for n in tool_names
    )


def test_intent_verbs_absent_by_default_condensed_mode(monkeypatch):
    """Default MCP_TOOL_MODE=condensed is untouched — no intent verbs, nothing
    gated. (Regression guard: the new surface is opt-in only.)"""
    monkeypatch.delenv("MCP_TOOL_MODE", raising=False)

    _args, mcp, _middlewares = kg_server._build_server(bootstrap=False)

    assert "graph_query" in kg_server.REGISTERED_TOOLS
    assert gated_tool_names(mcp) == set()
