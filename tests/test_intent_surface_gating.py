"""Seam 8 (CONCEPT:AU-ECO.mcp.intent-surface-condensed-collapse / CONCEPT:AU-ECO.mcp.intent-surface-tool-lifecycle) —
the condensed-surface local-tool gate + the load->use->unload lifecycle.

Under ``MCP_TOOL_MODE=intent`` graph-os's OWN granular tools stay fully
registered (REST/_execute_tool/REGISTERED_TOOLS all unaffected — see
``tests/unit/test_intent_surface.py`` and ``test_gateway_mcp_parity.py``) but
are held back from a session's default tool list. This module proves the
mechanism that holds them back and the ``load_tools``/``unload_tools``
escape hatch that reveals/retracts them — reusing the SAME session-visibility
infra the fleet multiplexer already uses for external servers, extended to the
host's own tools (``MCPMultiplexer._local_gated``).
"""

from __future__ import annotations

import json

import pytest
from fastmcp import Client, FastMCP

from agent_utilities.mcp.multiplexer import (
    MCPMultiplexer,
    SessionVisibilityMiddleware,
    _register_meta_tools,
)


def _write_config(tmp_path) -> object:
    path = tmp_path / "mcp_config.json"
    path.write_text(json.dumps({"mcpServers": {}}), encoding="utf-8")
    return path


def _mux_with_local_gated(tmp_path, mcp: FastMCP, gated_tags: dict[str, set[str]]):
    """A mux with no fleet children, wired the way ``attach_fleet_loader`` wires
    the host's own gated tools (``_local_gated``) — register real FastMCP tools
    on ``mcp`` first (as ``register_tool_surface`` would), then seed the gate."""
    mux = MCPMultiplexer(_write_config(tmp_path))
    mux._skip_servers = {"mcp-multiplexer", "graph-os"}
    for name, tags in gated_tags.items():

        async def _fn(name=name) -> str:  # default-bind for closure safety
            return f"ran {name}"

        mcp.tool(name=name, tags=set(tags))(_fn)
    mux._local_gated = set(gated_tags)
    _register_meta_tools(mcp, mux)
    mux._global_visible = {
        "find_tools",
        "list_catalog",
        "load_tools",
        "unload_tools",
        "multiplexer_status",
    }
    mcp.add_middleware(SessionVisibilityMiddleware(mux, mcp))
    return mux


async def _visible_names(mcp: FastMCP) -> set[str]:
    return {t.name for t in await mcp.list_tools()}


@pytest.mark.asyncio
async def test_gated_local_tool_hidden_by_default(tmp_path):
    mcp = FastMCP("graph-os-test")
    await_names_before = {t.name for t in await mcp.list_tools()}
    _mux_with_local_gated(
        tmp_path, mcp, {"graph_query": {"query", "granular", "gated"}}
    )

    names = await _visible_names(mcp)
    assert "graph_query" not in names
    assert {"load_tools", "unload_tools", "find_tools"} <= names
    assert names - await_names_before  # meta-tools were newly added


@pytest.mark.asyncio
async def test_load_tools_reveals_a_gated_local_tool(tmp_path):
    mcp = FastMCP("graph-os-test")
    _mux_with_local_gated(
        tmp_path, mcp, {"graph_query": {"query", "granular", "gated"}}
    )

    async with Client(mcp) as client:
        before = {t.name for t in await client.list_tools()}
        assert "graph_query" not in before

        load_result = await client.call_tool("load_tools", {"tools": ["graph_query"]})
        assert "graph_query" in load_result.structured_content["newly_exposed"]

        after = {t.name for t in await client.list_tools()}
        assert "graph_query" in after

        result = await client.call_tool("graph_query", {})
        assert "ran graph_query" in str(result.content)


@pytest.mark.asyncio
async def test_unload_by_whole_server_name_retracts_every_local_gated_tool(tmp_path):
    mcp = FastMCP("graph-os-test")
    mux = _mux_with_local_gated(
        tmp_path,
        mcp,
        {"graph_query": {"query", "gated"}, "graph_write": {"write_ingest", "gated"}},
    )
    load = await mcp.get_tool("load_tools")
    await load.fn(tools=["graph_query", "graph_write"])
    assert {"graph_query", "graph_write"} <= mux.session_loaded("__default__")

    unload = await mcp.get_tool("unload_tools")
    result = await unload.fn(servers=["graph-os"])
    assert set(result.structured_content["unloaded"]) == {"graph_query", "graph_write"}
    assert not ({"graph_query", "graph_write"} & mux.session_loaded("__default__"))

    names = await _visible_names(mcp)
    assert "graph_query" not in names
    assert "graph_write" not in names


@pytest.mark.asyncio
async def test_unload_by_toolset_tag_retracts_only_matching_tools(tmp_path):
    mcp = FastMCP("graph-os-test")
    mux = _mux_with_local_gated(
        tmp_path,
        mcp,
        {"graph_query": {"query", "gated"}, "graph_write": {"write_ingest", "gated"}},
    )
    load = await mcp.get_tool("load_tools")
    await load.fn(tools=["graph_query", "graph_write"])

    unload = await mcp.get_tool("unload_tools")
    result = await unload.fn(toolsets=["query"])
    assert result.structured_content["unloaded"] == ["graph_query"]
    loaded = mux.session_loaded("__default__")
    assert "graph_query" not in loaded
    assert "graph_write" in loaded  # untouched — different toolset tag


@pytest.mark.asyncio
async def test_auto_unload_retracts_the_tool_after_its_next_call(tmp_path):
    """Responsible tool usage (CONCEPT:AU-ECO.mcp.intent-surface-tool-lifecycle): a tool
    loaded with auto_unload=True is a ONE-SHOT — it vanishes from the session's
    tool list right after the call that used it, so a long session doesn't
    accumulate schemas for tools it only needed once. Nothing is destroyed —
    load_tools brings it right back."""
    mcp = FastMCP("graph-os-test")
    mux = _mux_with_local_gated(tmp_path, mcp, {"graph_query": {"query", "gated"}})

    async with Client(mcp) as client:
        await client.call_tool(
            "load_tools", {"tools": ["graph_query"], "auto_unload": True}
        )
        tools_after_load = {t.name for t in await client.list_tools()}
        assert "graph_query" in tools_after_load

        await client.call_tool("graph_query", {})

        tools_after_use = {t.name for t in await client.list_tools()}
        assert "graph_query" not in tools_after_use

    # It is NOT gone forever — load_tools brings it straight back.
    assert "graph_query" not in mux.session_loaded("__default__")
    load = await mcp.get_tool("load_tools")
    await load.fn(tools=["graph_query"])
    assert "graph_query" in mux.session_loaded("__default__")


@pytest.mark.asyncio
async def test_manage_verb_lifecycle_action_loads_and_unloads(tmp_path):
    """The `manage` intent verb's action='unload'/'load' shortcut (CONCEPT:AU-ECO.mcp.intent-surface-tool-lifecycle)
    reaches the SAME lifecycle core as the load_tools/unload_tools meta-tools."""
    from agent_utilities.mcp.tools import intent_tools

    mcp = FastMCP("graph-os-test")
    mux = _mux_with_local_gated(tmp_path, mcp, {"graph_query": {"query", "gated"}})
    mcp._fleet_mux = mux

    loaded = await intent_tools._manage_lifecycle(
        mcp, {"action": "load", "tools": ["graph_query"]}
    )
    assert "graph_query" in mux.session_loaded("__default__")
    assert "graph_query" in loaded["newly_exposed"]

    unloaded = await intent_tools._manage_lifecycle(
        mcp, {"action": "unload", "tools": ["graph_query"]}
    )
    assert unloaded["unloaded"] == ["graph_query"]
    assert "graph_query" not in mux.session_loaded("__default__")

    # Not a lifecycle action -> returns None so the caller falls through to the
    # normal capability resolver.
    assert await intent_tools._manage_lifecycle(mcp, {}) is None
