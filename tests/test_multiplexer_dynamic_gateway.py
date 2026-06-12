"""Tests for the dynamic tool gateway (CONCEPT:ECO-4.36).

Covers the lazy-mount refactor, KG-backed discovery + (server, tool) mapping,
the load/unload resolution logic, and the FastMCP meta-tool wiring. Children and
the knowledge-graph are mocked so these run with no processes or live engine.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

from agent_utilities.mcp.multiplexer import (
    MCPMultiplexer,
    _register_meta_tools,
    get_server_prefix,
)

CNT = "container-manager-mcp"
CNT_TOOL = "cm_container_operations"
CNT_PREFIXED = "cnt__cm_container_operations"


def _write_config(tmp_path, servers: dict) -> object:
    path = tmp_path / "mcp_config.json"
    path.write_text(json.dumps({"mcpServers": servers}), encoding="utf-8")
    return path


def _fake_tool(name: str, description: str = "", schema: dict | None = None):
    tool = MagicMock()
    tool.name = name
    tool.description = description
    tool.inputSchema = schema if schema is not None else {}
    return tool


def _mux_with_children(tmp_path, tool_map: dict[str, list[tuple[str, str]]]):
    """Build a mux whose ``_start_child`` yields the given ``server -> [(tool,
    desc)]`` map instead of spawning real processes."""
    servers = {name: {"command": "python", "args": ["-m", name]} for name in tool_map}
    servers["mcp-multiplexer"] = {"command": "self"}  # must be excluded
    servers["off"] = {"command": "python", "disabled": True}  # must be excluded
    mux = MCPMultiplexer(_write_config(tmp_path, servers))

    async def fake_start_child(server_name, cfg):
        tools = [_fake_tool(n, d) for n, d in tool_map.get(server_name, [])]
        session = AsyncMock()
        return server_name, session, tools, cfg

    mux._start_child = AsyncMock(side_effect=fake_start_child)  # type: ignore[method-assign]
    return mux


# --------------------------------------------------------------------------- #
# Catalog + lazy mount
# --------------------------------------------------------------------------- #


def test_load_catalog_excludes_self_and_disabled(tmp_path):
    mux = _mux_with_children(tmp_path, {CNT: [(CNT_TOOL, "containers")]})
    catalog = mux.load_catalog()
    assert CNT in catalog
    assert "mcp-multiplexer" not in catalog
    assert "off" not in catalog
    # idempotent / cached
    assert mux.load_catalog() is catalog


async def test_mount_child_lazy_and_idempotent(tmp_path):
    mux = _mux_with_children(tmp_path, {CNT: [(CNT_TOOL, "containers")]})

    tools = await mux.mount_child(CNT)
    assert [t.name for t in tools] == [CNT_PREFIXED]
    assert CNT in mux.children
    assert mux.tool_to_server[CNT_PREFIXED] == (CNT, CNT_TOOL)
    assert CNT_PREFIXED in {t.name for t in mux.aggregated_tools}
    assert mux._start_child.await_count == 1

    # Second mount must not re-spawn the child.
    again = await mux.mount_child(CNT)
    assert [t.name for t in again] == [CNT_PREFIXED]
    assert mux._start_child.await_count == 1


async def test_mount_child_unknown_server(tmp_path):
    mux = _mux_with_children(tmp_path, {CNT: [(CNT_TOOL, "containers")]})
    assert await mux.mount_child("does-not-exist") == []
    assert mux._start_child.await_count == 0


async def test_start_children_eager_unchanged(tmp_path):
    mux = _mux_with_children(
        tmp_path,
        {CNT: [(CNT_TOOL, "containers")], "graph-os": [("graph_query", "query")]},
    )
    await mux.start_children()
    names = {t.name for t in mux.aggregated_tools}
    assert CNT_PREFIXED in names
    assert "kg__graph_query" in names
    assert mux._start_child.await_count == 2


# --------------------------------------------------------------------------- #
# Discovery + mapping
# --------------------------------------------------------------------------- #


def _kg_responder(index_rows, search_hits=None):
    async def _call(bare_tool, arguments):
        if bare_tool == "graph_query":
            return index_rows
        if bare_tool == "graph_search":
            return search_hits or []
        return None

    return AsyncMock(side_effect=_call)


async def test_discover_tools_ranks_and_maps(tmp_path):
    mux = _mux_with_children(tmp_path, {CNT: [(CNT_TOOL, "containers")]})
    rows = [
        {"server": CNT, "tool": CNT_TOOL, "description": "manage docker containers"},
        {"server": CNT, "tool": "cm_image_operations", "description": "build images"},
        {
            "server": "ghost-mcp",
            "tool": "ghost_op",
            "description": "manage docker swarm",
        },
    ]
    mux._kg_call = _kg_responder(  # type: ignore[method-assign]
        rows, search_hits=[{"name": CNT_TOOL, "score": 5.0}]
    )

    results = await mux.discover_tools("manage docker containers", top_k=5)
    assert results, "expected ranked results"
    top = results[0]
    assert top["tool"] == CNT_TOOL
    assert top["prefixed_name"] == CNT_PREFIXED
    assert top["server"] == CNT
    assert top["mountable"] is True
    assert top["mounted"] is False

    # A tool whose server isn't in this multiplexer's config is flagged unmountable.
    ghost = [r for r in results if r["server"] == "ghost-mcp"]
    assert ghost and ghost[0]["mountable"] is False


async def test_discover_tools_server_level_fallback_when_kg_cold(tmp_path):
    mux = _mux_with_children(tmp_path, {CNT: [(CNT_TOOL, "containers")]})
    # KG returns nothing for every call (cold / unavailable).
    mux._kg_call = AsyncMock(return_value=None)  # type: ignore[method-assign]

    results = await mux.discover_tools("anything", top_k=5)
    assert results
    assert all(r["tool"] == "*" for r in results)
    assert CNT in {r["server"] for r in results}


async def test_discover_marks_mounted(tmp_path):
    mux = _mux_with_children(tmp_path, {CNT: [(CNT_TOOL, "containers")]})
    rows = [{"server": CNT, "tool": CNT_TOOL, "description": "docker"}]
    mux._kg_call = _kg_responder(rows)  # type: ignore[method-assign]
    await mux.mount_child(CNT)
    mux._exposed.add(CNT_PREFIXED)

    results = await mux.discover_tools("docker", top_k=5)
    assert results[0]["mounted"] is True


# --------------------------------------------------------------------------- #
# load/unload resolution
# --------------------------------------------------------------------------- #


async def test_resolve_and_mount_by_server(tmp_path):
    mux = _mux_with_children(
        tmp_path, {CNT: [(CNT_TOOL, "a"), ("cm_image_operations", "b")]}
    )
    servers, to_expose = await mux.resolve_and_mount(servers=[CNT])
    assert servers == [CNT]
    assert set(to_expose) == {CNT_PREFIXED, "cnt__cm_image_operations"}
    assert CNT in mux.children


async def test_resolve_and_mount_by_prefixed_tool(tmp_path):
    mux = _mux_with_children(
        tmp_path, {CNT: [(CNT_TOOL, "a"), ("cm_image_operations", "b")]}
    )
    # Request a single tool by its prefixed name (server derived from prefix).
    servers, to_expose = await mux.resolve_and_mount(tools=[CNT_PREFIXED])
    assert servers == [CNT]
    assert to_expose == [CNT_PREFIXED]  # only the requested subset


async def test_resolve_and_mount_skips_already_exposed(tmp_path):
    mux = _mux_with_children(tmp_path, {CNT: [(CNT_TOOL, "a")]})
    await mux.resolve_and_mount(servers=[CNT])
    mux._exposed.add(CNT_PREFIXED)
    _servers, to_expose = await mux.resolve_and_mount(servers=[CNT])
    assert to_expose == []


def test_forget_tool(tmp_path):
    mux = _mux_with_children(tmp_path, {CNT: [(CNT_TOOL, "a")]})
    mux.tool_to_server[CNT_PREFIXED] = (CNT, CNT_TOOL)
    mux.aggregated_tools.append(_fake_tool(CNT_PREFIXED))
    mux._exposed.add(CNT_PREFIXED)

    owner = mux.forget_tool(CNT_PREFIXED)
    assert owner == CNT
    assert CNT_PREFIXED not in mux.tool_to_server
    assert CNT_PREFIXED not in mux._exposed
    assert CNT_PREFIXED not in {t.name for t in mux.aggregated_tools}


def test_server_for_prefixed_reverses_prefix(tmp_path):
    mux = _mux_with_children(tmp_path, {CNT: [(CNT_TOOL, "a")]})
    # Not yet mounted: resolved purely from the prefix → catalog.
    assert mux._server_for_prefixed(CNT_PREFIXED) == CNT


# --------------------------------------------------------------------------- #
# FastMCP meta-tool wiring (end-to-end through a real FastMCP instance)
# --------------------------------------------------------------------------- #


async def _registered_tool_names(mcp) -> set[str]:
    tools = await mcp.list_tools()
    return {t.name for t in tools}


async def test_meta_tools_registered_and_load_exposes(tmp_path):
    from fastmcp import FastMCP

    mux = _mux_with_children(tmp_path, {CNT: [(CNT_TOOL, "manage containers")]})
    mcp = FastMCP("test-mux")
    _register_meta_tools(mcp, mux)

    names = await _registered_tool_names(mcp)
    assert {"find_tools", "load_tools", "unload_tools", "multiplexer_status"} <= names
    # Nothing from the child is exposed yet.
    assert CNT_PREFIXED not in names

    # Invoke load_tools' underlying fn to mount + expose the container tool.
    load = await mcp.get_tool("load_tools")
    await load.fn(servers=[CNT])

    names_after = await _registered_tool_names(mcp)
    assert CNT_PREFIXED in names_after
    assert CNT_PREFIXED in mux._exposed

    # unload retracts it again.
    unload = await mcp.get_tool("unload_tools")
    await unload.fn(tools=[CNT_PREFIXED])
    names_final = await _registered_tool_names(mcp)
    assert CNT_PREFIXED not in names_final
    assert CNT_PREFIXED not in mux._exposed


async def test_find_tools_meta_returns_structured(tmp_path):
    from fastmcp import FastMCP

    mux = _mux_with_children(tmp_path, {CNT: [(CNT_TOOL, "containers")]})
    rows = [
        {"server": CNT, "tool": CNT_TOOL, "description": "manage docker containers"}
    ]
    mux._kg_call = _kg_responder(rows)  # type: ignore[method-assign]

    mcp = FastMCP("test-mux")
    _register_meta_tools(mcp, mux)
    find = await mcp.get_tool("find_tools")
    result = await find.fn(query="docker containers", top_k=3)

    payload = result.structured_content
    assert payload["count"] >= 1
    assert payload["results"][0]["prefixed_name"] == CNT_PREFIXED


def test_prefix_sanity():
    # Guards the (server -> prefix) assumption the rest of the suite relies on.
    assert get_server_prefix(CNT) == "cnt"
