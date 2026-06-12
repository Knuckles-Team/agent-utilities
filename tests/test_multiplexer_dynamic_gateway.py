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


def _seed_probe(mux, mapping):
    """Pre-populate the self-catalog probe cache. Values are either a list of
    ``(tool, description)`` (reachable) or an error string (unreachable)."""
    for server, val in mapping.items():
        if isinstance(val, str):
            mux._probe_cache[server] = {"tools": [], "error": val}
        else:
            mux._probe_cache[server] = {
                "tools": [
                    {"name": n, "description": d, "inputSchema": {}} for n, d in val
                ],
                "error": None,
            }


def _fake_session(tools):
    sess = AsyncMock()
    res = MagicMock()
    res.tools = [_fake_tool(n, d) for n, d in tools]
    sess.list_tools = AsyncMock(return_value=res)
    return sess


async def test_discover_tools_ranks_and_maps(tmp_path):
    mux = _mux_with_children(tmp_path, {CNT: [(CNT_TOOL, "containers")]})
    mux._kg_call = AsyncMock(return_value=None)  # type: ignore[method-assign]
    _seed_probe(
        mux,
        {
            CNT: [
                (CNT_TOOL, "manage docker containers"),
                ("cm_image_operations", "build images"),
            ],
            "ghost-mcp": [("ghost_op", "manage docker swarm")],  # not in catalog
        },
    )

    discovery = await mux.discover_tools("manage docker containers", top_k=5)
    results = discovery["results"]
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


async def test_discover_reports_unavailable_servers(tmp_path):
    mux = _mux_with_children(tmp_path, {CNT: [(CNT_TOOL, "c")], "leanix-mcp": []})
    mux._kg_call = AsyncMock(return_value=None)  # type: ignore[method-assign]
    _seed_probe(
        mux, {CNT: [(CNT_TOOL, "manage docker")], "leanix-mcp": "timeout after 15s"}
    )

    discovery = await mux.discover_tools("docker", top_k=5)
    assert "leanix-mcp" in discovery["unavailable"]
    assert "timeout" in discovery["unavailable"]["leanix-mcp"]
    assert all(r["server"] != "leanix-mcp" for r in discovery["results"])


async def test_discover_server_level_fallback_on_no_match(tmp_path):
    mux = _mux_with_children(tmp_path, {CNT: [(CNT_TOOL, "x")]})
    mux._kg_call = AsyncMock(return_value=None)  # type: ignore[method-assign]
    _seed_probe(mux, {CNT: [(CNT_TOOL, "containers")]})

    # Query matches nothing, but the server probed fine → list servers to load.
    discovery = await mux.discover_tools("zzznomatch", top_k=5)
    results = discovery["results"]
    assert results and all(r["tool"] == "*" for r in results)


async def test_discover_all_unreachable_yields_empty_results(tmp_path):
    mux = _mux_with_children(tmp_path, {"leanix-mcp": []})
    mux._kg_call = AsyncMock(return_value=None)  # type: ignore[method-assign]
    _seed_probe(mux, {"leanix-mcp": "timeout after 15s"})

    discovery = await mux.discover_tools("anything", top_k=5)
    assert discovery["results"] == []
    assert "leanix-mcp" in discovery["unavailable"]


async def test_discover_marks_mounted(tmp_path):
    mux = _mux_with_children(tmp_path, {CNT: [(CNT_TOOL, "containers")]})
    mux._kg_call = AsyncMock(return_value=None)  # type: ignore[method-assign]
    await mux.mount_child(CNT)  # mounted server is probed from live tools
    mux._exposed.add(CNT_PREFIXED)

    discovery = await mux.discover_tools("containers", top_k=5)
    assert discovery["results"][0]["mounted"] is True


# --------------------------------------------------------------------------- #
# Self-catalog probe
# --------------------------------------------------------------------------- #


async def test_probe_server_success_and_caches(tmp_path):
    mux = _mux_with_children(tmp_path, {CNT: [(CNT_TOOL, "c")]})

    async def _open(server, cfg, stack):
        return _fake_session([(CNT_TOOL, "manage containers")])

    mux._open_one_session = AsyncMock(side_effect=_open)  # type: ignore[method-assign]
    info = await mux.probe_server(CNT)
    assert info["error"] is None
    assert info["tools"][0]["name"] == CNT_TOOL
    assert mux._probe_cache[CNT] is info  # cached


async def test_probe_server_records_unreachable(tmp_path):
    mux = _mux_with_children(tmp_path, {CNT: [(CNT_TOOL, "c")]})

    async def _boom(server, cfg, stack):
        raise OSError("connection refused")

    mux._open_one_session = AsyncMock(side_effect=_boom)  # type: ignore[method-assign]
    info = await mux.probe_server(CNT, timeout=1)
    assert info["tools"] == []
    assert "connection refused" in info["error"]


async def test_probe_server_uses_live_tools_when_mounted(tmp_path):
    mux = _mux_with_children(tmp_path, {CNT: [(CNT_TOOL, "containers")]})
    await mux.mount_child(CNT)
    # Should NOT reconnect for an already-mounted child.
    mux._open_one_session = AsyncMock(  # type: ignore[method-assign]
        side_effect=AssertionError("must not reconnect")
    )
    info = await mux.probe_server(CNT)
    assert info["error"] is None
    assert info["tools"][0]["name"] == CNT_TOOL


# --------------------------------------------------------------------------- #
# Catalog browse
# --------------------------------------------------------------------------- #


async def test_list_catalog_all_servers(tmp_path):
    mux = _mux_with_children(tmp_path, {CNT: [(CNT_TOOL, "c")], "leanix-mcp": []})
    mux._kg_call = AsyncMock(return_value=None)  # type: ignore[method-assign]
    _seed_probe(
        mux,
        {
            CNT: [(CNT_TOOL, "containers"), ("cm_image_operations", "images")],
            "leanix-mcp": "timeout after 15s",
        },
    )

    cat = await mux.list_catalog()
    assert cat["total_servers"] == 2
    assert cat["total_tools"] == 2  # only the reachable server's tools
    assert "leanix-mcp" in cat["unavailable"]

    by_name = {s["server"]: s for s in cat["servers"]}
    assert by_name[CNT]["tool_count"] == 2
    assert by_name[CNT]["available"] is True
    assert CNT_PREFIXED in by_name[CNT]["tools"]  # prefixed names in all-view
    assert by_name["leanix-mcp"]["available"] is False
    assert "error" in by_name["leanix-mcp"]


async def test_list_catalog_single_server_drilldown(tmp_path):
    mux = _mux_with_children(tmp_path, {CNT: [(CNT_TOOL, "c")]})
    mux._kg_call = AsyncMock(return_value=None)  # type: ignore[method-assign]
    _seed_probe(mux, {CNT: [(CNT_TOOL, "manage containers")]})

    cat = await mux.list_catalog(server=CNT)
    assert cat["server"] == CNT
    assert cat["available"] is True
    tool = cat["tools"][0]
    assert tool["prefixed_name"] == CNT_PREFIXED
    assert tool["tool"] == CNT_TOOL
    assert tool["description"] == "manage containers"


async def test_list_catalog_unknown_server(tmp_path):
    mux = _mux_with_children(tmp_path, {CNT: [(CNT_TOOL, "c")]})
    mux._kg_call = AsyncMock(return_value=None)  # type: ignore[method-assign]
    _seed_probe(mux, {CNT: [(CNT_TOOL, "c")]})
    cat = await mux.list_catalog(server="does-not-exist")
    assert "error" in cat


async def test_list_catalog_meta_tool_registered(tmp_path):
    from fastmcp import FastMCP

    mux = _mux_with_children(tmp_path, {CNT: [(CNT_TOOL, "containers")]})
    mux._kg_call = AsyncMock(return_value=None)  # type: ignore[method-assign]
    _seed_probe(mux, {CNT: [(CNT_TOOL, "containers")]})

    mcp = FastMCP("test-mux")
    _register_meta_tools(mcp, mux)
    names = {t.name for t in await mcp.list_tools()}
    assert "list_catalog" in names

    tool = await mcp.get_tool("list_catalog")
    result = await tool.fn()
    assert result.structured_content["total_servers"] == 1


# --------------------------------------------------------------------------- #
# load/unload resolution
# --------------------------------------------------------------------------- #


async def test_resolve_and_mount_by_server(tmp_path):
    mux = _mux_with_children(
        tmp_path, {CNT: [(CNT_TOOL, "a"), ("cm_image_operations", "b")]}
    )
    servers, to_expose, failed = await mux.resolve_and_mount(servers=[CNT])
    assert servers == [CNT]
    assert set(to_expose) == {CNT_PREFIXED, "cnt__cm_image_operations"}
    assert failed == {}
    assert CNT in mux.children


async def test_resolve_and_mount_by_prefixed_tool(tmp_path):
    mux = _mux_with_children(
        tmp_path, {CNT: [(CNT_TOOL, "a"), ("cm_image_operations", "b")]}
    )
    # Request a single tool by its prefixed name (server derived from prefix).
    servers, to_expose, failed = await mux.resolve_and_mount(tools=[CNT_PREFIXED])
    assert servers == [CNT]
    assert to_expose == [CNT_PREFIXED]  # only the requested subset
    assert failed == {}


async def test_resolve_and_mount_skips_already_exposed(tmp_path):
    mux = _mux_with_children(tmp_path, {CNT: [(CNT_TOOL, "a")]})
    await mux.resolve_and_mount(servers=[CNT])
    mux._exposed.add(CNT_PREFIXED)
    _servers, to_expose, _failed = await mux.resolve_and_mount(servers=[CNT])
    assert to_expose == []


async def test_resolve_and_mount_reports_failed(tmp_path):
    mux = _mux_with_children(tmp_path, {CNT: [(CNT_TOOL, "a")], "leanix-mcp": []})
    good = mux._start_child.side_effect

    async def _fail_leanix(server_name, cfg):
        if server_name == "leanix-mcp":
            return None  # mount fails
        return await good(server_name, cfg)

    mux._start_child = AsyncMock(side_effect=_fail_leanix)  # type: ignore[method-assign]
    _seed_probe(mux, {"leanix-mcp": "timeout after 15s"})  # reason for the failure

    mounted, to_expose, failed = await mux.resolve_and_mount(
        servers=[CNT, "leanix-mcp"]
    )
    assert mounted == [CNT]
    assert CNT_PREFIXED in to_expose
    assert "leanix-mcp" in failed and "timeout" in failed["leanix-mcp"]


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
    mux._kg_call = AsyncMock(return_value=None)  # type: ignore[method-assign]
    _seed_probe(mux, {CNT: [(CNT_TOOL, "manage docker containers")]})

    mcp = FastMCP("test-mux")
    _register_meta_tools(mcp, mux)
    find = await mcp.get_tool("find_tools")
    result = await find.fn(query="docker containers", top_k=3)

    payload = result.structured_content
    assert payload["count"] >= 1
    assert payload["results"][0]["prefixed_name"] == CNT_PREFIXED
    assert "unavailable" in payload


def test_prefix_sanity():
    # Guards the (server -> prefix) assumption the rest of the suite relies on.
    assert get_server_prefix(CNT) == "cnt"
