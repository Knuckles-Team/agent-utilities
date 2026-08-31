"""Governed single-child MCP runtime refresh regressions."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import mcp.types as mcp_types
import pytest
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError

from agent_utilities.mcp.multiplexer import (
    MCPMultiplexer,
    _register_forwarder,
    _register_meta_tools,
)


def _write_config(tmp_path) -> object:
    path = tmp_path / "mcp_config.json"
    path.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "alpha-mcp": {"command": "python", "prefix": "alpha"},
                    "beta-mcp": {"command": "python", "prefix": "beta"},
                }
            }
        ),
        encoding="utf-8",
    )
    return path


def _tool(name: str, property_name: str) -> mcp_types.Tool:
    return mcp_types.Tool(
        name=name,
        description=f"{property_name} generation",
        inputSchema={
            "type": "object",
            "properties": {property_name: {"type": "string"}},
        },
    )


def _resource(uri: str, description: str) -> MagicMock:
    item = MagicMock()
    item.uri = uri
    item.description = description
    return item


def _session(*, resources: list[MagicMock] | None = None) -> AsyncMock:
    session = AsyncMock()
    session.list_resources = AsyncMock(
        return_value=SimpleNamespace(resources=list(resources or []))
    )

    async def _read(uri):
        return SimpleNamespace(contents=[SimpleNamespace(text=f"fresh body for {uri}")])

    session.read_resource = AsyncMock(side_effect=_read)
    return session


async def _wait_until(predicate, timeout: float = 1.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        assert asyncio.get_running_loop().time() < deadline
        await asyncio.sleep(0.005)


async def test_refresh_recovers_failed_child_and_replaces_only_its_live_catalog(
    tmp_path,
) -> None:
    mux = MCPMultiplexer(_write_config(tmp_path))
    host = FastMCP("refresh-host")
    mux._host_mcp = host

    alpha_old_session = _session()
    beta_session = _session()
    alpha_new_session = _session(
        resources=[
            _resource("skill://alpha-fresh/SKILL.md", "fresh skill"),
            _resource("prompt://alpha/review", "fresh prompt"),
        ]
    )
    starts = {
        "alpha-mcp": [(alpha_old_session, _tool("work", "legacy"))],
        "beta-mcp": [(beta_session, _tool("inspect", "stable"))],
    }

    async def _start(server_name, cfg):
        session, tool = starts[server_name].pop(0)
        return server_name, session, [tool], cfg

    mux._start_child = AsyncMock(side_effect=_start)  # type: ignore[method-assign]
    alpha_tools = await mux.mount_child("alpha-mcp")
    await mux.mount_child("beta-mcp")
    alpha_name = alpha_tools[0].name
    _register_forwarder(host, mux, alpha_tools[0])
    mux.session_loaded("session-a").add(alpha_name)
    mux._auto_unload["session-a"] = {alpha_name}

    failed_runtime = mux.children["alpha-mcp"]
    failed_runtime.state = "failed"
    sibling_runtime = mux.children["beta-mcp"]
    sibling_routes = {
        name: route
        for name, route in mux.tool_to_server.items()
        if route[0] == "beta-mcp"
    }
    mux._probe_cache["alpha-mcp"] = {
        "tools": [{"name": "work", "inputSchema": {"legacy": True}}],
        "skills": [],
        "prompts": [],
        "error": None,
    }
    starts["alpha-mcp"] = [(alpha_new_session, _tool("work", "fresh"))]
    writer = AsyncMock(return_value={"status": "ok", "servers_written": 1})
    mux._fleet_catalog_writer = writer

    result = await mux.refresh_child("alpha-mcp")

    assert result["status"] == "refreshed"
    assert result["state"] == "up"
    assert result["catalog_revision"] == 2
    assert result["changed_tools"] == [alpha_name]
    assert result["skill_count"] == 1
    assert result["prompt_count"] == 1
    assert result["kg_reingest"]["status"] == "ok"
    assert failed_runtime.state == "closed"
    assert mux.children["alpha-mcp"] is not failed_runtime
    assert mux.children["beta-mcp"] is sibling_runtime
    assert {
        name: route
        for name, route in mux.tool_to_server.items()
        if route[0] == "beta-mcp"
    } == sibling_routes
    assert alpha_name in mux.session_loaded("session-a")
    assert alpha_name in mux._auto_unload["session-a"]
    assert "session-a" in mux._pending_tool_list_changes
    forwarded = await host.get_tool(alpha_name)
    assert "fresh" in forwarded.parameters["properties"]

    writer.assert_awaited_once()
    written_catalog, written_configs, written_bindings = writer.await_args.args
    assert written_configs == {"alpha-mcp": mux.load_catalog()["alpha-mcp"]}
    assert set(written_bindings) == {"alpha-mcp"}
    assert written_catalog["alpha-mcp"]["skills"][0]["instructions"].startswith(
        "fresh body"
    )
    assert written_catalog["alpha-mcp"]["prompts"][0]["body"].startswith("fresh body")
    await mux.aclose()


async def test_concurrent_refreshes_of_one_child_are_singleflight(tmp_path) -> None:
    mux = MCPMultiplexer(_write_config(tmp_path))
    first_session = _session()
    fresh_session = _session()
    entered = asyncio.Event()
    release = asyncio.Event()
    calls = 0

    async def _start(server_name, cfg):
        nonlocal calls
        calls += 1
        if calls == 1:
            return server_name, first_session, [_tool("work", "legacy")], cfg
        entered.set()
        await release.wait()
        return server_name, fresh_session, [_tool("work", "fresh")], cfg

    mux._start_child = AsyncMock(side_effect=_start)  # type: ignore[method-assign]
    await mux.mount_child("alpha-mcp")
    mux._fleet_catalog_writer = AsyncMock(return_value={"status": "ok"})

    first = asyncio.create_task(mux.refresh_child("alpha-mcp"))
    await entered.wait()
    assert isinstance(mux._refresh_inflight["alpha-mcp"], asyncio.Task)
    second = asyncio.create_task(mux.refresh_child("alpha-mcp"))
    await asyncio.sleep(0)
    release.set()
    first_result, second_result = await asyncio.gather(first, second)

    assert calls == 2  # one initial mount + exactly one shared refresh mount
    assert first_result == second_result
    assert mux._fleet_catalog_writer.await_count == 1
    await mux.aclose()


async def test_cancelling_creator_after_retire_does_not_cancel_owned_refresh(
    tmp_path,
) -> None:
    mux = MCPMultiplexer(_write_config(tmp_path))
    entered_remount = asyncio.Event()
    release_remount = asyncio.Event()
    starts = 0

    async def _start(server_name, cfg):
        nonlocal starts
        starts += 1
        if starts == 1:
            return server_name, _session(), [_tool("work", "legacy")], cfg
        entered_remount.set()
        await release_remount.wait()
        return server_name, _session(), [_tool("work", "fresh")], cfg

    mux._start_child = AsyncMock(side_effect=_start)  # type: ignore[method-assign]
    await mux.mount_child("alpha-mcp")
    retired = mux.children["alpha-mcp"]
    mux._fleet_catalog_writer = AsyncMock(return_value={"status": "ok"})

    creator = asyncio.create_task(mux.refresh_child("alpha-mcp"))
    await entered_remount.wait()
    assert retired.state == "closed"
    assert "alpha-mcp" not in mux.children

    creator.cancel()
    with pytest.raises(asyncio.CancelledError):
        await creator
    # The creator is only a shielded waiter; the independently-owned refresh
    # remains registered and continues the full lifecycle.
    assert not mux._refresh_inflight["alpha-mcp"].cancelled()
    release_remount.set()
    await _wait_until(lambda: "alpha-mcp" not in mux._refresh_inflight)

    assert mux.children["alpha-mcp"].state == "up"
    assert mux.children["alpha-mcp"] is not retired
    assert mux._fleet_catalog_writer.await_count == 1
    await mux.aclose()


async def test_remount_failure_stays_retired_and_notifies_snapshot_sessions(
    tmp_path,
) -> None:
    mux = MCPMultiplexer(_write_config(tmp_path))
    host = FastMCP("failed-remount-host")
    mux._host_mcp = host
    attempts = 0

    async def _start(server_name, cfg):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return server_name, _session(), [_tool("work", "legacy")], cfg
        return None

    mux._start_child = AsyncMock(side_effect=_start)  # type: ignore[method-assign]
    mounted = await mux.mount_child("alpha-mcp")
    name = mounted[0].name
    _register_forwarder(host, mux, mounted[0])
    mux.session_loaded("affected").add(name)
    retired = mux.children["alpha-mcp"]

    with pytest.raises(RuntimeError, match="could not remount"):
        await mux.refresh_child("alpha-mcp")

    assert retired.state == "closed"
    assert "alpha-mcp" not in mux.children
    assert name not in mux._exposed
    assert name not in mux.session_loaded("affected")
    assert "affected" in mux._pending_tool_list_changes
    assert await host.get_tool(name) is None
    await mux.aclose()


async def test_refresh_waits_for_racing_initial_mount_then_retires_that_generation(
    tmp_path,
) -> None:
    mux = MCPMultiplexer(_write_config(tmp_path))
    initial_entered = asyncio.Event()
    release_initial = asyncio.Event()
    starts = 0
    initial_session = _session()
    fresh_session = _session()
    captured_runtime = None
    original_retire = mux._retire_child_for_refresh

    async def _start(server_name, cfg):
        nonlocal starts
        starts += 1
        if starts == 1:
            initial_entered.set()
            await release_initial.wait()
            return server_name, initial_session, [_tool("work", "initial")], cfg
        return server_name, fresh_session, [_tool("work", "fresh")], cfg

    async def _capture_retire(server_name):
        nonlocal captured_runtime
        captured_runtime = mux.children[server_name]
        return await original_retire(server_name)

    mux._start_child = AsyncMock(side_effect=_start)  # type: ignore[method-assign]
    mux._retire_child_for_refresh = _capture_retire  # type: ignore[method-assign]
    mux._fleet_catalog_writer = AsyncMock(return_value={"status": "ok"})

    mounting = asyncio.create_task(mux.mount_child("alpha-mcp"))
    await initial_entered.wait()
    refreshing = asyncio.create_task(mux.refresh_child("alpha-mcp"))
    await asyncio.sleep(0)
    assert starts == 1
    release_initial.set()
    await mounting
    result = await refreshing

    assert starts == 2
    assert captured_runtime is not None
    assert captured_runtime.state == "closed"
    assert mux.sessions["alpha-mcp"] is fresh_session
    assert result["status"] == "refreshed"
    await mux.aclose()


async def test_shutdown_cancels_and_awaits_owned_refresh_task(tmp_path) -> None:
    mux = MCPMultiplexer(_write_config(tmp_path))
    remount_entered = asyncio.Event()
    never_release = asyncio.Event()
    starts = 0

    async def _start(server_name, cfg):
        nonlocal starts
        starts += 1
        if starts == 1:
            return server_name, _session(), [_tool("work", "legacy")], cfg
        remount_entered.set()
        await never_release.wait()

    mux._start_child = AsyncMock(side_effect=_start)  # type: ignore[method-assign]
    await mux.mount_child("alpha-mcp")
    waiter = asyncio.create_task(mux.refresh_child("alpha-mcp"))
    await remount_entered.wait()

    await mux.aclose()

    with pytest.raises(asyncio.CancelledError):
        await waiter
    assert mux._refresh_inflight == {}
    with pytest.raises(RuntimeError, match="shutting down"):
        await mux.refresh_child("alpha-mcp")


async def test_two_tool_exposure_restore_failure_is_atomic_and_truthful(
    tmp_path, monkeypatch
) -> None:
    mux = MCPMultiplexer(_write_config(tmp_path))
    host = FastMCP("atomic-refresh-host")
    mux._host_mcp = host
    starts = 0

    async def _start(server_name, cfg):
        nonlocal starts
        starts += 1
        generation = "legacy" if starts == 1 else "fresh"
        return (
            server_name,
            _session(),
            [
                _tool("query", f"{generation}_query"),
                _tool("search", f"{generation}_search"),
            ],
            cfg,
        )

    mux._start_child = AsyncMock(side_effect=_start)  # type: ignore[method-assign]
    mounted = await mux.mount_child("alpha-mcp")
    names = {tool.name for tool in mounted}
    for tool in mounted:
        _register_forwarder(host, mux, tool)
    mux.session_loaded("affected").update(names)
    retired = mux.children["alpha-mcp"]

    original_add = host.add_tool
    add_calls = 0

    def _accept_one_then_fail(tool):
        nonlocal add_calls
        add_calls += 1
        if add_calls == 1:
            return original_add(tool)
        raise RuntimeError("synthetic second registration failure")

    monkeypatch.setattr(host, "add_tool", _accept_one_then_fail)
    with pytest.raises(RuntimeError, match="second registration failure"):
        await mux.refresh_child("alpha-mcp")

    assert add_calls == 2
    assert retired.state == "closed"
    assert mux.children["alpha-mcp"] is not retired
    assert mux.children["alpha-mcp"].state == "up"
    assert not names & mux._exposed
    assert not names & mux.session_loaded("affected")
    assert "affected" in mux._pending_tool_list_changes
    # The fresh runtime catalog remains mountable, but no partial executable
    # provider exposure or stale session claim survived the atomic refusal.
    assert names <= set(mux.tool_to_server)
    for name in names:
        assert await host.get_tool(name) is None
        assert "fresh" in mux.tool_object(name).description
    await mux.aclose()


async def test_refresh_meta_tool_is_admin_only_and_uses_same_mux_method(
    tmp_path, monkeypatch
) -> None:
    mux = MCPMultiplexer(_write_config(tmp_path))
    mux.refresh_child = AsyncMock(
        return_value={"status": "refreshed", "server": "alpha-mcp"}
    )  # type: ignore[method-assign]
    host = FastMCP("refresh-auth-host")
    _register_meta_tools(host, mux)
    tool = await host.get_tool("refresh_mcp_server")

    monkeypatch.setattr(
        "agent_utilities.mcp.multiplexer._request_capabilities",
        lambda: frozenset({"mcp:delegate"}),
    )
    with pytest.raises(ToolError, match="manage capability required"):
        await tool.fn(server_name="alpha-mcp")

    monkeypatch.setattr(
        "agent_utilities.mcp.multiplexer._request_capabilities",
        lambda: frozenset({"mcp:admin"}),
    )
    result = await tool.fn(server_name="alpha-mcp")
    assert result.structured_content == {
        "status": "refreshed",
        "server": "alpha-mcp",
    }
    mux.refresh_child.assert_awaited_once_with("alpha-mcp")
