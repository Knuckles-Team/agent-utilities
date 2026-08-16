"""Regression tests for D-CDX-52: a child that adds a NEW tool after
recovery does not silently vanish from a session that already loaded the
whole server.

The D-CDX-35 schema-refresh lane correctly refreshes CHANGED and REMOVED
tools on an already-loaded session (``_replace_exposed_forwarders`` /
``_queue_tools_changed``). Independent rereview found a distinct gap: a
brand-new tool name (present only in the refreshed schema, never in the
prior one) is catalogued and mountable on a fresh ``load_tools`` call, but a
session that already loaded the WHOLE SERVER is not remembered as a
subscriber, so the addition is never auto-exposed or announced to it.

The chosen fix makes the snapshot semantics EXPLICIT rather than adding a
standing per-server subscription: ``load_session_tools`` / ``AlwaysLoadResult``
now report ``server_catalog_revisions`` — each mounted server's
``_child_schema_revisions`` counter at snapshot time — so a caller can
detect that a server's catalog moved on (by comparing against
``status_snapshot()``'s live ``catalog_revision`` for that server) without
polling the whole fleet, and re-``load_tools`` to pick up the addition.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from fastmcp import FastMCP

from agent_utilities.mcp.multiplexer import (
    MCPMultiplexer,
    _perform_always_load,
    _session_key,
    load_session_tools,
)
from tests.unit.mcp.test_multiplexer_dynamic_gateway import (
    _schema_tool,
    _SchemaGenerationSession,
    _wait_for_condition,
    _write_config,
)


@pytest.mark.asyncio
async def test_addition_is_catalogued_but_not_auto_exposed_to_prior_loader(
    tmp_path,
) -> None:
    """A tool added by a recovered child is NOT retroactively pushed to a
    session that already loaded the whole server — documents the explicit
    snapshot contract this fix establishes (the alternative to a standing
    per-server subscription)."""
    server_name = "addition-mcp"
    config_path = _write_config(
        tmp_path, {server_name: {"command": "addition-child", "args": []}}
    )
    original = _SchemaGenerationSession(
        [_schema_tool("alpha", "field_a")], fail_calls=True, tag="gen1"
    )
    recovered = _SchemaGenerationSession(
        [_schema_tool("alpha", "field_a"), _schema_tool("beta", "field_b")],
        tag="gen2",
    )
    generations = [original, recovered]
    mux = MCPMultiplexer(config_path)

    async def fake_open_one_session(*_args):
        return generations.pop(0)

    mux._open_one_session = AsyncMock(side_effect=fake_open_one_session)  # type: ignore[method-assign]
    host = FastMCP("addition-host")
    mux._host_mcp = host

    session_key = _session_key()
    first = await load_session_tools(host, mux, servers=[server_name])
    assert first["mounted_servers"] == [server_name]
    assert first["server_catalog_revisions"] == {server_name: 1}
    loaded_before = set(mux.session_loaded(session_key))
    assert len(loaded_before) == 1  # just "alpha"

    # Recover the child with an ADDED tool ("beta") alongside the original.
    runtime = mux.children[server_name]
    runtime.restart_backoff_base = 0.005
    runtime.restart_backoff_cap = 0.005
    # Any live tool call re-checks the connection and picks up the new
    # generation, exactly like the D-CDX-35 recovery lane's own test.
    prefixed_alpha = next(iter(loaded_before))
    await mux.call_proxied_tool(prefixed_alpha, {})
    await _wait_for_condition(lambda: mux.sessions[server_name] is recovered)

    # The new tool IS catalogued (discoverable/mountable) —
    beta_prefixed = next(
        name
        for name, (srv, orig) in mux.tool_to_server.items()
        if srv == server_name and orig == "beta"
    )
    assert beta_prefixed in {t.name for t in mux.aggregated_tools}

    # — but it is NOT retroactively pushed into the prior loader's session.
    assert beta_prefixed not in mux.session_loaded(session_key)

    # The per-server revision advanced, so staleness IS detectable.
    live_revision = mux.status_snapshot()["children"][server_name]["catalog_revision"]
    assert live_revision > first["server_catalog_revisions"][server_name]

    # Re-loading the server picks the addition up.
    second = await load_session_tools(host, mux, servers=[server_name])
    assert beta_prefixed in second["newly_exposed"]
    assert beta_prefixed in mux.session_loaded(session_key)
    assert second["server_catalog_revisions"][server_name] == live_revision


@pytest.mark.asyncio
async def test_server_catalog_revisions_reported_on_first_load(tmp_path) -> None:
    from tests.unit.mcp.test_multiplexer_dynamic_gateway import (
        CNT,
        CNT_TOOL,
        _mux_with_children,
    )

    mux = _mux_with_children(tmp_path, {CNT: [(CNT_TOOL, "containers")]})
    host = FastMCP("revisions-host")
    mux._host_mcp = host

    result = await load_session_tools(host, mux, servers=[CNT])
    assert result["mounted_servers"] == [CNT]
    assert result["server_catalog_revisions"] == {
        CNT: mux._child_schema_revisions.get(CNT, 0)
    }


@pytest.mark.asyncio
async def test_always_load_result_reports_server_catalog_revisions(tmp_path) -> None:
    from tests.unit.mcp.test_multiplexer_dynamic_gateway import (
        CNT,
        CNT_TOOL,
        _mux_with_children,
    )

    mux = _mux_with_children(tmp_path, {CNT: [(CNT_TOOL, "containers")]})
    mux._always_load_servers = [CNT]
    host = FastMCP("always-load-revisions-host")
    mux._host_mcp = host

    result = await _perform_always_load(host, mux, "synthetic-session")
    assert result["mounted_servers"] == [CNT]
    assert result["server_catalog_revisions"] == {
        CNT: mux._child_schema_revisions.get(CNT, 0)
    }
