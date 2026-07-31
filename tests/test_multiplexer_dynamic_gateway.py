"""Tests for the dynamic tool gateway (CONCEPT:AU-ECO.multiplexer.tool-gateway-catalog).

Covers the lazy-mount refactor, KG-backed discovery + (server, tool) mapping,
the load/unload resolution logic, and the FastMCP meta-tool wiring. Children and
the knowledge-graph are mocked so these run with no processes or live engine.
"""

from __future__ import annotations

import asyncio
import json
import time
from unittest.mock import AsyncMock, MagicMock

import mcp.types
import pytest
from fastmcp.exceptions import ToolError

from agent_utilities.mcp.multiplexer import (
    MCPMultiplexer,
    SessionVisibilityMiddleware,
    _make_forwarder,
    _register_forwarder,
    _register_meta_tools,
    _session_key,
    _tool_is_verbose,
    get_server_prefix,
)

CNT = "container-manager-mcp"
CNT_TOOL = "cm_container_operations"
# container-manager-mcp auto-derives prefix "cm"; clean_tool_name then strips the
# redundant leading "cm_" from the tool name → "cm__container_operations".
CNT_PREFIXED = "cm__container_operations"
CNT_IMAGE_PREFIXED = "cm__image_operations"
CNT_INFO_PREFIXED = "cm__info_operations"


def _write_config(tmp_path, servers: dict) -> object:
    path = tmp_path / "mcp_config.json"
    path.write_text(json.dumps({"mcpServers": servers}), encoding="utf-8")
    return path


@pytest.mark.asyncio
async def test_forwarder_preserves_child_tool_error_as_outer_error(tmp_path) -> None:
    mux = MCPMultiplexer(tmp_path / "mcp_config.json")
    mux.call_proxied_tool = AsyncMock(
        return_value=mcp.types.CallToolResult(
            content=[mcp.types.TextContent(type="text", text="private child detail")],
            isError=True,
        )
    )

    with pytest.raises(ToolError, match="delegated_child_tool_failed"):
        await _make_forwarder(mux, "synthetic__tool")()


def _fake_tool(
    name: str,
    description: str = "",
    schema: dict | None = None,
    tags: list[str] | None = None,
):
    tool = MagicMock()
    tool.name = name
    tool.description = description
    tool.input_schema = schema if schema is not None else {}
    tool.annotations = None
    # FastMCP propagates tags via _meta; the multiplexer reads tool.meta.
    tool.meta = {"fastmcp": {"tags": list(tags)}} if tags is not None else None
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


def test_reload_catalog_drops_stale_routing_and_reparses(tmp_path):
    config_path = _write_config(
        tmp_path,
        {"first": {"command": "first-server"}},
    )
    mux = MCPMultiplexer(config_path)
    assert set(mux.load_catalog()) == {"first"}
    mux.tool_to_server["first__query"] = ("first", "query")

    config_path.write_text(
        json.dumps({"mcpServers": {"second": {"command": "second-server"}}})
    )
    catalog = mux.reload_catalog()

    assert set(catalog) == {"second"}
    assert mux.tool_to_server == {}


def test_load_catalog_auto_registers_native_langfuse_mcp(tmp_path, monkeypatch):
    config_path = tmp_path / "missing.json"
    native = {
        "command": "langfuse-mcp",
        "args": [],
        "env": {
            "LANGFUSE_HOST": "https://telemetry.example.test",
            "LANGFUSE_PUBLIC_KEY": "synthetic-public",
            "LANGFUSE_SECRET_KEY": "synthetic-secret",
            "LANGFUSE_PERSISTENCE_HMAC_KEY": "synthetic-persistence-material",
            "LANGFUSE_PERSISTENCE_HMAC_MATERIALIZED": "true",
        },
    }
    monkeypatch.setattr(
        "agent_utilities.observability.langfuse_trust.native_langfuse_mcp_config",
        lambda: native,
    )
    prepare = MagicMock(side_effect=AssertionError("native config prepared twice"))
    monkeypatch.setattr(
        "agent_utilities.observability.langfuse_trust.prepare_langfuse_mcp_config",
        prepare,
    )

    catalog = MCPMultiplexer(config_path).load_catalog()

    assert catalog["langfuse-mcp"]["command"] == "langfuse-mcp"
    assert "UV_NATIVE_TLS" not in catalog["langfuse-mcp"]["env"]
    assert prepare.call_count == 0
    assert catalog["langfuse-mcp"]["_runtime_materialized_secret_keys"] == [
        "LANGFUSE_PERSISTENCE_HMAC_KEY",
        "LANGFUSE_SECRET_KEY",
    ]


def test_load_catalog_classifies_native_langfuse_failure(tmp_path, monkeypatch, caplog):
    from agent_utilities.observability.langfuse_trust import LangfuseTrustError

    monkeypatch.setattr(
        "agent_utilities.observability.langfuse_trust.native_langfuse_mcp_config",
        MagicMock(side_effect=LangfuseTrustError("langfuse_credentials_missing")),
    )

    catalog = MCPMultiplexer(tmp_path / "missing.json").load_catalog()

    assert catalog == {}
    assert "credentials configuration invalid" in caplog.text
    assert "invalid CA bundle" not in caplog.text
    # The anti-pattern this regression guards: only logging the generic
    # category ("credentials") discards *why* -- the actual reason string
    # must also reach the log text.
    assert "langfuse_credentials_missing" in caplog.text


def test_load_catalog_admits_explicit_remote_langfuse_entry_via_direct_key_pair(
    tmp_path, monkeypatch
):
    """Regression test for the graph-os fleet-catalog gap.

    An explicit remote (streamable-http) ``langfuse-mcp`` entry ships with no
    ``env`` block, so credential resolution falls through to the parent
    process environment. That resolution must accept the direct
    ``LANGFUSE_PUBLIC_KEY`` / ``LANGFUSE_SECRET_KEY`` pair -- the same names
    ``langfuse_agent.auth.get_client`` reads for the standalone agent/MCP
    server -- not only the ``LANGFUSE_PUBLIC_KEY_REF`` / ``LANGFUSE_SECRET_KEY_REF``
    secret-reference form.
    """
    config_path = _write_config(
        tmp_path,
        {
            "langfuse-mcp": {
                "transport": "streamable-http",
                "url": "http://langfuse-mcp.example.test/mcp",
                "timeout": 120,
                "call_timeout": 600,
                "disabled": False,
            }
        },
    )
    for name in ("LANGFUSE_PUBLIC_KEY_REF", "LANGFUSE_SECRET_KEY_REF"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LANGFUSE_HOST", "https://langfuse.example.test")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-synthetic")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-synthetic")

    catalog = MCPMultiplexer(config_path).load_catalog()

    assert "langfuse-mcp" in catalog
    assert catalog["langfuse-mcp"]["transport"] == "streamable-http"
    assert catalog["langfuse-mcp"]["env"]["LANGFUSE_PUBLIC_KEY"] == "pk-lf-synthetic"
    assert catalog["langfuse-mcp"]["env"]["LANGFUSE_SECRET_KEY"] == "sk-lf-synthetic"


def test_load_catalog_still_rejects_when_neither_ref_nor_direct_key_present(
    tmp_path, monkeypatch, caplog
):
    config_path = _write_config(
        tmp_path,
        {
            "langfuse-mcp": {
                "transport": "streamable-http",
                "url": "http://langfuse-mcp.example.test/mcp",
                "disabled": False,
            }
        },
    )
    for name in (
        "LANGFUSE_PUBLIC_KEY_REF",
        "LANGFUSE_SECRET_KEY_REF",
        "LANGFUSE_PUBLIC_KEY",
        "LANGFUSE_SECRET_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LANGFUSE_HOST", "https://langfuse.example.test")

    catalog = MCPMultiplexer(config_path).load_catalog()

    assert "langfuse-mcp" not in catalog
    assert "langfuse_credentials_missing" in caplog.text


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
    assert "go__graph_query" in names
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


async def test_probe_catalog_returns_partial_results_within_budget(tmp_path):
    mux = _mux_with_children(tmp_path, {CNT: [(CNT_TOOL, "c")], "slow-mcp": []})

    async def slow_probe(server, **_kwargs):
        if server == CNT:
            return {
                "tools": [{"name": CNT_TOOL, "description": "containers"}],
                "error": None,
            }
        await asyncio.sleep(1)
        return {"tools": [], "error": None}

    mux.probe_server = AsyncMock(side_effect=slow_probe)  # type: ignore[method-assign]
    result = await mux.probe_catalog(budget=0.01)

    # Honest, per-server "still working" -- NOT the old fixed "budget
    # exceeded" claim stamped identically on every straggler regardless of
    # its real state (CONCEPT:AU-ECO.multiplexer.tool-gateway-catalog).
    assert result["slow-mcp"]["pending"] is True
    assert "still probing" in result["slow-mcp"]["error"]
    await mux.aclose()  # let the still-running background probe be cancelled cleanly


async def test_slow_server_degrades_alone_fast_servers_unaffected(tmp_path):
    """Requirement: one slow server must not blank the other 60 -- each
    reports its own state with its own reason (CONCEPT:AU-ECO.multiplexer.tool-gateway-catalog)."""
    mux = _mux_with_children(tmp_path, {CNT: [(CNT_TOOL, "c")], "slow-mcp": []})

    async def mixed_probe(server, **_kwargs):
        if server == CNT:
            info = {
                "tools": [{"name": CNT_TOOL, "description": "containers"}],
                "error": None,
            }
            mux._probe_cache[server] = info
            return info
        await asyncio.sleep(1)
        info = {"tools": [], "error": None}
        mux._probe_cache[server] = info
        return info

    mux.probe_server = AsyncMock(side_effect=mixed_probe)  # type: ignore[method-assign]
    result = await mux.probe_catalog(budget=0.05)

    assert result[CNT]["error"] is None
    assert result[CNT]["tools"][0]["name"] == CNT_TOOL
    assert result["slow-mcp"]["pending"] is True
    await mux.aclose()


async def test_discovery_succeeds_for_a_previously_timed_out_server(tmp_path):
    """The core regression this lane fixes: a server that misses the
    interactive budget must become available on a LATER call once its
    background probe finishes, via cache -- not be re-probed (and time out
    again) from scratch on every subsequent call forever
    (CONCEPT:AU-ECO.multiplexer.tool-gateway-catalog)."""
    mux = _mux_with_children(tmp_path, {"slow-mcp": []})
    release = asyncio.Event()
    calls = 0

    async def eventually_probe(server, **_kwargs):
        nonlocal calls
        calls += 1
        await release.wait()
        info = {
            "tools": [{"name": "slow_op", "description": "eventually works"}],
            "error": None,
        }
        mux._probe_cache[server] = info
        return info

    mux.probe_server = AsyncMock(side_effect=eventually_probe)  # type: ignore[method-assign]

    first = await mux.probe_catalog(budget=0.01)
    assert first["slow-mcp"]["pending"] is True
    assert "slow-mcp" not in mux._probe_cache

    # The background probe from the FIRST call is still running (it was
    # never cancelled) -- let it finish now.
    release.set()
    await asyncio.sleep(0.05)

    second = await mux.probe_catalog(budget=0.01)
    assert second["slow-mcp"]["error"] is None
    assert second["slow-mcp"]["tools"][0]["name"] == "slow_op"
    # Probed exactly once: the second call joined the SAME in-flight probe
    # rather than starting a duplicate.
    assert calls == 1


async def test_cached_results_are_labelled_with_age(tmp_path, monkeypatch):
    mux = _mux_with_children(tmp_path, {CNT: [(CNT_TOOL, "manage docker containers")]})
    mux._kg_call = AsyncMock(return_value=None)  # type: ignore[method-assign]
    fixed_now = 1_000_000.0
    mux._probe_cache[CNT] = {
        "tools": [
            {
                "name": CNT_TOOL,
                "description": "manage docker containers",
                "inputSchema": {},
            }
        ],
        "error": None,
        "probed_at": fixed_now,
    }

    monkeypatch.setattr(time, "time", lambda: fixed_now + 42.0)
    discovery = await mux.discover_tools("manage docker containers", top_k=5)
    top = discovery["results"][0]
    assert top["tool"] == CNT_TOOL
    assert top["age_s"] == 42.0
    assert top["stale"] is False

    cat = await mux.list_catalog(server=CNT)
    assert cat["age_s"] == 42.0


async def test_concurrent_probes_honor_their_own_per_server_deadline(tmp_path):
    """Two servers probed concurrently must each fail/succeed on THEIR OWN
    configured timeout, not a shared ceiling -- and truly concurrently, not
    serialized (CONCEPT:AU-ECO.multiplexer.tool-gateway-catalog)."""
    servers = {
        "quick-timeout-mcp": {
            "command": "python",
            "args": ["-m", "x"],
            "probe_timeout": 0.05,
        },
        "slow-answer-mcp": {
            "command": "python",
            "args": ["-m", "y"],
            "probe_timeout": 5.0,
        },
    }
    mux = MCPMultiplexer(_write_config(tmp_path, servers))

    async def _open(server, cfg, stack):
        if server == "quick-timeout-mcp":
            await asyncio.sleep(10)  # always exceeds its OWN 0.05s deadline
        else:
            await asyncio.sleep(0.1)  # comfortably inside its OWN 5s deadline
        return _fake_session([("op", "does a thing")])

    mux._open_one_session = AsyncMock(side_effect=_open)  # type: ignore[method-assign]

    t0 = asyncio.get_running_loop().time()
    result = await mux.probe_catalog()  # no shared budget: purely per-server
    elapsed = asyncio.get_running_loop().time() - t0

    assert result["quick-timeout-mcp"]["error"] == "timeout after 0.05s"
    assert result["slow-answer-mcp"]["error"] is None
    # Ran concurrently: total elapsed tracks the SLOWER server (~0.1s), not
    # the sum of both (which would be > 10s if serialized).
    assert elapsed < 2.0
    await mux.aclose()


@pytest.mark.parametrize(
    ("server", "query", "tool", "description"),
    (
        (
            "github-mcp",
            "read and comment on a GitHub pull request",
            "github_pull_requests",
            "Read and comment on GitHub pull requests.",
        ),
        (
            "mattermost-mcp",
            "post a Mattermost message",
            "mattermost_post_message",
            "Post a Mattermost message.",
        ),
    ),
)
async def test_discover_prioritizes_named_server_inside_shared_budget(
    tmp_path,
    monkeypatch,
    server,
    query,
    tool,
    description,
):
    """A named domain is probed before unrelated fleet fan-out consumes the budget."""

    from agent_utilities.core.config import config as agent_config

    tool_map = {
        **{f"slow-{index}-mcp": [] for index in range(24)},
        server: [(tool, description)],
    }
    mux = _mux_with_children(tmp_path, tool_map)
    calls: list[str] = []

    async def staged_probe(server_name, **_kwargs):
        calls.append(server_name)
        if server_name == server:
            info = {
                "tools": [
                    {
                        "name": tool,
                        "description": description,
                        "inputSchema": {},
                    }
                ],
                "error": None,
            }
            mux._probe_cache[server_name] = info
            return info
        await asyncio.sleep(1)
        return {"tools": [], "error": None}

    monkeypatch.setattr(agent_config, "mcp_dynamic_discovery_timeout", 0.05)
    mux.probe_server = AsyncMock(side_effect=staged_probe)  # type: ignore[method-assign]

    discovery = await mux.discover_tools(query, top_k=5)

    assert calls[0] == server
    assert discovery["results"][0]["server"] == server
    assert discovery["results"][0]["tool"] == tool
    assert server not in discovery["unavailable"]


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
    assert info["error"] == "OSError: connection refused"


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
# Unique prefixes (collision-free at any scale)
# --------------------------------------------------------------------------- #


def test_server_prefix_resolves_derived_collisions(tmp_path):
    # foo-bar-mcp and foo-baz-mcp both auto-derive the acronym "fb" → the
    # resolver must hand one a distinct prefix.
    mux = _mux_with_children(
        tmp_path,
        {"foo-bar-mcp": [("a", "")], "foo-baz-mcp": [("b", "")]},
    )
    p1 = mux.server_prefix("foo-bar-mcp")
    p2 = mux.server_prefix("foo-baz-mcp")
    assert p1 != p2
    assert "fb" in (p1, p2)  # one keeps the preferred prefix
    prefixes = [mux.server_prefix(s) for s in mux.load_catalog()]
    assert len(prefixes) == len(set(prefixes)), "prefixes must be unique"


def test_config_prefix_override_wins(tmp_path):
    mux = _mux_with_children(tmp_path, {"foo-bar-mcp": [("a", "")]})
    mux.load_catalog()["foo-bar-mcp"]["prefix"] = "myfoo"
    mux._build_prefix_map()
    assert mux.server_prefix("foo-bar-mcp") == "myfoo"


def test_reverse_prefix_resolves_owner(tmp_path):
    mux = _mux_with_children(
        tmp_path,
        {"foo-bar-mcp": [("a", "")], "foo-baz-mcp": [("b", "")]},
    )
    for name in ["foo-bar-mcp", "foo-baz-mcp"]:
        prefix = mux.server_prefix(name)
        assert mux._server_for_prefixed(f"{prefix}__x") == name


# --------------------------------------------------------------------------- #
# Enabled / disabled annotation
# --------------------------------------------------------------------------- #


def _seed_disabled(mux, server, disabled):
    mux.load_catalog()[server]["disabledTools"] = disabled


async def test_list_catalog_splits_enabled_disabled(tmp_path):
    mux = _mux_with_children(
        tmp_path, {CNT: [(CNT_TOOL, "a"), ("cm_info_operations", "b")]}
    )
    _seed_disabled(mux, CNT, ["cm_info_operations"])
    mux._kg_call = AsyncMock(return_value=None)  # type: ignore[method-assign]
    _seed_probe(mux, {CNT: [(CNT_TOOL, "containers"), ("cm_info_operations", "info")]})

    cat = await mux.list_catalog()
    s = {x["server"]: x for x in cat["servers"]}[CNT]
    assert s["tool_count"] == 2
    assert s["enabled_count"] == 1
    assert CNT_PREFIXED in s["tools"]
    assert CNT_INFO_PREFIXED in s.get("disabled_tools", [])
    assert CNT_INFO_PREFIXED not in s["tools"]


async def test_list_catalog_drilldown_enabled_flag(tmp_path):
    mux = _mux_with_children(
        tmp_path, {CNT: [(CNT_TOOL, "a"), ("cm_info_operations", "b")]}
    )
    _seed_disabled(mux, CNT, ["cm_info_operations"])
    mux._kg_call = AsyncMock(return_value=None)  # type: ignore[method-assign]
    _seed_probe(mux, {CNT: [(CNT_TOOL, "c"), ("cm_info_operations", "i")]})

    cat = await mux.list_catalog(server=CNT)
    by_tool = {t["tool"]: t for t in cat["tools"]}
    assert by_tool[CNT_TOOL]["enabled"] is True
    assert by_tool["cm_info_operations"]["enabled"] is False


async def test_discover_skips_disabled_tools(tmp_path):
    mux = _mux_with_children(
        tmp_path, {CNT: [(CNT_TOOL, "a"), ("cm_info_operations", "b")]}
    )
    _seed_disabled(mux, CNT, ["cm_info_operations"])
    mux._kg_call = AsyncMock(return_value=None)  # type: ignore[method-assign]
    _seed_probe(
        mux,
        {CNT: [(CNT_TOOL, "manage docker"), ("cm_info_operations", "manage docker")]},
    )

    discovery = await mux.discover_tools("manage docker", top_k=10)
    tools = [r["tool"] for r in discovery["results"]]
    assert CNT_TOOL in tools
    assert "cm_info_operations" not in tools


# --------------------------------------------------------------------------- #
# Probe error formatting
# --------------------------------------------------------------------------- #


def test_format_probe_error_unwraps_exceptiongroup():
    from agent_utilities.mcp.multiplexer import _format_probe_error

    eg = ExceptionGroup(
        "boom",
        [RuntimeError("HTTP 502 Bad Gateway"), RuntimeError("HTTP 502 Bad Gateway")],
    )
    msg = _format_probe_error(eg)
    # The leaf type AND its message are preserved (deduped across identical
    # leaves) — a bare "RuntimeError" gives no diagnostic signal (this exact
    # anti-pattern made the whole MCP fleet's outage undiagnosable).
    assert msg == "RuntimeError: HTTP 502 Bad Gateway"
    assert _format_probe_error(ConnectionError("refused")) == "ConnectionError: refused"
    assert _format_probe_error(RuntimeError("")) == "RuntimeError"


# --------------------------------------------------------------------------- #
# load/unload resolution
# --------------------------------------------------------------------------- #


async def test_resolve_and_mount_by_server(tmp_path):
    mux = _mux_with_children(
        tmp_path, {CNT: [(CNT_TOOL, "a"), ("cm_image_operations", "b")]}
    )
    servers, to_expose, failed = await mux.resolve_and_mount(servers=[CNT])
    assert servers == [CNT]
    assert set(to_expose) == {CNT_PREFIXED, CNT_IMAGE_PREFIXED}
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
    assert CNT_PREFIXED in names_after  # forwarder registered process-globally
    assert CNT_PREFIXED in mux._exposed
    # ...and made visible to this (default, context-less) session.
    session_key = _session_key()
    assert CNT_PREFIXED in mux.session_loaded(session_key)

    # unload retracts it from THIS session; the forwarder stays registered
    # (other sessions may still have it loaded) — visibility is the middleware's job.
    unload = await mcp.get_tool("unload_tools")
    await unload.fn(tools=[CNT_PREFIXED])
    assert CNT_PREFIXED not in mux.session_loaded(session_key)
    assert CNT_PREFIXED in mux._exposed  # still globally registered


async def test_per_session_disclosure_isolation(tmp_path):
    """Plan Phase 5: one session's load_tools must not leak to another session."""
    from fastmcp import Client, FastMCP

    mux = _mux_with_children(tmp_path, {CNT: [(CNT_TOOL, "manage containers")]})
    mcp = FastMCP("test-mux")
    _register_meta_tools(mcp, mux)
    mux._global_visible = {
        "find_tools",
        "list_catalog",
        "load_tools",
        "unload_tools",
        "multiplexer_status",
    }
    mcp.add_middleware(SessionVisibilityMiddleware(mux))

    async with Client(mcp) as a:
        # Session A loads the container server's tools.
        await a.call_tool("load_tools", {"servers": [CNT]})
        a_tools = {t.name for t in await a.list_tools()}
        # A fresh session B has loaded nothing.
        async with Client(mcp) as b:
            b_tools = {t.name for t in await b.list_tools()}
            # B cannot even call A's tool (gated until B loads it).
            with pytest.raises(ToolError):
                await b.call_tool(CNT_PREFIXED, {})

    # A sees its loaded tool + the meta-tools; B sees only the meta-tools.
    assert CNT_PREFIXED in a_tools
    assert CNT_PREFIXED not in b_tools
    assert {"find_tools", "load_tools"} <= a_tools
    assert {"find_tools", "load_tools"} <= b_tools


async def test_one_shot_tool_call_prunes_session_visibility_state(tmp_path):
    """Auto-unload must leave no process-global key after the one-shot call."""
    from fastmcp import Client, FastMCP

    mux = MCPMultiplexer(tmp_path / "mcp_config.json")
    mcp = FastMCP("test-mux")

    @mcp.tool()
    async def graph_jobs() -> str:
        return "ok"

    mux._local_gated = {"graph_jobs"}
    _register_meta_tools(mcp, mux)
    mux._global_visible = {
        "find_tools",
        "list_catalog",
        "load_tools",
        "unload_tools",
        "multiplexer_status",
    }
    mcp.add_middleware(SessionVisibilityMiddleware(mux, mcp))

    async with Client(mcp) as client:
        await client.call_tool(
            "load_tools",
            {"tools": ["graph_jobs"], "auto_unload": True},
        )
        assert len(mux._session_loaded) == 1
        assert len(mux._auto_unload) == 1
        await client.call_tool("graph_jobs", {})

    assert mux._session_loaded == {}
    assert mux._auto_unload == {}


async def test_explicit_unload_prunes_session_visibility_state(tmp_path):
    """List-only one-shot sessions retract visibility before termination."""
    from fastmcp import Client, FastMCP

    mux = MCPMultiplexer(tmp_path / "mcp_config.json")
    mcp = FastMCP("test-mux")
    mux._local_gated = {"graph_jobs"}
    _register_meta_tools(mcp, mux)

    async with Client(mcp) as client:
        await client.call_tool(
            "load_tools",
            {"tools": ["graph_jobs"], "auto_unload": True},
        )
        await client.call_tool("unload_tools", {"tools": ["graph_jobs"]})

    assert mux._session_loaded == {}
    assert mux._auto_unload == {}


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
    assert get_server_prefix(CNT) == "cm"


# --------------------------------------------------------------------------- #
# Verbose tools held in catalog (load on demand) — CONCEPT:AU-ECO.mcp.tool-mode-standardization
# --------------------------------------------------------------------------- #
def test_tool_is_verbose_tag_detection():
    assert _tool_is_verbose(_fake_tool("x", tags=["graph_write", "verbose"])) is True
    assert _tool_is_verbose(_fake_tool("x", tags=["write"])) is False
    assert _tool_is_verbose(_fake_tool("x")) is False  # no meta -> not verbose


@pytest.mark.asyncio
async def test_always_on_holds_verbose_tools_in_catalog(tmp_path):
    """An always-on child's verbose tools are mounted (in the catalog, loadable)
    but NOT auto-exposed at boot — only the condensed surface is."""
    from fastmcp import FastMCP

    mux = _mux_with_children(tmp_path, {"graph-os": []})

    async def fake_start_child(server_name, cfg):
        tools = [
            _fake_tool("graph_write", "condensed", tags=["graph-os", "write"]),
            _fake_tool(
                "graph_write_add_node", "verbose", tags=["graph_write", "verbose"]
            ),
        ]
        return server_name, AsyncMock(), tools, cfg

    mux._start_child = AsyncMock(side_effect=fake_start_child)

    mcp = FastMCP("mux")
    tools = await mux.mount_child("graph-os")
    # Replicate the dynamic always-on boot filter:
    for tool in tools:
        if _tool_is_verbose(tool):
            continue
        _register_forwarder(mcp, mux, tool)

    prefixes = {t.name for t in mux.aggregated_tools}
    # Both tools are mounted/aggregated (in the catalog → loadable on demand)
    assert any("graph_write_add_node" in n for n in prefixes)
    # …but only the condensed one is auto-exposed; the verbose one is held back.
    exposed = mux._exposed
    assert any(n.endswith("graph_write") for n in exposed)
    assert not any("graph_write_add_node" in n for n in exposed)


async def test_server_load_holds_verbose(tmp_path):
    """CONCEPT:AU-ECO.mcp.tool-mode-standardization — load_tools(servers=[X]) exposes only X's condensed tools; verbose
    1:1 tools stay loadable only by EXPLICIT name, so a server-load never floods context."""
    servers = {"svc": {"command": "python", "args": ["-m", "svc"]}}
    mux = MCPMultiplexer(_write_config(tmp_path, servers))

    async def fake_start_child(server_name, cfg):
        tools = [
            _fake_tool("svc_action", "condensed action-routed"),
            _fake_tool("svc_get_thing", "verbose 1:1", tags=["verbose"]),
        ]
        return server_name, AsyncMock(), tools, cfg

    mux._start_child = AsyncMock(side_effect=fake_start_child)

    _s, to_expose, _f = await mux.resolve_and_mount(servers=["svc"])
    cond = [n for n in to_expose if n.endswith("action")]
    verb = [n for n in to_expose if n.endswith("get_thing")]
    assert cond and not verb  # condensed exposed, verbose held

    # verbose IS loadable by explicit name
    verbose_name = next(n for n in mux.tool_to_server if n.endswith("get_thing"))
    _s2, to_expose2, _f2 = await mux.resolve_and_mount(tools=[verbose_name])
    assert verbose_name in to_expose2


# --------------------------------------------------------------------------- #
# Control-plane truthfulness (lane-mcp-desync): reported "mounted"/"loaded"
# state must derive from the SAME predicate the dispatch gate enforces, never
# from a parallel bookkeeping structure that can drift from it.
# --------------------------------------------------------------------------- #


async def test_resolve_and_mount_reports_unknown_tool_name_as_failed(tmp_path):
    """A prefixed name that resolves to no configured server must never be
    silently dropped from both ``to_expose`` and ``failed`` — the caller needs
    to know exactly which of its requested names didn't make it."""
    mux = _mux_with_children(tmp_path, {CNT: [(CNT_TOOL, "a")]})
    mounted, to_expose, failed = await mux.resolve_and_mount(
        tools=["zz__totally_bogus"]
    )
    assert mounted == []
    assert to_expose == []
    assert "zz__totally_bogus" in failed


async def test_resolve_and_mount_reports_disabled_requested_tool_as_failed(tmp_path):
    """A tool explicitly requested by name, whose owning server mounts fine
    but that config disables, must show up in ``failed`` — not vanish while
    ``mounted_servers`` still claims success."""
    mux = _mux_with_children(
        tmp_path, {CNT: [(CNT_TOOL, "a"), ("cm_info_operations", "b")]}
    )
    _seed_disabled(mux, CNT, ["cm_info_operations"])

    mounted, to_expose, failed = await mux.resolve_and_mount(tools=[CNT_INFO_PREFIXED])

    assert mounted == [CNT]
    assert CNT_INFO_PREFIXED not in to_expose
    assert CNT_INFO_PREFIXED in failed


async def test_tool_dispatchable_false_for_catalogued_but_unmounted_tool(tmp_path):
    """A tool the catalog KNOWS about (via the prefix map) but that has never
    been mounted/exposed must report as NOT dispatchable — the optimistic
    'not gated -> visible' fallback is only valid for tools the multiplexer's
    own bookkeeping has no opinion on at all (e.g. native host tools)."""
    mux = _mux_with_children(tmp_path, {CNT: [(CNT_TOOL, "a")]})
    assert mux.tool_dispatchable(CNT_PREFIXED, session_key="any-session") is False


async def test_tool_dispatchable_is_session_scoped_after_expose(tmp_path):
    """Reproduces the reported parent-vs-subagent asymmetry at the state layer:
    once a tool is globally exposed (registered as a live forwarder), it is
    dispatchable ONLY for the session(s) that actually loaded it — a second,
    brand-new session must see it as not-yet-dispatchable even though the
    forwarder already exists process-globally."""
    mux = _mux_with_children(tmp_path, {CNT: [(CNT_TOOL, "a")]})
    _mounted, to_expose, failed = await mux.resolve_and_mount(servers=[CNT])
    assert failed == {}
    for name in to_expose:
        mux._exposed.add(name)  # simulates _register_forwarder's bookkeeping

    mux.session_loaded("session-A").add(CNT_PREFIXED)

    assert mux.tool_dispatchable(CNT_PREFIXED, session_key="session-A") is True
    assert mux.tool_dispatchable(CNT_PREFIXED, session_key="session-B") is False


async def test_list_catalog_mounted_matches_dispatch_reality_across_sessions(
    tmp_path,
):
    """End-to-end reproduction of the control-plane desync report: session A's
    load_tools call makes a tool globally registered, but list_catalog must
    tell a DIFFERENT session B the truth about what B can dispatch — not what
    the child process happens to be doing."""
    from fastmcp import Client, FastMCP

    mux = _mux_with_children(tmp_path, {CNT: [(CNT_TOOL, "manage containers")]})
    mux._kg_call = AsyncMock(return_value=None)  # type: ignore[method-assign]
    _seed_probe(mux, {CNT: [(CNT_TOOL, "manage containers")]})
    mcp = FastMCP("test-mux")
    _register_meta_tools(mcp, mux)
    mux._global_visible = {
        "find_tools",
        "list_catalog",
        "load_tools",
        "unload_tools",
        "multiplexer_status",
    }
    mcp.add_middleware(SessionVisibilityMiddleware(mux, mcp))

    async with Client(mcp) as a:
        await a.call_tool("load_tools", {"servers": [CNT]})

        async with Client(mcp) as b:
            cat_b = await b.call_tool("list_catalog", {"server": CNT})
            payload_b = cat_b.structured_content
            entry_b = next(
                t for t in payload_b["tools"] if t["prefixed_name"] == CNT_PREFIXED
            )
            # The child process IS running (session A mounted it)...
            assert payload_b["process_running"] is True
            # ...but session B never loaded it, so list_catalog must NOT claim
            # it's dispatchable — and an actual call must agree with that claim.
            assert entry_b["mounted"] is False
            with pytest.raises(ToolError):
                await b.call_tool(CNT_PREFIXED, {})

        # Session A, which DID load it, is told the truth the other way.
        cat_a = await a.call_tool("list_catalog", {"server": CNT})
        entry_a = next(
            t
            for t in cat_a.structured_content["tools"]
            if t["prefixed_name"] == CNT_PREFIXED
        )
        assert entry_a["mounted"] is True


async def test_notify_tools_changed_returns_false_without_request_context():
    from agent_utilities.mcp.multiplexer import _notify_tools_changed

    sent = await _notify_tools_changed(None)
    assert sent is False


async def test_notify_tools_changed_surfaces_send_failure(monkeypatch, caplog):
    """A LIVE client that rejects/drops the push must be visible via the
    return value (and logged with its real cause) — never swallowed into
    silence the caller can't observe (CLAUDE.md: never swallow an exception)."""
    import logging

    from agent_utilities.mcp.multiplexer import _notify_tools_changed

    class _FailingContext:
        async def send_notification(self, _note):
            raise RuntimeError("client transport gone")

    monkeypatch.setattr(
        "fastmcp.server.dependencies.get_context", lambda: _FailingContext()
    )

    with caplog.at_level(logging.WARNING, logger="mcp_multiplexer"):
        sent = await _notify_tools_changed(None)

    assert sent is False
    assert any("list_changed" in r.message for r in caplog.records)


async def test_load_tools_reports_notified_true_inside_a_live_session(tmp_path):
    from fastmcp import Client, FastMCP

    mux = _mux_with_children(tmp_path, {CNT: [(CNT_TOOL, "manage containers")]})
    mcp = FastMCP("test-mux")
    _register_meta_tools(mcp, mux)
    mcp.add_middleware(SessionVisibilityMiddleware(mux, mcp))

    async with Client(mcp) as client:
        result = await client.call_tool("load_tools", {"servers": [CNT]})

    assert result.structured_content["newly_exposed"]
    assert result.structured_content["notified"] is True


async def test_load_tools_reports_not_notified_outside_a_request_context(tmp_path):
    """Calling the core helper with no live client context (e.g. the tool's
    own ``.fn()`` invoked directly, as in ``test_meta_tools_registered_and_load_exposes``)
    must be truthful that the client was never actually told — a caller that
    only checked ``newly_exposed`` would otherwise wrongly assume its own
    client's tool list is already fresh."""
    from agent_utilities.mcp.multiplexer import load_session_tools

    mux = _mux_with_children(tmp_path, {CNT: [(CNT_TOOL, "manage containers")]})
    from fastmcp import FastMCP

    mcp = FastMCP("test-mux")
    _register_meta_tools(mcp, mux)

    payload = await load_session_tools(mcp, mux, servers=[CNT])

    assert payload["newly_exposed"]
    assert payload["notified"] is False


async def test_forced_reprobe_does_not_evict_a_live_joinable_probe(tmp_path):
    """A forced re-probe runs as its OWN task; when it settles it must not
    retract a DIFFERENT, still-running non-forced probe from the join map.

    Regression: ``_settle_probe_task`` popped ``_probe_inflight[server]``
    unconditionally, so a fast forced probe finishing while a slow shared probe
    was still running left that shared probe untracked -- a later call spawned a
    duplicate of an already-in-flight probe, and ``aclose`` no longer knew to
    cancel it.
    """
    mux = _mux_with_children(tmp_path, {"slow-mcp": []})
    release = asyncio.Event()

    async def probe(server, force=False, timeout=None):
        if not force:
            await release.wait()
        return {"tools": [], "error": None}

    mux.probe_server = AsyncMock(side_effect=probe)  # type: ignore[method-assign]

    shared = mux._ensure_probing("slow-mcp", force=False, timeout=None)
    forced = mux._ensure_probing("slow-mcp", force=True, timeout=None)
    assert forced is not shared
    await forced

    # The slow shared probe is still running and still joinable.
    assert mux._probe_inflight.get("slow-mcp") is shared
    assert mux._ensure_probing("slow-mcp", force=False, timeout=None) is shared

    release.set()
    await shared
    assert "slow-mcp" not in mux._probe_inflight
    await mux.aclose()


async def test_aclose_cancels_a_forced_probe_too(tmp_path):
    """``aclose`` must cancel EVERY live probe, not just the joinable ones --
    a forced re-probe is never in ``_probe_inflight`` and used to outlive the
    multiplexer that spawned it."""
    mux = _mux_with_children(tmp_path, {"slow-mcp": []})

    async def never_finishes(server, force=False, timeout=None):
        await asyncio.Event().wait()
        return {"tools": [], "error": None}

    mux.probe_server = AsyncMock(side_effect=never_finishes)  # type: ignore[method-assign]

    forced = mux._ensure_probing("slow-mcp", force=True, timeout=None)
    await asyncio.sleep(0)
    assert "slow-mcp" not in mux._probe_inflight  # forced probes are not joinable
    assert forced in mux._probe_tasks

    await mux.aclose()
    assert forced.cancelled()
    assert not mux._probe_tasks
