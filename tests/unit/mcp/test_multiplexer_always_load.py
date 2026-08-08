"""Always-load: eager mounting of core MCP servers/tools (CONCEPT:AU-ECO.multiplexer.tool-gateway-catalog).

Covers the four properties the feature exists for:

1. **Server granularity** — ``MCP_ALWAYS_LOAD`` names whole servers, mounted on a
   session's first contact with no ``find_tools`` round trip.
2. **Tool granularity** — ``MCP_ALWAYS_LOAD_TOOLS`` names individual tools, so a
   large server (github/gitlab) contributes its issue + PR/MR tools without
   flooding the session with its whole surface.
3. **Fail-soft** — a broken always-load server degrades to lazy discovery. It
   must not raise out of ``tools/list``, must not stop the OTHER always-load
   entries mounting, and must be reported as degraded rather than silently
   dropped.
4. **Shipped defaults** — the operator's four servers and the GitHub/GitLab
   issue + PR/MR tools are the out-of-the-box value of the config fields.

Every test here is written to FAIL against the pre-change behaviour (nothing is
eagerly mounted, and there are no config fields at all).
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_utilities.mcp.multiplexer import (
    MCPMultiplexer,
    SessionVisibilityMiddleware,
    ensure_always_loaded,
)

CNT = "container-manager-mcp"
BIG = "github-mcp"


def _write_config(tmp_path, servers: dict):
    path = tmp_path / "mcp_config.json"
    path.write_text(json.dumps({"mcpServers": servers}), encoding="utf-8")
    return path


def _fake_tool(name: str, description: str = ""):
    tool = MagicMock()
    tool.name = name
    tool.description = description
    tool.input_schema = {}
    tool.annotations = None
    tool.meta = None
    return tool


def _mux(tmp_path, tool_map: dict[str, list[str]], *, broken: set[str] | None = None):
    """Mux whose ``_start_child`` yields fake tools, or raises for ``broken``."""
    servers = {name: {"command": "python", "args": ["-m", name]} for name in tool_map}
    for name in broken or ():
        servers.setdefault(name, {"command": "python", "args": ["-m", name]})
    mux = MCPMultiplexer(_write_config(tmp_path, servers))

    async def fake_start_child(server_name, cfg):
        if server_name in (broken or ()):
            raise RuntimeError("child crash-looping on a fastmcp version mismatch")
        tools = [_fake_tool(n) for n in tool_map.get(server_name, [])]
        return server_name, AsyncMock(), tools, cfg

    mux._start_child = AsyncMock(side_effect=fake_start_child)  # type: ignore[method-assign]
    return mux


def _fake_mcp():
    """Minimal FastMCP stand-in: forwarder registration + change notification."""
    server = MagicMock()
    server.add_tool = MagicMock()
    return server


def _prefixed(mux, server: str, original: str) -> str:
    """The aggregated name a child's tool gets, computed the way the mux does."""
    from agent_utilities.mcp.multiplexer import clean_tool_name

    return clean_tool_name(mux.server_prefix(server), server, original)


# --------------------------------------------------------------------------- #
# 4. Shipped defaults
# --------------------------------------------------------------------------- #


def test_shipped_defaults_are_the_operators_four_servers():
    from agent_utilities.core.config import AgentConfig

    default = AgentConfig.model_fields["mcp_always_load"].get_default(
        call_default_factory=True
    )
    assert default == [
        "tunnel-manager-mcp",
        "systems-manager-mcp",
        "repository-manager-mcp",
        "container-manager-mcp",
    ]


def test_shipped_default_tools_cover_github_and_gitlab_issues_and_prs():
    from agent_utilities.core.config import AgentConfig

    default = AgentConfig.model_fields["mcp_always_load_tools"].get_default(
        call_default_factory=True
    )
    # Server-qualified so a shift in the multiplexer's DERIVED prefixes cannot
    # silently break the defaults.
    assert default == [
        "github-mcp:github_issues",
        "github-mcp:github_pulls",
        "gitlab-mcp:gitlab_issues",
        "gitlab-mcp:gitlab_merge_requests",
    ]


def test_default_server_names_use_catalog_keys_not_friendly_names():
    """Guards the specific failure of naming servers "container-manager" rather
    than the catalog key "container-manager-mcp", which would make every
    default silently unmountable."""
    from agent_utilities.core.config import (
        DEFAULT_MCP_ALWAYS_LOAD,
        DEFAULT_MCP_ALWAYS_LOAD_TOOLS,
    )

    assert all(name.endswith("-mcp") for name in DEFAULT_MCP_ALWAYS_LOAD)
    assert all(
        spec.split(":", 1)[0].endswith("-mcp") for spec in DEFAULT_MCP_ALWAYS_LOAD_TOOLS
    )


def test_always_load_lists_accept_comma_separated_env_form():
    from agent_utilities.core.config import AgentConfig

    cfg = AgentConfig(MCP_ALWAYS_LOAD="a-mcp, b-mcp ,, c-mcp")
    assert cfg.mcp_always_load == ["a-mcp", "b-mcp", "c-mcp"]


def test_always_load_can_be_disabled_entirely():
    from agent_utilities.core.config import AgentConfig

    assert AgentConfig(MCP_ALWAYS_LOAD="[]").mcp_always_load == []


# --------------------------------------------------------------------------- #
# 1. Server granularity
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_always_load_server_is_mounted_without_find_tools(tmp_path):
    mux = _mux(tmp_path, {CNT: ["cm_container_operations"]})
    mux._always_load_servers = [CNT]

    assert CNT not in mux.children  # nothing eager before the first session
    result = await ensure_always_loaded(_fake_mcp(), mux, session_key="s1")

    assert result["mounted_servers"] == [CNT]
    assert CNT in mux.children
    assert result["degraded"] == {}
    assert _prefixed(mux, CNT, "cm_container_operations") in mux.session_loaded("s1")


@pytest.mark.asyncio
async def test_always_load_runs_once_per_session(tmp_path):
    mux = _mux(tmp_path, {CNT: ["cm_container_operations"]})
    mux._always_load_servers = [CNT]
    fake = _fake_mcp()

    first = await ensure_always_loaded(fake, mux, session_key="s1")
    second = await ensure_always_loaded(fake, mux, session_key="s1")

    assert first["mounted_servers"] == [CNT]
    # The second call is memoized: same answer, and NO second mounting pass.
    assert second == first
    assert mux._start_child.await_count == 1


@pytest.mark.asyncio
async def test_middleware_mounts_always_load_before_the_first_tools_list(tmp_path):
    """The client's FIRST tools/list already contains the always-load surface.

    This is the whole user-visible contract: "loaded by default when graph-os
    first connects", not after a find_tools round trip.
    """
    mux = _mux(tmp_path, {CNT: ["cm_container_operations"]})
    mux._always_load_servers = [CNT]
    prefixed = _prefixed(mux, CNT, "cm_container_operations")
    middleware = SessionVisibilityMiddleware(mux, _fake_mcp())

    async def call_next(_context):
        return [_fake_tool(prefixed)]

    visible = await middleware.on_list_tools(MagicMock(), call_next)

    assert [t.name for t in visible] == [prefixed]


# --------------------------------------------------------------------------- #
# 2. Tool granularity
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_tool_level_always_load_exposes_only_the_named_tools(tmp_path):
    """A large server contributes ONLY its named tools.

    This is the point of the second config field: mounting github-mcp whole to
    reach issues and pulls would flood exactly the context the multiplexer
    exists to protect.
    """
    mux = _mux(
        tmp_path,
        {BIG: ["github_issues", "github_pulls", "github_actions", "github_releases"]},
    )
    mux._always_load_tool_specs = [f"{BIG}:github_issues", f"{BIG}:github_pulls"]

    result = await ensure_always_loaded(_fake_mcp(), mux, session_key="s1")

    assert result["mounted_servers"] == [BIG]
    assert result["degraded"] == {}
    assert set(result["exposed"]) == {
        _prefixed(mux, BIG, "github_issues"),
        _prefixed(mux, BIG, "github_pulls"),
    }
    # The server's OTHER tools are mounted but NOT exposed to the session.
    loaded = mux.session_loaded("s1")
    assert _prefixed(mux, BIG, "github_actions") not in loaded
    assert _prefixed(mux, BIG, "github_releases") not in loaded


@pytest.mark.asyncio
async def test_tool_spec_accepts_an_already_prefixed_name(tmp_path):
    mux = _mux(tmp_path, {BIG: ["github_issues", "github_actions"]})
    # Resolve the prefixed form the way an operator reading list_catalog would.
    mux.load_catalog()
    prefixed = _prefixed(mux, BIG, "github_issues")
    mux._always_load_tool_specs = [prefixed]

    result = await ensure_always_loaded(_fake_mcp(), mux, session_key="s1")

    assert result["exposed"] == [prefixed]


@pytest.mark.asyncio
async def test_tool_absent_from_its_server_is_reported_not_silently_dropped(tmp_path):
    mux = _mux(tmp_path, {BIG: ["github_issues"]})
    mux._always_load_tool_specs = [f"{BIG}:github_issues", f"{BIG}:github_nonexistent"]

    result = await ensure_always_loaded(_fake_mcp(), mux, session_key="s1")

    assert result["exposed"] == [_prefixed(mux, BIG, "github_issues")]
    assert f"{BIG}:github_nonexistent" in result["degraded"]


@pytest.mark.asyncio
async def test_tool_spec_for_an_unknown_server_is_reported(tmp_path):
    mux = _mux(tmp_path, {CNT: ["cm_container_operations"]})
    mux._always_load_tool_specs = ["not-a-real-name"]

    result = await ensure_always_loaded(_fake_mcp(), mux, session_key="s1")

    assert result["mounted_servers"] == []
    assert "not-a-real-name" in result["degraded"]


# --------------------------------------------------------------------------- #
# 3. Fail-soft — the property that keeps graph-os up
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_broken_always_load_server_degrades_and_does_not_raise(tmp_path):
    """A crash-looping always-load server must not fail the pass.

    58 fleet pods were simultaneously crash-looping on a fastmcp mismatch on the
    day this landed; eager-loading them must not take graph-os down.
    """
    mux = _mux(tmp_path, {CNT: ["cm_container_operations"]}, broken={"broken-mcp"})
    mux._always_load_servers = ["broken-mcp", CNT]

    result = await ensure_always_loaded(_fake_mcp(), mux, session_key="s1")

    # The healthy server still mounted — one bad entry does not poison the rest.
    assert result["mounted_servers"] == [CNT]
    assert "broken-mcp" in result["degraded"]
    assert "broken-mcp" not in mux.children


@pytest.mark.asyncio
async def test_broken_always_load_server_does_not_break_tools_list(tmp_path):
    """The restored-bug check for fail-soft: an eager mount that raises must not
    propagate out of the serving path."""
    mux = _mux(tmp_path, {}, broken={"broken-mcp"})
    mux._always_load_servers = ["broken-mcp"]
    middleware = SessionVisibilityMiddleware(mux, _fake_mcp())
    mux._global_visible = {"find_tools"}

    async def call_next(_context):
        return [_fake_tool("find_tools")]

    visible = await middleware.on_list_tools(MagicMock(), call_next)

    # graph-os still serves; the fleet stays reachable through the meta-tools.
    assert [t.name for t in visible] == ["find_tools"]


@pytest.mark.asyncio
async def test_a_pass_that_fails_wholesale_is_still_survivable(tmp_path):
    """Even a defect INSIDE the always-load implementation must not fail a
    request — the outer guard returns a degraded result rather than raising."""
    mux = _mux(tmp_path, {CNT: ["cm_container_operations"]})
    mux._always_load_servers = [CNT]
    mux.load_catalog = MagicMock(side_effect=RuntimeError("catalog exploded"))

    result = await ensure_always_loaded(_fake_mcp(), mux, session_key="s1")

    assert result["mounted_servers"] == []
    assert result["degraded"]


@pytest.mark.asyncio
async def test_no_declaration_means_nothing_is_mounted(tmp_path):
    """Fully-lazy remains reachable: an empty declaration mounts nothing."""
    mux = _mux(tmp_path, {CNT: ["cm_container_operations"]})

    assert mux.always_load_declared() is False
    result = await ensure_always_loaded(_fake_mcp(), mux, session_key="s1")

    assert result["mounted_servers"] == []
    assert mux._start_child.await_count == 0


# --------------------------------------------------------------------------- #
# Wiring: attach_fleet_loader reads the config
# --------------------------------------------------------------------------- #


def test_attach_fleet_loader_reads_the_declaration_from_config(tmp_path, monkeypatch):
    from agent_utilities.mcp import multiplexer as mux_mod

    monkeypatch.setenv("MCP_ALWAYS_LOAD", "alpha-mcp,beta-mcp")
    monkeypatch.setenv("MCP_ALWAYS_LOAD_TOOLS", '["alpha-mcp:one"]')

    assert mux_mod._always_load_setting("mcp_always_load", "MCP_ALWAYS_LOAD") == [
        "alpha-mcp",
        "beta-mcp",
    ]
    assert mux_mod._always_load_setting(
        "mcp_always_load_tools", "MCP_ALWAYS_LOAD_TOOLS"
    ) == ["alpha-mcp:one"]


def test_always_load_setting_falls_back_to_the_shipped_default(monkeypatch):
    """With nothing in the environment the SHIPPED defaults apply — otherwise
    Deliverable 3 would be inert in every real deployment."""
    from agent_utilities.mcp import multiplexer as mux_mod

    monkeypatch.delenv("MCP_ALWAYS_LOAD", raising=False)

    assert mux_mod._always_load_setting("mcp_always_load", "MCP_ALWAYS_LOAD") == [
        "tunnel-manager-mcp",
        "systems-manager-mcp",
        "repository-manager-mcp",
        "container-manager-mcp",
    ]


def test_malformed_declaration_degrades_to_lazy_rather_than_raising(monkeypatch):
    from agent_utilities.mcp import multiplexer as mux_mod

    monkeypatch.setenv("MCP_ALWAYS_LOAD", "[not valid json")

    assert mux_mod._always_load_setting("mcp_always_load", "MCP_ALWAYS_LOAD") == []
