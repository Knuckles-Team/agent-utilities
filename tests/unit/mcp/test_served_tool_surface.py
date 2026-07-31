"""The SERVED graph-os MCP tool surface, pinned per ``MCP_TOOL_MODE``.

CONCEPT:AU-ECO.mcp.fleet-meta-tools-always-on

Existing coverage stops at :func:`kg_server._build_server`, which registers only
the *mode-selected* tools. But what a client actually sees over ``tools/list`` is
``_build_server`` **plus** :func:`~agent_utilities.mcp.multiplexer.attach_fleet_loader`
(the five fleet meta-tools + the per-session visibility middleware), and that
attach happens later, in :func:`kg_server.mcp_server`. A regression that broke the
attach therefore changed the served surface from ~11 tools to 118 without a single
one of the ~9.9k existing tests noticing.

Two invariants are pinned here:

1. **The five fleet meta-tools are mode-independent infrastructure.**
   ``find_tools`` / ``list_catalog`` / ``load_tools`` / ``unload_tools`` /
   ``multiplexer_status`` are the ONLY way to reach anything the active mode holds
   back, so they must be served under ``intent``, ``condensed``, ``verbose`` AND
   ``both``. They are registered outside the mode switch on purpose — this test
   exists so no future change can quietly fold them into one mode's branch.
2. **``intent`` serves exactly the six verbs plus those five meta-tools** — the
   granular ``graph_*`` surface stays *registered* (REST/``REGISTERED_TOOLS`` are
   unaffected) but hidden, reachable only through ``load_tools``.

``bootstrap=False`` skips engine/daemon startup, so no live engine is needed —
this exercises tool *registration + visibility*, never execution.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from agent_utilities.mcp import kg_server
from agent_utilities.mcp.multiplexer import (
    SessionVisibilityMiddleware,
    attach_fleet_loader,
)
from agent_utilities.mcp.verbose_tools import VALID_TOOL_MODES, _provider_tools

#: Fleet meta-tools. Infrastructure, not a mode's tool set — always served.
FLEET_META_TOOLS = frozenset(
    {
        "find_tools",
        "list_catalog",
        "load_tools",
        "unload_tools",
        "multiplexer_status",
    }
)

#: The collapsed intent surface (CONCEPT:AU-ECO.mcp.intent-surface-condensed-collapse).
INTENT_VERBS = frozenset({"ask", "find", "act", "why", "write", "manage"})


def _served_surface(monkeypatch, tmp_path, mode: str) -> tuple[Any, Any, set[str]]:
    """Build graph-os exactly as ``mcp_server()`` does and return what it serves.

    Returns ``(mcp, mux, visible_tool_names)``. ``visible`` is computed through the
    REAL :class:`SessionVisibilityMiddleware` predicate that ``on_list_tools``
    filters with, for a fresh session that has loaded nothing.
    """
    config_path = tmp_path / "mcp_config.json"
    config_path.write_text(json.dumps({"mcpServers": {}}), encoding="utf-8")
    monkeypatch.setenv("MCP_TOOL_MODE", mode)
    monkeypatch.setenv("MCP_CONFIG", str(config_path))

    _args, mcp, _middlewares = kg_server._build_server(bootstrap=False)
    mux = attach_fleet_loader(mcp, config_path=str(config_path))
    middleware = SessionVisibilityMiddleware(mux, mcp)
    registered = set(_provider_tools(mcp))
    return mcp, mux, {name for name in registered if middleware._visible(name)}


@pytest.mark.parametrize("mode", sorted(VALID_TOOL_MODES))
def test_fleet_meta_tools_are_served_in_every_tool_mode(monkeypatch, tmp_path, mode):
    """The five meta-tools survive EVERY mode — they sit outside the mode switch.

    Parameterised over ``VALID_TOOL_MODES`` itself so a newly added mode is forced
    to honour the invariant rather than silently skipping it.
    """
    _mcp, _mux, visible = _served_surface(monkeypatch, tmp_path, mode)

    missing = FLEET_META_TOOLS - visible
    assert not missing, (
        f"MCP_TOOL_MODE={mode!r} does not serve the fleet meta-tools {sorted(missing)}; "
        "they are mode-independent infrastructure and must always be registered."
    )


def test_intent_mode_serves_exactly_the_verbs_and_the_meta_tools(monkeypatch, tmp_path):
    """The whole point of ``intent``: ~11 tool schemas, not ~118.

    An exact-set assertion (not a superset one) — the regression this pins leaked
    107 granular ``graph_*`` tools into the default view, which a superset check
    would have happily passed.
    """
    _mcp, _mux, visible = _served_surface(monkeypatch, tmp_path, "intent")

    assert visible == set(INTENT_VERBS) | set(FLEET_META_TOOLS)


def test_intent_mode_keeps_the_granular_surface_registered_and_load_tools_reachable(
    monkeypatch, tmp_path
):
    """Hidden, never lost: the granular tools are registered and ``load_tools``-able."""
    mcp, mux, visible = _served_surface(monkeypatch, tmp_path, "intent")

    registered = set(_provider_tools(mcp))
    # A representative granular tool is registered but held back from the view.
    assert "graph_query" in registered
    assert "graph_query" not in visible
    # ...and the gate knows it is locally revealable (no child mount needed), which
    # is exactly what load_tools flips.
    assert "graph_query" in mux._local_gated

    middleware = SessionVisibilityMiddleware(mux, mcp)
    session_key = "test-session"
    mux._session_loaded[session_key] = {"graph_query"}
    monkeypatch.setattr(
        "agent_utilities.mcp.multiplexer._session_key", lambda: session_key
    )
    assert middleware._visible("graph_query")


@pytest.mark.parametrize("mode", ["condensed", "verbose", "both"])
def test_non_intent_modes_serve_their_granular_surface_plus_the_meta_tools(
    monkeypatch, tmp_path, mode
):
    """Nothing is gated outside ``intent`` — the granular tools are served directly,
    and the meta-tools ride alongside them."""
    _mcp, mux, visible = _served_surface(monkeypatch, tmp_path, mode)

    assert FLEET_META_TOOLS <= visible
    assert not mux._local_gated, (
        f"MCP_TOOL_MODE={mode!r} must not gate the host's own tools; "
        f"gating is intent-only, got {sorted(mux._local_gated)[:5]}"
    )
    # The mode's own surface is actually served (condensed action tools in
    # condensed/both; the 1:1 expansion in verbose/both).
    assert len(visible) > len(FLEET_META_TOOLS) + len(INTENT_VERBS)


def test_mcp_protocol_error_resolves_on_the_installed_sdk():
    """The child-resilience layer binds a REAL exception class on either SDK line.

    ``mcp.shared.exceptions.McpError`` (SDK v1) was renamed ``MCPError`` in SDK v2.
    A hard import of one spelling raises ``ImportError`` at module scope on the
    other, and ``multiplexer.py`` imports ``child_resilience`` at module scope —
    which is how the whole fleet loader was taken down.
    """
    from agent_utilities.mcp.child_resilience import MCPError
    from agent_utilities.mcp.protocol_compat import mcp_protocol_error

    resolved = mcp_protocol_error()
    assert isinstance(resolved, type) and issubclass(resolved, BaseException)
    # Never a benign placeholder: `()`/`Exception` would make is_session_dead()
    # answer for every exception instead of the MCP protocol error.
    assert resolved is not Exception
    assert MCPError is resolved


def test_mcp_protocol_error_raises_when_neither_spelling_exists(monkeypatch):
    """No silent fallback — an SDK exposing neither name fails loudly."""
    from mcp.shared import exceptions as mcp_exceptions

    from agent_utilities.mcp.protocol_compat import mcp_protocol_error

    for name in ("MCPError", "McpError"):
        monkeypatch.delattr(mcp_exceptions, name, raising=False)

    with pytest.raises(ImportError, match="MCPError"):
        mcp_protocol_error()


def test_fleet_loader_attach_failure_is_fatal_not_swallowed(monkeypatch):
    """A failed attach must abort startup, not serve a silently wrong surface.

    The regression this pins was survivable *by design*: the attach was wrapped in
    ``except Exception: logger.error(...)``, so graph-os happily served 118 ungated
    tools with no ``load_tools`` at all. The meta-tools are infrastructure — losing
    them is a startup failure, and ``__cause__`` must survive to name the reason.
    """
    from unittest.mock import MagicMock, patch

    monkeypatch.setenv("IS_KG_SERVER", "false")

    args = MagicMock()
    args.transport = "stdio"
    args.host = "127.0.0.1"
    args.port = 8000
    args.auth_type = "none"
    mcp = MagicMock()
    boom = ImportError("cannot import name 'MCPError' from 'mcp.shared.exceptions'")

    with (
        patch("agent_utilities.core.config.load_config"),
        patch.object(kg_server, "_configure_graphos_otel"),
        patch.object(kg_server, "_configure_telemetry_engine_otel"),
        patch.object(kg_server, "_build_server", return_value=(args, mcp, [])),
        patch.object(kg_server, "_fleet_embed_fn", return_value=None),
        patch("agent_utilities.mcp.multiplexer.attach_fleet_loader", side_effect=boom),
        pytest.raises(RuntimeError, match="fleet loader attach failed") as captured,
    ):
        kg_server.mcp_server()

    assert captured.value.__cause__ is boom
    mcp.run.assert_not_called()
