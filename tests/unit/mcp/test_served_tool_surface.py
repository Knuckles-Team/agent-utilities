"""The SERVED graph-os MCP tool surface: one intent contract.

CONCEPT:AU-ECO.mcp.fleet-meta-tools-always-on

What a client sees over ``tools/list`` is :func:`kg_server._build_server`
**plus** :func:`~agent_utilities.mcp.multiplexer.attach_fleet_loader`, which
attaches later, in :func:`kg_server.mcp_server`. The fleet is reached through
the intent tools (``find`` discovers, ``act`` calls, ``manage`` loads), so the
attach must add NO tool and no visibility layer — the served list stays the six
verbs plus the two MCP Apps launchers. A regression in the attach once changed
the served surface from ~14 tools to 118 without any other test noticing.

``bootstrap=False`` skips engine/daemon startup, so no live engine is needed.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from agent_utilities.mcp import kg_server, shared_multiplexer
from agent_utilities.mcp.multiplexer import attach_fleet_loader
from agent_utilities.mcp.verbose_tools import _provider_tools

#: MCP Apps entry-point tools (``mcp/tools/mcp_apps.py``): the MCP Apps
#: extension binds a UI to a listed tool's ``_meta.ui.resourceUri``.
MCP_APP_TOOLS = frozenset({"graph_task_progress_app", "graph_trace_waterfall_app"})

#: The retired fleet meta-tools — served through the intent verbs now.
RETIRED_META_TOOLS = frozenset(
    {
        "find_tools",
        "list_catalog",
        "load_tools",
        "unload_tools",
        "catalog_refresh",
        "catalog_dispatch",
        "catalog_session_resume",
        "multiplexer_status",
    }
)

INTENT_VERBS = frozenset({"ask", "find", "act", "why", "write", "manage"})


def _served_surface(monkeypatch, tmp_path) -> tuple[Any, Any, set[str]]:
    """Build graph-os exactly as ``mcp_server()`` does; return what it serves."""
    shared_multiplexer._reset_served_multiplexer_for_tests()
    config_path = tmp_path / "mcp_config.json"
    config_path.write_text(json.dumps({"mcpServers": {}}), encoding="utf-8")
    monkeypatch.setenv("MCP_CONFIG", str(config_path))

    _args, mcp, _middlewares = kg_server._build_server(bootstrap=False)
    mux = attach_fleet_loader(mcp, config_path=str(config_path))
    return mcp, mux, set(_provider_tools(mcp))


def test_served_surface_is_exactly_the_verbs_and_the_app_launchers(
    monkeypatch, tmp_path
):
    """An exact-set assertion: the regression this pins leaked 107 granular
    ``graph_*`` tools into the default view, which a superset check passes."""
    _mcp, _mux, served = _served_surface(monkeypatch, tmp_path)
    assert served == set(INTENT_VERBS) | set(MCP_APP_TOOLS)
    assert not served & RETIRED_META_TOOLS


def test_the_fleet_multiplexer_is_bound_for_the_intent_tools(monkeypatch, tmp_path):
    """``find``/``act``/``manage`` reach the fleet through ``mcp._fleet_mux``."""
    mcp, mux, _served = _served_surface(monkeypatch, tmp_path)
    assert mcp._fleet_mux is mux
    assert callable(mux.require_capability)


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


def test_mcp_protocol_exception_uses_v2_constructor_shape(monkeypatch):
    """SDK v2 receives code/message/data directly, never an ErrorData wrapper."""
    from mcp.shared import exceptions as mcp_exceptions

    from agent_utilities.mcp.protocol_compat import mcp_protocol_exception

    class _V2Error(BaseException):
        def __init__(self, code, message, data=None):
            self.code = code
            self.message = message
            self.data = data

    monkeypatch.setattr(mcp_exceptions, "MCPError", _V2Error, raising=False)
    error = mcp_protocol_exception(-32602, "invalid task", {"task": "one"})

    assert isinstance(error, _V2Error)
    assert (error.code, error.message, error.data) == (
        -32602,
        "invalid task",
        {"task": "one"},
    )


def test_fleet_loader_attach_failure_is_fatal_not_swallowed(monkeypatch):
    """A failed attach must abort startup, not serve a silently wrong surface.

    The regression this pins was survivable *by design*: the attach was wrapped in
    ``except Exception: logger.error(...)``, so graph-os happily served 118 ungated
    tools with no fleet access at all. Fleet access is infrastructure — losing it
    is a startup failure, and ``__cause__`` must survive to name the reason.
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
