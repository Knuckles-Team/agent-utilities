"""AU-CONTROL-R031: ``ensure_tools_registered()``'s headless build attaches
AU's own fleet loader, or records a precise, surfaced refusal when it cannot.

Before this fix, any process whose *only* build path was
``ensure_tools_registered()`` (a gateway-only process, e.g. standalone
agent-webui) got a ``mcp`` with no fleet multiplexer ever attached, so every
``find``/``act`` fleet call raised a generic "embedded/headless build"
error forever — even though AU already owns a complete, working fleet-attach
implementation (:func:`agent_utilities.mcp.multiplexer.attach_fleet_loader`),
used with these exact arguments by this module's own full composed server.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent_utilities.mcp import kg_server
from agent_utilities.mcp.tools.intent_tools import _fleet_mux


@pytest.mark.spec("AU-CONTROL-R031")
def test_attach_headless_fleet_loader_calls_attach_fleet_loader_with_composed_args() -> (
    None
):
    """Same arguments the full composed server passes at kg_server.py's
    ``state.fleet_mux = attach_fleet_loader(...)`` call."""
    mcp = SimpleNamespace()
    sentinel_embed = object()
    with (
        patch.object(kg_server, "_fleet_embed_fn", return_value=sentinel_embed),
        patch("agent_utilities.mcp.multiplexer.attach_fleet_loader") as mock_attach,
    ):
        kg_server._attach_headless_fleet_loader(mcp)

    mock_attach.assert_called_once_with(
        mcp,
        embed_fn=sentinel_embed,
        authority_scope=kg_server.verified_tool_session_scope,
        catalog_writer=kg_server._write_refreshed_fleet_catalog,
    )
    assert not hasattr(mcp, "_fleet_mux_unavailable_reason")


@pytest.mark.spec("AU-CONTROL-R031")
def test_attach_headless_fleet_loader_records_the_exact_failure_cause(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """No silent fallback: a failed attach leaves ``mcp`` unattached and
    records why, rather than a degraded or empty multiplexer.

    The exact cause (the exception's own message) reaches the server-side
    log -- an ``agent_utilities.*`` logger, sanitized by the process-wide
    log-privacy boundary before it leaves the process -- but the value
    stored on ``mcp`` is read verbatim by a served caller
    (``intent_tools._fleet_mux``), so it carries only the exception's CLASS
    name, never its message
    (``test_exception_surface_static_gate.py::test_served_packages_do_not_expose_raw_exception_text``).
    """
    mcp = SimpleNamespace()
    with (
        caplog.at_level(logging.WARNING, logger="agent_utilities.mcp.kg_server"),
        patch(
            "agent_utilities.mcp.multiplexer.attach_fleet_loader",
            side_effect=RuntimeError("no mcp_config.json found"),
        ),
    ):
        kg_server._attach_headless_fleet_loader(mcp)

    assert getattr(mcp, "_fleet_mux", None) is None
    reason = mcp._fleet_mux_unavailable_reason
    assert "RuntimeError" in reason
    assert "no mcp_config.json found" not in reason
    assert "no mcp_config.json found" in caplog.text
    assert "RuntimeError" in caplog.text


@pytest.mark.spec("AU-CONTROL-R031")
def test_ensure_tools_registered_attaches_fleet_loader_to_the_built_mcp() -> None:
    """``ensure_tools_registered()`` now attaches a fleet loader to the exact
    ``mcp`` its headless ``_build_server(bootstrap=False)`` call returns."""
    saved = dict(kg_server.REGISTERED_TOOLS)
    kg_server.REGISTERED_TOOLS.clear()
    built_mcp = SimpleNamespace()
    try:
        with (
            patch.object(
                kg_server,
                "_build_server",
                return_value=(SimpleNamespace(), built_mcp, []),
            ) as mock_build,
            patch.object(kg_server, "_attach_headless_fleet_loader") as mock_attach,
        ):
            kg_server.ensure_tools_registered()

        mock_build.assert_called_once_with(bootstrap=False)
        mock_attach.assert_called_once_with(built_mcp)
    finally:
        kg_server.REGISTERED_TOOLS.clear()
        kg_server.REGISTERED_TOOLS.update(saved)


def test_ensure_tools_registered_skips_rebuilding_when_already_populated() -> None:
    """The idempotency guard still short-circuits before any attach attempt."""
    saved = dict(kg_server.REGISTERED_TOOLS)
    kg_server.REGISTERED_TOOLS.clear()
    kg_server.REGISTERED_TOOLS["find"] = object()
    try:
        with (
            patch.object(kg_server, "_build_server") as mock_build,
            patch.object(kg_server, "_attach_headless_fleet_loader") as mock_attach,
        ):
            kg_server.ensure_tools_registered()

        mock_build.assert_not_called()
        mock_attach.assert_not_called()
    finally:
        kg_server.REGISTERED_TOOLS.clear()
        kg_server.REGISTERED_TOOLS.update(saved)


def test_fleet_mux_returns_the_attached_multiplexer() -> None:
    sentinel = object()
    mcp = SimpleNamespace(_fleet_mux=sentinel)
    assert _fleet_mux(mcp) is sentinel


def test_fleet_mux_raises_the_recorded_reason_when_unattached() -> None:
    mcp = SimpleNamespace(
        _fleet_mux=None,
        _fleet_mux_unavailable_reason=(
            "fleet loader attach failed in this headless build (RuntimeError: boom)"
        ),
    )
    with pytest.raises(ValueError, match="fleet loader attach failed"):
        _fleet_mux(mcp)


def test_fleet_mux_raises_the_generic_message_when_no_reason_recorded() -> None:
    mcp = SimpleNamespace()
    with pytest.raises(ValueError, match="embedded/headless build"):
        _fleet_mux(mcp)
