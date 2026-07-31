"""Unit tests for ``agent_utilities.mcp.tools.mcp_apps`` (CONCEPT:AU-ECO.ui.mcp-apps-host).

Covers BOTH shipped MCP Apps -- task-progress (pre-existing, previously untested)
and trace-waterfall (D-25-1, new): each entry-point tool registers with the
correct ``AppConfig``/``resource_uri``, returns the exact prop shape its HTML
expects on init, and its paired ``ui://`` resource returns self-contained,
dependency-free HTML that names the tool(s) it polls over the postMessage
bridge -- never a raw external network/script/style load.
"""

from __future__ import annotations

from typing import Any

import pytest

from agent_utilities.mcp.tools import mcp_apps


class _CollectingMCP:
    """Minimal FastMCP stand-in that captures ``@mcp.tool``/``@mcp.resource``
    registered functions and their config, mirroring
    ``test_engine_surface_tools.py``'s ``_CollectingMCP``."""

    def __init__(self) -> None:
        self.tools: dict[str, Any] = {}
        self.tool_configs: dict[str, dict[str, Any]] = {}
        self.resources: dict[str, Any] = {}
        self.resource_configs: dict[str, dict[str, Any]] = {}

    def tool(self, *, name: str, description: str = "", tags=None, app=None):
        def _deco(fn):
            self.tools[name] = fn
            self.tool_configs[name] = {
                "description": description,
                "tags": tags,
                "app": app,
            }
            return fn

        return _deco

    def resource(
        self, *, uri: str, name: str = "", description: str = "", mime_type: str = ""
    ):
        def _deco(fn):
            self.resources[uri] = fn
            self.resource_configs[uri] = {
                "name": name,
                "description": description,
                "mime_type": mime_type,
            }
            return fn

        return _deco


@pytest.fixture
def mcp() -> _CollectingMCP:
    server = _CollectingMCP()
    mcp_apps.register_mcp_apps_tools(server)
    return server


# ── registration parity: every app has BOTH a tool and a matching resource ──


def test_both_apps_register_a_tool_and_matching_resource(mcp: _CollectingMCP) -> None:
    assert set(mcp.tools) == {"graph_task_progress_app", "graph_trace_waterfall_app"}
    assert set(mcp.resources) == {
        mcp_apps.TASK_PROGRESS_RESOURCE_URI,
        mcp_apps.TRACE_WATERFALL_RESOURCE_URI,
    }
    for name, uri in (
        ("graph_task_progress_app", mcp_apps.TASK_PROGRESS_RESOURCE_URI),
        ("graph_trace_waterfall_app", mcp_apps.TRACE_WATERFALL_RESOURCE_URI),
    ):
        app_cfg = mcp.tool_configs[name]["app"]
        assert app_cfg.resource_uri == uri
        assert app_cfg.visibility == ["model"]


def test_register_mcp_apps_tools_skips_resource_on_a_tool_only_test_double() -> None:
    """server_factory.py's own custom_route registration is defended the same
    way: several unit tests build a server against a minimal FastMCP stand-in
    exposing only ``.tool()``, not ``.resource()`` -- registration must not
    raise in that case (the resource half is just skipped)."""

    class _ToolOnlyMCP:
        def __init__(self) -> None:
            self.tools: dict[str, Any] = {}

        def tool(self, *, name: str, description: str = "", tags=None, app=None):
            def _deco(fn):
                self.tools[name] = fn
                return fn

            return _deco

    server = _ToolOnlyMCP()
    mcp_apps.register_mcp_apps_tools(server)
    assert set(server.tools) == {"graph_task_progress_app", "graph_trace_waterfall_app"}


# ── task-progress app ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_task_progress_tool_returns_job_id_prop(mcp: _CollectingMCP) -> None:
    result = await mcp.tools["graph_task_progress_app"](job_id="orch-abc123")
    assert result == {"jobId": "orch-abc123"}


@pytest.mark.asyncio
async def test_task_progress_resource_is_self_contained_and_polls_graph_jobs(
    mcp: _CollectingMCP,
) -> None:
    html = await mcp.resources[mcp_apps.TASK_PROGRESS_RESOURCE_URI]()
    assert "<script>" in html
    assert "graph_jobs" in html
    assert "mcpapp/ready" in html and "mcpapp/init" in html
    # Self-contained: no external network/script/style/font load.
    for banned in ("http://", "https://", "<link ", "<script src"):
        assert banned not in html


# ── trace-waterfall app (D-25-1) ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_trace_waterfall_tool_returns_trace_id_prop(mcp: _CollectingMCP) -> None:
    result = await mcp.tools["graph_trace_waterfall_app"](trace_id="trace:1")
    assert result == {"traceId": "trace:1"}


@pytest.mark.asyncio
async def test_trace_waterfall_resource_is_self_contained_and_polls_graph_traces(
    mcp: _CollectingMCP,
) -> None:
    html = await mcp.resources[mcp_apps.TRACE_WATERFALL_RESOURCE_URI]()
    assert "<script>" in html
    assert "graph_traces" in html
    assert "waterfall" in html
    assert "mcpapp/ready" in html and "mcpapp/init" in html
    for banned in ("http://", "https://", "<link ", "<script src"):
        assert banned not in html
