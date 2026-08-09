"""Wiring tests for the WebUI's host-injected MCP delegation.

CONCEPT:AU-ECO.mcp.webui-governed-mcp-delegation

These drive the helpers end to end against a **real FastMCP server** (one tool
plus one ``ui://`` MCP App resource) over FastMCP's in-memory transport: only
the ``mcp_config`` lookup is substituted, so the argument marshalling, the
protocol round trip and the result decoding are the production ones.
"""

from __future__ import annotations

from typing import Any

import pytest

fastmcp = pytest.importorskip("fastmcp")

from agent_utilities.protocols.source_connectors.connectors.mcp_tool import (  # noqa: E402
    McpToolSourceConnector,
    McpToolSourceError,
    read_resource_once,
)
from agent_utilities.server.webui_mcp_delegation import (  # noqa: E402
    webui_mcp_delegation_helpers,
)

APP_URI = "ui://graph-os/task-progress.html"
APP_HTML = "<html><head></head><body><div id='jobId'>-</div></body></html>"


@pytest.fixture
def graph_os(monkeypatch) -> Any:
    """A faithful stand-in for graph-os, resolved as if it were in mcp_config."""
    server = fastmcp.FastMCP("graph-os-test")

    @server.tool(name="graph_jobs")
    async def graph_jobs(action: str, job_id: str = "") -> dict[str, Any]:
        return {"action": action, "jobId": job_id, "status": "working"}

    @server.resource(uri=APP_URI, name="Task Progress", mime_type="text/html")
    async def task_progress() -> str:
        return APP_HTML

    def _target(self: McpToolSourceConnector) -> Any:
        if self.server != "graph-os":
            raise McpToolSourceError(
                f"MCP server {self.server!r} not found in mcp_config.json"
            )
        return server

    monkeypatch.setattr(McpToolSourceConnector, "_client_target", _target)
    return server


def test_helpers_expose_all_three_delegation_entry_points() -> None:
    helpers = webui_mcp_delegation_helpers()
    assert set(helpers) == {
        "list_mcp_server_tools",
        "call_mcp_tool",
        "read_mcp_resource",
    }


@pytest.mark.asyncio
async def test_call_mcp_tool_reaches_the_server_with_verbatim_arguments(
    graph_os,
) -> None:
    """The WebUI's callers already build the fleet envelope themselves.

    So the helper must forward ``arguments`` as plain tool arguments — a second
    ``action``/``params_json`` wrap would arrive as an unknown argument.
    """
    call_mcp_tool = webui_mcp_delegation_helpers()["call_mcp_tool"]

    result = await call_mcp_tool(
        server_name="graph-os",
        tool_name="graph_jobs",
        arguments={"action": "status", "job_id": "orch-1"},
    )

    assert result == {"action": "status", "jobId": "orch-1", "status": "working"}


@pytest.mark.asyncio
async def test_list_mcp_server_tools_reads_the_shared_multiplexer(
    monkeypatch,
) -> None:
    """GOC-60-W04b: ``list_mcp_server_tools`` must read the SAME shared
    standalone multiplexer as the REST catalog surface (W03), not a second,
    independent inventory mechanism that could drift from it.
    """
    from agent_utilities.mcp import shared_multiplexer as shared_mux_mod

    class _StubMux:
        async def probe_server(self, server_name: str) -> dict:
            assert server_name == "github-api"
            return {
                "tools": [
                    {
                        "name": "create_issue",
                        "description": "Open an issue",
                        "inputSchema": {"type": "object"},
                    }
                ],
                "error": None,
            }

    async def _get_stub() -> Any:
        return _StubMux()

    monkeypatch.setattr(shared_mux_mod, "get_shared_multiplexer", _get_stub)

    list_mcp_server_tools = webui_mcp_delegation_helpers()["list_mcp_server_tools"]
    tools = await list_mcp_server_tools(server_name="github-api")

    assert tools == [
        {
            "name": "create_issue",
            "description": "Open an issue",
            "input_schema": {"type": "object"},
        }
    ]


@pytest.mark.asyncio
async def test_list_mcp_server_tools_raises_on_a_probe_error_rather_than_returning_empty(
    monkeypatch,
) -> None:
    """A failed probe must raise (surfaced by the caller as a typed 503), not
    come back as an indistinguishable empty tool list (GOC-60 invariant 1)."""
    from agent_utilities.mcp import shared_multiplexer as shared_mux_mod

    class _StubMux:
        async def probe_server(self, server_name: str) -> dict:
            return {"tools": [], "error": "not in catalog"}

    async def _get_stub() -> Any:
        return _StubMux()

    monkeypatch.setattr(shared_mux_mod, "get_shared_multiplexer", _get_stub)

    list_mcp_server_tools = webui_mcp_delegation_helpers()["list_mcp_server_tools"]
    with pytest.raises(RuntimeError, match="not in catalog"):
        await list_mcp_server_tools(server_name="not-a-real-server")


@pytest.mark.asyncio
async def test_read_mcp_resource_returns_the_app_html(graph_os) -> None:
    read_mcp_resource = webui_mcp_delegation_helpers()["read_mcp_resource"]

    resource = await read_mcp_resource(server_name="graph-os", uri=APP_URI)

    assert resource["uri"] == APP_URI
    assert resource["text"] == APP_HTML
    assert resource["mimeType"] == "text/html"


@pytest.mark.asyncio
async def test_an_unconfigured_server_is_refused_before_any_connection(
    graph_os,
) -> None:
    """The reachable set is exactly what ``mcp_config`` declares."""
    call_mcp_tool = webui_mcp_delegation_helpers()["call_mcp_tool"]

    with pytest.raises(McpToolSourceError):
        await call_mcp_tool(
            server_name="not-a-configured-server",
            tool_name="graph_jobs",
            arguments={"action": "status"},
        )


@pytest.mark.asyncio
async def test_a_missing_resource_raises_rather_than_returning_blank(
    graph_os,
) -> None:
    with pytest.raises(McpToolSourceError):
        await read_resource_once(server="graph-os", uri="ui://graph-os/absent.html")


def test_the_web_ui_host_injects_the_delegation_helpers() -> None:
    """``server/app.py`` must actually add them to the helper bundle.

    Without this the WebUI mounts with a delegation seam nobody filled and every
    fleet-tool route answers 501 — the regression this module exists to close.
    """
    from pathlib import Path

    import agent_utilities.server.app as app_module

    source = Path(app_module.__file__).read_text(encoding="utf-8")
    assert "webui_mcp_delegation_helpers()" in source
    injected = source.index("helpers.update(webui_mcp_delegation_helpers())")
    created = source.index("web_app = create_agent_web_app(")
    assert injected < created, (
        "the delegation helpers must be merged into `helpers` BEFORE it is "
        "handed to create_agent_web_app"
    )
