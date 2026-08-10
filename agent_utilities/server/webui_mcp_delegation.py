"""The host-injected MCP delegation the WebUI backend delegates through.

CONCEPT:AU-ECO.mcp.webui-governed-mcp-delegation

``agent_webui.api_extensions`` never builds an MCP client of its own. It calls
three host-supplied workspace helpers — ``list_mcp_server_tools``,
``call_mcp_tool`` and ``read_mcp_resource`` — and reports 501 when one is
absent, so that the allow-list, credentials and transport stay with the host
that mounts the UI. This module is that host side.

Until now nothing supplied them: ``app.py`` injected seventeen workspace
helpers and none of these three, so **every** WebUI route that dispatches a
fleet MCP tool (the Jira/GitHub/GitLab/data-science/scholarx/searxng/
home-assistant/calendar ecosystem panels, and the MCP Apps host's
``tools/call`` bridge) raised ``RuntimeError('Governed MCP delegation is not
configured')`` on its first line and rendered an error state.

Transport and authentication are not re-invented here: ``call_mcp_tool`` and
``read_mcp_resource`` go through
:func:`agent_utilities.protocols.source_connectors.connectors.mcp_tool.call_tool_once`
/ :func:`~...mcp_tool.read_resource_once`, which resolve the server through the
fleet ``mcp_config`` and authenticate with the same OIDC client-credentials
bearer the multiplexer uses (``mcp.client_credentials.child_auth``). Because
the target URL comes from ``mcp_config`` it is the server's configured
authority, which is what a graph-os listener's ``MCP_ALLOWED_HOSTS`` check
accepts — a pod IP would be rejected with ``400 host rejected``.

``list_mcp_server_tools`` (GOC-60-W04b) instead reads from the process-wide
standalone :class:`~agent_utilities.mcp.multiplexer.MCPMultiplexer` in
:mod:`agent_utilities.mcp.shared_multiplexer` — the SAME instance the REST
catalog surface (``/api/mcp/catalog``, GOC-60-W03) reads from — so a server's
tool list can never drift between the two surfaces. It probes the one
requested server (connect → ``list_tools`` → release) rather than opening a
persistent client per WebUI request.

**What bounds this.** The reachable set is exactly the servers declared in the
fleet ``mcp_config``: an unknown name raises ``McpToolSourceError`` (for
``call_mcp_tool``/``read_mcp_resource``) or a typed probe error (for
``list_mcp_server_tools``) before any connection is opened. On the WebUI side
the routes additionally require ``kg:admin``
(``agent_webui.server._ADMIN_MUTATION_ROUTE_PREFIXES``), because invoking an
arbitrary fleet tool is at least as powerful as any other admin mutation.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable
from typing import TYPE_CHECKING, Any, Protocol, TypedDict

if TYPE_CHECKING:  # the runtime import stays lazy, inside the helpers
    from agent_utilities.protocols.source_connectors.connectors.mcp_tool import (
        McpResourceContent,
    )

logger = logging.getLogger(__name__)

__all__ = ["WebUiMcpDelegation", "webui_mcp_delegation_helpers"]


class _ListMcpServerTools(Protocol):
    def __call__(self, *, server_name: str) -> Awaitable[list[dict[str, Any]]]: ...


class _CallMcpTool(Protocol):
    def __call__(
        self,
        *,
        server_name: str,
        tool_name: str,
        arguments: dict[str, Any],
        timeout: float = ...,
    ) -> Awaitable[Any]: ...


class _ReadMcpResource(Protocol):
    def __call__(
        self, *, server_name: str, uri: str, timeout: float = ...
    ) -> Awaitable[McpResourceContent]: ...


class WebUiMcpDelegation(TypedDict):
    """The three workspace-helper keys the WebUI looks up by name.

    A typed seam on purpose: ``agent_webui.api_extensions`` resolves these with
    ``get_helper('list_mcp_server_tools')`` / ``get_helper('call_mcp_tool')`` /
    ``get_helper('read_mcp_resource')`` and answers 501 when one is missing, so
    a rename here must be a type error rather than a silently-unwired route.
    """

    list_mcp_server_tools: _ListMcpServerTools
    call_mcp_tool: _CallMcpTool
    read_mcp_resource: _ReadMcpResource


async def _list_mcp_server_tools(*, server_name: str) -> list[dict[str, Any]]:
    """List one fleet server's tools with descriptions and input schemas.

    Reads the process-wide standalone multiplexer (GOC-60-W03's
    ``shared_multiplexer``) rather than opening a dedicated client — the same
    catalog/probe-cache state the REST ``/api/mcp/catalog`` route reads, so
    this can never report a different tool list for the same server. Probing
    is a bounded connect -> ``list_tools`` -> release, cached by the
    multiplexer itself; an unknown/unreachable server raises rather than
    returning an empty list, so the caller's existing 503 handling
    (``api_extensions.list_mcp_server_tools``) sees a real failure instead of
    an indistinguishable "no tools" response.
    """
    from agent_utilities.mcp.shared_multiplexer import get_shared_multiplexer

    logger.debug("WebUI MCP delegation: tool inventory for %r", server_name)
    mux = await get_shared_multiplexer()
    info = await mux.probe_server(server_name)
    if info.get("error"):
        raise RuntimeError(f"MCP server {server_name!r} probe failed: {info['error']}")

    tools: list[dict[str, Any]] = []
    for tool in info.get("tools", []) or []:
        if isinstance(tool, dict):
            name = tool.get("name")
            description = tool.get("description", "") or ""
            schema = tool.get("inputSchema") or tool.get("input_schema") or {}
            meta = tool.get("meta")
        else:
            name = getattr(tool, "name", None)
            description = getattr(tool, "description", "") or ""
            schema = getattr(tool, "inputSchema", None) or {}
            meta = getattr(tool, "meta", None)
        if not name:
            continue
        entry: dict[str, Any] = {
            "name": name,
            "description": description,
            "input_schema": schema,
        }
        # BUG-071: an MCP Apps tool's UI binding (``meta["ui"]["resourceUri"]``,
        # the ``io.modelcontextprotocol/ui`` extension) is declared on the tool
        # DESCRIPTOR (this is a `tools/list`-time field, not a `tools/call`
        # result field — the entry-point tools in `mcp/tools/mcp_apps.py`
        # answer their calls with plain `{"jobId": ...}`/`{"traceId": ...}`, no
        # meta at all). Forward it so a WebUI app-launcher can discover which
        # `ui://` resource to fetch and render for a given tool, exactly as it
        # already does for `read_mcp_resource`. Omitted when absent so an
        # ordinary tool's shape is unchanged.
        if meta:
            entry["meta"] = meta
        tools.append(entry)
    return tools


async def _call_mcp_tool(
    *,
    server_name: str,
    tool_name: str,
    arguments: dict[str, Any],
    timeout: float = 30.0,
) -> Any:
    """Invoke one fleet MCP tool and return its decoded result.

    ``arguments`` is passed through verbatim (``params_style="args"``, no
    ``action`` envelope): the WebUI's callers already assemble the fleet
    ``action``/``params_json`` convention themselves, so wrapping again here
    would nest one envelope inside another.
    """
    from agent_utilities.protocols.source_connectors.connectors.mcp_tool import (
        call_tool_once,
    )

    logger.debug("WebUI MCP delegation: tool call on %r", server_name)
    return await call_tool_once(
        server=server_name,
        tool=tool_name,
        params=dict(arguments or {}),
        params_style="args",
        timeout=timeout,
    )


async def _read_mcp_resource(
    *,
    server_name: str,
    uri: str,
    timeout: float = 30.0,
) -> McpResourceContent:
    """Read one fleet MCP resource (e.g. an MCP App's ``ui://`` HTML)."""
    from agent_utilities.protocols.source_connectors.connectors.mcp_tool import (
        read_resource_once,
    )

    logger.debug("WebUI MCP delegation: resource read on %r", server_name)
    return await read_resource_once(server=server_name, uri=uri, timeout=timeout)


def webui_mcp_delegation_helpers() -> WebUiMcpDelegation:
    """The ``list_mcp_server_tools`` / ``call_mcp_tool`` / ``read_mcp_resource``
    workspace helpers."""
    return WebUiMcpDelegation(
        list_mcp_server_tools=_list_mcp_server_tools,
        call_mcp_tool=_call_mcp_tool,
        read_mcp_resource=_read_mcp_resource,
    )
