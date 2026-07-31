"""The host-injected MCP delegation the WebUI backend delegates through.

CONCEPT:AU-ECO.mcp.webui-governed-mcp-delegation

``agent_webui.api_extensions`` never builds an MCP client of its own. It calls
two host-supplied workspace helpers — ``call_mcp_tool`` and
``read_mcp_resource`` — and reports 501 when they are absent, so that the
allow-list, credentials and transport stay with the host that mounts the UI.
This module is that host side.

Until now nothing supplied them: ``app.py`` injected seventeen workspace
helpers and neither of these two, so **every** WebUI route that dispatches a
fleet MCP tool (the Jira/GitHub/GitLab/data-science/scholarx/searxng/
home-assistant/calendar ecosystem panels, and the MCP Apps host's
``tools/call`` bridge) raised ``RuntimeError('Governed MCP delegation is not
configured')`` on its first line and rendered an error state.

Transport and authentication are not re-invented here: both helpers go through
:func:`agent_utilities.protocols.source_connectors.connectors.mcp_tool.call_tool_once`
/ :func:`~...mcp_tool.read_resource_once`, which resolve the server through the
fleet ``mcp_config`` and authenticate with the same OIDC client-credentials
bearer the multiplexer uses (``mcp.client_credentials.child_auth``). Because
the target URL comes from ``mcp_config`` it is the server's configured
authority, which is what a graph-os listener's ``MCP_ALLOWED_HOSTS`` check
accepts — a pod IP would be rejected with ``400 host rejected``.

**What bounds this.** The reachable set is exactly the servers declared in the
fleet ``mcp_config``: an unknown name raises ``McpToolSourceError`` before any
connection is opened. On the WebUI side the routes additionally require
``kg:admin`` (``agent_webui.server._ADMIN_MUTATION_ROUTE_PREFIXES``), because
invoking an arbitrary fleet tool is at least as powerful as any other admin
mutation.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

__all__ = ["webui_mcp_delegation_helpers"]


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
) -> dict[str, Any]:
    """Read one fleet MCP resource (e.g. an MCP App's ``ui://`` HTML)."""
    from agent_utilities.protocols.source_connectors.connectors.mcp_tool import (
        read_resource_once,
    )

    logger.debug("WebUI MCP delegation: resource read on %r", server_name)
    return await read_resource_once(server=server_name, uri=uri, timeout=timeout)


def webui_mcp_delegation_helpers() -> dict[str, Any]:
    """The ``call_mcp_tool`` / ``read_mcp_resource`` workspace helpers."""
    return {
        "call_mcp_tool": _call_mcp_tool,
        "read_mcp_resource": _read_mcp_resource,
    }
