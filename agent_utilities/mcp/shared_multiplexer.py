"""Process binding for the one multiplexer actually served by GraphOS.

CONCEPT:AU-ECO.multiplexer.tool-gateway-catalog

``attach_fleet_loader`` builds a fresh :class:`MCPMultiplexer` for a directly
served graph-os MCP process (``mcp_server()`` in ``kg_server.py``) — that
process owns the FastMCP serving loop the multiplexer's live-forwarder
bookkeeping is normally wired against. Two OTHER consumers need the SAME
dispatchable-truth catalog with no serving loop of their own:

* the REST twin of ``list_catalog``/``multiplexer_status``
  (:mod:`agent_utilities.server.routers.mcp_catalog`, GOC-60-W03), and
* the WebUI's governed MCP delegation seam's ``list_mcp_server_tools`` helper
  (:mod:`agent_utilities.server.webui_mcp_delegation`, GOC-60-W04b), and the
  WebUI's own MCP-servers inventory panel
  (``agent_webui.api_extensions.list_all_tools``), consuming this in-process
  rather than over HTTP.

REST and WebUI are adapters over the served catalog authority. They must not
construct a detached multiplexer with an independent cache, generation, child
pool, or writer. Composition binds the served instance once; consumers fail
closed until that binding exists.
"""

from __future__ import annotations

from agent_utilities.mcp.multiplexer import MCPMultiplexer

__all__ = [
    "bind_served_multiplexer",
    "get_served_multiplexer",
]

_served_multiplexer: MCPMultiplexer | None = None


def bind_served_multiplexer(multiplexer: MCPMultiplexer) -> None:
    """Bind exactly one process-local served authority, idempotently."""
    global _served_multiplexer
    if _served_multiplexer is not None and _served_multiplexer is not multiplexer:
        raise RuntimeError("a different MCP catalog authority is already bound")
    _served_multiplexer = multiplexer


async def get_served_multiplexer() -> MCPMultiplexer:
    """Return the served authority or fail closed before composition."""
    if _served_multiplexer is None:
        raise RuntimeError("served MCP catalog authority is not bound")
    return _served_multiplexer


def _reset_served_multiplexer_for_tests() -> None:
    """Test-only reset for independent application compositions."""
    global _served_multiplexer
    _served_multiplexer = None
