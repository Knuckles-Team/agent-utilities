"""A process-wide standalone :class:`MCPMultiplexer` for non-serving consumers.

CONCEPT:AU-ECO.mcp.shared-multiplexer-singleton — GOC-60-W03/W04b.

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

:class:`MCPMultiplexer` is explicitly designed to support this — its
``_host_mcp`` attribute is documented as optional specifically to "preserve
the standalone probe and unit-test paths, which do not own a FastMCP server
or live forwarders" (see ``multiplexer.py``). This module gives those
consumers ONE shared instance rather than several independent ones, so a REST
payload, an MCP payload, and the WebUI's own in-process read all come from the
SAME catalog / probe-cache / session-visibility state instead of drifting
against each other — which is the whole point of "the REST payload equals the
MCP tool payload for the same session" (GOC-60-W03 acceptance evidence).

``multiplexer.py`` is READ-ONLY from this lane's perspective (GOC-60 worker
instructions) — this module only ever *consumes* its public
:class:`MCPMultiplexer` constructor and methods.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from agent_utilities.core.config import setting
from agent_utilities.mcp.multiplexer import MCPMultiplexer

__all__ = [
    "get_shared_multiplexer",
]

_shared_multiplexer: MCPMultiplexer | None = None
_construction_lock = asyncio.Lock()


def _default_config_path() -> Path:
    """Resolve the fleet ``mcp_config.json`` the same way ``attach_fleet_loader``
    does (``multiplexer._resolve_config_path``), duplicated in miniature here
    rather than importing that private helper — ``multiplexer.py`` stays
    read-only and unmodified, and this is the same three-branch precedence
    (explicit path, ``MCP_CONFIG`` setting, XDG config dir) documented there.
    """
    explicit = setting("MCP_CONFIG")
    if explicit:
        return Path(explicit)
    from agent_utilities.core.paths import config_dir

    return config_dir() / "mcp_config.json"


def _new_multiplexer() -> MCPMultiplexer:
    mux = MCPMultiplexer(_default_config_path())
    mux.load_catalog()  # parse config into the mountable-server catalog; spawns nothing
    return mux


async def get_shared_multiplexer() -> MCPMultiplexer:
    """Return the process-wide standalone multiplexer, constructing it once.

    Safe to call from any async context (REST route, in-process WebUI helper).
    Never spawns a child process and never owns a FastMCP serving loop — it is
    exactly the "standalone probe" shape :class:`MCPMultiplexer` already
    documents supporting.
    """
    global _shared_multiplexer
    if _shared_multiplexer is not None:
        return _shared_multiplexer
    async with _construction_lock:
        if _shared_multiplexer is None:
            _shared_multiplexer = _new_multiplexer()
        return _shared_multiplexer


def _reset_shared_multiplexer_for_tests() -> None:
    """Test-only: drop the cached singleton so the next call builds a fresh one.

    Without this, the FIRST test in a process to call
    :func:`get_shared_multiplexer` would pin every later test to its catalog
    (and its monkeypatches), silently cross-contaminating unrelated tests.
    """
    global _shared_multiplexer
    _shared_multiplexer = None
