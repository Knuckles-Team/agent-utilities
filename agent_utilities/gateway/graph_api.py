"""Centralized Knowledge Graph REST surface for the API gateway.

CONCEPT:ECO-4.0 — Knowledge Graph API Gateway

The full Knowledge Graph REST API (``/graph/*``, ``/sessions``, ``/goals``,
``/tools``) is now owned by this gateway rather than the ``graph-os`` MCP server,
which has been slimmed to a thin FastMCP wrapper (MCP tools only). Funnelling all
graph HTTP traffic through this single persistent process eliminates the
embedded-DB file-lock contention that arises when many clients
(agent-terminal-ui, geniusbot, subagents, ingestion scripts) each open the graph
store directly.

The route table and handler implementations are the canonical ones defined in
:mod:`agent_utilities.mcp.kg_server`; we import and mount them here so the two
never drift. :func:`register_graph_routes` is the single entry point used by
:mod:`agent_utilities.server.app`.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def register_graph_routes(app, prefix: str = "/api") -> None:
    """Mount the centralized Knowledge Graph REST surface onto ``app``.

    Args:
        app: The FastAPI/Starlette application to mount routes on.
        prefix: Path prefix for every route (default ``/api`` → e.g.
            ``/api/graph/query``, ``/api/sessions``).

    Registers all KG tools (so the handlers can dispatch via ``REGISTERED_TOOLS``)
    and adds the lock-bypassing ``POST /cypher`` fast-path middleware
    (``CentralizedCypherMiddleware``) which routes raw Cypher straight to the
    active engine with backpressure + a short read cache.
    """
    from agent_utilities.mcp import kg_server

    # Populate REGISTERED_TOOLS without starting the MCP server's own engine
    # bootstrap — the gateway owns the engine/daemon lifecycle.
    kg_server.ensure_tools_registered()

    # Lock-bypassing direct-Cypher fast path (POST /cypher). Added as ASGI
    # middleware so it short-circuits before route matching; it passes through
    # every non-/cypher request untouched.
    try:
        app.add_middleware(kg_server.CentralizedCypherMiddleware)
    except Exception as exc:  # pragma: no cover - best-effort, never fatal
        logger.warning("Could not attach CentralizedCypherMiddleware: %s", exc)

    kg_server._mount_rest_routes(app, prefix=prefix)
    logger.info(
        "Mounted centralized Knowledge Graph REST routes under %r (graph-os MCP "
        "is now a thin wrapper).",
        prefix,
    )
