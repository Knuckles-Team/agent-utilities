"""Centralized Knowledge Graph REST surface for the API gateway.

CONCEPT:AU-ECO.mcp.knowledge-graph-exposure — Knowledge Graph API Gateway

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
from typing import Any

logger = logging.getLogger(__name__)

# Cached OWL/RDF bridge for the local SPARQL endpoint (built lazily from the
# active engine + a local owlready2 backend). Its rdflib materialization is
# cache-invalidated on LPG changes, so a single instance is safe to reuse.
_SPARQL_BRIDGE: Any = None


def _get_sparql_bridge() -> Any:
    """Return a cached OWLBridge for SPARQL, or ``None`` if unavailable."""
    global _SPARQL_BRIDGE
    if _SPARQL_BRIDGE is not None:
        return _SPARQL_BRIDGE
    try:
        from agent_utilities.knowledge_graph.backends.owl import create_owl_backend
        from agent_utilities.knowledge_graph.core.owl_bridge import OWLBridge
        from agent_utilities.mcp import kg_server

        engine = kg_server._get_engine()
        try:
            owl_backend = create_owl_backend()  # local owlready2 if installed
        except Exception:
            # owlready2 absent → still serve SPARQL via rdflib materialization of
            # the live LPG (query_sparql falls back when self.owl has no query_sparql).
            owl_backend = None
        _SPARQL_BRIDGE = OWLBridge(
            graph=engine.graph_compute,
            owl_backend=owl_backend,
            backend=getattr(engine, "backend", None),
        )
    except Exception as exc:  # pragma: no cover - SPARQL is best-effort
        logger.warning("Local SPARQL bridge unavailable: %s", exc)
        return None
    return _SPARQL_BRIDGE


def _mount_sparql_route(app, prefix: str = "/api") -> None:
    """Mount ``{prefix}/sparql`` — a local, zero-dependency SPARQL endpoint."""
    from starlette.requests import Request
    from starlette.responses import JSONResponse

    async def sparql_endpoint(request: Request) -> JSONResponse:
        # Query from ?query= (GET) or JSON body {"query": ...} / raw body (POST).
        query = request.query_params.get("query")
        if not query and request.method == "POST":
            try:
                body = await request.json()
                query = body.get("query") if isinstance(body, dict) else None
            except Exception:
                query = (await request.body()).decode("utf-8", "replace") or None
        if not query:
            return JSONResponse(
                {"status": "error", "message": "missing 'query'"}, status_code=400
            )
        bridge = _get_sparql_bridge()
        if bridge is None:
            return JSONResponse(
                {
                    "status": "error",
                    "message": "SPARQL layer unavailable (install agent-utilities[owl])",
                },
                status_code=503,
            )
        try:
            bindings = bridge.query_sparql(query)
            # W3C SPARQL-JSON-ish envelope (vars derived from the first binding).
            varnames = list(bindings[0].keys()) if bindings else []
            return JSONResponse(
                {
                    "status": "success",
                    "head": {"vars": varnames},
                    "results": {"bindings": bindings},
                }
            )
        except Exception as exc:
            return JSONResponse(
                {"status": "error", "message": str(exc)}, status_code=500
            )

    path = f"{prefix}/sparql"
    # Starlette-style add_route (works on FastAPI too): raw Request→Response with
    # no FastAPI body/param validation, matching the other kg_server endpoints.
    app.add_route(path, sparql_endpoint, methods=["GET", "POST"])
    logger.info("Mounted local SPARQL endpoint at %s", path)


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

    # Per-tenant token-bucket rate limiting (CONCEPT:AU-OS.observability.no-op-without-metrics). Added BEFORE
    # the identity middleware so it sits INSIDE it (Starlette: last added =
    # outermost) — the server-minted ActorContext is already in scope when
    # the bucket key (tenant → actor → client IP) is resolved. Disabled by
    # default (GATEWAY_RATE_LIMIT=0); /metrics and health paths are exempt.
    from agent_utilities.core.config import config

    if config.gateway_rate_limit > 0:
        from agent_utilities.gateway.rate_limit import GatewayRateLimitMiddleware

        app.add_middleware(GatewayRateLimitMiddleware)

    # Server-minted JWT identity (CONCEPT:AU-OS.identity.authenticated-identity-enforcement). Added AFTER the cypher
    # middleware so it sits OUTSIDE it — the lock-bypassing ``POST /cypher``
    # fast path is identity-scoped too. Validates ``Authorization: Bearer`` via
    # the existing JWKS machinery, scopes the request to an authenticated
    # ActorContext, and (with KG_AUTH_REQUIRED) rejects unauthenticated
    # requests with 401.
    from agent_utilities.security.request_identity import (
        ActorIdentityMiddleware,
        warn_unauthenticated_identity_once,
    )

    app.add_middleware(ActorIdentityMiddleware)
    if not config.kg_auth_required:
        warn_unauthenticated_identity_once()

    # Python-tier Prometheus metrics (CONCEPT:AU-OS.observability.no-op-without-metrics). Added LAST so the
    # metrics middleware is OUTERMOST — auth rejections (401) and rate-limit
    # rejections (429) are counted too. GET /metrics is exempt from the
    # identity middleware (scrapers cannot mint JWTs). Both the gateway and
    # the agent-webui backend mount through here, so both get the same
    # instrumentation. With prometheus_client absent (optional ``metrics``
    # extra) everything degrades to a no-op.
    if config.gateway_metrics:
        from agent_utilities.observability.gateway_metrics import (
            GatewayMetricsMiddleware,
            metrics_asgi_endpoint,
            metrics_endpoint,
        )

        app.add_middleware(GatewayMetricsMiddleware)
        if not any(getattr(r, "path", None) == "/metrics" for r in app.routes):
            if hasattr(app, "add_api_route"):  # FastAPI
                app.add_api_route(
                    "/metrics",
                    metrics_endpoint,
                    methods=["GET"],
                    include_in_schema=False,
                )
            else:  # plain Starlette
                app.add_route("/metrics", metrics_asgi_endpoint, methods=["GET"])

    kg_server._mount_rest_routes(app, prefix=prefix)

    # Local SPARQL endpoint (CONCEPT:AU-KG.query.vendor-agnostic-traversal) — served over the OWL/RDF bridge with
    # ZERO external dependencies (rdflib materialization of the live LPG + OWL
    # inferences); an external Fuseki/Stardog is optional enterprise scale-out, not
    # required. Works in the zero-dep tiny profile.
    _mount_sparql_route(app, prefix=prefix)

    # Native swarm supervisory plane (CONCEPT:AU-OS.safety.ontological-guardrail): /fleet/* + approvals.
    from agent_utilities.gateway.fleet import mount_fleet_routes

    mount_fleet_routes(app, prefix=prefix)

    # Granular, typed, OpenAPI-visible ontology/object reads layered on top of
    # the collapsed action-routed twins (resource-style GET-by-id + history).
    from agent_utilities.gateway.ontology_api import register_ontology_routes

    register_ontology_routes(app, prefix=prefix)

    # Granular, typed research surface (ARA over the one ontology-driven KG,
    # CONCEPT:AU-KG.research.best-effort-lightweight-never/2.80) — dispatches through the same research_artifact MCP tool.
    from agent_utilities.gateway.research_api import register_research_routes

    register_research_routes(app, prefix=prefix)

    logger.info(
        "Mounted centralized Knowledge Graph REST routes + fleet supervisory "
        "plane under %r (graph-os MCP is now a thin wrapper).",
        prefix,
    )
