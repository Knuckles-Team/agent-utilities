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

from starlette.requests import Request
from starlette.responses import JSONResponse

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
        logger.warning("Local SPARQL bridge unavailable (%s)", type(exc).__name__)
        return None
    return _SPARQL_BRIDGE


def _mount_sparql_route(app, prefix: str = "/api") -> None:
    """Mount ``{prefix}/sparql`` — a local, zero-dependency SPARQL endpoint."""

    async def sparql_endpoint(request: Request) -> JSONResponse:
        from agent_utilities.knowledge_graph.core.session import (
            resolve_session,
            use_session,
        )

        session = resolve_session(required_scope="kg:read")
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
            # ``bridge.query_sparql`` targets whatever named graph the AMBIENT
            # session is pinned to (OWLBridge -> GraphComputeEngine.sparql ->
            # the session-routed engine client) -- retarget per graph the
            # actor may read (GOC-61: org graph(s) then commons, same
            # ``accessible_graphs`` ordering ``tenant_sharing.read_union``
            # uses for Cypher) so a SPARQL query issued under a tenant-pinned
            # session still sees the commons graph. ``read_union`` itself is
            # NOT reused here: it additionally applies the Cypher-shaped
            # ``filter_commons_catalog`` node-type restriction to commons
            # rows, which would drop every SPARQL binding (a binding carries
            # no Cypher ``node_type``) -- see its docstring.
            from agent_utilities.knowledge_graph.core.tenant_sharing import (
                accessible_graphs,
            )

            bindings: list[dict[str, Any]] = []
            for graph_name in accessible_graphs(session.actor):
                try:
                    with use_session(session.with_graph(graph_name)):
                        bindings.extend(bridge.query_sparql(query))
                except Exception as exc:  # noqa: BLE001 — one graph down ≠ whole read down
                    logger.debug(
                        "sparql union: graph %s unavailable (%s)",
                        graph_name,
                        type(exc).__name__,
                    )
                    continue
            # W3C SPARQL-JSON-ish envelope (vars derived from the first binding).
            varnames = list(bindings[0].keys()) if bindings else []
            return JSONResponse(
                {
                    "status": "success",
                    "head": {"vars": varnames},
                    "results": {"bindings": bindings},
                }
            )
        except Exception:
            return JSONResponse(
                {"status": "error", "message": "SPARQL query failed"},
                status_code=500,
            )

    path = f"{prefix}/sparql"
    # Starlette-style add_route (works on FastAPI too): raw Request→Response with
    # no FastAPI body/param validation, matching the other kg_server endpoints.
    app.add_route(path, sparql_endpoint, methods=["GET", "POST"])
    logger.info("Mounted local SPARQL endpoint")


SQL_SCHEMA_PATH = "/graph/sql-schema"

SQL_SCHEMA_SUMMARY = "Introspect the engine SQL catalog (read-only, no caller SQL)"

SQL_SCHEMA_DESCRIPTION = (
    "Return the `catalogs -> tables/views -> columns` projection of the "
    "engine's synthesized, read-only `information_schema` -- the schema tree a "
    "catalog browser renders. Accepts an OPTIONAL JSON body "
    '`{"schema": "<name>"}`; it carries **no SQL**. Every statement issued is a '
    "server-authored constant "
    "(`agent_utilities.mcp.tools.graph_tools.CATALOG_STATEMENTS`) and the schema "
    "filter is validated as a SQL identifier and applied in Python, so there is "
    "no interpolation and no path to row data. Dispatches through the same "
    "`_execute_tool` action core as the MCP `graph_table` tool, under the "
    "caller's own GraphSession (`kg:read`) and the engine's row-level "
    "isolation. Fails closed: an unreadable catalog is a typed 4xx/5xx error, "
    "never an empty success. The response's `capabilities` block reports which "
    "facets the engine can actually answer (primary keys and nullability are "
    "currently not tracked engine-side)."
)


async def _sql_schema_endpoint(request: Request) -> JSONResponse:
    """Serve ``POST /graph/sql-schema`` — the read-only catalog projection."""
    from agent_utilities.knowledge_graph.core.session import resolve_session
    from agent_utilities.mcp.tools.graph_tools import SqlSchemaUnavailable, sql_schema

    try:
        resolve_session(required_scope="kg:read")
    except Exception as exc:  # noqa: BLE001 — authz failures are 403, never 500
        logger.info("sql-schema denied (%s)", type(exc).__name__)
        return JSONResponse(
            {
                "status": "error",
                "code": "forbidden",
                "message": "SQL catalog introspection denied",
            },
            status_code=403,
        )

    try:
        schema = await _sql_schema_filter(request)
        return JSONResponse(await sql_schema(schema=schema))
    except SqlSchemaUnavailable as exc:
        return JSONResponse(exc.as_payload(), status_code=exc.status_code)


async def _sql_schema_filter(request: Request) -> str | None:
    """The optional ``schema`` name from the request body (never SQL text)."""
    try:
        body = await request.json()
    except Exception:  # noqa: BLE001 — an absent/blank body means "every schema"
        return None
    if not isinstance(body, dict):
        return None
    return body.get("schema")


def _mount_sql_schema_route(app, prefix: str = "/api") -> None:
    """Mount ``{prefix}/graph/sql-schema``.

    Registered with FastAPI's ``add_api_route`` (not raw ``add_route``) so the
    route is visible in the generated OpenAPI spec -- ``scripts/
    check_openapi_coverage.py`` ratchets on undocumented raw-Starlette routes,
    and a new one would be a regression.
    """
    path = f"{prefix}{SQL_SCHEMA_PATH}"
    if any(getattr(r, "path", None) == path for r in getattr(app, "routes", [])):
        return
    if hasattr(app, "add_api_route"):  # FastAPI
        app.add_api_route(
            path,
            _sql_schema_endpoint,
            methods=["POST"],
            name="graph_sql_schema",
            summary=SQL_SCHEMA_SUMMARY,
            description=SQL_SCHEMA_DESCRIPTION,
        )
    else:  # plain Starlette
        app.add_route(path, _sql_schema_endpoint, methods=["POST"])
    logger.info("Mounted read-only SQL catalog introspection at %s", path)


def register_graph_routes(app, prefix: str = "/api") -> None:
    """Mount the centralized Knowledge Graph REST surface onto ``app``.

    Args:
        app: The FastAPI/Starlette application to mount routes on.
        prefix: Path prefix for every route (default ``/api`` → e.g.
            ``/api/graph/query``, ``/api/sessions``).

    Registers all KG tools so handlers dispatch through ``REGISTERED_TOOLS``.
    Graph queries use the typed ``/api/graph/query`` action surface; there is no
    second raw query route.
    """
    from agent_utilities.mcp import kg_server

    # Populate REGISTERED_TOOLS without starting the MCP server's own engine
    # bootstrap — the gateway owns the engine/daemon lifecycle.
    kg_server.ensure_tools_registered()

    # Per-tenant token-bucket rate limiting (CONCEPT:AU-OS.observability.no-op-without-metrics). Added BEFORE
    # the identity middleware so it sits INSIDE it (Starlette: last added =
    # outermost) — the server-minted ActorContext is already in scope when
    # the bucket key (tenant → actor → client IP) is resolved. Disabled by
    # default (GATEWAY_RATE_LIMIT=0); /metrics and health paths are exempt.
    from agent_utilities.core.config import config

    if config.gateway_rate_limit > 0:
        from agent_utilities.gateway.rate_limit import GatewayRateLimitMiddleware

        app.add_middleware(GatewayRateLimitMiddleware)

    # Server-minted JWT identity (CONCEPT:AU-OS.identity.authenticated-identity-enforcement).
    # Validates ``Authorization: Bearer`` via
    # the existing JWKS machinery, scopes the request to an authenticated
    # ActorContext + GraphSession, and rejects unauthenticated requests with 401.
    from agent_utilities.security.request_identity import ActorIdentityMiddleware

    app.add_middleware(ActorIdentityMiddleware)

    # Python-tier Prometheus metrics (CONCEPT:AU-OS.observability.no-op-without-metrics). Added LAST so the
    # metrics middleware is OUTERMOST — auth rejections (401) and rate-limit
    # rejections (429) are counted too. GET /metrics remains behind the
    # identity middleware when KG authentication is required. Both the gateway and
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

    # Read-only SQL catalog introspection (CONCEPT:AU-KG.query.raw-python) — the
    # catalogs → tables/views → columns projection the agent-webui catalog
    # browser renders. Deliberately NOT a raw SQL passthrough: it takes no query
    # text, only an optional identifier-validated schema filter, and dispatches
    # through the same ``_execute_tool`` action core as the MCP ``graph_table``
    # tool. See agent_utilities/mcp/tools/graph_tools.py.
    _mount_sql_schema_route(app, prefix=prefix)

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

    # Read-only tenant/principal-scoped view over the engine-native fleet
    # catalog.  The registry module owns only transport/schema shaping; its
    # writer remains ``fleet_catalog_tables`` and no live MCP probing occurs.
    from agent_utilities.gateway.registry_api import register_registry_routes

    register_registry_routes(app, prefix=prefix)

    # Browser OAuth callbacks are part of the same authenticated gateway
    # surface; pass the graph prefix explicitly so providers register the
    # exact deployed callback path without accidental double-prefixing.
    from agent_utilities.gateway.remote_oauth_api import register_remote_oauth_routes

    register_remote_oauth_routes(app, prefix=prefix)

    logger.info(
        "Mounted centralized Knowledge Graph REST routes + fleet supervisory "
        "plane under %r (graph-os MCP is now a thin wrapper).",
        prefix,
    )
