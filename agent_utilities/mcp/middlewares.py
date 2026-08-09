#!/usr/bin/python
"""FastMCP Middlewares Module.

This module defines custom middlewares for FastMCP servers. It handles
user token extraction for delegation and JWT claims logging to provide
enhanced observability and authorization context during tool execution.
"""

import time

from fastmcp.server.middleware import Middleware, MiddlewareContext
from fastmcp.utilities.logging import get_logger

logger = get_logger(name="TokenMiddleware")


class ToolMetricsMiddleware(Middleware):
    """Record per-tool call count, latency and error outcome as Prometheus
    metrics for a standalone MCP server (CONCEPT:AU-OS.observability.no-op-without-metrics).

    Mounted by ``create_mcp_server`` on every server the factory builds, so the
    server's own ``GET /metrics`` route exposes per-tool request rate, p95
    latency and error rate — the server-side complement to the multiplexer's
    ``MCP_CHILD_*`` view. Degrades to a no-op when the optional ``metrics``
    extra (``prometheus_client``) is absent.
    """

    async def on_call_tool(self, context: MiddlewareContext, call_next):
        from agent_utilities.observability.gateway_metrics import (
            MCP_TOOL_CALLS,
            MCP_TOOL_DURATION,
            MCP_TOOL_IN_FLIGHT,
        )

        tool = getattr(context.message, "name", None) or "unknown"
        MCP_TOOL_IN_FLIGHT.inc()
        start = time.perf_counter()
        outcome = "ok"
        try:
            return await call_next(context)
        except Exception:
            outcome = "error"
            raise
        finally:
            MCP_TOOL_DURATION.labels(tool).observe(time.perf_counter() - start)
            MCP_TOOL_CALLS.labels(tool, outcome).inc()
            MCP_TOOL_IN_FLIGHT.dec()


class UserTokenMiddleware(Middleware):
    """Middleware to extract and store user tokens for downstream delegation.

    If delegation is enabled, this middleware accepts the token and claims
    exposed by FastMCP's configured authentication provider and binds them to
    request-scoped context variables for downstream delegation.
    """

    def __init__(self, config: dict):
        self.config = config

    async def on_call_tool(self, context: MiddlewareContext, call_next):
        if not self.config["enable_delegation"]:
            return await call_next(context)

        # Never trust a raw message/header copy. FastMCP exposes this object only
        # after its configured auth provider has validated the credential.
        try:
            from fastmcp.server.dependencies import get_access_token

            access_token = get_access_token()
        except Exception:
            access_token = None
        raw_token = getattr(access_token, "token", None)
        claims = getattr(access_token, "claims", None)
        if (
            not isinstance(raw_token, str)
            or not raw_token
            or len(raw_token) > 16_384
            or not isinstance(claims, dict)
        ):
            logger.warning("Delegation denied: validated user token unavailable")
            raise PermissionError("Validated user token required for delegation")

        from agent_utilities.mcp.delegated_auth import (
            _reset_delegated_identity,
            _set_delegated_identity,
        )

        tokens = _set_delegated_identity(raw_token, claims)
        try:
            return await call_next(context)
        finally:
            _reset_delegated_identity(tokens)


class JWTClaimsLoggingMiddleware(Middleware):
    """Middleware for logging JWT authentication claims on responses.

    Captures and logs the subject, client ID, and scopes from validated
    JWT claims to provide an audit trail for successful authentications.
    """

    async def on_response(self, context: MiddlewareContext, call_next):
        response = await call_next(context)
        if hasattr(context, "auth") and hasattr(context.auth, "claims"):
            logger.info("JWT authentication succeeded")
        return response


class ActorContextMiddleware(Middleware):
    """Bridge a validated JWT into the ambient actor and ``GraphSession``.

    ``create_mcp_server`` already validates inbound JWTs (multi-realm), but
    nothing carried that identity into the agent-utilities execution plane — so
    no server could scope resources or authorization to *who is calling*. This
    middleware is the fleet-wide identity bridge:
    it mints the actor once per tool call from the already-validated claims
    (:func:`~agent_utilities.security.request_identity.actor_from_claims`, the
    one IdP-agnostic mapping), mints the immutable graph authority, and scopes
    the call to both, so
    :func:`~agent_utilities.security.entitlements.identity_scoped_resources` and
    the KG permissioning layer all see the real caller.

    Mounted on every server the factory builds. Two modes, selected by
    ``require_verified_session`` (default ``False``, unchanged fleet-wide
    behavior — see below):

    * **``require_verified_session=False`` (every non-graph-os server).**
      When there is no validated token this stays a genuine no-op — the
      ambient actor/session are left exactly as they were. Each of these
      servers keeps its own authorization contract (API keys, network-level
      trust, or none, by design); this middleware's job for them is opportunistic
      identity *enrichment*, not gating, and changing that fleet-wide would
      be a separate, much larger change requiring its own audit of ~60
      independently-owned packages.
    * **``require_verified_session=True`` (graph-os only, wired in
      ``server_factory._configure_middleware``).** graph-os's own tools reach
      privileged Knowledge-Graph reads/writes, and BUG-036 proved that relying
      on each tool to remember to call ``resolve_session`` itself is not
      sufficient — the federation dialect (``engine_federation.py``) reached a
      backend's ``execute_read`` with zero identity check because this
      middleware's old unconditional no-op left nothing else in the MCP-native
      dispatch path (FastMCP → this middleware → the ``@mcp.tool``-decorated
      function) to catch it; only the *separate* ``_execute_tool`` dispatch
      core (REST gateway / internal delegation) had its own gate
      (``verified_tool_session_scope``). In this mode a call with no validated
      token is refused — never silently proceeds — UNLESS a verified authority
      is *already* bound some other legitimate way:
      (a) an ambient :class:`~agent_utilities.knowledge_graph.core.session.GraphSession`
      is already set (a nested call under an already-authenticated caller), or
      (b) the tiny-profile local stdio process authority
      (:func:`~agent_utilities.security.request_identity.mint_local_process_session`,
      bound as ``agent_utilities.mcp.kg_server._PROCESS_SESSION``) is bound —
      stdio has no per-call bearer token by protocol, so that process-lifetime,
      already-verified authority IS this transport's carrier, not an absence
      of one. Every other case raises
      :class:`~agent_utilities.knowledge_graph.core.session.SessionRequiredError`
      (GOC-15/GOC-62: "a surface that cannot mint a caller must refuse, never
      proceed").
    """

    def __init__(self, *, require_verified_session: bool = False) -> None:
        self.require_verified_session = require_verified_session

    def _claims(self, context: MiddlewareContext) -> dict | None:
        auth = getattr(context, "auth", None)
        claims = getattr(auth, "claims", None) if auth is not None else None
        if claims:
            return dict(claims)
        # FastMCP 3.x: the validated access token is exposed via a dependency.
        try:
            from fastmcp.server.dependencies import get_access_token

            token = get_access_token()
        except Exception:
            return None
        claims = getattr(token, "claims", None) if token is not None else None
        return dict(claims) if claims else None

    async def on_call_tool(self, context: MiddlewareContext, call_next):
        from agent_utilities.knowledge_graph.core.session import (
            SessionRequiredError,
            current_session,
            reset_session,
            set_session,
        )
        from agent_utilities.security.brain_context import reset_actor, set_actor
        from agent_utilities.security.request_identity import (
            actor_from_claims,
            mint_graph_session,
        )

        claims = self._claims(context)
        if not claims:
            if not self.require_verified_session:
                return await call_next(context)
            # BUG-036/GOC-15: a surface that cannot mint a caller must refuse,
            # never proceed. The two narrow, named exemptions above (docstring)
            # are: an authority already ambient (nested call), or the tiny
            # stdio local-process bootstrap.
            if current_session() is not None:
                return await call_next(context)
            from agent_utilities.mcp import kg_server

            if getattr(kg_server, "_PROCESS_SESSION", None) is not None:
                return await call_next(context)
            raise SessionRequiredError(
                "Verified caller identity required: no validated bearer "
                "credential was presented for this tool call and no local "
                "process authority is bound"
            )
        actor = actor_from_claims(claims)
        try:
            session = mint_graph_session(actor)
            session.engine_verified_context()
        except PermissionError:
            raise PermissionError("Verified graph authority is incomplete") from None
        actor_token = set_actor(actor)
        session_token = set_session(session)
        try:
            return await call_next(context)
        finally:
            reset_session(session_token)
            reset_actor(actor_token)


class EntityLinkingMiddleware(Middleware):
    """Middleware for intercepting tool calls for cross-entity relationship resolution.

    CONCEPT:AU-ECO.mcp.fastmcp-middleware - Assimilated from FastMCP 'cross-cutting concern interception layer'.
    """

    async def on_request(self, context: MiddlewareContext, call_next):
        logger.debug("EntityLinkingMiddleware: intercepting request")
        # Example interception: dynamically link entities when writing to the KG
        if hasattr(context, "message") and getattr(
            context.message, "method", ""
        ).endswith("tools/call"):
            params = getattr(context.message, "params", {})
            if params and getattr(params, "name", "") == "kg_write":
                logger.info(
                    "Entity linking capabilities applied to kg_write interception."
                )
        return await call_next(context)
