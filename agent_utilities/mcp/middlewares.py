#!/usr/bin/python
"""FastMCP Middlewares Module.

This module defines custom middlewares for FastMCP servers. It handles
user token extraction for delegation and JWT claims logging to provide
enhanced observability and authorization context during tool execution.
"""

import threading
import time

from fastmcp.server.middleware import Middleware, MiddlewareContext
from fastmcp.utilities.logging import get_logger

local = threading.local()
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

    If delegation is enabled, this middleware captures the 'Authorization'
    header from incoming requests and stores the Bearer token in
    thread-local storage.
    """

    def __init__(self, config: dict):
        self.config = config

    async def on_request(self, context: MiddlewareContext, call_next):
        logger.debug(f"Delegation enabled: {self.config['enable_delegation']}")
        if self.config["enable_delegation"]:
            headers = getattr(context.message, "headers", {})
            auth = headers.get("Authorization")
            if auth and auth.startswith("Bearer "):
                token = auth.split(" ")[1]
                local.user_token = token
                local.user_claims = None

                if hasattr(context, "auth") and hasattr(context.auth, "claims"):
                    local.user_claims = context.auth.claims
                    logger.info(
                        "Stored JWT claims for delegation",
                        extra={"subject": context.auth.claims.get("sub")},
                    )
                else:
                    logger.debug("JWT claims not yet available (will be after auth)")

                logger.info("Extracted Bearer token for delegation")
            else:
                logger.error("Missing or invalid Authorization header")
                raise ValueError("Missing or invalid Authorization header")
        return await call_next(context)


class JWTClaimsLoggingMiddleware(Middleware):
    """Middleware for logging JWT authentication claims on responses.

    Captures and logs the subject, client ID, and scopes from validated
    JWT claims to provide an audit trail for successful authentications.
    """

    async def on_response(self, context: MiddlewareContext, call_next):
        response = await call_next(context)
        logger.info(f"JWT Response: {response}")
        if hasattr(context, "auth") and hasattr(context.auth, "claims"):
            logger.info(
                "JWT Authentication Success",
                extra={
                    "subject": context.auth.claims.get("sub"),
                    "client_id": context.auth.claims.get("client_id"),
                    "scopes": context.auth.claims.get("scope"),
                },
            )


class ActorContextMiddleware(Middleware):
    """Bridge a server's validated JWT into the ambient ``ActorContext`` so every
    tool call runs scoped to the caller's Okta/Keycloak identity.

    ``create_mcp_server`` already validates inbound JWTs (multi-realm), but
    nothing carried that identity into the agent-utilities execution plane — so
    ``current_actor()`` was always the privileged ``SYSTEM_ACTOR`` and no server
    could scope resources or authorization to *who is calling*. This middleware
    is the fleet-wide fix (CONCEPT:AU-OS.identity.idp-agnostic-role-inheritance):
    it mints the actor once per tool call from the already-validated claims
    (:func:`~agent_utilities.security.request_identity.actor_from_claims`, the
    one IdP-agnostic mapping) and scopes the call to it, so
    :func:`~agent_utilities.security.entitlements.identity_scoped_resources` and
    the KG permissioning layer all see the real caller.

    Native + back-compat: mounted on every server the factory builds. When there
    is no validated token (stdio/local/unauthenticated), it is a no-op and the
    ambient ``SYSTEM_ACTOR`` is unchanged — behaviour is identical to today until
    a real authenticated identity arrives.
    """

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
        from agent_utilities.security.brain_context import reset_actor, set_actor
        from agent_utilities.security.request_identity import actor_from_claims

        claims = self._claims(context)
        if not claims:
            return await call_next(context)
        token = set_actor(actor_from_claims(claims))
        try:
            return await call_next(context)
        finally:
            reset_actor(token)


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
