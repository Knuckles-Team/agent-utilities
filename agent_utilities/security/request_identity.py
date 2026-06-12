#!/usr/bin/python
from __future__ import annotations

"""Server-minted request identity for the Knowledge Graph (CONCEPT:OS-5.14).

Closes the honor-system identity gap: until now the MCP/REST surface built
:class:`~agent_utilities.security.brain_context.ActorContext` from
caller-supplied ``_actor``/``_roles``/``_tenant`` kwargs — unsigned and
trivially spoofable. This module mints the actor **server-side** from a
validated JWT instead, reusing the existing JWKS machinery in
:mod:`agent_utilities.security.auth` (no second validator):

* :class:`ActorIdentityMiddleware` — pure-ASGI middleware (mounted by
  ``gateway.graph_api.register_graph_routes``) that validates an
  ``Authorization: Bearer`` token against ``AUTH_JWT_JWKS_URI`` and scopes the
  request to the minted, ``authenticated=True`` actor via the existing
  contextvar. An invalid token is always a 401. With ``KG_AUTH_REQUIRED`` on,
  requests without a valid token are rejected (401) except for health probes.
* :func:`actor_from_claims` — the single claims→ActorContext mapping.
* :func:`mint_actor_from_token_sync` — synchronous helper for stdio MCP
  startup, where identity comes from a validated ``KG_AUTH_TOKEN`` (stdio has
  no Authorization header).
* :func:`warn_unauthenticated_identity_once` — the one-time, prominent
  startup warning emitted when ``KG_AUTH_REQUIRED`` is off.

Configuration (typed, on ``AgentConfig``): ``KG_AUTH_REQUIRED`` (default off,
back-compat), ``KG_AUTH_TOKEN``, and the pre-existing ``AUTH_JWT_JWKS_URI`` /
``AUTH_JWT_ISSUER`` / ``AUTH_JWT_AUDIENCE``.
"""

import json
import logging
import os
from typing import Any

from agent_utilities.core.config import setting

from ..models.company_brain import ActorType
from .brain_context import ActorContext, reset_actor, set_actor

logger = logging.getLogger(__name__)

# Paths that must stay reachable without credentials (container/LB liveness).
HEALTH_PATHS: frozenset[str] = frozenset(
    {"/health", "/healthz", "/api/health", "/api/healthz"}
)

# Prometheus scrapers cannot mint JWTs; /metrics exposes only aggregate
# counters (no graph data), so it is exempt like the health probes.
# (CONCEPT:OS-5.23 — Gateway Middle-Tier Hardening)
UNAUTHENTICATED_PATHS: frozenset[str] = HEALTH_PATHS | {"/metrics"}

_warned_unauthenticated = False


def warn_unauthenticated_identity_once() -> None:
    """Log a one-time, prominent warning that KG identity is honor-system."""
    global _warned_unauthenticated  # noqa: PLW0603
    if _warned_unauthenticated:
        return
    _warned_unauthenticated = True
    logger.warning(
        "KG identity is UNAUTHENTICATED (KG_AUTH_REQUIRED=0): caller-supplied "
        "_actor/_roles/_tenant kwargs are trusted as-is. Set KG_AUTH_REQUIRED=1 "
        "and AUTH_JWT_JWKS_URI to enforce server-validated JWT identity. "
        "(CONCEPT:OS-5.14)"
    )


_TRUTHY = frozenset({"1", "true", "yes", "on"})

# Network MCP transports expose graph-os to many clients at once. Serving them
# without server-validated identity is the "security fails open" condition, so
# the served profile is enforced for these transports (CONCEPT:OS-5.14).
SERVED_TRANSPORTS: frozenset[str] = frozenset({"streamable-http", "sse"})


def apply_served_security_profile(transport: str, config: Any = None) -> None:
    """Turn on fail-closed identity + enforcement for a network MCP transport.

    graph-os over ``streamable-http``/``sse`` is a multi-client surface, so it
    must not run in the honor-system mode that is fine for ``stdio``/local dev.
    When serving such a transport this:

    * **refuses to start** if ``AUTH_JWT_JWKS_URI`` is unset — a public surface
      with no way to validate a JWT is fail-open, so we fail loud instead;
    * turns on ``KG_AUTH_REQUIRED`` (rejects unauthenticated HTTP — which also
      makes the privileged ``SYSTEM_ACTOR`` ambient fallback unreachable over
      the network) and ``KG_BRAIN_ENFORCE`` (tenant scoping + ACLs), unless an
      operator has explicitly pinned either flag.

    Opt out for local network dev with ``KG_SERVED_PROFILE=0`` (logs a loud
    warning and leaves the honor-system behaviour in place). No-op for
    ``stdio``/unknown transports, so existing deployments are byte-for-byte
    unaffected. Idempotent.
    """
    if transport not in SERVED_TRANSPORTS:
        return

    if config is None:
        from agent_utilities.core.config import config as _config

        config = _config

    if not getattr(config, "kg_served_profile", True):
        logger.warning(
            "KG_SERVED_PROFILE=0: serving graph-os over %s WITHOUT enforced "
            "identity. Every caller is trusted as-is — use only for local dev. "
            "(CONCEPT:OS-5.14)",
            transport,
        )
        warn_unauthenticated_identity_once()
        return

    jwks = getattr(config, "auth_jwt_jwks_uri", None)
    if not jwks:
        raise RuntimeError(
            f"Refusing to serve graph-os over {transport}: AUTH_JWT_JWKS_URI is "
            "not configured, so client identity cannot be validated and the "
            "surface would be fail-open. Set AUTH_JWT_JWKS_URI (and AUTH_JWT_"
            "ISSUER/AUDIENCE), or set KG_SERVED_PROFILE=0 to override for local "
            "dev only. (CONCEPT:OS-5.14)"
        )

    # Honor an operator who has explicitly pinned a flag (even to off); only
    # supply the secure default when the flag is unset.
    if setting("KG_AUTH_REQUIRED") is None:
        os.environ["KG_AUTH_REQUIRED"] = "1"
    # The live middleware reads the singleton config, so mutate it too.
    try:
        config.kg_auth_required = setting("KG_AUTH_REQUIRED", False)
    except Exception:  # noqa: BLE001 - frozen/odd config must not block startup
        pass
    if setting("KG_BRAIN_ENFORCE") is None:
        os.environ["KG_BRAIN_ENFORCE"] = "1"

    logger.warning(
        "graph-os served profile ACTIVE over %s: KG_AUTH_REQUIRED=%s, "
        "KG_BRAIN_ENFORCE=%s, JWKS=%s. Unauthenticated requests are rejected "
        "and reads/writes are tenant-scoped. (CONCEPT:OS-5.14)",
        transport,
        setting("KG_AUTH_REQUIRED"),
        setting("KG_BRAIN_ENFORCE"),
        jwks,
    )


def actor_from_claims(claims: dict[str, Any]) -> ActorContext:
    """Mint an ``authenticated`` :class:`ActorContext` from validated JWT claims.

    Mapping (first match wins):

    * ``actor_id`` ← ``sub`` | ``client_id`` | ``azp``
    * ``roles``    ← ``roles`` | ``realm_access.roles`` (Keycloak) |
      space-separated ``scope``/``scp``
    * ``tenant_id``← ``tenant_id`` | ``tenant`` | ``org_id`` | ``tid``
    * ``actor_type`` ← HUMAN when an ``email`` claim is present, else
      AUTOMATED_SERVICE (provenance only — not used for access decisions).
    """
    actor_id = str(
        claims.get("sub") or claims.get("client_id") or claims.get("azp") or "jwt"
    )

    roles: Any = claims.get("roles")
    if not roles:
        realm = claims.get("realm_access")
        if isinstance(realm, dict):
            roles = realm.get("roles")
    if not roles:
        scope = claims.get("scope") or claims.get("scp")
        if isinstance(scope, str):
            roles = scope.split()
        elif isinstance(scope, list):
            roles = scope
    if isinstance(roles, str):
        roles = [r.strip() for r in roles.split(",") if r.strip()]
    role_tuple = tuple(str(r) for r in (roles or ()))

    tenant = str(
        claims.get("tenant_id")
        or claims.get("tenant")
        or claims.get("org_id")
        or claims.get("tid")
        or ""
    )

    actor_type = ActorType.HUMAN if claims.get("email") else ActorType.AUTOMATED_SERVICE
    return ActorContext(
        actor_id=actor_id,
        actor_type=actor_type,
        roles=role_tuple,
        tenant_id=tenant,
        authenticated=True,
    )


async def actor_from_bearer_token(token: str) -> ActorContext:
    """Validate ``token`` against the configured JWKS and mint the actor.

    Raises ``fastapi.HTTPException`` (401) on any validation failure, and
    ``RuntimeError`` when no ``AUTH_JWT_JWKS_URI`` is configured (the caller
    decides whether that is fatal).
    """
    from agent_utilities.core.config import config

    from .auth import _decode_jwt, _fetch_jwks

    if not config.auth_jwt_jwks_uri:
        raise RuntimeError(
            "Cannot validate Bearer token: AUTH_JWT_JWKS_URI is not configured."
        )
    jwks = await _fetch_jwks(config.auth_jwt_jwks_uri)
    claims = _decode_jwt(
        token,
        jwks,
        issuer=config.auth_jwt_issuer,
        audience=config.auth_jwt_audience,
    )
    return actor_from_claims(claims)


def mint_actor_from_token_sync(token: str) -> ActorContext | None:
    """Synchronously validate ``token`` and mint an actor (stdio MCP startup).

    Returns ``None`` when validation is impossible or fails — the caller falls
    back to the restricted read-only identity. Never raises.
    """
    import asyncio

    try:
        return asyncio.run(actor_from_bearer_token(token))
    except Exception as exc:  # noqa: BLE001 — startup must not crash on a bad token
        logger.warning("KG_AUTH_TOKEN validation failed (%s); ignoring token.", exc)
        return None


async def _send_json(send: Any, status: int, payload: dict[str, Any]) -> None:
    body = json.dumps(payload).encode("utf-8")
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [
                (b"content-type", b"application/json"),
                (b"www-authenticate", b"Bearer"),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})


class ActorIdentityMiddleware:
    """Pure-ASGI middleware that mints the request's ActorContext from a JWT.

    CONCEPT:OS-5.14 — Authenticated Identity Enforcement.

    Behaviour matrix:

    * Bearer token present + JWKS configured → validate; valid → scope the
      request to the minted ``authenticated`` actor; invalid → 401 (always,
      regardless of ``KG_AUTH_REQUIRED``).
    * No valid identity + ``KG_AUTH_REQUIRED=1`` → 401 (health paths exempt).
    * No valid identity + ``KG_AUTH_REQUIRED=0`` → pass through unauthenticated
      (legacy behaviour) with a one-time warning.

    Mounted *outside* ``CentralizedCypherMiddleware`` so the lock-bypassing
    ``POST /cypher`` fast path is covered too.
    """

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        from agent_utilities.core.config import config

        path = scope.get("path", "")
        token: str | None = None
        for key, value in scope.get("headers") or []:
            if key.decode("latin-1").lower() == "authorization":
                raw = value.decode("latin-1")
                if raw.lower().startswith("bearer "):
                    token = raw[7:].strip()
                break

        actor: ActorContext | None = None
        if token and config.auth_jwt_jwks_uri:
            try:
                actor = await actor_from_bearer_token(token)
            except Exception as exc:  # noqa: BLE001 — any failure = invalid credential
                detail = getattr(exc, "detail", None) or "Token validation failed"
                await _send_json(send, 401, {"error": str(detail)})
                return

        if actor is None:
            if config.kg_auth_required and path not in UNAUTHENTICATED_PATHS:
                await _send_json(
                    send,
                    401,
                    {
                        "error": (
                            "Authentication required (KG_AUTH_REQUIRED=1): provide "
                            "a valid JWT Bearer token"
                            + (
                                ""
                                if config.auth_jwt_jwks_uri
                                else " (server misconfigured: AUTH_JWT_JWKS_URI unset)"
                            )
                        )
                    },
                )
                return
            warn_unauthenticated_identity_once()
            await self.app(scope, receive, send)
            return

        ctx_token = set_actor(actor)
        try:
            await self.app(scope, receive, send)
        finally:
            reset_actor(ctx_token)


__all__ = [
    "ActorIdentityMiddleware",
    "HEALTH_PATHS",
    "SERVED_TRANSPORTS",
    "UNAUTHENTICATED_PATHS",
    "actor_from_bearer_token",
    "actor_from_claims",
    "apply_served_security_profile",
    "mint_actor_from_token_sync",
    "warn_unauthenticated_identity_once",
]
