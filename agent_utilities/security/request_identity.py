#!/usr/bin/python
from __future__ import annotations

"""Server-minted request identity for the Knowledge Graph (CONCEPT:AU-OS.identity.authenticated-identity-enforcement).

MCP/REST identity is minted **server-side** from a validated JWT, reusing the
existing JWKS machinery in
:mod:`agent_utilities.security.auth` (no second validator):

* :class:`ActorIdentityMiddleware` — pure-ASGI middleware (mounted by
  ``gateway.graph_api.register_graph_routes``) that validates an
  ``Authorization: Bearer`` token against ``AUTH_JWT_JWKS_URI`` and scopes the
  request to both the minted, ``authenticated=True`` actor and a server-owned
  ``GraphSession``. Missing or invalid credentials are rejected (401) except
  for non-fingerprinting health probes.
* :func:`actor_from_claims` — the single claims→ActorContext mapping.
* :func:`mint_local_process_session` — creates the private, in-memory authority
  used only by tiny packaged-local stdio startup.
* :func:`acquire_process_identity_token` — resolves a configured secret
  reference or performs OAuth2 client credentials for every external topology.
* :func:`mint_actor_from_token_sync` — validates that acquired token.
External authority configuration (typed, on ``AgentConfig``):
``KG_AUTH_TOKEN_REF`` or ``KG_IDENTITY_OAUTH2``, ``AUTH_JWT_JWKS_URI``,
``AUTH_JWT_ISSUER``, ``AUTH_JWT_AUDIENCE``, and ``KG_POLICY_VERSION``.
"""

import json
import logging
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from ..models.company_brain import ActorType
from .brain_context import (
    ActorContext,
    CredentialExpiredError,
    reset_actor,
    set_actor,
    use_actor,
)

if TYPE_CHECKING:
    from agent_utilities.knowledge_graph.core.session import GraphSession

logger = logging.getLogger(__name__)

# Paths that must stay reachable without credentials (container/LB liveness).
HEALTH_PATHS: frozenset[str] = frozenset(
    {"/health", "/healthz", "/api/health", "/api/healthz"}
)

# Only non-fingerprinting liveness routes bypass the identity boundary.
# Remote metrics must authenticate just like any other operational endpoint.
UNAUTHENTICATED_PATHS: frozenset[str] = HEALTH_PATHS

# Network MCP transports expose graph-os to many clients at once. Serving them
# without server-validated identity is the "security fails open" condition, so
# the served profile is enforced for these transports (CONCEPT:AU-OS.identity.authenticated-identity-enforcement).
SERVED_TRANSPORTS: frozenset[str] = frozenset({"streamable-http", "sse"})

# The only graph authorization scopes a served identity may project into a
# GraphSession. They come from validated JWT capabilities (``ActorContext.roles``),
# never from request JSON/headers. Only the explicit ``kg:admin`` capability —
# supplied directly or through the configured identity mapping — grants graph
# administration; a generic application role named ``admin`` is not equivalent.
_GRAPH_AUTH_SCOPES: frozenset[str] = frozenset({"kg:read", "kg:write", "kg:admin"})

_LOCAL_PROCESS_ISSUER = "urn:agent-utilities:graph-os:local-process"
_LOCAL_PROCESS_AUDIENCE = "graph-os-local"
_LOCAL_PROCESS_POLICY_VERSION = "local-ephemeral-v1"
_LOCAL_PROCESS_SUBJECT = "graph-os:local-process"
_LOCAL_PROCESS_TENANT = "local"


def local_process_authority_enabled(config: Any) -> bool:
    """Return whether the zero-infrastructure stdio authority is available.

    The local authority is deliberately narrow: only the ``tiny`` profile with
    a packaged-local engine and no configured external process identity may use
    it. Network transports never consult this predicate as an authentication
    fallback, and a configured-but-invalid external identity still fails loud.
    """
    profile = getattr(config, "deployment_profile", "tiny")
    endpoints = getattr(config, "graph_service_endpoints", None)
    return bool(
        isinstance(profile, str)
        and profile.strip().lower() == "tiny"
        and not endpoints
        and not getattr(config, "kg_auth_token_ref", None)
        and not getattr(config, "kg_identity_oauth2", None)
    )


def mint_graph_session(actor: ActorContext) -> GraphSession:
    """Mint the server-owned :class:`GraphSession` for an authenticated actor.

    This is the single served-boundary claims-to-session projection.  Tenant,
    graph, scopes, policy revision, and trace context are derived exclusively
    from the validated actor and server configuration; caller-supplied session
    fields are never consulted.

    Raises:
        PermissionError: If ``actor`` was not created from validated
            credentials.
    """
    from agent_utilities.core.config import config

    audience = str(config.auth_jwt_audience or config.mcp_jwt_audience or "").strip()
    policy_version = str(config.kg_policy_version or "").strip()
    return _mint_graph_session(
        actor,
        audience=audience,
        policy_version=policy_version,
    )


def _mint_graph_session(
    actor: ActorContext,
    *,
    audience: str,
    policy_version: str,
) -> GraphSession:
    """Project one already-verified actor into the authoritative graph route."""
    if not actor.authenticated:
        raise PermissionError(
            "A GraphSession can only be minted from authenticated identity"
        )
    if not str(actor.actor_id or "").strip():
        raise PermissionError("Verified identity is missing its subject")

    from agent_utilities.knowledge_graph.core.placement_catalog import (
        resolve_placement,
    )
    from agent_utilities.knowledge_graph.core.session import GraphSession, use_session
    from agent_utilities.knowledge_graph.core.shard_topology import (
        default_graph_name,
        resolve_endpoints,
        tenant_graph_name,
    )
    from agent_utilities.observability import correlation

    scopes = frozenset(str(role) for role in actor.roles) & _GRAPH_AUTH_SCOPES
    # Coarse KG scopes are hierarchical.  A writer necessarily performs
    # authorization-safe precondition reads, while an administrator may do
    # both.  Expand the hierarchy once at the trusted claims boundary so the
    # facade and the engine receive the same capability set.
    if "kg:admin" in scopes:
        scopes |= frozenset({"kg:read", "kg:write"})
    elif "kg:write" in scopes:
        scopes |= frozenset({"kg:read"})
    tenant = str(actor.tenant_id or "").strip()
    if not tenant:
        raise PermissionError(
            "Authenticated graph requests require a verified tenant claim"
        )
    graph = tenant_graph_name(tenant, base=default_graph_name())
    from agent_utilities.core.config import config

    audience = str(audience or "").strip()
    policy_version = str(policy_version or "").strip()
    if not audience or not policy_version:
        raise PermissionError(
            "Verified graph authority is missing audience or policy revision"
        )
    # Placement is itself an authenticated engine read.  Mint the verified but
    # unrouted session first, bind it only for that catalog lookup, and then
    # narrow it to the authoritative route.  This removes the bootstrap cycle
    # without inventing a system actor or weakening the placement boundary.
    unrouted = GraphSession(
        actor=actor,
        tenant=tenant,
        scopes=scopes,
        graph=graph,
        policy_version=policy_version,
        trace_context=correlation.ensure_correlation_id(),
        audience=audience,
    )
    endpoints = resolve_endpoints(config)
    with use_session(unrouted):
        if not config.graph_service_endpoints:
            # The packaged-local topology has no external coordinator to answer
            # the first placement read.  Provision the one process transport
            # through the canonical resolver while the verified bootstrap
            # session is bound; explicit topologies remain connect-only.
            from agent_utilities.knowledge_graph.core.graph_compute import (
                GraphComputeEngine,
            )

            GraphComputeEngine.get_or_create(graph_name=graph)
        placement = resolve_placement(graph, endpoints, config)
    return unrouted.with_route(
        endpoint=placement.endpoint,
        placement_group=(int(placement.group) if placement.group is not None else None),
        catalog_epoch=int(placement.epoch),
    )


def apply_served_security_profile(
    transport: str,
    config: Any = None,
    *,
    transport_auth_configured: bool = False,
) -> None:
    """Turn on fail-closed identity + enforcement for a network MCP transport.

    A network MCP transport is a multi-client surface. When serving such a
    transport this:

    * **refuses to start** unless either ``AUTH_JWT_JWKS_URI`` is set or the
      FastMCP transport already has a successfully constructed auth provider;
    * verifies the audience/policy inputs required by mandatory tenant, ACL,
      session, and single-client enforcement.

    No-op for ``stdio``/unknown transports. Idempotent.
    """
    if transport not in SERVED_TRANSPORTS:
        return

    if config is None:
        from agent_utilities.core.config import config as _config

        config = _config

    jwks = getattr(config, "auth_jwt_jwks_uri", None)
    if not jwks and not transport_auth_configured:
        raise RuntimeError(
            f"Refusing to serve graph-os over {transport}: neither a transport "
            "authentication provider nor AUTH_JWT_JWKS_URI is configured, so "
            "client identity cannot be validated. Configure JWT/OIDC auth. "
            "(CONCEPT:AU-OS.identity.authenticated-identity-enforcement)"
        )

    audience = str(
        getattr(config, "auth_jwt_audience", None)
        or getattr(config, "mcp_jwt_audience", None)
        or ""
    ).strip()
    policy_version = str(getattr(config, "kg_policy_version", None) or "").strip()
    if not audience or not policy_version:
        raise RuntimeError(
            f"Refusing to serve graph-os over {transport}: verified authority "
            "requires a configured audience and policy revision."
        )

    logger.warning(
        "graph-os served profile ACTIVE over %s: verified identity/session, "
        "tenant scoping, ACL enforcement, and one graph client are mandatory. "
        "Unauthenticated requests are rejected. "
        "(CONCEPT:AU-OS.identity.authenticated-identity-enforcement)",
        transport,
    )


def actor_from_claims(claims: dict[str, Any]) -> ActorContext:
    """Mint an ``authenticated`` :class:`ActorContext` from validated JWT claims.

    Delegates claim parsing to the one IdP-agnostic normalizer
    (:func:`agent_utilities.security.identity.normalize_identity`) so **Okta
    groups and Keycloak roles are first-class and interchangeable**
    (CONCEPT:AU-OS.identity.idp-agnostic-role-inheritance):

    * ``actor_id`` ← ``sub`` | ``client_id`` | ``azp``
    * ``roles`` ← the *base capability set* = role claims (``roles`` /
      Keycloak ``realm_access.roles`` / ``resource_access.*.roles``) ∪
      ``scope``/``scp`` ∪ group-derived capabilities (Okta ``groups`` /
      Keycloak group mapper, via the optional ``IDENTITY_GROUP_CAPABILITY_MAP``
      config; a group defaults to a same-named capability). This is what ACL
      checks read, so a group membership natively grants access with no
      per-consumer change.
    * ``groups`` ← the raw normalized group set (retained for k8s impersonation).
    * ``tenant_id`` ← ``tenant_id`` | ``tenant`` | ``org_id`` | ``tid`` | ``org``
    * ``actor_type`` ← HUMAN when an ``email`` claim is present, else
      AUTOMATED_SERVICE (provenance only — not used for access decisions).
    """
    from agent_utilities.core.config import config

    from .identity import base_capabilities, normalize_identity

    identity = normalize_identity(claims)
    group_map = getattr(config, "identity_group_capability_map", None)

    actor_type = ActorType.HUMAN if identity.email else ActorType.AUTOMATED_SERVICE
    credential_expires_at: int | None = None
    if "exp" in claims:
        raw_expiry = claims["exp"]
        if isinstance(raw_expiry, bool) or not isinstance(raw_expiry, (int, float)):
            raise ValueError("validated identity has an invalid expiry claim")
        if raw_expiry < 0:
            raise ValueError("validated identity has an invalid expiry claim")
        try:
            credential_expires_at = int(raw_expiry)
        except (OverflowError, ValueError):
            raise ValueError("validated identity has an invalid expiry claim") from None
    return ActorContext(
        actor_id=identity.subject,
        actor_type=actor_type,
        roles=base_capabilities(identity, group_map),
        tenant_id=identity.tenant,
        authenticated=True,
        groups=identity.groups,
        credential_expires_at=credential_expires_at,
    )


def mint_local_process_session() -> GraphSession:
    """Mint a private, process-ephemeral authority for tiny stdio GraphOS.

    An asymmetric key signs one short-lived JWT entirely in memory. The token is
    validated by the same decoder used for external bearer identities, then the
    token and private-key references are discarded before the resulting actor is
    projected into a graph session. Claims are fixed neutral service values; no
    username, host name, filesystem path, endpoint, or other local identifier is
    represented or persisted.
    """
    import secrets
    import time

    from joserfc import jwt
    from joserfc.jwk import RSAKey

    from .auth import _decode_jwt

    key = RSAKey.generate_key(2048)
    public_jwks = {"keys": [key.as_dict(is_private=False)]}
    now = int(time.time())
    token = jwt.encode(
        {"alg": "RS256", "typ": "at+jwt"},
        {
            "aud": _LOCAL_PROCESS_AUDIENCE,
            "exp": now + 120,
            "iat": now,
            "iss": _LOCAL_PROCESS_ISSUER,
            "jti": secrets.token_hex(16),
            "nbf": now - 1,
            "roles": ["kg:admin"],
            "scope": "kg:admin",
            "sub": _LOCAL_PROCESS_SUBJECT,
            "tenant_id": _LOCAL_PROCESS_TENANT,
        },
        key,
    )
    del key
    claims = _decode_jwt(
        token,
        public_jwks,
        issuer=_LOCAL_PROCESS_ISSUER,
        audience=_LOCAL_PROCESS_AUDIENCE,
    )
    del token
    # The local JWT is a one-time bootstrap attestation. Its private key and
    # token are destroyed, so the resulting process session is intentionally
    # bounded by the process lifetime rather than the 120-second proof window.
    actor = replace(actor_from_claims(claims), credential_expires_at=None)
    return _mint_graph_session(
        actor,
        audience=_LOCAL_PROCESS_AUDIENCE,
        policy_version=_LOCAL_PROCESS_POLICY_VERSION,
    )


async def actor_from_bearer_token(token: str) -> ActorContext:
    """Validate ``token`` against the configured JWKS and mint the actor.

    Raises ``fastapi.HTTPException`` (401) on any validation failure, and
    ``RuntimeError`` when no ``AUTH_JWT_JWKS_URI`` is configured (the caller
    decides whether that is fatal).
    """
    from agent_utilities.core.config import config

    from .auth import _decode_jwt, _fetch_jwks

    if not isinstance(token, str) or not token or len(token.encode("utf-8")) > 16_384:
        raise RuntimeError("Bearer token is invalid")
    if not config.auth_jwt_jwks_uri:
        raise RuntimeError(
            "Cannot validate Bearer token: AUTH_JWT_JWKS_URI is not configured."
        )
    if not str(config.auth_jwt_audience or "").strip():
        raise RuntimeError("Cannot validate Bearer token: audience is not configured")
    jwks = await _fetch_jwks(config.auth_jwt_jwks_uri)
    claims = _decode_jwt(
        token,
        jwks,
        issuer=config.auth_jwt_issuer,
        audience=config.auth_jwt_audience,
    )
    return actor_from_claims(claims)


def acquire_process_identity_token(config: Any = None) -> str:
    """Acquire the graph process JWT from exactly one runtime identity source.

    Configuration stores only secret references or an OAuth2 client-credentials
    block. Token material is resolved/minted at startup and is never logged.
    """
    if config is None:
        from agent_utilities.core.config import config as live_config

        config = live_config
    token_ref = str(getattr(config, "kg_auth_token_ref", None) or "").strip()
    oauth2 = getattr(config, "kg_identity_oauth2", None)
    if bool(token_ref) == bool(oauth2):
        raise RuntimeError(
            "Configure exactly one graph process identity source: "
            "KG_AUTH_TOKEN_REF or KG_IDENTITY_OAUTH2"
        )
    try:
        if token_ref:
            from .cli_secrets import resolve_runtime_secret_reference

            token = resolve_runtime_secret_reference(token_ref)
        else:
            from .oauth_client_credentials import build_provider_from_config

            token = build_provider_from_config(oauth2).get_token()
    except Exception:
        raise RuntimeError("Graph process identity acquisition failed") from None
    if (
        not isinstance(token, str)
        or not token
        or len(token.encode("utf-8")) > 16_384
        or any(ord(character) < 32 or ord(character) == 127 for character in token)
    ):
        raise RuntimeError("Graph process identity acquisition returned invalid material")
    return token


def mint_actor_from_token_sync(token: str) -> ActorContext:
    """Synchronously validate a graph process JWT and mint an actor."""
    import asyncio

    try:
        return asyncio.run(actor_from_bearer_token(token))
    except Exception:
        raise RuntimeError("Graph process identity token validation failed") from None


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

    CONCEPT:AU-OS.identity.authenticated-identity-enforcement — Authenticated Identity Enforcement.

    Behaviour matrix:

    * Bearer token present + JWKS configured → validate; valid → scope the
      request to the minted ``authenticated`` actor and verified session;
      invalid → 401.
    * No valid identity → 401 (health paths exempt).

    Mounted outside graph routes so every served REST/MCP request receives the
    same server-minted session authority.
    """

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        from agent_utilities.core.config import config

        path = scope.get("path", "")
        state = scope.get("state") or {}
        prevalidated = state.get("user_claims") if isinstance(state, dict) else None
        from .auth import parse_bearer_authorization

        authorization = [
            value
            for key, value in scope.get("headers") or []
            if isinstance(key, bytes) and key.lower() == b"authorization"
        ]
        try:
            token = parse_bearer_authorization(authorization)
        except PermissionError:
            await _send_json(send, 401, {"error": "Verified Bearer identity required"})
            return

        actor: ActorContext | None = None
        if isinstance(prevalidated, dict) and prevalidated.get("auth_type") == "jwt":
            # The outer HTTP authentication boundary already verified this
            # credential. Reuse its claims rather than making a second JWKS
            # lookup/verification pass with potentially different timing.
            try:
                actor = actor_from_claims(prevalidated)
            except (TypeError, ValueError):
                await _send_json(send, 401, {"error": "Token validation failed"})
                return
        if token and not config.auth_jwt_jwks_uri:
            await _send_json(send, 401, {"error": "Token validation unavailable"})
            return

        if token and actor is None:
            try:
                actor = await actor_from_bearer_token(token)
            except Exception:  # noqa: BLE001 — any failure = invalid credential
                await _send_json(send, 401, {"error": "Token validation failed"})
                return

        if actor is None:
            if path not in UNAUTHENTICATED_PATHS:
                await _send_json(
                    send,
                    401,
                    {"error": "Verified Bearer identity required"},
                )
                return
            # Liveness is deliberately unauthenticated.  Scope it to an
            # explicit non-authoritative actor and suppress any GraphSession
            # inherited from a parent task so health handlers cannot observe or
            # accidentally reuse process/request authority.
            from agent_utilities.knowledge_graph.core.session import suspend_session

            with (
                use_actor(
                    ActorContext(
                        actor_id="health-probe",
                        tenant_id="health",
                        authenticated=False,
                    )
                ),
                suspend_session(),
            ):
                await self.app(scope, receive, send)
            return

        # The authenticated request establishes both ambient currencies.  The
        # session is minted here, outside every served route, so async tasks and
        # worker-thread dispatch inherit one immutable, verified authority.
        from agent_utilities.knowledge_graph.core.session import (
            SessionExpiredError,
            reset_session,
            set_session,
        )

        try:
            session = mint_graph_session(actor)
        except SessionExpiredError:
            await _send_json(send, 401, {"error": "Bearer credential expired"})
            return
        except PermissionError:
            await _send_json(send, 403, {"error": "Verified tenant claim required"})
            return
        try:
            ctx_token = set_actor(actor)
            try:
                session_token = set_session(session)
            except Exception:
                reset_actor(ctx_token)
                raise
        except (CredentialExpiredError, SessionExpiredError):
            await _send_json(send, 401, {"error": "Bearer credential expired"})
            return

        response_started = False

        async def tracked_send(message: Any) -> None:
            nonlocal response_started
            if message.get("type") == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.app(scope, receive, tracked_send)
        except (CredentialExpiredError, SessionExpiredError):
            if response_started:
                raise
            await _send_json(send, 401, {"error": "Bearer credential expired"})
        finally:
            reset_session(session_token)
            reset_actor(ctx_token)


__all__ = [
    "ActorIdentityMiddleware",
    "HEALTH_PATHS",
    "SERVED_TRANSPORTS",
    "UNAUTHENTICATED_PATHS",
    "acquire_process_identity_token",
    "actor_from_bearer_token",
    "actor_from_claims",
    "apply_served_security_profile",
    "local_process_authority_enabled",
    "mint_local_process_session",
    "mint_graph_session",
    "mint_actor_from_token_sync",
]
