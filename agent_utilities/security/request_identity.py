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

Identity, not topology (D-SP-1)
-------------------------------
Minting a session establishes **who the caller is**; it never resolves **where
the data lives**. :func:`_mint_graph_session` used to end with a live
``placement_catalog.resolve_placement`` round-trip to the engine, which made
*engine cluster-admin authority a precondition for authenticating any request
on every agent-utilities served surface*: the engine's capability ledger
declares ``PlacementRoute`` with ``authz_action = "admin:cluster-read"``
(``epistemic-graph/crates/eg-capabilities/src/lib.rs:2274``) and
``src/server/dispatch.rs`` enforces it twice — once against the caller's token
scopes (``dispatch.rs:2185``) and again against the engine's own
``IsolationLayer`` via ``require_admin_capability`` (``dispatch.rs:2199`` →
``src/server/access.rs:858``), which **no JWT claim can satisfy**. graph-os only
looked healthy because every credential in use held cluster admin; the first
genuinely non-admin principal got a 500 out of the identity middleware
(``PlacementAuthorityError`` is a ``RuntimeError``, so it fell through the
middleware's 401/403 arms).

Dropping the mint-time route is not a workaround, it is the correct placement of
the responsibility, and the route it produced was already dead data:

* ``knowledge_graph/core/graph_compute.py`` re-resolves placement **per call**
  at the real data-plane call site and binds it to the outgoing session, and
  skips placement entirely when no route config/endpoints exist;
* for the one payload where the epoch is load-bearing — the fencing token and
  ``placement_epoch`` on ``ApplyChangeEnvelope`` — ``_route_bound_params``
  **overwrites** whatever the session carried with the live per-call route, so a
  mint-time epoch could only ever be stale, never authoritative;
* every other reader of ``GraphSession.catalog_epoch`` already treats ``None``
  as "not tracked" and falls back to the data-plane epoch.

So ``endpoint`` / ``placement_group`` / ``catalog_epoch`` are left unbound
here. Everything that constitutes *authority* — the verified actor, the tenant
claim, the graph name, the scope set and its hierarchy expansion, the policy
revision, the audience, the trace correlation, and every ``PermissionError``
this function raises — is unchanged, so the served 401/403 boundary is
byte-for-byte the same.
"""

import json
import logging
import threading
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from .actor_identity import ActorType
from .brain_context import (
    ActorContext,
    CredentialExpiredError,
    CredentialLease,
    reset_actor,
    set_actor,
    use_actor,
)

if TYPE_CHECKING:
    from agent_utilities.knowledge_graph.core.session import GraphSession

logger = logging.getLogger(__name__)

# Paths that must stay reachable without credentials (container/LB probes).
HEALTH_PATHS: frozenset[str] = frozenset(
    {"/health", "/health/ready", "/healthz", "/api/health", "/api/healthz"}
)

# Only non-fingerprinting liveness/readiness routes bypass the identity boundary.
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

_MAX_AUTHORITY_TEXT_LENGTH = 512
_MAX_AUTHORITY_GROUPS = 128
_MAX_CREDENTIAL_EXPIRY = (1 << 63) - 1


def _bounded_authority_text(value: object, *, field_name: str) -> str:
    text = value if isinstance(value, str) else ""
    if (
        not text
        or text != text.strip()
        or len(text) > _MAX_AUTHORITY_TEXT_LENGTH
        or any(ord(character) < 32 or ord(character) == 127 for character in text)
    ):
        raise PermissionError(f"Verified authority has an invalid {field_name}")
    return text


def _bounded_authority_groups(values: object) -> tuple[str, ...]:
    if not isinstance(values, tuple) or len(values) > _MAX_AUTHORITY_GROUPS:
        raise PermissionError("Verified authority has invalid groups")
    groups = tuple(
        _bounded_authority_text(value, field_name="group") for value in values
    )
    if len(set(groups)) != len(groups):
        raise PermissionError("Verified authority has duplicate groups")
    return groups


def _bounded_integer_expiry(value: object) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 0 <= value <= _MAX_CREDENTIAL_EXPIRY
    ):
        raise PermissionError("Verified authority requires a bounded expiry")
    return value


def _actor_expiry(actor: ActorContext) -> int:
    lease = actor.credential_lease
    raw_expiry = getattr(lease, "expires_at", actor.credential_expires_at)
    return _bounded_integer_expiry(raw_expiry)


def _claim_expiry(claims: dict[str, Any]) -> int | None:
    if "exp" not in claims:
        return None
    raw_expiry = claims["exp"]
    if isinstance(raw_expiry, bool) or not isinstance(raw_expiry, int | float):
        raise ValueError("validated identity has an invalid expiry claim")
    try:
        expiry = int(raw_expiry)
    except (OverflowError, ValueError):
        raise ValueError("validated identity has an invalid expiry claim") from None
    if raw_expiry < 0 or expiry > _MAX_CREDENTIAL_EXPIRY:
        raise ValueError("validated identity has an invalid expiry claim")
    return expiry


def _actor_with_credential_lease(actor: ActorContext) -> ActorContext:
    return replace(
        actor,
        credential_lease=CredentialLease(_actor_expiry(actor)),
    )


@dataclass(frozen=True, slots=True)
class VerifiedRequestAuthority:
    """Immutable lower authority consumed by application projections.

    This security-owned DTO is deliberately graph-agnostic. It contains no
    graph name, placement, trace, session, engine, transport, or persistence
    implementation. ``knowledge_graph.core.session`` owns the sole projection
    from this object into its graph-session currency.

    The embedded actor is retained for existing actor-context consumers, while
    every authorization-bearing member is snapshotted and rechecked by
    :meth:`ensure_current`. A renewable credential lease therefore cannot
    silently mutate authority after construction: renewal must build a fresh
    DTO, and subject/tenant/capability/group drift fails closed.
    """

    actor: ActorContext
    subject: str
    tenant: str
    scopes: frozenset[str]
    groups: tuple[str, ...]
    audience: str
    policy_version: str
    credential_expires_at: int

    def __post_init__(self) -> None:
        _bounded_authority_text(self.subject, field_name="subject")
        _bounded_authority_text(self.tenant, field_name="tenant")
        _bounded_authority_text(self.audience, field_name="audience")
        _bounded_authority_text(self.policy_version, field_name="policy revision")
        _bounded_authority_groups(self.groups)
        self.ensure_current()

    def ensure_current(self) -> None:
        """Reject expired authority or drift from the verified actor snapshot."""
        self.actor.ensure_credential_current()
        if not self.actor.authenticated:
            raise PermissionError("Verified authority actor is not authenticated")
        if self.subject != self.actor.actor_id:
            raise PermissionError("Verified authority actor identity drifted")
        if self.tenant != self.actor.tenant_id:
            raise PermissionError("Verified authority tenant drifted")
        if self.scopes != _resolve_authenticated_scopes(self.actor):
            raise PermissionError("Verified authority scopes drifted")
        if self.groups != _bounded_authority_groups(self.actor.groups):
            raise PermissionError("Verified authority groups drifted")
        if self.credential_expires_at != _actor_expiry(self.actor):
            raise PermissionError("Verified authority expiry drifted")


def build_verified_request_authority(
    actor: ActorContext,
    *,
    audience: str,
    policy_version: str,
) -> VerifiedRequestAuthority:
    """Snapshot one already-authenticated actor into the lower authority DTO.

    External credential verification must happen before this call. This
    function performs only the fail-closed structural/narrowing checks owned by
    security; it does not mint graph, session, placement, or trace state.
    """
    _assert_actor_authenticated(actor)
    credential_expires_at = _actor_expiry(actor)
    actor.ensure_credential_current()
    verified_audience, verified_policy = _assert_graph_authority(
        audience, policy_version
    )
    authority = VerifiedRequestAuthority(
        actor=actor,
        subject=_bounded_authority_text(actor.actor_id, field_name="subject"),
        tenant=_bounded_authority_text(actor.tenant_id, field_name="tenant"),
        scopes=_resolve_authenticated_scopes(actor),
        groups=_bounded_authority_groups(actor.groups),
        audience=_bounded_authority_text(verified_audience, field_name="audience"),
        policy_version=_bounded_authority_text(
            verified_policy, field_name="policy revision"
        ),
        credential_expires_at=credential_expires_at,
    )
    return authority


# The exact claim-key shape of the ONE live verified-identity carrier this
# codebase ships (GOC-15 carrier contract,
# CONCEPT:AU-OS.identity.verified-carrier-contract): the dict
# `GraphSession.engine_verified_context()` produces and
# `crates/eg-types/src/acl.rs::RequestContextClaims` deserializes
# (`#[serde(deny_unknown_fields)]` on the Rust side — an unrecognized key is a
# hard reject there, so a caller-supplied claims dict with an extra key would
# already fail on the wire; this validator gives Python-side callers the same
# fail-closed shape check *before* they propagate a claims dict to a new
# adapter, e.g. an auxiliary SPARQL/federation/observability surface, instead
# of the engine round-trip being the first place a malformed carrier is
# caught). See docs/architecture/verified-identity-carrier-contract.md for
# the full field-level contract, including why request/trace ids and
# expiry are deliberately NOT part of this set.
CARRIER_CLAIM_FIELDS: frozenset[str] = frozenset(
    {
        "principal",
        "tenant",
        "audience",
        "agent_id",
        "roles",
        "scopes",
        "delegation",
        "policy_version",
    }
)

# Present only when applicable; never required, never rejected for being
# absent. Present together they form the full key set a claims dict may use.
OPTIONAL_CARRIER_CLAIM_FIELDS: frozenset[str] = frozenset(
    {"node", "priority", "oidc_token"}
)


def _validate_carrier_scalar_fields(claims: dict[str, Any]) -> None:
    scalar_fields = ("principal", "tenant", "audience", "agent_id", "policy_version")
    for scalar_field in scalar_fields:
        value = claims[scalar_field]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"carrier claim {scalar_field!r} must be a non-empty string"
            )


def _validate_carrier_list_fields(claims: dict[str, Any]) -> None:
    for list_field in ("roles", "scopes", "delegation"):
        value = claims[list_field]
        is_str_list = isinstance(value, list) and all(
            isinstance(item, str) for item in value
        )
        if not is_str_list:
            raise ValueError(f"carrier claim {list_field!r} must be a list of strings")


def validate_carrier_claims(claims: dict[str, Any]) -> None:
    """Fail closed if ``claims`` is not a well-formed verified-carrier dict.

    This checks **shape** (required keys present and non-empty, no
    unrecognized key), not cryptographic validity — it is not a substitute
    for the engine's own MAC/audience/tenant/policy verification. It exists
    for a new adapter (SPARQL/federation/observability, GOC-18/19/20/GOC-85)
    that needs to confirm a claims dict it is about to propagate is actually
    carrier-shaped before forwarding it, without hand-copying the key list
    from :func:`GraphSession.engine_verified_context`'s docstring.

    Raises:
        ValueError: ``claims`` is missing a required field, a required field
            is empty/wrong-typed, or an unrecognized key is present.
    """
    if not isinstance(claims, dict):
        raise ValueError("carrier claims must be a dict")
    allowed = CARRIER_CLAIM_FIELDS | OPTIONAL_CARRIER_CLAIM_FIELDS
    unknown = set(claims) - allowed
    if unknown:
        raise ValueError(
            f"carrier claims contain unrecognized field(s): {sorted(unknown)}"
        )
    missing = CARRIER_CLAIM_FIELDS - set(claims)
    if missing:
        raise ValueError(f"carrier claims missing required field(s): {sorted(missing)}")
    _validate_carrier_scalar_fields(claims)
    _validate_carrier_list_fields(claims)


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


def _assert_actor_authenticated(actor: ActorContext) -> None:
    if not actor.authenticated:
        raise PermissionError(
            "A GraphSession can only be minted from authenticated identity"
        )
    if not str(actor.actor_id or "").strip():
        raise PermissionError("Verified identity is missing its subject")


def _resolve_authenticated_scopes(actor: ActorContext) -> frozenset[str]:
    """Coarse KG scopes are hierarchical. A writer necessarily performs
    authorization-safe precondition reads, while an administrator may do
    both. Expand the hierarchy once at the trusted claims boundary so the
    facade and the engine receive the same capability set."""
    scopes = frozenset(str(role) for role in actor.roles) & _GRAPH_AUTH_SCOPES
    if "kg:admin" in scopes:
        return scopes | frozenset({"kg:read", "kg:write"})
    if "kg:write" in scopes:
        return scopes | frozenset({"kg:read"})
    return scopes


def _resolve_verified_tenant(actor: ActorContext) -> str:
    tenant = str(actor.tenant_id or "").strip()
    if not tenant:
        raise PermissionError(
            "Authenticated graph requests require a verified tenant claim"
        )
    return tenant


def _assert_graph_authority(audience: str, policy_version: str) -> tuple[str, str]:
    audience = str(audience or "").strip()
    policy_version = str(policy_version or "").strip()
    if not audience or not policy_version:
        raise PermissionError(
            "Verified graph authority is missing audience or policy revision"
        )
    return audience, policy_version


def _mint_graph_session(
    actor: ActorContext,
    *,
    audience: str,
    policy_version: str,
) -> GraphSession:
    """Project one already-verified actor into its verified graph authority.

    Authentication establishes **identity**, never **topology**. This function
    therefore binds actor, tenant, graph, scopes, policy revision, audience and
    trace context — and deliberately leaves ``endpoint`` / ``placement_group`` /
    ``catalog_epoch`` unbound. See the module docstring's *"Identity, not
    topology"* note for why binding a route here was a security defect.
    """
    authority = build_verified_request_authority(
        actor,
        audience=audience,
        policy_version=policy_version,
    )

    from agent_utilities.knowledge_graph.core.session import GraphSession
    from agent_utilities.knowledge_graph.core.shard_topology import (
        default_graph_name,
        tenant_graph_name,
    )
    from agent_utilities.observability import correlation

    scopes = authority.scopes
    tenant = authority.tenant
    graph = tenant_graph_name(tenant, base=default_graph_name())

    # No route, no engine contact, no transport provisioning: the data plane
    # (``knowledge_graph/core/graph_compute.py``) binds the authoritative route
    # per call with the calling session already bound, and overwrites the
    # envelope's placement epoch/fencing token from it. ``endpoint=None`` is the
    # documented "resolve normally" value on GraphSession. See the module
    # docstring's "Identity, not topology" note (D-SP-1).
    return GraphSession(
        actor=authority.actor,
        tenant=tenant,
        scopes=scopes,
        graph=graph,
        policy_version=authority.policy_version,
        trace_context=correlation.ensure_correlation_id(),
        audience=authority.audience,
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

    * **refuses to start** unless the FastMCP transport already has a
      successfully constructed auth provider (``transport_auth_configured``,
      i.e. ``--auth-type``/``AUTH_TYPE`` resolved to something other than
      ``none``). ``AUTH_JWT_JWKS_URI`` being *set* is deliberately **not**
      treated as a substitute: an operator can populate every JWKS/issuer/
      audience variable and still leave ``AUTH_TYPE`` unset (or set the wrong
      env var name, e.g. a stray ``MCP_AUTH_TYPE``) — that combination looks
      "configured" but FastMCP never actually attaches a token verifier, so
      the ``/mcp`` endpoint would silently accept every request unauthenticated
      while this function's own log line claims the opposite. Requiring the
      constructed provider closes that fail-open gap
      (CONCEPT:AU-OS.identity.authenticated-identity-enforcement);
    * verifies the audience/policy inputs required by mandatory tenant, ACL,
      session, and single-client enforcement.

    No-op for ``stdio``/unknown transports. Idempotent.
    """
    if transport not in SERVED_TRANSPORTS:
        return

    if config is None:
        from agent_utilities.core.config import config as _config

        config = _config

    if not transport_auth_configured:
        raise RuntimeError(
            f"Refusing to serve graph-os over {transport}: no transport "
            "authentication provider is configured, so client identity "
            "cannot be validated. Set --auth-type/AUTH_TYPE to a real scheme "
            "(jwt, oidc-proxy, oauth-proxy, remote-oauth, or static) — having "
            "AUTH_JWT_JWKS_URI/OIDC_* set is not sufficient by itself. "
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
    credential_expires_at = _claim_expiry(claims)
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
    # The proof key and token are destroyed, but their validated expiry remains
    # authority. Long-running process callers renew by minting a fresh proof;
    # destroying proof material must never turn a bounded credential into an
    # indefinite process grant.
    actor = _actor_with_credential_lease(actor_from_claims(claims))
    return _mint_graph_session(
        actor,
        audience=_LOCAL_PROCESS_AUDIENCE,
        policy_version=_LOCAL_PROCESS_POLICY_VERSION,
    )


_system_write_session: GraphSession | None = None
_system_write_session_lock = threading.Lock()

# Headroom demanded of a CACHED system session before it is reused. A session
# handed back with two seconds left would expire mid-RPC in the caller's hands;
# re-minting slightly early is cheap (the OAuth2 provider caches its own token)
# and makes the handed-out authority usable for the whole call it was fetched
# for. See :func:`system_write_session` (BUG-PE-053).
_SYSTEM_SESSION_MIN_TTL_S = 30


def _session_authority_usable(session: GraphSession) -> bool:
    """Whether a cached system session's credential is still good to hand out.

    ``ensure_authority_current`` is the single existing definition of "this
    verified authority is still valid" (expired JWT, a lease inside its
    minimum TTL, or lapsed engine-route continuity) -- this reuses it rather
    than re-deriving expiry semantics here. Any other exception type means
    the session is not something this helper can vouch for either, so it is
    treated the same way: drop it and mint a fresh one.
    """
    try:
        session.ensure_authority_current(minimum_ttl_seconds=_SYSTEM_SESSION_MIN_TTL_S)
    except Exception:  # noqa: BLE001 - any unusable cached authority is re-minted
        logger.info(
            "cached system write session is no longer current; re-minting",
            exc_info=True,
        )
        return False
    return True


def system_write_session(config: Any = None) -> GraphSession:
    """Resolve the verified authority for a background/system-triggered graph write.

    BUG-033/BUG-039 (CONCEPT:AU-OS.identity.authenticated-identity-enforcement):
    three write paths (chat persistence, multi-IDE conversation ingestion,
    messaging post-conversation enrichment) used to call the engine with
    whatever ambient actor happened to be bound -- often none, since a bare
    ``asyncio`` scheduler loop or a standalone script never threads one
    through. That silently landed ``Thread``/``Message``/``Concept`` nodes
    with no owner (the downstream ``stamp_ownership`` call raised
    ``PermissionError`` and was swallowed).

    Two-step resolution, preferring the caller's own authority so a write made
    inside an already-authenticated context (a served request, a task-worker
    thread started via ``_authorized_background_thread``, a messaging
    co-service thread whose event loop inherited its daemon's bound session)
    is correctly attributed to that real caller, not a generic system actor:

    1. Prefer the already-verified ambient :class:`~agent_utilities.
       knowledge_graph.core.session.GraphSession` (:meth:`GraphSession.
       from_ambient`).
    2. Else mint -- then cache for as long as the minted credential stays
       valid -- this process's OWN verified system identity through the
       exact mechanism every other background/served entrypoint in this
       codebase already uses to establish its process authority
       (:func:`mint_local_process_session` for the ``tiny`` local profile
       with no configured external identity, else
       :func:`acquire_process_identity_token` ->
       :func:`mint_actor_from_token_sync` -> :func:`mint_graph_session` for a
       configured external identity -- see :func:`messaging.daemon.
       mint_process_identity` and ``kg_server.py``'s own bootstrap, which run
       this identical sequence). This is a REAL, validated, authenticated
       actor -- never a synthesized or unauthenticated one -- so it is never
       a bypass of the ownership-stamping seam it is meant to satisfy.

    THE CACHE IS REVALIDATED, NOT PERMANENT (BUG-PE-053, measured live
    2026-08-25). The session minted in step 2 wraps a bearer JWT with a
    finite lifetime (Keycloak's access-token lifespan -- minutes, not the
    lifetime of a long-running gateway process). The cache originally
    returned that session forever, so every consumer began failing closed
    with ``SessionExpiredError: Verified graph authority has expired`` once
    the first minted token aged out, and NOTHING could recover it short of
    restarting the process. That stayed invisible while the only consumers
    were best-effort background writers that swallow their own failures;
    it became a hard, permanent, whole-surface outage the moment
    ``gateway/registry_api.py`` started routing every ``/api/registry/*``
    and ``/api/enhanced/tools`` catalog read through this helper
    (``8dc652039``) -- the dashboard's entire MCP tools/servers surface
    served ``503 catalog_unavailable`` / an empty ``mcp_tools`` list from
    roughly the token lifespan after each pod start onwards. So the cached
    session is checked with :meth:`GraphSession.ensure_authority_current`
    before it is handed back, and re-minted when that check says the
    credential has expired (or is about to). ``_SYSTEM_SESSION_MIN_TTL_S``
    buys enough headroom that a session handed to a caller does not expire
    part-way through the engine RPC it was fetched for.

    Never returns an unauthenticated session: every branch below either
    returns a verified :class:`GraphSession` or raises (``SessionRequiredError``
    from ``from_ambient``'s failure path is caught internally; a genuine
    minting failure -- e.g. neither ``KG_AUTH_TOKEN_REF`` nor
    ``KG_IDENTITY_OAUTH2`` configured outside the ``tiny`` profile --
    propagates loudly instead of returning a degraded identity).
    """
    from agent_utilities.knowledge_graph.core.session import (
        GraphSession,
        SessionRequiredError,
    )

    try:
        return GraphSession.from_ambient()
    except SessionRequiredError:  # noqa: BLE001 — expected control flow: no ambient session bound is the documented signal to fall through to minting this process's own verified system identity below, not a real failure (see docstring above)
        pass

    global _system_write_session
    with _system_write_session_lock:
        cached = _system_write_session
        if cached is not None and _session_authority_usable(cached):
            return cached
        if config is None:
            from agent_utilities.core.config import config as _config

            config = _config
        if local_process_authority_enabled(config):
            session = mint_local_process_session()
        else:
            token = acquire_process_identity_token(config)
            actor = mint_actor_from_token_sync(token)
            session = mint_graph_session(actor)
        _system_write_session = session
        return session


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


def _resolve_process_identity_source(config: Any) -> tuple[str, Any]:
    """``(token_ref, oauth2)`` — fail-closed unless EXACTLY one is configured."""
    token_ref = str(getattr(config, "kg_auth_token_ref", None) or "").strip()
    oauth2 = getattr(config, "kg_identity_oauth2", None)
    if bool(token_ref) == bool(oauth2):
        raise RuntimeError(
            "Configure exactly one graph process identity source: "
            "KG_AUTH_TOKEN_REF or KG_IDENTITY_OAUTH2"
        )
    return token_ref, oauth2


def _fetch_process_identity_token(token_ref: str, oauth2: Any) -> str:
    try:
        if token_ref:
            from .cli_secrets import resolve_runtime_secret_reference

            return resolve_runtime_secret_reference(token_ref)
        from .oauth_client_credentials import build_provider_from_config

        if oauth2 is None:
            raise RuntimeError(
                "No graph process identity source configured: expected "
                "KG_IDENTITY_OAUTH2 when KG_AUTH_TOKEN_REF is unset"
            )
        return build_provider_from_config(oauth2).get_token()
    except Exception as exc:
        # BUG-PE-028: was `from None`, discarding the real cause (a
        # transport/TLS/secret-lookup failure) entirely -- a
        # CERTIFICATE_VERIFY_FAILED once surfaced as this opaque message
        # with no way to find the actual root cause short of monkeypatching
        # `requests.post`. Every caller of this function is internal
        # server-side bootstrap code (gateway/messaging daemons, ingest
        # worker, MCP servers -- never an external/untrusted consumer), so
        # chaining the cause is safe: the OUTER message stays sanitised
        # (never echoes secret/token material), while `__cause__` keeps the
        # real exception available to server-side logs/tracebacks.
        raise RuntimeError("Graph process identity acquisition failed") from exc


def _validate_process_identity_token(token: str) -> str:
    if (
        not isinstance(token, str)
        or not token
        or len(token.encode("utf-8")) > 16_384
        or any(ord(character) < 32 or ord(character) == 127 for character in token)
    ):
        raise RuntimeError(
            "Graph process identity acquisition returned invalid material"
        )
    return token


def acquire_process_identity_token(config: Any = None) -> str:
    """Acquire the graph process JWT from exactly one runtime identity source.

    Configuration stores only secret references or an OAuth2 client-credentials
    block. Token material is resolved/minted at startup and is never logged.
    """
    if config is None:
        from agent_utilities.core.config import config as live_config

        config = live_config
    token_ref, oauth2 = _resolve_process_identity_source(config)
    token = _fetch_process_identity_token(token_ref, oauth2)
    return _validate_process_identity_token(token)


def mint_actor_from_token_sync(token: str) -> ActorContext:
    """Synchronously validate a graph process JWT and mint an actor.

    Callable from BOTH a plain synchronous context and one that already has a
    running event loop. `asyncio.run` refuses the latter outright
    (``RuntimeError: asyncio.run() cannot be called from a running event
    loop``), and this function's original form called it unconditionally.

    That was not a theoretical gap. `agent-webui`'s `ensure_tenant_admission`
    is `async`, and it calls the synchronous `_service_authority()`, which
    lands here — so every Keycloak sign-in raised, surfacing to the user as
    `{"detail": "Internal request failed", "error_id": ...}` with the real
    cause only visible in the pod log. The daemon callers
    (`gateway/daemon.py`, `messaging/daemon.py`, `mcp/kg_server.py`) run with
    no loop and never saw it.

    When a loop is already running, the coroutine is driven to completion on
    its own loop in a worker thread and this thread blocks on the result. That
    keeps the function's contract — synchronous in, actor out — instead of
    pushing `async` up through every caller, and it cannot deadlock: the
    worker owns a fresh loop and shares no state with the caller's.
    """
    import asyncio
    import concurrent.futures

    def _drive() -> ActorContext:
        return asyncio.run(actor_from_bearer_token(token))

    try:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return _drive()
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(_drive).result()
    except Exception as exc:
        # Preserve the cause: this is the same JWT verification path as the
        # per-request boundary, and discarding it here would hide the exact
        # class of fault this change targets (a dependency/config fault in
        # `_decode_jwt` reported as an opaque failure with no signal of why).
        raise RuntimeError("Graph process identity token validation failed") from exc


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


async def _resolve_prevalidated_actor(
    prevalidated: Any, send: Any
) -> tuple[ActorContext | None, bool]:
    """``(actor, handled)`` from the outer HTTP boundary's already-verified
    claims — ``handled=True`` means a 401 was already sent and the caller
    must return immediately without any further processing."""
    if not (isinstance(prevalidated, dict) and prevalidated.get("auth_type") == "jwt"):
        return None, False
    # The outer HTTP authentication boundary already verified this
    # credential. Reuse its claims rather than making a second JWKS
    # lookup/verification pass with potentially different timing.
    try:
        return actor_from_claims(prevalidated), False
    except (TypeError, ValueError):
        await _send_json(send, 401, {"error": "Token validation failed"})
        return None, True


async def _resolve_bearer_actor(
    token: str, send: Any
) -> tuple[ActorContext | None, bool]:
    """``(actor, handled)`` from a live JWKS bearer-token verification —
    same ``handled`` contract as :func:`_resolve_prevalidated_actor`."""
    from fastapi import HTTPException
    from fastapi import status as http_status

    try:
        return await actor_from_bearer_token(token), False
    except HTTPException as exc:
        if exc.status_code == http_status.HTTP_401_UNAUTHORIZED:
            await _send_json(send, 401, {"error": "Token validation failed"})
            return None, True
        # A verification-path fault that is NOT a credential rejection
        # (currently only `_decode_jwt`'s 500 when its JWT dependency is
        # missing) must surface loudly and distinctly, never collapse
        # to a 401 — that collapse is exactly how a missing dependency
        # was misreported as a rejected credential.
        logger.error(
            "JWT verification path failed (status=%s): %s",
            exc.status_code,
            exc.detail,
        )
        await _send_json(send, exc.status_code, {"error": exc.detail})
        return None, True
    except Exception:  # noqa: BLE001 — any other failure = invalid credential
        await _send_json(send, 401, {"error": "Token validation failed"})
        return None, True


async def _resolve_request_actor(
    token: str | None, prevalidated: Any, config: Any, send: Any
) -> tuple[ActorContext | None, bool]:
    """``(actor, handled)`` — tries prevalidated claims first, then a live
    bearer-token verification, in the SAME order and under the SAME JWKS
    availability gate as the original inline sequence."""
    actor, handled = await _resolve_prevalidated_actor(prevalidated, send)
    if handled:
        return None, True
    if token and not config.auth_jwt_jwks_uri:
        await _send_json(send, 401, {"error": "Token validation unavailable"})
        return None, True
    if token and actor is None:
        actor, handled = await _resolve_bearer_actor(token, send)
        if handled:
            return None, True
    return actor, False


def _extract_authorization_headers(scope: Any) -> list[Any]:
    return [
        value
        for key, value in scope.get("headers") or []
        if isinstance(key, bytes) and key.lower() == b"authorization"
    ]


def _extract_prevalidated_claims(scope: Any) -> Any:
    state = scope.get("state") or {}
    return state.get("user_claims") if isinstance(state, dict) else None


async def _mint_request_session(actor: ActorContext, send: Any) -> GraphSession | None:
    """The server-owned GraphSession, or ``None`` once the matching 401/403 has
    already been sent.

    A ``None`` return therefore means "response already handled" — the caller
    must return immediately. The previous shape returned ``(session, handled)``,
    but ``handled`` was exactly ``session is None`` on every path, and the caller
    needed a narrowing ``assert`` to use the session. ``assert`` is stripped
    under ``-O``, which on this authenticated dispatch path would have let a
    ``None`` session reach the route; making the check load-bearing removes that.
    """
    from agent_utilities.knowledge_graph.core.session import SessionExpiredError

    try:
        return mint_graph_session(actor)
    except (CredentialExpiredError, SessionExpiredError):
        await _send_json(send, 401, {"error": "Bearer credential expired"})
        return None
    except PermissionError:
        await _send_json(send, 403, {"error": "Verified tenant claim required"})
        return None


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

    async def _serve_unauthenticated_path(
        self, scope: Any, receive: Any, send: Any, path: str
    ) -> None:
        """Handle the ``actor is None`` case: 401 unless ``path`` is exempt,
        in which case it is served under an explicit non-authoritative
        health-probe actor with no inherited GraphSession."""
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

    async def _dispatch_authenticated(
        self,
        scope: Any,
        receive: Any,
        send: Any,
        actor: ActorContext,
        session: GraphSession,
    ) -> None:
        """Bind actor+session context vars for the app call, dispatch, and
        unwind them — degrading a credential/session expiry to 401 only if
        no response has been sent yet (a started response cannot be
        rewritten, so it must re-raise instead)."""
        from agent_utilities.knowledge_graph.core.session import (
            SessionExpiredError,
            reset_session,
            set_session,
        )

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

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        from agent_utilities.core.config import config

        from .auth import parse_bearer_authorization

        path = scope.get("path", "")
        prevalidated = _extract_prevalidated_claims(scope)

        try:
            token = parse_bearer_authorization(_extract_authorization_headers(scope))
        except PermissionError:
            await _send_json(send, 401, {"error": "Verified Bearer identity required"})
            return

        actor, handled = await _resolve_request_actor(token, prevalidated, config, send)
        if handled:
            return

        if actor is None:
            await self._serve_unauthenticated_path(scope, receive, send, path)
            return

        # The authenticated request establishes both ambient currencies.  The
        # session is minted here, outside every served route, so async tasks and
        # worker-thread dispatch inherit one immutable, verified authority.
        session = await _mint_request_session(actor, send)
        if session is None:
            return

        await self._dispatch_authenticated(scope, receive, send, actor, session)


__all__ = [
    "ActorIdentityMiddleware",
    "CARRIER_CLAIM_FIELDS",
    "HEALTH_PATHS",
    "OPTIONAL_CARRIER_CLAIM_FIELDS",
    "SERVED_TRANSPORTS",
    "UNAUTHENTICATED_PATHS",
    "VerifiedRequestAuthority",
    "acquire_process_identity_token",
    "actor_from_bearer_token",
    "actor_from_claims",
    "apply_served_security_profile",
    "build_verified_request_authority",
    "local_process_authority_enabled",
    "mint_local_process_session",
    "mint_graph_session",
    "mint_actor_from_token_sync",
    "system_write_session",
    "validate_carrier_claims",
]
