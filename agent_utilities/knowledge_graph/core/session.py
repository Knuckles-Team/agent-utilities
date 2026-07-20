#!/usr/bin/python
from __future__ import annotations

"""``GraphSession`` — the one explicit currency threaded through KG entrypoints (AU-P0-1).

Identity, policy, placement, and trace originate in distinct subsystems:

* :class:`~agent_utilities.security.brain_context.ActorContext` carries *who*
  (``actor_id``/``actor_type``/``roles``/``tenant_id``/``authenticated``), set
  ambiently via :func:`~agent_utilities.security.brain_context.use_actor` /
  :func:`~agent_utilities.security.brain_context.current_actor`.
* :mod:`agent_utilities.observability.correlation` carries *what run this is
  part of* (the W3C ``traceparent`` / correlation id), also ambient.
* Policy (``action_policy``/``permissioning``) is resolved per-call with no
  stable version stamped onto the request at all.

None of these is an explicit object a caller threads through a method
signature on its own — so a query several layers deep would otherwise have no
single value to hand to a nested call, log, or audit row that says "this is
the session this happened under".

``GraphSession`` composes those authorities with scope, graph, transaction,
placement, and audience targeting in one immutable object. Served boundaries
mint it from verified process or bearer identity and propagate it through the
entire graph call. Callers cannot construct authority from request fields.

MCP/REST dispatch and in-process graph operations require the immutable session
minted by authentication middleware or the configured stdio process identity.
There is no development bypass, implicit actor, or authority-widening helper.
"""

import contextvars
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from typing import Any

from agent_utilities.security.brain_context import (
    ActorContext,
    CredentialExpiredError,
)

__all__ = [
    "GraphSession",
    "SessionExpiredError",
    "SessionRequiredError",
    "current_session",
    "graph_session_required",
    "resolve_session",
    "suspend_session",
    "use_session",
    "set_session",
    "reset_session",
    "ScopeError",
]


class ScopeError(PermissionError):
    """Raised by :meth:`GraphSession.require_scope` when a scope is missing."""


class SessionRequiredError(PermissionError):
    """Raised when a production graph operation bypasses its verified session."""


class SessionExpiredError(SessionRequiredError):
    """Raised when validated bearer authority is no longer current."""


def graph_session_required() -> bool:
    """Return the mandatory graph-session invariant.

    Kept as a named predicate for callers that share the invariant; it is not a
    runtime feature switch.
    """
    return True


@dataclass(frozen=True)
class GraphSession:
    """The one explicit currency for a unit of KG work: who, under what policy,
    against which graph/snapshot, correlated to which trace.

    Attributes:
        actor: *Who* is doing the work (wraps :class:`ActorContext`; does not
            replace it — ``session.actor.tenant_id``/``.roles`` etc. still work
            exactly as before).
        tenant: The required tenant id this session operates under; it must
            equal the verified actor's tenant.
        scopes: The permission scopes granted to this session (e.g.
            ``{"kg:read", "kg:write"}``). Empty means "no explicit scopes
            recorded" — :meth:`require_scope` only enforces when the caller
            actually checks, so an empty ``scopes`` is not itself a denial.
        graph: The target named graph/namespace (CONCEPT:AU-KG.sharding.tenant-partitioned-sharding-hrw
            — e.g. ``tenant_graph_name(tenant)``). Empty string means "use the
            backend's configured default".
        endpoint: Optional explicit backend endpoint/shard this session is
            pinned to (host, connection string, or shard label). ``None``
            means "resolve normally" through the active backend and engine catalog.
        placement_group: Authoritative Raft group returned by the engine
            placement catalog. ``None`` means no catalog route has been bound yet.
        catalog_epoch: Optional schema/catalog version fingerprint the session
            was opened against, for staleness detection against a moving
            ontology. ``None`` when not tracked.
        txn: Optional live backend-native transaction/snapshot handle (e.g. a
            ``CheckedOutSubgraph`` or another backend-native txn object) this
            session's writes/reads should route through. ``None`` means
            "no active txn — hit the backend directly".
        policy_version: The authorization/policy revision this session was
            authorized under, for audit and for detecting a stale grant when
            policy changes mid-run. Defaults to the ``KG_POLICY_VERSION``
            setting (empty string when unset).
        trace_context: The W3C ``traceparent`` (or correlation id fallback)
            this session's work should be attributed to. Populated from the
            ambient correlation module by :meth:`from_ambient`.
        audience: The server-validated audience this authority was minted for.
            It is forwarded to the engine's v2 verified request context; it is
            never accepted from an HTTP/tool payload in the served profile.
    """

    actor: ActorContext
    tenant: str
    scopes: frozenset[str] = field(default_factory=frozenset)
    graph: str = ""
    endpoint: str | None = None
    placement_group: int | None = None
    catalog_epoch: int | None = None
    txn: Any | None = None
    policy_version: str | int = ""
    trace_context: str | None = None
    audience: str = ""

    def __post_init__(self) -> None:
        actor_id = str(getattr(self.actor, "actor_id", "") or "").strip()
        actor_tenant = str(getattr(self.actor, "tenant_id", "") or "").strip()
        tenant = str(self.tenant or "").strip()
        if not getattr(self.actor, "authenticated", False) or not actor_id:
            raise SessionRequiredError(
                "GraphSession authority must be minted from verified identity"
            )
        if not tenant or not actor_tenant or tenant != actor_tenant:
            raise SessionRequiredError(
                "GraphSession requires one verified, matching tenant authority"
            )

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    @classmethod
    def from_ambient(
        cls,
    ) -> GraphSession:
        """Return the already-bound verified session; never synthesize one."""
        ambient_session = _current.get()
        if ambient_session is None:
            raise SessionRequiredError(
                "A verified ambient GraphSession is required; production graph "
                "operations may not synthesize authority"
            )
        return ambient_session

    # ------------------------------------------------------------------
    # Immutable "with" helpers
    # ------------------------------------------------------------------
    def with_graph(self, graph: str) -> GraphSession:
        """Return a copy of this session targeting a different named graph."""
        return replace(self, graph=graph)

    def with_txn(self, txn: Any) -> GraphSession:
        """Return a copy of this session bound to a live transaction/snapshot handle."""
        return replace(self, txn=txn)

    def with_actor(self, actor: ActorContext) -> GraphSession:
        """Return a copy of this session scoped to a different (verified) actor.

        Re-runs :meth:`__post_init__` via :func:`dataclasses.replace`, so the
        new actor must still be authenticated and its tenant must still match.
        """
        return replace(self, actor=actor, tenant=actor.tenant_id or self.tenant)

    def with_scopes(self, *scopes: str) -> GraphSession:
        """Return a copy of this session with ``scopes`` added to its scope set."""
        return replace(self, scopes=self.scopes | frozenset(scopes))

    def with_route(
        self,
        *,
        endpoint: str,
        placement_group: int | None,
        catalog_epoch: int,
    ) -> GraphSession:
        """Return a copy carrying server-resolved placement metadata.

        This helper may narrow routing only; it does not change identity,
        tenant, graph, scopes, policy, or trace authority.
        """
        return replace(
            self,
            endpoint=endpoint,
            placement_group=placement_group,
            catalog_epoch=int(catalog_epoch),
        )

    def engine_verified_context(self) -> dict[str, Any]:
        """Return the current engine authority claims for this immutable session.

        The engine hashes ``principal`` before persisting ChangeEnvelope
        provenance.  ``agent_id`` remains the authenticated ACL subject used by
        the engine; no filesystem path, workstation username, or caller-supplied
        display name is introduced here. The native client accepts exactly these
        eight claims; trace correlation remains in the governed ChangeEnvelope
        context rather than being smuggled into the authority object.
        """
        self.ensure_authority_current()
        if not getattr(self.actor, "authenticated", False):
            raise SessionRequiredError(
                "GraphSession authority was not minted from verified identity"
            )
        actor_id = str(getattr(self.actor, "actor_id", "") or "").strip()
        if not actor_id:
            raise SessionRequiredError("GraphSession has no authenticated principal")
        tenant = str(self.tenant or "").strip()
        audience = str(self.audience or "").strip()
        policy_version = str(self.policy_version or "").strip()
        if not tenant or not audience or not policy_version:
            raise SessionRequiredError(
                "GraphSession lacks tenant, audience, or policy-version authority"
            )
        return {
            "principal": actor_id,
            "tenant": tenant,
            "audience": audience,
            "agent_id": actor_id,
            "roles": sorted(
                {
                    rendered
                    for role in getattr(self.actor, "roles", ())
                    if (rendered := str(role).strip())
                }
            ),
            "scopes": sorted(
                {rendered for scope in self.scopes if (rendered := str(scope).strip())}
            ),
            "delegation": [],
            "policy_version": policy_version,
        }

    def ensure_authority_current(self, *, minimum_ttl_seconds: int = 0) -> None:
        """Fail closed when the session's validated bearer JWT has expired.

        The expiry remains in memory and is neither forwarded to the engine nor
        persisted as provenance. ``minimum_ttl_seconds`` supports proactive
        OAuth renewal before a new stdio tool call begins.
        """
        import time

        try:
            self.actor.ensure_credential_current()
        except (CredentialExpiredError, TypeError, ValueError):
            raise SessionExpiredError("Verified graph authority has expired") from None
        lease = getattr(self.actor, "credential_lease", None)
        expiry = (
            lease.expires_at
            if lease is not None
            else getattr(self.actor, "credential_expires_at", None)
        )
        if expiry is not None and int(time.time()) + max(
            0, int(minimum_ttl_seconds)
        ) >= int(expiry):
            raise SessionExpiredError("Verified graph authority expires too soon")

    # ------------------------------------------------------------------
    # Enforcement
    # ------------------------------------------------------------------
    def require_scope(self, scope: str) -> None:
        """Raise :class:`ScopeError` when ``scope`` is not granted.

        The coarse KG scopes are hierarchical: ``kg:admin`` satisfies
        ``kg:write`` and ``kg:read``; ``kg:write`` also satisfies
        ``kg:read`` because mutations require authorization-safe precondition
        reads.  Fine-grained engine capabilities remain exact matches.

        Roles never bypass scopes: trusted administrative identities are
        normalized to an explicit ``kg:admin`` scope at the identity boundary.
        """
        coarse_grants = {
            "kg:read": frozenset({"kg:read", "kg:write", "kg:admin"}),
            "kg:write": frozenset({"kg:write", "kg:admin"}),
            "kg:admin": frozenset({"kg:admin"}),
        }
        accepted = coarse_grants.get(scope, frozenset({scope}))
        if self.scopes.isdisjoint(accepted):
            raise ScopeError(f"GraphSession is missing required scope {scope!r}")


# ---------------------------------------------------------------------------
# Ambient propagation — mirrors agent_utilities.security.brain_context
# ---------------------------------------------------------------------------
_current: contextvars.ContextVar[GraphSession | None] = contextvars.ContextVar(
    "graph_session", default=None
)


def current_session() -> GraphSession | None:
    """Return the ambient :class:`GraphSession` for this execution context, if any.

    ``None`` when nothing has scoped one. Authority is never synthesized.
    """
    return _current.get()


def resolve_session(
    session: GraphSession | None = None,
    *,
    graph: str | None = None,
    required_scope: str | None = None,
) -> GraphSession:
    """Resolve the sole authority for a graph operation.

    Every caller must inherit the authentication middleware's ambient
    session.  Supplying a different actor/tenant/scope/policy/graph context is
    rejected even if its dataclass fields claim to be authenticated.  A caller
    may attach a transaction handle to the same authority; it may not widen or
    retarget that authority.
    """
    ambient = _current.get()
    if ambient is None:
        raise SessionRequiredError(
            "A verified ambient GraphSession is required for this graph operation"
        )
    candidate = session or ambient
    authority_fields = (
        "actor",
        "tenant",
        "scopes",
        "graph",
        "endpoint",
        "placement_group",
        "catalog_epoch",
        "policy_version",
        "trace_context",
        "audience",
    )
    if any(
        getattr(candidate, name) != getattr(ambient, name) for name in authority_fields
    ):
        raise SessionRequiredError(
            "The supplied GraphSession does not match the verified ambient authority"
        )
    resolved = candidate
    resolved.ensure_authority_current()

    if graph is not None and graph != resolved.graph:
        raise SessionRequiredError(
            "A graph operation may not retarget its verified graph authority"
        )
    if required_scope is not None:
        resolved.require_scope(required_scope)
    return resolved


def set_session(session: GraphSession) -> contextvars.Token[GraphSession | None]:
    """Set a verified session, returning a token for :func:`reset_session`."""
    # Revalidate at the ambient boundary even if a caller bypassed normal
    # dataclass construction (for example via unsafe deserialization).
    session.__post_init__()
    session.ensure_authority_current()
    return _current.set(session)


def reset_session(token: contextvars.Token[GraphSession | None]) -> None:
    """Restore the prior session (inverse of :func:`set_session`)."""
    _current.reset(token)


@contextmanager
def suspend_session() -> Iterator[None]:
    """Temporarily remove ambient graph authority and restore it on exit.

    This is used at explicitly unauthenticated boundaries such as liveness
    probes.  It prevents a parent task's verified session from being inherited
    by code that is intentionally executing without graph authority.
    """
    token = _current.set(None)
    try:
        yield None
    finally:
        _current.reset(token)


@contextmanager
def use_session(session: GraphSession) -> Iterator[GraphSession]:
    """Scope a block of work to ``session`` (restores the previous one on exit).

    Example::

        session = mint_graph_session(verified_identity)
        with use_session(session):
            engine.query_cypher("MATCH (n) RETURN n LIMIT 1")  # inherits it
    """
    token = set_session(session)
    try:
        yield session
    finally:
        _current.reset(token)
