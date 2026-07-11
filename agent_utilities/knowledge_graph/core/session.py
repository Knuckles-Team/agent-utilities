#!/usr/bin/python
from __future__ import annotations

"""``GraphSession`` — the one explicit currency threaded through KG entrypoints (AU-P0-1).

**The gap this closes.** Identity, policy, and trace today are three separate,
mostly-ambient authorities:

* :class:`~agent_utilities.security.brain_context.ActorContext` carries *who*
  (``actor_id``/``actor_type``/``roles``/``tenant_id``/``authenticated``), set
  ambiently via :func:`~agent_utilities.security.brain_context.use_actor` /
  :func:`~agent_utilities.security.brain_context.current_actor`.
* :mod:`agent_utilities.observability.correlation` carries *what run this is
  part of* (the W3C ``traceparent`` / correlation id), also ambient.
* Policy (``action_policy``/``permissioning``) is resolved per-call with no
  stable version stamped onto the request at all.

None of the three is an explicit object a caller threads through a method
signature — so a query 3 layers deep has no single value to hand to a nested
call, log, or audit row that says "this is the session this happened under".

``GraphSession`` wraps all three (plus scope/graph/txn targeting) in ONE
dataclass. It does not replace :class:`ActorContext` or the correlation
module — it composes them. Two ways to get one:

* :meth:`GraphSession.from_ambient` — the back-compat bridge. Reads
  :func:`current_actor` and the active correlation id, so existing call sites
  (which set ambient state via ``use_actor``/``bind_carrier`` and never knew
  about a session object) still produce a fully-populated session when a
  callee asks for one.
* Construct one explicitly and pass it down, or scope a block of work to it
  with :func:`use_session` — mirrors :func:`~agent_utilities.security.brain_context.use_actor`.

Entrypoints adopt this incrementally: add ``session: GraphSession | None =
None`` and, when the caller didn't supply one, derive it via
``GraphSession.from_ambient()``. Nothing existing breaks; new/updated callers
get one explicit object to pass instead of juggling actor + trace + policy
separately.
"""

import contextvars
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from typing import Any

from agent_utilities.core.config import setting
from agent_utilities.observability import correlation
from agent_utilities.security.brain_context import (
    SYSTEM_ACTOR,
    ActorContext,
    current_actor,
)

__all__ = [
    "GraphSession",
    "current_session",
    "use_session",
    "set_session",
    "reset_session",
    "ScopeError",
]


class ScopeError(PermissionError):
    """Raised by :meth:`GraphSession.require_scope` when a scope is missing."""


@dataclass(frozen=True)
class GraphSession:
    """The one explicit currency for a unit of KG work: who, under what policy,
    against which graph/snapshot, correlated to which trace.

    All fields are optional/defaulted so a bare ``GraphSession()`` behaves like
    today's unscoped ambient default (:data:`SYSTEM_ACTOR`, default graph, no
    txn) — constructing one never requires a running service.

    Attributes:
        actor: *Who* is doing the work (wraps :class:`ActorContext`; does not
            replace it — ``session.actor.tenant_id``/``.roles`` etc. still work
            exactly as before).
        tenant: The resolved tenant id this session operates under. Defaults to
            ``actor.tenant_id`` when unset (see :meth:`__post_init__`), so
            existing single-tenant callers see ``tenant == ""`` unchanged.
        scopes: The permission scopes granted to this session (e.g.
            ``{"kg:read", "kg:write"}``). Empty means "no explicit scopes
            recorded" — :meth:`require_scope` only enforces when the caller
            actually checks, so an empty ``scopes`` is not itself a denial.
        graph: The target named graph/namespace (CONCEPT:AU-KG.sharding.tenant-partitioned-sharding-hrw
            — e.g. ``tenant_graph_name(tenant)``). Empty string means "use the
            backend's configured default".
        endpoint: Optional explicit backend endpoint/shard this session is
            pinned to (host, connection string, or shard label). ``None``
            means "resolve normally" (active backend / HRW routing).
        catalog_epoch: Optional schema/catalog version fingerprint the session
            was opened against, for staleness detection against a moving
            ontology. ``None`` when not tracked.
        txn: Optional live transaction/snapshot handle (e.g. a
            ``CheckedOutSubgraph`` or a backend-native txn object) this
            session's writes/reads should route through. ``None`` means
            "no active txn — hit the backend directly".
        policy_version: The authorization/policy revision this session was
            authorized under, for audit and for detecting a stale grant when
            policy changes mid-run. Defaults to the ``KG_POLICY_VERSION``
            setting (empty string when unset).
        trace_context: The W3C ``traceparent`` (or correlation id fallback)
            this session's work should be attributed to. Populated from the
            ambient correlation module by :meth:`from_ambient`.
    """

    actor: ActorContext = field(default_factory=lambda: SYSTEM_ACTOR)
    tenant: str = ""
    scopes: frozenset[str] = field(default_factory=frozenset)
    graph: str = ""
    endpoint: str | None = None
    catalog_epoch: int | None = None
    txn: Any | None = None
    policy_version: str | int = ""
    trace_context: str | None = None

    def __post_init__(self) -> None:
        # Default tenant from the actor so existing single-field callers
        # (which only ever set ``actor.tenant_id``) don't have to also set
        # ``tenant`` for it to be correct.
        if not self.tenant and self.actor is not None and self.actor.tenant_id:
            object.__setattr__(self, "tenant", self.actor.tenant_id)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    @classmethod
    def from_ambient(
        cls,
        graph: str | None = None,
        **overrides: Any,
    ) -> GraphSession:
        """Build a session from today's ambient state (the back-compat bridge).

        Reads :func:`current_actor` (set by the MCP server / agent runner via
        ``use_actor``) and the active correlation id / traceparent (set by
        :mod:`agent_utilities.observability.correlation`), so an entrypoint that
        adopts the explicit ``session`` parameter but whose caller never
        constructed one still gets a fully-populated :class:`GraphSession`
        that matches what ``current_actor()``/``current_source()`` would have
        told it separately.

        Args:
            graph: Optional named graph override (maps to the ``graph`` field).
            **overrides: Any other :class:`GraphSession` field to override
                (e.g. ``policy_version="v3"``, ``scopes=frozenset({"kg:write"})``).

        Returns:
            A new :class:`GraphSession`.
        """
        ambient_session = _current.get()
        actor = overrides.pop("actor", None) or (
            ambient_session.actor if ambient_session is not None else current_actor()
        )
        tenant = overrides.pop("tenant", None)
        if tenant is None:
            tenant = (
                ambient_session.tenant
                if ambient_session is not None
                else actor.tenant_id
            )
        trace_context = overrides.pop("trace_context", None)
        if trace_context is None:
            trace_context = correlation.ensure_correlation_id()
        policy_version = overrides.pop("policy_version", None)
        if policy_version is None:
            policy_version = (
                ambient_session.policy_version
                if ambient_session is not None
                else setting("KG_POLICY_VERSION", "")
            )
        resolved_graph = graph if graph is not None else overrides.pop("graph", "")
        if not resolved_graph and ambient_session is not None:
            resolved_graph = ambient_session.graph

        return cls(
            actor=actor,
            tenant=tenant,
            graph=resolved_graph,
            trace_context=trace_context,
            policy_version=policy_version,
            **overrides,
        )

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
        """Return a copy of this session scoped to a different actor."""
        return replace(self, actor=actor, tenant=actor.tenant_id or self.tenant)

    def with_scopes(self, *scopes: str) -> GraphSession:
        """Return a copy of this session with ``scopes`` added to its scope set."""
        return replace(self, scopes=self.scopes | frozenset(scopes))

    # ------------------------------------------------------------------
    # Enforcement
    # ------------------------------------------------------------------
    def require_scope(self, scope: str) -> None:
        """Raise :class:`ScopeError` if ``scope`` is not in :attr:`scopes`.

        System actors (``actor.actor_type is ActorType.SYSTEM`` with the
        privileged default roles) are exempt — this mirrors
        :data:`~agent_utilities.security.brain_context.SYSTEM_ACTOR` being the
        ambient default everywhere enforcement is off, so a session built with
        no scopes at all behaves exactly like today's unscoped callers instead
        of denying everything by default.
        """
        if self.actor is not None and "admin" in self.actor.roles:
            return
        if scope not in self.scopes:
            raise ScopeError(
                f"GraphSession is missing required scope {scope!r} "
                f"(actor={self.actor.actor_id if self.actor else '?'!r}, "
                f"granted={sorted(self.scopes)!r})"
            )


# ---------------------------------------------------------------------------
# Ambient propagation — mirrors agent_utilities.security.brain_context
# ---------------------------------------------------------------------------
_current: contextvars.ContextVar[GraphSession | None] = contextvars.ContextVar(
    "graph_session", default=None
)


def current_session() -> GraphSession | None:
    """Return the ambient :class:`GraphSession` for this execution context, if any.

    ``None`` (rather than a privileged default) when nothing has scoped one —
    callers that need a concrete session should use
    ``current_session() or GraphSession.from_ambient()``.
    """
    return _current.get()


def set_session(session: GraphSession) -> contextvars.Token[GraphSession | None]:
    """Set the current session, returning a token for :func:`reset_session`."""
    return _current.set(session)


def reset_session(token: contextvars.Token[GraphSession | None]) -> None:
    """Restore the prior session (inverse of :func:`set_session`)."""
    _current.reset(token)


@contextmanager
def use_session(session: GraphSession) -> Iterator[GraphSession]:
    """Scope a block of work to ``session`` (restores the previous one on exit).

    Example::

        session = GraphSession.from_ambient(graph="acme:__commons__")
        with use_session(session):
            engine.query_cypher("MATCH (n) RETURN n LIMIT 1")  # inherits it
    """
    token = _current.set(session)
    try:
        yield session
    finally:
        _current.reset(token)
