#!/usr/bin/python
from __future__ import annotations

"""Ambient actor context for Company Brain enforcement (CONCEPT:KG-2.6).

The Company Brain's trust, permission, and audit layers need to know *who* is
reading or writing. Threading an actor through hundreds of existing call sites
would be invasive and break compatibility, so identity is carried in a
``contextvars.ContextVar`` instead: callers that care (the MCP server, agent
runner) set it with :func:`use_actor`; everything else transparently inherits
the privileged :data:`SYSTEM_ACTOR`, preserving today's behaviour until a
deployment turns enforcement on.

This module is deliberately dependency-light (only the shared ``ActorType``
enum) so it can be imported from any layer without cycles.
"""

import contextvars
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field, replace

from ..models.company_brain import ActorType


@dataclass(frozen=True)
class ActorContext:
    """The identity of whoever is currently reading or writing the graph.

    ``roles`` drives role-based ACL checks; ``tenant_id`` drives multi-tenant
    isolation. Both default to permissive values so unscoped callers behave
    exactly as they do today.

    ``authenticated`` is True only when the identity was minted server-side
    from a validated credential (JWT via the gateway middleware or a validated
    ``KG_AUTH_TOKEN`` — CONCEPT:OS-5.14). Caller-supplied identities and the
    ambient :data:`SYSTEM_ACTOR` are unauthenticated; when an authenticated
    actor is in scope, caller-supplied ``_actor``/``_roles``/``_tenant``
    kwargs are ignored.
    """

    actor_id: str = "system"
    actor_type: ActorType = ActorType.SYSTEM
    roles: tuple[str, ...] = field(default_factory=tuple)
    tenant_id: str = ""
    authenticated: bool = False

    def with_tenant(self, tenant_id: str) -> ActorContext:
        return replace(self, tenant_id=tenant_id)


# Privileged default: full access, used whenever no caller has scoped the
# context. Enforcement (KG_BRAIN_ENFORCE) is what makes scoping matter; until
# then every read/write runs as the system actor and nothing is filtered.
SYSTEM_ACTOR = ActorContext(
    actor_id="system",
    actor_type=ActorType.SYSTEM,
    roles=("admin", "system"),
    tenant_id="",
)


_current: contextvars.ContextVar[ActorContext] = contextvars.ContextVar(
    "company_brain_actor", default=SYSTEM_ACTOR
)


def current_actor() -> ActorContext:
    """Return the actor for the current execution context."""
    return _current.get()


def set_actor(ctx: ActorContext) -> contextvars.Token[ActorContext]:
    """Set the current actor, returning a token for :func:`reset_actor`."""
    return _current.set(ctx)


def reset_actor(token: contextvars.Token[ActorContext]) -> None:
    """Restore the actor to its prior value."""
    _current.reset(token)


# ---------------------------------------------------------------------------
# Source-system context — "which system did this write come from?"
# ---------------------------------------------------------------------------
# Distinct from the actor ("who"): a single agent may ingest from ServiceNow
# then ERPNext in one run. Ingestion paths set this so the write-path guard can
# resolve source authority/trust. Defaults to "" (falls back to the actor id).
_source: contextvars.ContextVar[str] = contextvars.ContextVar(
    "company_brain_source", default=""
)


def current_source() -> str:
    """Return the source-system label for the current write context."""
    return _source.get()


@contextmanager
def use_source(source_system: str) -> Iterator[str]:
    """Scope writes to a named source system (for trust/provenance)."""
    token = _source.set(source_system or "")
    try:
        yield source_system
    finally:
        _source.reset(token)


@contextmanager
def use_actor(ctx: ActorContext) -> Iterator[ActorContext]:
    """Scope a block of work to ``ctx`` (restores the previous actor on exit).

    Example::

        with use_actor(ActorContext("agent:marketing", ActorType.AI_AGENT,
                                    roles=("marketing",), tenant_id="acme")):
            kg.search("...")  # reads filtered to what marketing may see
    """
    token = _current.set(ctx)
    try:
        yield ctx
    finally:
        _current.reset(token)
