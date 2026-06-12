#!/usr/bin/python
from __future__ import annotations

"""Read-path enforcement helpers for the Company Brain (CONCEPT:KG-2.6).

Thin, reusable functions that apply data-level permissions, tenant scoping, and
read auditing on top of the dormant :class:`CompanyBrain` managers. Every helper
is a **no-op unless ``KG_BRAIN_ENFORCE`` is on**, so default behaviour is
unchanged and callers can sprinkle these without conditionals.

Identity comes from the ambient :func:`current_actor` (set by the MCP server /
agent runner via ``use_actor``); callers may override per-call.
"""

import logging
from typing import Any

from ...security.brain_context import ActorContext, current_actor
from .company_brain_runtime import brain_enforcement_enabled, get_company_brain

logger = logging.getLogger(__name__)


def permit(node_ids: list[str], actor: ActorContext | None = None) -> list[str]:
    """Return only the node ids ``actor`` is permitted to read.

    No-op (returns input) when enforcement is off. ACL default is *allow*, so
    only nodes with an explicit restricting ACL are filtered.
    """
    if not brain_enforcement_enabled() or not node_ids:
        return node_ids
    actor = actor or current_actor()
    try:
        return get_company_brain().permissions.filter_nodes(
            node_ids,
            actor.actor_id,
            actor.actor_type,
            action="read",
            actor_roles=list(actor.roles),
        )
    except Exception as exc:  # pragma: no cover - fail safe = deny nothing extra
        logger.debug("permit() failed, returning unfiltered: %s", exc)
        return node_ids


def audit_read(
    node_ids: list[str], summary: str = "", actor: ActorContext | None = None
) -> None:
    """Record a read-access audit entry (mandatory for RESTRICTED nodes)."""
    if not brain_enforcement_enabled():
        return
    actor = actor or current_actor()
    try:
        get_company_brain().provenance.record_read(
            actor_id=actor.actor_id,
            actor_type=actor.actor_type,
            nodes_accessed=list(node_ids),
            query_summary=summary,
            tenant_id=actor.tenant_id,
        )
    except Exception as exc:  # pragma: no cover - audit best-effort
        logger.debug("audit_read() failed: %s", exc)


def scope(cypher: str, actor: ActorContext | None = None) -> str:
    """Tenant-scope a Cypher read query for ``actor`` (``n.tenant_id = <tenant>``).

    Cross-org isolation, the primary boundary (KG-2.6). Kept to a SIMPLE equality
    so even the lightweight in-memory operational-subset interpreter can parse it;
    the finer owner/scope visibility (KG-2.60) is applied as a Python post-filter
    in :func:`visible` so it is backend-agnostic. No-op when enforcement is off or
    the actor has no tenant.
    """
    if not brain_enforcement_enabled():
        return cypher
    actor = actor or current_actor()
    if not actor.tenant_id:
        return cypher
    try:
        return get_company_brain().tenancy.scope_cypher_query(cypher, actor.tenant_id)
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("tenant scope() failed, leaving unscoped: %s", exc)
        return cypher


def visible(
    rows: list[dict[str, Any]], actor: ActorContext | None = None
) -> list[dict[str, Any]]:
    """Drop rows the actor may not see by owner/scope (KG-2.60), Python-side.

    The backend-agnostic companion to :func:`scope`: applies private-by-default
    owner/scope visibility on the returned rows so it works on any backend
    (including the in-memory interpreter that cannot parse a compound predicate).
    No-op when enforcement is off or for a privileged actor.
    """
    if not brain_enforcement_enabled() or not rows:
        return rows
    try:
        from .tenant_sharing import filter_visible

        return filter_visible(rows, actor)
    except Exception as exc:  # pragma: no cover - never break a read
        logger.debug("visible() filter skipped: %s", exc)
        return rows


_CLASS_ORDER = {"public": 0, "internal": 1, "confidential": 2, "restricted": 3}


def inherit_inferred_acl(subject: str, obj: str) -> None:
    """Entailment-aware scoping: an inferred fact inherits its parents' secrecy.

    Sets the inferred target's classification to the *most restrictive* of the
    two endpoints, so OWL reasoning can't leak a RESTRICTED parent through a
    derived edge. No-op unless enforcement is on.
    """
    if not brain_enforcement_enabled():
        return
    try:
        perms = get_company_brain().permissions
        levels = []
        for nid in (subject, obj):
            acl = perms.get_acl(nid)
            if acl is not None:
                levels.append(acl.classification)
        if not levels:
            return
        strictest = max(levels, key=lambda c: _CLASS_ORDER.get(str(c), 0))
        target_acl = perms.get_acl(obj)
        if target_acl is None or _CLASS_ORDER.get(
            str(target_acl.classification), 0
        ) < _CLASS_ORDER.get(str(strictest), 0):
            perms.classify_node(obj, strictest)
    except Exception as exc:  # pragma: no cover - best-effort propagation
        logger.debug("inherit_inferred_acl failed for %s->%s: %s", subject, obj, exc)


def _row_node_id(row: dict[str, Any]) -> str | None:
    """Best-effort extraction of a node id from a result row."""
    for key in ("id", "node_id", "n.id", "_id"):
        val = row.get(key)
        if isinstance(val, str):
            return val
    for val in row.values():  # Cypher often returns a node dict under an alias
        if isinstance(val, dict):
            inner = val.get("id") or val.get("node_id")
            if isinstance(inner, str):
                return inner
    return None


def filter_rows(
    rows: list[dict[str, Any]], actor: ActorContext | None = None
) -> list[dict[str, Any]]:
    """Drop result rows whose identifiable node id is ACL-denied for ``actor``.

    Rows whose id can't be determined are kept (we never silently lose data we
    can't classify); auditing still records the access.
    """
    if not brain_enforcement_enabled() or not rows:
        return rows
    actor = actor or current_actor()
    ids = [nid for r in rows if (nid := _row_node_id(r))]
    if not ids:
        return rows
    allowed = set(permit(ids, actor))
    return [r for r in rows if (nid := _row_node_id(r)) is None or nid in allowed]
