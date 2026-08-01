#!/usr/bin/python
from __future__ import annotations

"""Fail-closed read-path enforcement helpers for the Company Brain.

Thin, reusable functions that apply data-level permissions, tenant scoping, and
read auditing on top of the :class:`CompanyBrain` managers. Enforcement is a
current contract, not a feature switch.

Identity comes from the ambient :func:`current_actor` (set by the MCP server /
agent runner via ``use_actor``); callers may override per-call.
"""

import json
from typing import Any

from ...security.brain_context import ActorContext, current_actor
from .company_brain_runtime import get_company_brain


def _verified_actor(actor: ActorContext | None) -> ActorContext:
    resolved = actor or current_actor()
    resolved.ensure_credential_current()
    actor_id = str(getattr(resolved, "actor_id", "") or "").strip()
    tenant_id = str(getattr(resolved, "tenant_id", "") or "").strip()
    if not getattr(resolved, "authenticated", False) or not actor_id or not tenant_id:
        raise PermissionError("Graph reads require verified tenant authority")
    return resolved


def permit(
    node_ids: list[str],
    actor: ActorContext | None = None,
) -> list[str]:
    """Return only the node ids ``actor`` is permitted to read.

    Nodes without an ACL are denied. Authorization infrastructure failures are
    surfaced rather than returning unfiltered data.
    """
    if not node_ids:
        return []
    actor = _verified_actor(actor)
    try:
        permissions = get_company_brain().permissions
        missing = [
            node_id for node_id in node_ids if permissions.get_acl(node_id) is None
        ]
        if missing:
            _hydrate_missing_acls(missing, actor)
        return permissions.filter_nodes(
            node_ids,
            actor.actor_id,
            actor.actor_type,
            action="read",
            actor_roles=list(actor.roles),
        )
    except Exception as exc:  # pragma: no cover - defensive boundary
        raise PermissionError("Node permission evaluation failed") from exc


def _durable_access_rows(node_ids: list[str]) -> dict[str, dict[str, Any]]:
    """Fetch durable ACL material for cache misses in one bounded round-trip.

    No active process-owned graph means there is no hydration authority and the
    caller remains default-denied. An active authority without a readable
    backend is a configuration failure, not permission to fall back to N
    per-node reads.

    Reads through ``active.backend`` — the SAME authority every node write
    (``IngestionMixin._upsert_node``, used by ``ingest_mcp_server`` and every
    other platform-node ingestion path) targets. Earlier this read
    ``active.graph_compute`` instead: that object is only the SAME store as
    ``active.backend`` when the backend happens to expose a reusable
    ``.graph`` (the single-process EpistemicGraphBackend chain); for any other
    backend it is a distinct, never-written-to compute scratchpad, so newly
    ingested platform nodes had no durable ACL material to hydrate and the
    fail-closed guard denied them. Reading the backend directly removes that
    asymmetry instead of loosening the guard.
    """

    from .engine import IntelligenceGraphEngine

    active = IntelligenceGraphEngine.get_active()
    if active is None:
        return {}
    backend = getattr(active, "backend", None)
    execute_read = getattr(backend, "execute_read", None)
    if not callable(execute_read):
        raise PermissionError("Durable ACL hydration authority is unavailable")
    try:
        rows = execute_read(
            "MATCH (n) WHERE n.id IN $ids RETURN n.id AS id, "
            "n.tenant_id AS tenant_id, n.classification AS classification, "
            "n.external_access AS external_access",
            {"ids": list(node_ids)},
        )
    except Exception as exc:
        raise PermissionError("Durable ACL hydration query failed") from exc
    if not isinstance(rows, list):
        raise PermissionError("Durable ACL hydration response is invalid")

    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise PermissionError("Durable ACL hydration response is invalid")
        node_id = row.get("id")
        if not isinstance(node_id, str):
            continue
        external_access = row.get("external_access")
        if isinstance(external_access, str):
            try:
                external_access = json.loads(external_access)
            except (TypeError, ValueError):
                external_access = None
        result[node_id] = {
            "tenant_id": row.get("tenant_id"),
            "classification": row.get("classification"),
            "external_access": external_access,
        }
    return result


def _hydrate_missing_acls(node_ids: list[str], actor: ActorContext) -> None:
    """Rebuild process-local ACL entries from governed durable node metadata."""

    from ...models.company_brain import DataClassification
    from ...protocols.source_connectors.base import ExternalAccess
    from ...protocols.source_connectors.permission_sync import sync_access

    rows = _durable_access_rows(node_ids)
    for node_id in node_ids:
        properties = rows.get(node_id)
        if properties is None:
            continue
        if str(properties.get("tenant_id") or "") != actor.tenant_id:
            continue
        raw_access = properties.get("external_access")
        if not isinstance(raw_access, dict):
            continue
        try:
            access = ExternalAccess.model_validate(raw_access)
            classification = DataClassification(
                str(properties.get("classification") or "")
            )
            sync_access(node_id, access, classification=classification)
        except Exception as exc:
            raise PermissionError("Durable ACL metadata is invalid") from exc


def audit_read(
    node_ids: list[str],
    summary: str = "",
    actor: ActorContext | None = None,
) -> None:
    """Record a read-access audit entry (mandatory for RESTRICTED nodes)."""
    actor = _verified_actor(actor)
    try:
        get_company_brain().provenance.record_read(
            actor_id=actor.actor_id,
            actor_type=actor.actor_type,
            nodes_accessed=list(node_ids),
            query_summary=summary,
            tenant_id=actor.tenant_id,
        )
    except Exception as exc:  # pragma: no cover - defensive boundary
        raise PermissionError("Read audit recording failed") from exc


def scope(
    cypher: str,
    actor: ActorContext | None = None,
) -> str:
    """Tenant-scope a Cypher read query for ``actor`` (``n.tenant_id = <tenant>``).

    Cross-org isolation, the primary boundary (KG-2.6). Kept to a simple,
    portable equality; finer owner/scope visibility (KG-2.60) is applied as a
    mandatory post-filter in :func:`visible`.
    """
    actor = _verified_actor(actor)
    try:
        return get_company_brain().tenancy.scope_cypher_query(cypher, actor.tenant_id)
    except Exception as exc:  # pragma: no cover - defensive boundary
        raise PermissionError("Tenant query scoping failed") from exc


def visible(
    rows: list[dict[str, Any]],
    actor: ActorContext | None = None,
) -> list[dict[str, Any]]:
    """Drop rows the actor may not see by owner/scope (KG-2.60), Python-side.

    The backend-agnostic companion to :func:`scope`: applies private-by-default
    owner/scope visibility on the returned rows. Missing identity or visibility
    infrastructure denies the read.
    """
    actor = _verified_actor(actor)
    if not rows:
        return []
    try:
        from .tenant_sharing import filter_visible

        return filter_visible(rows, actor)
    except Exception as exc:  # pragma: no cover - defensive boundary
        raise PermissionError("Row visibility evaluation failed") from exc


_CLASS_ORDER = {"public": 0, "internal": 1, "confidential": 2, "restricted": 3}


def inherit_inferred_acl(subject: str, obj: str) -> None:
    """Entailment-aware scoping: an inferred fact inherits its parents' secrecy.

    Sets the inferred target's classification to the *most restrictive* of the
    two endpoints, so OWL reasoning can't leak a RESTRICTED parent through a
    derived edge.
    """
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
    except Exception as exc:  # pragma: no cover - defensive boundary
        raise PermissionError("Inferred ACL propagation failed") from exc


def _row_node_id(row: dict[str, Any]) -> str | None:
    """Best-effort extraction of a node id from a result row."""
    for key in ("id", "node_id", "n.id", "_id"):
        val = row.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    for val in row.values():  # Cypher often returns a node dict under an alias
        if isinstance(val, dict):
            inner = val.get("id") or val.get("node_id")
            if isinstance(inner, str) and inner.strip():
                return inner.strip()
    return None


def row_node_ids(rows: list[dict[str, Any]]) -> list[str]:
    """Return the governed node id carried by every result row.

    Public graph projections must retain an ``id`` (or a node mapping that
    contains one) so authorization and audit refer to the same objects. A
    projection that removes identity is not governable and is denied.
    """
    ids = [_row_node_id(row) for row in rows]
    if any(node_id is None for node_id in ids):
        raise PermissionError("Graph result contains a row without a governed node id")
    return [node_id for node_id in ids if node_id is not None]


def filter_rows(
    rows: list[dict[str, Any]],
    actor: ActorContext | None = None,
) -> list[dict[str, Any]]:
    """Drop result rows whose identifiable node id is ACL-denied for ``actor``.

    Every row must expose a governable node id. Unclassifiable rows are rejected
    so a projection cannot bypass ACL evaluation.
    """
    actor = _verified_actor(actor)
    if not rows:
        return []
    governed_ids = row_node_ids(rows)
    allowed = set(permit(governed_ids, actor))
    return [
        row
        for row, node_id in zip(rows, governed_ids, strict=True)
        if node_id in allowed
    ]
