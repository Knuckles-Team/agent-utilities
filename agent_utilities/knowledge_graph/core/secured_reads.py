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
import logging
from typing import TYPE_CHECKING, Any

from ...security.brain_context import ActorContext, current_actor
from .company_brain_runtime import get_company_brain

if TYPE_CHECKING:
    from ...models.company_brain import DataClassification

logger = logging.getLogger(__name__)


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

    R-22/GOC-67 (defect 1): ``active.backend`` is the PROCESS-DEFAULT engine's
    backend — fine as long as the caller's verified work is against that same
    default graph. Once explicit-graph selection (``resolve_explicit_graph`` /
    ``bound_to_graph``) narrows the ambient :class:`~.session.GraphSession` to a
    DIFFERENT physical graph, the real row read (routed through the wire
    layer's own session-graph resolution) correctly comes from the selected
    graph, but this ACL hydration kept reading the default backend regardless
    — the selected graph's nodes are simply absent there, so they were
    (correctly, given the wrong backend) default-denied. The requested graph
    is derived ONLY from the verified ambient session — never accepted as a
    parameter here — so this layer cannot be pointed at an unverified,
    caller-supplied graph; the session was already validated fail-closed
    against the engine's own graph catalog before it could ever carry an
    unknown or unauthorized name (``resolve_explicit_graph``). When the
    session's graph differs from the active engine's own bound graph, hydrate
    from ``IntelligenceGraphEngine.for_graph(<trusted-graph>)`` instead — a
    lightweight, zero-transport view over the SAME process transport (no new
    socket/connection), never a second authorization mechanism: the engine's
    RBAC/RLS still evaluates every RPC server-side exactly as it does for the
    row read itself. A missing per-graph view or a hydration failure on that
    view is surfaced as :class:`PermissionError`, identically to every other
    failure mode here — never a silent fallback to the wrong graph's backend.
    """

    from .engine import IntelligenceGraphEngine
    from .session import current_session

    active = IntelligenceGraphEngine.get_active()
    if active is None:
        return {}

    session = current_session()
    requested_graph = (
        str(getattr(session, "graph", "") or "") if session is not None else ""
    )
    active_graph = str(
        getattr(getattr(active, "graph_compute", None), "graph_name", "") or ""
    )

    hydration_authority = active
    if requested_graph and requested_graph != active_graph:
        view_factory = getattr(active, "for_graph", None)
        if not callable(view_factory):
            raise PermissionError("Durable ACL hydration authority is unavailable")
        try:
            hydration_authority = view_factory(requested_graph)
        except Exception as exc:
            raise PermissionError(
                "Durable ACL hydration authority is unavailable"
            ) from exc
        if hydration_authority is None:
            raise PermissionError("Durable ACL hydration authority is unavailable")

    backend = getattr(hydration_authority, "backend", None)
    execute_read = getattr(backend, "execute_read", None)
    if not callable(execute_read):
        raise PermissionError("Durable ACL hydration authority is unavailable")
    try:
        rows = execute_read(
            "MATCH (n) WHERE n.id IN $ids RETURN n.id AS id, "
            "n.tenant_id AS tenant_id, n.classification AS classification, "
            "n.external_access AS external_access, n._owner_id AS owner_id, "
            "n._shared_scope AS shared_scope",
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
            "owner_id": row.get("owner_id"),
            # D-P0-U119: the write-time governance stamp
            # (`tenant_sharing.stamp_ownership`) always writes `_owner_id` AND
            # `_shared_scope` together -- an org-/commons-shared node has NO
            # `external_access` descriptor (that shape is connector-only, see
            # `_hydrate_missing_acls`'s docstring) and its private-by-default
            # `_owner_id` alone denies every non-owner same-tenant reader. This
            # field was previously dropped here, so `_hydrate_missing_acls` had
            # no organization-sharing evidence and every non-owner reader in the
            # SAME tenant was (incorrectly) default-denied post-restart/cache-miss
            # even though `tenant_sharing.visible()`'s raw-row post-filter (a
            # DIFFERENT, already-correct enforcement path) would have shown the
            # row. Must be preserved through this mapping for
            # `_hydrate_missing_acls` to reconcile it.
            "shared_scope": row.get("shared_scope"),
        }
    return result


def _parse_classification(raw: Any) -> DataClassification | None:
    """Best-effort ``DataClassification`` parse; ``None`` for anything unrecognized."""
    from ...models.company_brain import DataClassification

    try:
        return DataClassification(str(raw or ""))
    except ValueError:
        return None


def _hydrate_missing_acls(node_ids: list[str], actor: ActorContext) -> None:
    """Rebuild process-local ACL entries from governed durable node metadata.

    Two independent durable-metadata shapes are understood, mirroring the two
    write-time governance stamps (CONCEPT:AU-KG.backend.company-brain-write-guard):

    1. **Connector-sourced access** (``external_access``, a source-connector
       ``ExternalAccess`` descriptor) — synced via :func:`sync_access`,
       unchanged from before.
    2. **First-party write-time stamp** (``classification`` +
       ``_owner_id``/``tenant_id``, stamped by
       ``tenant_sharing.stamp_classification``/``stamp_ownership`` at the
       ``IntelligenceGraphEngine._upsert_node`` / ``GraphComputeEngine.add_node``
       chokepoints) — synthesized directly into a :class:`NodeACL` here. First-
       party (non-connector) nodes never carry an ``external_access``
       descriptor — only Documents ingested through a source connector do — so
       without this fallback EVERY internally-created node (Memory, Episode,
       Skill, CallableResource, Concept, ...) had no durable ACL material the
       connector-oriented gate above understood, and stayed permanently
       denied — even to its own owning actor, in production, not just tests.

    A node with neither ``external_access`` nor a stamped ``_owner_id``/
    ``PUBLIC``/organization-shared classification (unowned/system data, or
    first-party data written before this fix existed) is left unregistered
    and stays denied — fail-closed, exactly as before this fallback existed.
    This only unblocks a node's own real, verified owner, genuinely PUBLIC
    data, or data explicitly org-/commons-shared BY ITS OWNER; it never
    widens access for anyone else, and the cross-tenant check above still
    gates every branch.

    3. **Organization/commons sharing** (``_shared_scope`` — D-P0-U119).
       ``tenant_sharing.stamp_ownership`` always writes ``_owner_id`` and
       ``_shared_scope`` together, and the raw-row post-filter
       (``tenant_sharing.visible``/``visibility_predicate``) already treats
       ``_shared_scope in ('org', 'commons')`` as visible to every reader in
       the SAME tenant. The per-node-ACL gate here (``permit``/
       ``filter_rows`` → :class:`NodeACL`) had no equivalent: it only ever
       granted the recorded owner or PUBLIC classification, so a same-tenant
       non-owner reader was denied by THIS gate even though the raw-row
       filter would have shown the row — a governed multi-row projection
       (``filter_rows`` applies ``permit`` first) therefore silently dropped
       every organization-shared row it did not itself own, most visibly
       right after a process restart / ACL-cache miss. Fixed by granting the
       verified reading actor explicit per-actor read access
       (:attr:`NodeACL.read_actors`) whenever the durable row is
       org-/commons-shared — additive only: it never touches
       ``read_actors``/``read_roles``/``admin_actors`` for a private,
       unshared node, and the same-tenant gate above still applies first.
    """

    from ...models.company_brain import DataClassification
    from ...protocols.source_connectors.base import ExternalAccess
    from ...protocols.source_connectors.permission_sync import sync_access
    from .tenant_sharing import SCOPE_COMMONS, SCOPE_ORG

    org_shared_scopes = {SCOPE_ORG, SCOPE_COMMONS}

    rows = _durable_access_rows(node_ids)
    for node_id in node_ids:
        properties = rows.get(node_id)
        if properties is None:
            continue
        if str(properties.get("tenant_id") or "") != actor.tenant_id:
            continue
        raw_access = properties.get("external_access")
        if isinstance(raw_access, dict):
            try:
                access = ExternalAccess.model_validate(raw_access)
                classification = DataClassification(
                    str(properties.get("classification") or "")
                )
                sync_access(node_id, access, classification=classification)
            except Exception as exc:
                raise PermissionError("Durable ACL metadata is invalid") from exc
            continue

        # No connector descriptor: synthesize from the first-party write-time
        # stamp instead of leaving the node permanently unclassified.
        parsed_classification = _parse_classification(properties.get("classification"))
        owner_id = str(properties.get("owner_id") or "").strip()
        shared_scope = str(properties.get("shared_scope") or "").strip().lower()
        org_shared = shared_scope in org_shared_scopes
        if parsed_classification is DataClassification.PUBLIC:
            get_company_brain().permissions.classify_node(
                node_id, DataClassification.PUBLIC
            )
        elif owner_id or org_shared:
            acl = get_company_brain().permissions.classify_node(
                node_id,
                parsed_classification or DataClassification.CONFIDENTIAL,
                data_owner=owner_id,
            )
            if org_shared and actor.actor_id not in acl.read_actors:
                acl.read_actors.append(actor.actor_id)
                get_company_brain().permissions.set_acl(acl)
        # else: no owner, not PUBLIC, not org-/commons-shared -> nothing to
        # synthesize; the node stays denied (fail closed), identical to
        # pre-fix behavior.


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
) -> tuple[str, dict[str, Any]]:
    """Tenant-scope a Cypher read query for ``actor`` (``<bound var>.tenant_id = $_tenant_scope_id``).

    Cross-org isolation, the primary boundary (KG-2.6). Kept to a simple,
    portable equality; finer owner/scope visibility (KG-2.60) is applied as a
    mandatory post-filter in :func:`visible`.

    The injected predicate is written against the query's own first bound
    node variable (:func:`~.cypher_scoping.first_bound_node_variable`), never
    a hardcoded ``n`` — a caller-written query keeps whatever variable name it
    chose (``MATCH (p:Policy) ...``, ``MATCH (f:ProcessFlow) ...``). When
    ``cypher`` has a ``WHERE``/``RETURN`` clause to scope but no derivable
    bound node variable, ``scope_cypher_query`` raises
    :class:`~.cypher_scoping.UnscopableQueryError` (a ``PermissionError``
    subclass) rather than silently emitting an unscoped or mis-scoped read;
    this fails the same way here, wrapped below like every other scoping
    failure.

    Returns:
        ``(scoped_cypher, extra_params)`` (D-W2T-2 — the tenant id is a bound
        parameter, not a string-literal splice). The caller MUST merge
        ``extra_params`` into whatever params dict it executes
        ``scoped_cypher`` with.

    A :class:`~.cypher_scoping.UnscopableQueryError` (or any other
    ``PermissionError``) raised underneath is propagated AS-IS — it is
    already a deliberately-typed, specific fail-closed decision, and
    re-wrapping it here would flatten its actionable message (e.g. "every
    node pattern in the first `MATCH` clause is anonymous") into the generic
    "Tenant query scoping failed", making a query-shape problem
    indistinguishable from every other denial. Anything else (an
    infrastructure failure inside the tenancy manager/company brain) is
    still logged with its full cause and wrapped as ``PermissionError`` —
    that failure mode is a deliberate "cannot verify scope, so deny"
    fail-closed posture, not a code defect, so it keeps denying rather than
    surfacing as an internal-error type.
    """
    actor = _verified_actor(actor)
    try:
        return get_company_brain().tenancy.scope_cypher_query(cypher, actor.tenant_id)
    except PermissionError:
        raise
    except Exception as exc:  # pragma: no cover - defensive boundary
        logger.error(
            "Tenant query scoping failed: %s: %s",
            type(exc).__name__,
            exc,
            exc_info=True,
        )
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
