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
from ...security.identifiers import validate_identifier
from .company_brain_runtime import get_company_brain

if TYPE_CHECKING:
    from ...models.company_brain import DataClassification

logger = logging.getLogger(__name__)

# Node labels VERIFIED (this session, by reading the write paths — not
# guessed) to be written by the fleet-registration hot path that drove the
# "1-4s per tool" production incident: `source_sync`'s catalog-write loop
# builds entities with `"type": "MCPServer"/"Tool"/"Skill"` (spliced straight
# through as the node's Cypher label by `_ingest_graph_slice_via_envelope`),
# and `skill_workflow_ingest.ingest_runnable_skill` writes the runnable
# resource as `engine._upsert_node("CallableResource", resource_id, ...)`.
#
# PERF-SR-1 (row-governance clone-the-graph audit): this tuple is now a
# SECONDARY safety net, not the primary defense against the O(graph) unlabeled
# `MATCH (n)` scan below. The primary defense is `_id_indexed_batch_rows` —
# a label-independent, bounded-by-`len(node_ids)` lookup through the active
# backend's own id-primary-key node store (when the backend exposes one; see
# that function's docstring). Any id that primary path resolves NEVER reaches
# this tuple or the unlabeled scan at all, for ANY label — a non-allowlisted
# label (e.g. `Preference`) is no longer a cliff on a backend with that
# accelerator. This tuple, and the unlabeled fallback after it, remain the
# ONLY path for a backend that does not expose the accelerator (an
# `execute_read`-only Cypher surface with no id-indexed store beneath it) —
# for such a backend, an id whose label isn't in this tuple still pays the
# full unlabeled scan, because there is no generic way to bound an
# id-equality-in-list Cypher query without either an id index in the query
# engine itself (a parallel lane's dependency, see `_id_indexed_batch_rows`)
# or knowing the label in advance. Extending this tuple is still a valid
# (if narrow) mitigation for that remaining case; it is simply no longer
# load-bearing for the backend that hit the production incident.
_LABELED_HYDRATION_CANDIDATES: tuple[str, ...] = (
    "Tool",
    "MCPServer",
    "CallableResource",
    "Skill",
)


def _id_indexed_batch_rows(backend: Any, node_ids: list[str]) -> list[dict[str, Any]]:
    """Bounded, label-independent ACL-row hydration via the backend's own
    id-primary-key node store — the fix for the "no id index, only a label
    index" cliff described on `_durable_access_rows`, without needing an
    enumerated label allowlist at all.

    Every concrete node store beneath a backend is ALREADY id-keyed for its
    own CRUD surface — `has_node(id)`, `remove_node(id)`, and (critically)
    `GraphComputeEngine._get_node_properties_batch(ids)` all resolve by
    primary key, in ONE round trip for the batch form, with no per-label
    branching and no full-graph clone. That primitive is not new: it is the
    SAME one `EpistemicGraphBackend.semantic_search` already uses to
    hydrate a candidate id set's properties (see that method). What has "no
    id index" is specifically the CYPHER QUERY PLANNER's `MATCH (n) WHERE
    n.id IN $ids` form (`GraphCore::get_nodes()`, per the docstring below) —
    a distinct surface from the node store itself. This function reaches the
    node store directly, bypassing the query planner (and therefore the
    label question) entirely.

    Feature-detected via `backend.graph` (the backend's OWN public property
    exposing its `GraphComputeEngine`, e.g. `EpistemicGraphBackend.graph`)
    and `_get_node_properties_batch` on it. Absent on any backend that has no
    such store (a bare `execute_read`-only test double, or a future backend
    with a genuinely different storage shape) — those degrade to exactly the
    label-loop-then-unlabeled-scan path that ran before this function
    existed. Any exception, a non-dict response, or the capability being
    entirely missing all resolve to "not accelerated", never to a grant or a
    denial of their own: `_durable_access_rows`'s existing labeled/unlabeled
    Cypher path (already fail-closed, already covered by the tests in this
    module) is the sole source of truth whenever this returns a partial or
    empty result. A node with a genuinely empty property bag is deliberately
    left unresolved here (falls through to the Cypher path, which returns
    the same node with the same governance fields all `NULL`, denied the
    same way) — this function only ever shortcuts a lookup the Cypher path
    would answer identically, never changes the answer.

    Returns rows in the SAME shape `_durable_access_rows`'s Cypher
    `return_clause` produces (`id`/`tenant_id`/`classification`/
    `external_access`/`owner_id`/`shared_scope`) so the caller's existing
    row-assembly loop handles both sources uniformly — including the
    existing JSON-vs-native `external_access` normalization. `owner_id`/
    `shared_scope` are read from the underlying `_owner_id`/`_shared_scope`
    property names (the literal write-time stamp, per
    `tenant_sharing.stamp_ownership`) since this path reads raw node
    properties rather than an aliased Cypher `RETURN`.
    """
    if not node_ids:
        return []
    node_store = getattr(backend, "graph", None)
    batch_read = getattr(node_store, "_get_node_properties_batch", None)
    if not callable(batch_read):
        return []
    try:
        raw = batch_read(list(node_ids))
    except Exception:  # noqa: BLE001 — accelerator is a pure optimization, never authoritative on failure
        return []
    if not isinstance(raw, dict):
        return []
    rows: list[dict[str, Any]] = []
    for node_id in node_ids:
        props = raw.get(node_id)
        if not isinstance(props, dict) or not props:
            continue
        rows.append(
            {
                "id": node_id,
                "tenant_id": props.get("tenant_id"),
                "classification": props.get("classification"),
                "external_access": props.get("external_access"),
                "owner_id": props.get("_owner_id"),
                "shared_scope": props.get("_shared_scope"),
            }
        )
    return rows


def _ambient_tenant_id() -> str:
    """Return the authenticated ambient tenant for the SQL fast path."""
    try:
        actor = current_actor()
        if getattr(actor, "authenticated", False):
            return str(getattr(actor, "tenant_id", "") or "")
    except Exception:  # noqa: BLE001 — no actor means no SQL fast path
        pass
    return ""


def _catalog_acl_hits(
    active: Any, node_ids: list[str], tenant_id: str
) -> dict[str, dict[str, Any]]:
    """Best-effort SQL ACL projection; an unavailable projection is a miss."""
    from .fleet_catalog_tables import catalog_acl_rows

    try:
        return catalog_acl_rows(active, node_ids, tenant_id)
    except Exception:  # noqa: BLE001 — Cypher remains the authoritative fallback
        return {}


def _selected_hydration_authority(active: Any) -> Any:
    """Resolve durable ACL reads against the verified session graph."""
    from .session import current_session

    session = current_session()
    requested_graph = (
        str(getattr(session, "graph", "") or "") if session is not None else ""
    )
    active_graph = str(
        getattr(getattr(active, "graph_compute", None), "graph_name", "") or ""
    )
    if not requested_graph or requested_graph == active_graph:
        return active

    view_factory = getattr(active, "for_graph", None)
    if not callable(view_factory):
        raise PermissionError("Durable ACL hydration authority is unavailable")
    try:
        authority = view_factory(requested_graph)
    except Exception as exc:
        raise PermissionError("Durable ACL hydration authority is unavailable") from exc
    if authority is None:
        raise PermissionError("Durable ACL hydration authority is unavailable")
    return authority


def _hydration_reader(active: Any) -> tuple[Any, Any]:
    """Return the selected authority's backend and governed read method."""
    authority = _selected_hydration_authority(active)
    backend = getattr(authority, "backend", None)
    execute_read = getattr(backend, "execute_read", None)
    if not callable(execute_read):
        raise PermissionError("Durable ACL hydration authority is unavailable")
    return backend, execute_read


def _accelerated_hydration_rows(
    backend: Any, node_ids: list[str]
) -> tuple[list[dict[str, Any]], list[str]]:
    """Return id-indexed rows and ids that still need the Cypher path."""
    rows = _id_indexed_batch_rows(backend, node_ids)
    resolved_ids = {row["id"] for row in rows}
    pending = [node_id for node_id in node_ids if node_id not in resolved_ids]
    return rows, pending


def _validated_labeled_rows(
    found: Any,
) -> tuple[list[dict[str, Any]], set[str]]:
    """Validate a labeled query response and collect ids it resolved."""
    if not isinstance(found, list):
        raise PermissionError("Durable ACL hydration response is invalid")
    rows: list[dict[str, Any]] = []
    resolved_ids: set[str] = set()
    for row in found:
        if not isinstance(row, dict):
            raise PermissionError("Durable ACL hydration response is invalid")
        rows.append(row)
        row_id = row.get("id")
        if isinstance(row_id, str):
            resolved_ids.add(row_id)
    return rows, resolved_ids


def _read_cypher_hydration_rows(
    execute_read: Any, remaining: list[str], return_clause: str
) -> list[dict[str, Any]]:
    """Read labeled candidates, then preserve the general Cypher fallback."""
    rows: list[dict[str, Any]] = []
    pending = list(remaining)
    try:
        for candidate_label in _LABELED_HYDRATION_CANDIDATES:
            if not pending:
                break
            safe_label = validate_identifier(candidate_label, kind="label")
            found = execute_read(
                f"MATCH (n:{safe_label}) WHERE n.id IN $ids {return_clause}",
                {"ids": pending},
            )
            found_rows, resolved_ids = _validated_labeled_rows(found)
            rows.extend(found_rows)
            pending = [node_id for node_id in pending if node_id not in resolved_ids]
        if pending:
            found = execute_read(
                f"MATCH (n) WHERE n.id IN $ids {return_clause}",
                {"ids": pending},
            )
            if not isinstance(found, list):
                raise PermissionError("Durable ACL hydration response is invalid")
            rows.extend(found)
    except PermissionError:
        raise
    except Exception as exc:
        raise PermissionError("Durable ACL hydration query failed") from exc
    return rows


def _merge_hydration_rows(
    result: dict[str, dict[str, Any]], rows: list[dict[str, Any]]
) -> None:
    """Normalize durable query rows into the ACL hydrator's result shape."""
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
            "shared_scope": row.get("shared_scope"),
        }


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

    **SQL-authoritative fast path first** (CONCEPT:AU-KG.ingest.fleet-catalog-acl-projection).
    The production incident this closes measured two *unlabeled* Cypher full
    scans per fleet tool (1-4s each, ~12/min) — the label-scoped candidates
    below already fixed the "unlabeled" half; this fixes the "at all" half
    for the common case. ``fleet_catalog_tables.catalog_acl_rows`` carries a
    durable ACL stamp written from the SAME policy
    (``tenant_sharing.stamp_ownership``/``stamp_classification``) the
    matching KG node write uses, so for any id it can fully answer for, the
    Cypher round trip below is skipped entirely. It answers ONLY for an id
    whose catalog row was written with a real ACL stamp; every other id —
    not a fleet node, a fleet node the (possibly still-empty, see
    ``fleet_catalog_tables`` module docstring) catalog hasn't synced yet, or
    a legacy/un-stamped catalog row — falls through to the Cypher path
    below completely unchanged, so this can only ever make an id resolve
    FASTER, never resolve to something the Cypher path would not have
    granted.

    The SQL query's tenant scope is resolved from the SAME ambient
    :func:`~...security.brain_context.current_actor` every write-time stamp
    (:func:`~.fleet_catalog_tables._stamped_acl_fields`) and every other
    read helper in this module reads — never a caller-suppliable parameter
    (this function keeps the exact ``(node_ids)`` signature it always had:
    a graph, tenant, or actor is never accepted as a raw argument here, only
    ever resolved from verified ambient/session state). No bound actor (an
    unauthenticated context, or none at all) simply skips the fast path —
    every id then falls through to the Cypher path exactly as before this
    existed.

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

    active = IntelligenceGraphEngine.get_active()
    if active is None:
        return {}

    result: dict[str, dict[str, Any]] = {}
    remaining = list(dict.fromkeys(node_ids))

    tenant_id = _ambient_tenant_id()
    if tenant_id and remaining:
        sql_hits = _catalog_acl_hits(active, remaining, tenant_id)
        if sql_hits:
            result.update(sql_hits)
            remaining = [node_id for node_id in remaining if node_id not in sql_hits]

    if not remaining:
        return result

    backend, execute_read = _hydration_reader(active)

    # Label-scoped first: an unlabeled `MATCH (n)` resolves via
    # `GraphCore::get_nodes()`, which clones every node's property blob in
    # the ENTIRE graph on every call (no id index exists in the Cypher
    # engine, only a label index, `get_nodes_by_label`) — the dominant cost
    # of the measured "1-4s per tool during fleet registration" production
    # incident this fixes. `permit`/`_hydrate_missing_acls` hydrate ACLs for
    # ANY node type in the graph (Memory, Episode, Concept, Document, ... —
    # see `_hydrate_missing_acls`'s docstring), so this cannot be narrowed to
    # a single label the way a fleet-only helper can. `_LABELED_HYDRATION_
    # CANDIDATES` below are the labels VERIFIED this session to be written by
    # the fleet-registration hot path itself (`source_sync`'s catalog write
    # loop writes `MCPServer`/`Tool`/`Skill`; `ingest_runnable_skill` writes
    # `CallableResource`) — trying them first turns THAT hot path into a
    # handful of indexed lookups. Any id not resolved by one of those labels
    # falls through to the unlabeled query exactly as before, so every other
    # node type this function has ever supported keeps resolving correctly;
    # it just doesn't get the speedup. `execute_read` (session-scoped, same
    # authorization gate as every other read here) is used for every
    # attempt — no lower-level, unscoped read path is substituted.
    return_clause = (
        "RETURN n.id AS id, n.tenant_id AS tenant_id, "
        "n.classification AS classification, "
        "n.external_access AS external_access, n._owner_id AS owner_id, "
        "n._shared_scope AS shared_scope"
    )

    # Primary defense against the O(graph) unlabeled scan below: a bounded,
    # label-independent id lookup through the backend's own node store (see
    # `_id_indexed_batch_rows`'s docstring). Resolves ANY label, including one
    # never added to `_LABELED_HYDRATION_CANDIDATES` — e.g. `Preference`.
    # Purely additive: whatever it cannot resolve (capability absent,
    # transient failure, or a genuinely property-empty node) falls straight
    # through to the labeled/unlabeled Cypher path exactly as if this call
    # had never run.
    result_rows, remaining = _accelerated_hydration_rows(backend, remaining)

    result_rows.extend(
        _read_cypher_hydration_rows(execute_read, remaining, return_clause)
    )

    # `result` was pre-seeded above with the SQL fast path's hits (for a
    # disjoint id set — `remaining` never contained an id SQL already
    # answered), so this only ever ADDS entries, never overwrites one.
    _merge_hydration_rows(result, result_rows)
    return result


def _parse_classification(raw: Any) -> DataClassification | None:
    """Best-effort ``DataClassification`` parse; ``None`` for anything unrecognized."""
    from ...models.company_brain import DataClassification

    try:
        return DataClassification(str(raw or ""))
    except ValueError:
        return None


def _hydrate_connector_acl(node_id: str, properties: dict[str, Any]) -> None:
    """Restore an ACL backed by a source connector's access descriptor."""
    from ...models.company_brain import DataClassification
    from ...protocols.source_connectors.base import ExternalAccess
    from ...protocols.source_connectors.permission_sync import sync_access

    try:
        access = ExternalAccess.model_validate(properties["external_access"])
        classification = DataClassification(str(properties.get("classification") or ""))
        sync_access(node_id, access, classification=classification)
    except Exception as exc:
        raise PermissionError("Durable ACL metadata is invalid") from exc


def _hydrate_first_party_acl(
    node_id: str,
    properties: dict[str, Any],
    actor: ActorContext,
    org_shared_scopes: set[str],
) -> None:
    """Restore an ACL from first-party ownership/classification stamps."""
    from ...models.company_brain import DataClassification

    parsed_classification = _parse_classification(properties.get("classification"))
    owner_id = str(properties.get("owner_id") or "").strip()
    shared_scope = str(properties.get("shared_scope") or "").strip().lower()
    org_shared = shared_scope in org_shared_scopes
    if parsed_classification is DataClassification.PUBLIC:
        get_company_brain().permissions.classify_node(
            node_id, DataClassification.PUBLIC
        )
        return
    if not owner_id and not org_shared:
        return

    acl = get_company_brain().permissions.classify_node(
        node_id,
        parsed_classification or DataClassification.CONFIDENTIAL,
        data_owner=owner_id,
    )
    if org_shared and actor.actor_id not in acl.read_actors:
        acl.read_actors.append(actor.actor_id)
        get_company_brain().permissions.set_acl(acl)


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

    from .tenant_sharing import SCOPE_COMMONS, SCOPE_ORG

    org_shared_scopes = {SCOPE_ORG, SCOPE_COMMONS}

    rows = _durable_access_rows(node_ids)
    for node_id in node_ids:
        properties = rows.get(node_id)
        if (
            properties is None
            or str(properties.get("tenant_id") or "") != actor.tenant_id
        ):
            continue
        if isinstance(properties.get("external_access"), dict):
            _hydrate_connector_acl(node_id, properties)
            continue
        _hydrate_first_party_acl(node_id, properties, actor, org_shared_scopes)


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


def row_node_ids(
    rows: list[dict[str, Any]], *, trust_pushdown: bool = False
) -> list[str]:
    """Return the governed node id carried by every result row.

    Public graph projections must retain an ``id`` (or a node mapping that
    contains one) so authorization and audit refer to the same objects. A
    projection that removes identity is normally not governable and is
    denied — UNLESS ``trust_pushdown`` is set, meaning the CALLER has already
    pushed tenant scope + owner/scope visibility into the query text for
    this specific read (:func:`~.tenant_sharing.push_down_visibility`; see
    ``QueryMixin.query_cypher``/``KnowledgeGraph.query``). In that case an
    identity-less row (e.g. ``RETURN n.name AS name``) is a legitimate
    projection shape, not evidence of a bypass — the same trade-off already
    made for aggregate/scalar rows, which never carry a per-row id either —
    and is simply omitted from the returned id list (nothing governable to
    name for the audit trail) rather than rejecting the whole read.

    ``trust_pushdown`` defaults to ``False``: every EXISTING caller that
    does not pass it keeps the exact prior fail-closed behavior.
    """
    ids = [_row_node_id(row) for row in rows]
    if not trust_pushdown and any(node_id is None for node_id in ids):
        raise PermissionError("Graph result contains a row without a governed node id")
    return [node_id for node_id in ids if node_id is not None]


def filter_rows(
    rows: list[dict[str, Any]],
    actor: ActorContext | None = None,
    *,
    trust_pushdown: bool = False,
) -> list[dict[str, Any]]:
    """Drop result rows whose identifiable node id is ACL-denied for ``actor``.

    A row that DOES carry a governable node id is always classified against
    the fine-grained node ACL exactly as before, kept only if ``permit()``
    allows it. A row with NO identifiable node id (e.g. a plain
    ``RETURN n.name AS name`` projection, which carries no ``id`` column at
    all) cannot be evaluated against a per-node ACL — there is nothing to
    look up. Rejecting such a row is a POST-HOC authorization decision on
    data tenant ``scope()`` already bounded; whether that is safe depends on
    ``trust_pushdown``:

    * ``trust_pushdown=False`` (the default — every existing caller that
      does not pass it) — unclassifiable rows are rejected and the whole
      read raises, the historical behavior, preserved byte-for-byte.
    * ``trust_pushdown=True`` — the caller has ALREADY pushed owner/scope
      visibility into the query text for this specific read
      (:func:`~.tenant_sharing.push_down_visibility`), so an unclassifiable
      row is trusted and kept unfiltered rather than raised or silently
      dropped — mirroring :func:`~.tenant_sharing.filter_visible`'s own
      documented stance ("rows whose properties can't be located are kept —
      we never silently drop data we can't classify") and
      :func:`~.tenant_sharing.filter_commons_catalog`'s ``trust_pushdown``
      escape, the reference shape for this fix.
    """
    actor = _verified_actor(actor)
    if not rows:
        return []
    pairs = [(row, _row_node_id(row)) for row in rows]
    if not trust_pushdown and any(node_id is None for _, node_id in pairs):
        raise PermissionError("Graph result contains a row without a governed node id")
    governed_ids = [node_id for _, node_id in pairs if node_id is not None]
    allowed = set(permit(governed_ids, actor)) if governed_ids else set()
    return [row for row, node_id in pairs if node_id is None or node_id in allowed]
