# CONCEPT:AU-KG.compute.data-is-private-its - Hierarchical org-to-user data segmentation: private-by-default owner/scope markers over KG-2.58 tenant graphs, with in-place org sharing, cross-graph commons promotion, and marking-based sharing
"""Per-client data segmentation with explicit, private-by-default sharing.

Builds directly on **KG-2.58** (tenant → named-graph routing) and the markings
system in :mod:`...ontology.permissioning`. The locked model:

* **Org = the physical isolation boundary.** Each org routes to its own named
  graph ``tenant__<slug>__<base>`` (KG-2.58), so cross-org isolation is physical.
* **User-privacy within an org = logical.** Every guarded write stamps an
  ``_owner_id`` (the writing actor) and an ``_shared_scope`` marker. A user sees
  their own nodes plus anything shared to the org — so "share with my org" is a
  cheap in-place flip, never a data move.
* **The default graph is the COMMONS.** It is readable across orgs; *promoting*
  a node there is the one operation that copies across graphs ("share by **where**
  it is placed"). Markings ("share by **how** it is placed") are the orthogonal,
  mandatory control.

Visibility for a non-privileged verified actor (composed with
the existing tenant ``scope()`` predicate):

* ``own``      — ``n._owner_id == actor.actor_id``
* ``org``      — ``n._shared_scope IN ('org','commons')``  (inside the org graph)
* ``unowned``  — ``n._owner_id IS NULL``  (legacy/system data, never hidden)
* ``commons``  — anything in the commons graph (separate graph, read-union)

Verified actors carrying the explicit ``kg:admin`` capability are unrestricted
by owner/scope visibility; tenant and ACL boundaries still apply. A generic
application role named ``admin`` is not graph authority.

**BUG-052 / GOC-61 — this module's naming is the canonical one.** The
``epistemic-graph`` engine's own native RLS (``crates/eg-core/src/isolation.rs``)
independently reserves ``_owner``/``_visibility``/``_grants`` for the same
concept — a different naming convention that BUG-052 identified as disagreeing
with this module's ``_owner_id``/``_shared_scope``. That decision record
(``plans/graph-os-completion-program/decisions/GOC-61-ownership-property-convention.md``)
names **this** module's convention canonical (it is what actually decides a
live access for all traffic that reaches the engine through this codebase — the
engine's own RLS is bypassed for that traffic because this process's engine
connection is provisioned as ``AgentRole::System``) and adds READ-side
compatibility on the engine so a node tagged either way resolves consistently.
No change was needed here: this module already writes only the canonical keys.
"""

from __future__ import annotations

import contextvars
import logging
import re
from typing import Any

from ...models.company_brain import DataClassification
from ...security.brain_context import ActorContext, current_actor
from .shard_topology import default_graph_name, tenant_graph_name

logger = logging.getLogger(__name__)

__all__ = [
    "COMMONS_PRIVATE_NODE_TYPES",
    "COMMONS_SHAREABLE_NODE_TYPES",
    "OWNER_KEY",
    "PUBLIC_CATALOG_LABELS",
    "SCOPE_COMMONS",
    "SCOPE_KEY",
    "SCOPE_ORG",
    "SCOPE_PRIVATE",
    "TENANT_KEY",
    "accessible_graphs",
    "apply_commons_catalog_restriction",
    "apply_visibility",
    "check_system_graph_write",
    "commons_catalog_predicate",
    "commons_graph_name",
    "filter_commons_catalog",
    "filter_visible",
    "is_privileged",
    "make_private",
    "org_graph_name",
    "promote_to_commons",
    "read_union",
    "share",
    "share_with_org",
    "stamp_classification",
    "stamp_ownership",
    "visibility_predicate",
]

#: Node property carrying the owning org/tenant (drives the scope() predicate).
TENANT_KEY = "tenant_id"
#: Node property carrying the owning actor id (the user who created it).
OWNER_KEY = "_owner_id"
#: Node property carrying the share scope: one of the SCOPE_* values below.
SCOPE_KEY = "_shared_scope"
SCOPE_PRIVATE = "private"
SCOPE_ORG = "org"
SCOPE_COMMONS = "commons"

# Roles that bypass owner/scope visibility (mirror permissioning._PRIVILEGED_ROLES
# and request_identity._GRAPH_AUTH_SCOPES so every graph layer agrees on the one
# explicit administrative capability). A generic application ``admin`` role is
# deliberately not equivalent to graph administration.
_PRIVILEGED_ROLES: frozenset[str] = frozenset({"kg:admin"})

# Cypher-literal safety: ids come from JWT claims and must never break out of the
# quoted predicate. Same character class the KG-2.6 tenant scoper accepts.
_SAFE_ID_RE = re.compile(r"[A-Za-z0-9_:.\-@]+")


def _require_actor(actor: ActorContext | None) -> ActorContext:
    resolved = actor or current_actor()
    resolved.ensure_credential_current()
    if (
        not resolved.authenticated
        or not str(resolved.actor_id or "").strip()
        or not str(resolved.tenant_id or "").strip()
    ):
        raise PermissionError("Tenant sharing requires verified tenant authority")
    return resolved


def is_privileged(actor: ActorContext | None = None) -> bool:
    """True when a verified ``actor`` holds explicit graph administration."""
    actor = _require_actor(actor)
    return actor.authenticated and bool(_PRIVILEGED_ROLES.intersection(actor.roles))


# ---------------------------------------------------------------------------
# Graph-name resolution (the "where it is placed" axis)
# ---------------------------------------------------------------------------


def commons_graph_name(config: Any = None) -> str:
    """The shared, cross-org commons graph (= the configured default graph)."""
    return default_graph_name(config)


def org_graph_name(actor: ActorContext | None = None, config: Any = None) -> str:
    """The named graph an actor's org routes to (KG-2.58), or commons if tenantless."""
    actor = _require_actor(actor)
    base = default_graph_name(config)
    if not actor.tenant_id:
        return base
    return tenant_graph_name(actor.tenant_id, base=base)


def accessible_graphs(
    actor: ActorContext | None = None, config: Any = None
) -> list[str]:
    """Ordered, de-duplicated graphs an actor may read: org (+ancestors) then commons.

    The org graph comes first (most-specific, where the actor's writes land);
    the commons graph is always appended last. Org *ancestor* graphs (org→user
    hierarchies registered in the CompanyBrain ``TenancyManager``) are included
    between the two so a user inherits read access up the tenant tree.
    """
    actor = _require_actor(actor)
    base = default_graph_name(config)
    graphs: list[str] = []

    def _add(g: str) -> None:
        if g and g not in graphs:
            graphs.append(g)

    if actor.tenant_id:
        _add(tenant_graph_name(actor.tenant_id, base=base))
        for ancestor in _ancestor_tenants(actor.tenant_id):
            _add(tenant_graph_name(ancestor, base=base))
    _add(base)  # commons, always last
    return graphs


def _ancestor_tenants(tenant_id: str) -> list[str]:
    """Best-effort org→user ancestor chain from the CompanyBrain tenancy tree."""
    try:
        from .company_brain_runtime import get_company_brain

        brain = get_company_brain()
        # TenancyManager exposes the ancestor walk via the membership API; reuse
        # its private helper when present, else nothing (flat tenancy).
        walk = getattr(brain.tenancy, "_get_ancestor_tenants", None)
        if callable(walk):
            return list(walk(tenant_id))
    except Exception as exc:  # noqa: BLE001 — flat tenancy is a fine default
        logger.debug("ancestor lookup skipped for %s: %s", tenant_id, exc)
    return []


# ---------------------------------------------------------------------------
# Ownership stamping (write path) — private by default
# ---------------------------------------------------------------------------


def stamp_ownership(
    properties: dict[str, Any], actor: ActorContext | None = None
) -> None:
    """Stamp ``tenant_id``/``_owner_id``/``_shared_scope`` onto node props in place.

    Two layers:

    * **``tenant_id``** is stamped whenever the actor carries one — this is what
      makes the tenant ``scope()`` predicate (``n.tenant_id = <org>``) match, so
      cross-org isolation works on a shared backend graph, not only in the
      KG-2.58 named-graph/sharded mode.
    * **Private-by-default ownership** (``_owner_id`` + ``_shared_scope``) is
      added only for a real, non-privileged actor; privileged/system writes are
      left **unowned** so platform data stays visible to everyone in the tenant.

    Existing markers are never overwritten (a re-write or an explicit share is
    not silently reset to private).
    """
    actor = _require_actor(actor)
    if actor.tenant_id:
        properties.setdefault(TENANT_KEY, actor.tenant_id)
    if is_privileged(actor):
        return
    if not actor.actor_id or actor.actor_id == "system":
        return
    properties.setdefault(OWNER_KEY, actor.actor_id)
    properties.setdefault(SCOPE_KEY, SCOPE_PRIVATE)


# ---------------------------------------------------------------------------
# ACL classification stamping (write path) — private-by-default, label-aware
# ---------------------------------------------------------------------------

#: Node labels that are deliberately broadly discoverable platform/fleet
#: capability catalog data, not private user data. Both are the self-describing
#: tool registry (CONCEPT:AU-ECO.toolkit.self-describing-registry): MCP tool
#: metadata (``engine_ingestion.ingest_mcp_server``) and callable-resource /
#: function registrations (``core.registry.kg_adapter.register_callable_resource``,
#: whose own docstring says "discoverable through the same graph query") exist
#: specifically so any authenticated actor can discover what capabilities the
#: fleet exposes. PUBLIC here mirrors ``DataClassification.PUBLIC``'s documented
#: meaning ("visible to all authenticated actors") and
#: ``company_brain.DataLevelPermissions.check_permission``'s read-allow special
#: case for it. Deliberately a SHORT, evidence-backed list, not a default —
#: everything not named here gets the conservative private-by-default fallback
#: below.
PUBLIC_CATALOG_LABELS: frozenset[str] = frozenset({"ToolMetadata", "CallableResource"})


def stamp_classification(
    properties: dict[str, Any],
    label: str | None,
) -> None:
    """Stamp a durable ``classification`` ACL property onto node props in place.

    CONCEPT:AU-KG.backend.company-brain-write-guard — the write-time half of
    defence-in-depth ACL registration. ``secured_reads.permit()``'s governed
    read path is default-deny for any node with no registered ACL
    (``company_brain.DataLevelPermissions.check_permission``: "No ACL defined
    — default deny") and, before this, nothing on the generic
    ``add_node``/``_upsert_node``/``GraphComputeEngine.add_node`` write seams
    ever registered one — data was writable but permanently unreadable, even
    by its own creator (secured_reads._hydrate_missing_acls reads this durable
    property back at query time to reconstruct the actual ACL).

    Policy (deliberate, not a blanket assignment):

    * ``label`` in :data:`PUBLIC_CATALOG_LABELS` -> ``PUBLIC``.
    * Everything else -> ``CONFIDENTIAL`` (private-by-default): the safe,
      conservative default that fixes the "unreadable by its own creator"
      defect without over-exposing to any OTHER actor — read access still
      requires the node's stamped ``_owner_id`` (:func:`stamp_ownership`) to
      match the reading actor, or an explicit grant. Mirrors this module's own
      "data is private, its owner" default for the sibling tenant/owner stamp,
      and is intentionally the MOST conservative non-PUBLIC classification
      choice available (over-restrictive is a data-availability bug fixable
      later; over-permissive is a data-exposure vulnerability).

    Individual call sites with better first-hand knowledge than a bare label
    (e.g. ``core.sessions._persist_goal``, which knows a ``Concept`` node here
    is specifically a user's goal) may still call
    ``DataLevelPermissions.classify_node`` explicitly afterwards to refine the
    classification (e.g. to ``INTERNAL``) — that does not change *whether* the
    node is readable (this stamp already guarantees the owner can read it),
    only the label used for entailment-propagation strictness ordering
    (``secured_reads.inherit_inferred_acl``) and audit posture.

    Existing ``classification`` markers are never overwritten (a re-write is
    not silently reclassified).
    """
    if properties.get("classification"):
        return
    normalized = str(label or "").strip()
    if normalized in PUBLIC_CATALOG_LABELS:
        properties["classification"] = DataClassification.PUBLIC.value
    else:
        properties["classification"] = DataClassification.CONFIDENTIAL.value


# ---------------------------------------------------------------------------
# Visibility predicate (read path) — composes with the tenant scope()
# ---------------------------------------------------------------------------


def visibility_predicate(
    actor: ActorContext | None = None, var: str = "n"
) -> tuple[str, dict[str, Any]] | None:
    """A Cypher boolean fragment gating owner/scope, or ``None`` for full access.

    Returns ``None`` for privileged actors (no restriction). Otherwise returns
    ``(predicate, extra_params)`` where ``predicate`` is
    ``(n._owner_id = $_visibility_owner_id OR n._shared_scope IN ['org','commons']
    OR n._owner_id IS NULL)`` — own, org-shared/commons, or unowned data.

    D-W2T-2: the actor's owner id is injected as a **bound Cypher parameter**
    (``$_visibility_owner_id``) rather than spliced into the predicate text as
    a string literal — the ``_SAFE_ID_RE`` allowlist below already made the
    literal splice safe, but a bound parameter is real defense-in-depth
    instead of relying solely on that validation. ``SCOPE_ORG``/
    ``SCOPE_COMMONS`` stay as literals: they are fixed module-level constants,
    never actor-influenced. The caller MUST merge ``extra_params`` into
    whatever params dict it executes the scoped query with.
    """
    actor = _require_actor(actor)
    if is_privileged(actor):
        return None
    owner = actor.actor_id or ""
    if not _SAFE_ID_RE.fullmatch(owner):
        # An unsafe id can never equal a stored owner; fall back to the
        # share-scope branches only (fail closed on the owner test).
        owner = "__no_such_owner__"
    predicate = (
        f"({var}.{OWNER_KEY} = $_visibility_owner_id "
        f"OR {var}.{SCOPE_KEY} IN ['{SCOPE_ORG}', '{SCOPE_COMMONS}'] "
        f"OR {var}.{OWNER_KEY} IS NULL)"
    )
    return predicate, {"_visibility_owner_id": owner}


def _row_props(row: dict[str, Any]) -> dict[str, Any]:
    """Best-effort extraction of a node's properties from a result row."""
    for v in row.values():
        if isinstance(v, dict) and (OWNER_KEY in v or SCOPE_KEY in v or "id" in v):
            return v
    return row


def filter_visible(
    rows: list[dict[str, Any]], actor: ActorContext | None = None
) -> list[dict[str, Any]]:
    """Drop rows not visible to ``actor`` by owner/scope (Python post-filter).

    The backend-agnostic counterpart to :func:`visibility_predicate`: a row is
    visible when the actor is privileged, or the node is owned by the actor, is
    org-/commons-shared, or is unowned. Rows whose properties can't be located
    are kept (we never silently drop data we can't classify — tenant ``scope()``
    already bounded the set).
    """
    if is_privileged(actor):
        return rows
    me = _require_actor(actor).actor_id
    out: list[dict[str, Any]] = []
    for row in rows:
        props = _row_props(row)
        owner = props.get(OWNER_KEY)
        scope_val = props.get(SCOPE_KEY)
        # Unowned (no owner key at all) → visible; owned → only self / shared.
        if (
            OWNER_KEY not in props
            or owner is None
            or owner == me
            or scope_val in (SCOPE_ORG, SCOPE_COMMONS)
        ):
            out.append(row)
    return out


def apply_visibility(
    cypher: str, actor: ActorContext | None = None, var: str | None = None
) -> tuple[str, dict[str, Any]]:
    """AND the owner/scope visibility predicate into a Cypher read query.

    Mirrors the injection discipline of
    :meth:`~agent_utilities.knowledge_graph.core.company_brain.TenancyManager.scope_cypher_query`:
    insert after the first ``WHERE`` (case-insensitive) or, lacking one, before
    the first ``RETURN``. Queries with no ``WHERE``/``RETURN`` (writes/DDL) are
    returned unchanged.

    Args:
        cypher: The Cypher read query text.
        actor: The requesting actor (privileged actors get no restriction).
        var: Optional explicit node variable to scope against. ``None`` (the
            default) derives it from the query's own first ``MATCH (<var>...``
            clause (CONCEPT:AU-KG.backend.company-brain-write-guard) instead of
            the historical hardcoded ``"n"`` — a query binding its node under a
            different name (``MATCH (x:Entity) RETURN x``) previously got a
            predicate referencing an unbound ``n``, silently mis-scoping (or,
            on a lenient backend, silently returning zero rows) instead of
            raising.

    Returns:
        ``(scoped_cypher, extra_params)``. D-W2T-2: the visibility predicate's
        owner id is now a bound parameter (``$_visibility_owner_id``), not a
        string-literal splice — the caller MUST merge ``extra_params`` into
        whatever params dict it executes ``scoped_cypher`` with. ``extra_params``
        is ``{}`` whenever no predicate was injected (privileged actor, or no
        ``WHERE``/``RETURN`` site).

    Raises:
        UnscopableQueryError: ``var`` is omitted, the query has a
            ``WHERE``/``RETURN`` clause to inject into, and no bound node
            variable can be derived from it.
    """
    # Privileged bypass short-circuits BEFORE variable derivation: a privileged
    # actor gets no predicate at all (matching visibility_predicate's own
    # contract), so an unscopable query must not be refused on their behalf
    # for a restriction that was never going to be applied.
    if is_privileged(actor):
        return cypher, {}

    where_match = re.search(r"\bWHERE\b", cypher, flags=re.IGNORECASE)
    return_match = re.search(r"\bRETURN\b", cypher, flags=re.IGNORECASE)
    if not where_match and not return_match:
        return cypher, {}

    resolved_var = var
    if resolved_var is None:
        from .cypher_scoping import first_bound_node_variable

        resolved_var = first_bound_node_variable(cypher)

    predicate = visibility_predicate(actor, var=resolved_var)
    if predicate is None:
        return cypher, {}
    cond, extra_params = predicate

    from .cypher_scoping import inject_and_predicate

    # inject_and_predicate parenthesizes any PRE-EXISTING WHERE body as a unit
    # before ANDing `cond` in — a bare `cond AND <existing>` textual splice
    # would mis-group against a top-level OR already in `existing` (AND binds
    # tighter), letting a disjunct bypass this visibility predicate entirely.
    # `cond` itself is already a single parenthesized term (see
    # visibility_predicate above), so it never needs re-wrapping.
    return inject_and_predicate(cypher, cond), extra_params


# ---------------------------------------------------------------------------
# Read-union across the actor's accessible graphs (commons + org)
# ---------------------------------------------------------------------------


def read_union(
    cypher: str,
    params: dict[str, Any] | None,
    executor: Any,
    actor: ActorContext | None = None,
    config: Any = None,
    id_keys: tuple[str, ...] = ("id", "node_id", "n.id", "_id"),
) -> list[dict[str, Any]]:
    """Run ``cypher`` across every accessible graph and merge, de-duped by id.

    ``executor(graph_name, cypher, params) -> rows`` runs the query against one
    named graph. The actor's own (org) graph is queried first so its rows win on
    duplicate ids; commons rows fill in the rest. A per-graph failure is logged
    and skipped — a missing commons graph degrades to org-only, never an error.
    """
    seen: set[str] = set()
    merged: list[dict[str, Any]] = []
    for graph in accessible_graphs(actor, config):
        try:
            rows = executor(graph, cypher, params or {}) or []
        except Exception as exc:  # noqa: BLE001 — one graph down ≠ whole read down
            logger.debug("read_union: graph %s unavailable: %s", graph, exc)
            continue
        for row in rows:
            nid = _row_id(row, id_keys)
            if nid is None:
                merged.append(row)
                continue
            if nid in seen:
                continue
            seen.add(nid)
            merged.append(row)
    return merged


def _row_id(row: dict[str, Any], id_keys: tuple[str, ...]) -> str | None:
    for key in id_keys:
        val = row.get(key)
        if isinstance(val, str):
            return val
    for val in row.values():
        if isinstance(val, dict):
            inner = val.get("id") or val.get("node_id")
            if isinstance(inner, str):
                return inner
    return None


# ---------------------------------------------------------------------------
# Explicit sharing transitions (the user-facing verbs)
# ---------------------------------------------------------------------------


def _store(store: Any = None) -> Any:
    if store is not None:
        return store
    from ..facade import KnowledgeGraph

    return KnowledgeGraph().store


def _node_exists(node_id: str, store: Any) -> bool:
    """Whether ``node_id`` resolves to a real node in ``store`` (BUG-6 guard)."""
    rows = (
        store.execute("MATCH (n {id: $id}) RETURN n.id AS id LIMIT 1", {"id": node_id})
        or []
    )
    return bool(rows)


def _node_properties(node_id: str, store: Any) -> dict[str, Any] | None:
    """Load the governed ownership fields required before a share transition."""
    reader = getattr(store, "get_node_properties", None)
    try:
        if callable(reader):
            props = reader(node_id)
            if props is None:
                return None
            if not isinstance(props, dict):
                raise PermissionError("Node sharing authority is unavailable")
            return dict(props)
        rows = (
            store.execute(
                "MATCH (n {id: $id}) RETURN properties(n) AS props LIMIT 1",
                {"id": node_id},
            )
            or []
        )
    except PermissionError:
        raise
    except Exception as exc:
        raise PermissionError("Node sharing authority is unavailable") from exc
    if not rows:
        return None
    props = rows[0].get("props")
    if not isinstance(props, dict):
        raise PermissionError("Node sharing authority is unavailable")
    return dict(props)


def _require_share_authority(
    node_id: str,
    store: Any,
    actor: ActorContext | None,
) -> tuple[ActorContext, bool]:
    """Require current same-tenant owner/admin authority for a share mutation."""
    resolved = _require_actor(actor)
    props = _node_properties(node_id, store)
    if props is None:
        return resolved, False
    node_tenant = str(props.get(TENANT_KEY) or "").strip()
    owner = str(props.get(OWNER_KEY) or "").strip()
    if node_tenant != resolved.tenant_id:
        raise PermissionError("Node sharing requires same-tenant authority")
    if not is_privileged(resolved) and owner != resolved.actor_id:
        raise PermissionError("Node sharing requires owner or administrator authority")
    return resolved, True


def _set_scope(node_id: str, scope: str, store: Any, owner: str | None = None) -> bool:
    """Flip the share-scope property of ``node_id``, iff it exists.

    Returns ``True`` when the node was found and updated, ``False`` (no-op) when
    ``node_id`` does not resolve to a real node — the Cypher ``MATCH`` used to
    silently match zero rows with no signal back to the caller (BUG-6), letting
    ``graph_share`` report success for ids that were never written.
    """
    if not _node_exists(node_id, store):
        return False
    sets = [f"n.{SCOPE_KEY} = $scope"]
    params: dict[str, Any] = {"id": node_id, "scope": scope}
    if owner is not None:
        sets.append(f"n.{OWNER_KEY} = $owner")
        params["owner"] = owner
    store.execute(f"MATCH (n {{id: $id}}) SET {', '.join(sets)}", params)
    return True


def share_with_org(
    node_id: str, store: Any = None, actor: ActorContext | None = None
) -> bool:
    """Make ``node_id`` visible to everyone in the owner's org (in-place flip).

    Returns ``False`` (no-op) if ``node_id`` does not exist (BUG-6).
    """
    resolved_store = _store(store)
    _actor, exists = _require_share_authority(node_id, resolved_store, actor)
    if not exists:
        return False
    return _set_scope(node_id, SCOPE_ORG, resolved_store)


def make_private(
    node_id: str, store: Any = None, actor: ActorContext | None = None
) -> bool:
    """Restrict ``node_id`` back to its owner (defaults the owner to the caller).

    Returns ``False`` (no-op) if ``node_id`` does not exist (BUG-6).
    """
    resolved_store = _store(store)
    actor, exists = _require_share_authority(node_id, resolved_store, actor)
    if not exists:
        return False
    return _set_scope(
        node_id, SCOPE_PRIVATE, resolved_store, owner=actor.actor_id or None
    )


def share(node_id: str, marking: Any) -> None:
    """Share by **how** it is placed: attach a mandatory marking (mark-based).

    A thin pass-through to the markings system so callers have one sharing
    surface. Actors holding the ``marking:<name>`` role can then read the node
    regardless of org, complementing the place-based commons promotion.
    """
    from ..ontology.permissioning import apply_marking

    apply_marking(node_id, marking)


def promote_to_commons(
    node_id: str,
    store: Any = None,
    commons_store: Any = None,
    actor: ActorContext | None = None,
    config: Any = None,
) -> bool:
    """Share by **where** it is placed: copy ``node_id`` into the commons graph.

    Reads the node's properties from the actor's org graph (``store``) and
    writes them into the commons graph (``commons_store``), marking the copy
    ``_shared_scope='commons'`` so the cross-org read-union surfaces it. Returns
    ``False`` (and logs) if the node can't be found — promotion is best-effort
    and must never raise into a tool call. The org-local original is left in
    place (promotion shares; it does not move).
    """
    src = _store(store)
    _actor, exists = _require_share_authority(node_id, src, actor)
    if not exists:
        logger.warning("promote_to_commons: governed node was not found")
        return False
    rows = (
        src.execute(
            "MATCH (n {id: $id}) RETURN properties(n) AS props, labels(n) AS labels",
            {"id": node_id},
        )
        or []
    )
    if not rows:
        logger.warning("promote_to_commons: node %s not found in org graph", node_id)
        return False
    props = dict(rows[0].get("props") or {})
    props[SCOPE_KEY] = SCOPE_COMMONS
    props.setdefault("id", node_id)

    dst = commons_store if commons_store is not None else _commons_store(config)
    if dst is None:
        logger.warning("promote_to_commons: commons graph unavailable for %s", node_id)
        return False
    # GOC-61 phase-1 write gate (check_system_graph_write, below): the CONTENT
    # condition applies unconditionally, even here — promotion is authorized
    # (owner/admin already checked above), but authorization to SHARE a node
    # is not authorization to publish ANY node type into a system graph. The
    # AUTHORITY condition is the one this context legitimately bypasses (the
    # owner/admin check just above already satisfies it), signalled via the
    # ambient _SHARE_VERB_ACTIVE context.
    token = _SHARE_VERB_ACTIVE.set(True)
    try:
        if hasattr(dst, "add_node"):
            dst.add_node(node_id, **{k: v for k, v in props.items() if k != "id"})
        else:
            # ``SET n += $props`` is a map-merge assignment, outside the native
            # write subset (SET values must be literals;
            # epistemic-graph/crates/eg-query/src/cypher/parser.rs:1184).
            # ``materialization.set_clause`` is the one SET-clause builder used
            # everywhere else in this codebase for the portable per-property
            # equivalent (``IntelligenceGraphEngine._get_set_clause`` delegates
            # to the same function); params are the flattened props themselves
            # so each generated ``$key`` placeholder resolves.
            from .materialization import set_clause

            merge_props = {**props, "id": node_id}
            query = f"MERGE (n {{id: $id}}) {set_clause(merge_props, dst)}".rstrip()
            dst.execute(query, merge_props)
    finally:
        _SHARE_VERB_ACTIVE.reset(token)
    # Reflect the promotion on the org-local copy too, so it reads as shared.
    try:
        _set_scope(node_id, SCOPE_COMMONS, src)
    except Exception as exc:  # noqa: BLE001 — the commons copy is the source of truth
        logger.debug("promote_to_commons: org-local scope update skipped: %s", exc)
    return True


def _commons_store(config: Any = None) -> Any:
    """Resolve a store bound to the commons (default) graph, or None.

    Goes through the elastic engine pool (KG-2.62) so the commons engine is kept
    warm and shared rather than rebuilt per promotion; with the pool disabled
    (default) this is a plain per-use construction.
    """
    try:
        from .tenant_engine_pool import acquire_engine

        return acquire_engine(graph_name=commons_graph_name(config))
    except Exception as exc:  # noqa: BLE001 — degrade cleanly
        logger.debug("commons store unavailable: %s", exc)
        return None


# ---------------------------------------------------------------------------
# System-graph write authorization (GOC-61 phase 1, W04 — minimal stopgap)
# ---------------------------------------------------------------------------

# Ambient signal that a write is happening INSIDE an already-authorized
# sharing verb (currently only :func:`promote_to_commons`, which already ran
# ``_require_share_authority`` — owner-or-admin — on the SOURCE node before
# writing the copy). Bypasses ONLY the AUTHORITY condition of
# :func:`check_system_graph_write`; the CONTENT condition is never bypassed by
# this — see that function's docstring. A plain ``contextvars.ContextVar`` is
# correct here (not a mutable module global): it is task/request-local, so a
# concurrent unrelated write on another task is never affected by one
# in-flight ``promote_to_commons`` call.
_SHARE_VERB_ACTIVE: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "_SHARE_VERB_ACTIVE", default=False
)

#: WRITE-SIDE denylist (2026-08-09 owner ruling, superseding the original
#: allowlist-on-write design below the split point in this module's history —
#: see :data:`COMMONS_SHAREABLE_NODE_TYPES` for the READ-side allowlist that
#: replaced it). Confirmed private-class types only: refusing a WRITE of any
#: of these into commons is what closes the leak the 2026-08-09 pre-flight
#: check found (11 InboundMessage, 11 Thread, 4 Memory, 3 Memento observed
#: live). Deliberately narrow: an evidence-backed audit found ``WorkItem``/
#: ``RuntimeSignal`` writer functions are actor-agnostic and are reached from
#: live MCP tool entrypoints under real tenant-bound sessions
#: (``mcp/tasks_extension.py:377``, ``mcp/tools/job_tools.py:147``,
#: ``orchestration/agent_runner.py:467``, ``messaging/router.py:334``) — an
#: allowlist-on-write would have refused those already-working, user-
#: authenticated writes. Operational/provenance/ontology data is NOT refused
#: on write; it is restricted on READ instead (see below). ``Memory`` is
#: deliberately NOT listed here even though it is one of the four private-
#: class node types observed live — GOC-61 phase 2 will need to resolve why
#: (an existing ``store_memory()`` caller may legitimately need to write
#: episodic memory into a system-scoped graph); flagged, not silently solved,
#: pending the owner's phase-2 ruling on the read-side ``Memory`` handling.
COMMONS_PRIVATE_NODE_TYPES: frozenset[str] = frozenset(
    {
        "Message",
        "Thread",
        "Memento",
        "ChatSummary",
        "InboundMessage",
        "EvictedBlock",
    }
)

#: READ-side allowlist (2026-08-09 owner ruling): the ONLY node types a
#: DIFFERENT tenant may see when reading the commons graph — the capability
#: catalog. DENY-BY-DEFAULT still lives here: a new/unclassified node type is
#: invisible cross-tenant until explicitly added to this set; it no longer
#: blocks the type from being WRITTEN (see :data:`COMMONS_PRIVATE_NODE_TYPES`
#: above for the — much narrower — write-side refusal). Operational data
#: (``WorkItem``, ``RuntimeSignal``, ``Concept``, ``Evidence``, ...) is
#: deliberately NOT on this list: it stays visible only to a privileged
#: actor or to the row's own tenant, never to another tenant, via
#: :func:`filter_commons_catalog`/:func:`apply_commons_catalog_restriction`
#: below. ``CallableResource`` additionally requires ``resource_type ==
#: "AGENT_SKILL"`` (see those two functions) — a bare ``CallableResource`` is
#: not, by itself, sufficient. Source-verified against
#: ``agent_utilities/models/schema_definition.py`` entity names (not
#: guessed): Skill (:314), Tool/NativeTool/MCPServer/ToolMetadata/
#: CallableResource/Server (tool+server catalog), Prompt (:295),
#: LanguageModel/EmbeddingModel (:87/:109), WorkflowDefinition (skill-graph
#: runnables, GOC-60 E7's KG resource_type family). ``SystemPrompt`` is kept
#: even though it is not yet confirmed against schema_definition.py, per the
#: owner's explicit list — flagged, not silently dropped.
COMMONS_SHAREABLE_NODE_TYPES: frozenset[str] = frozenset(
    {
        # agent skills
        "Skill",
        "CallableResource",  # AGENT_SKILL only — see resource_type qualifier above
        "WorkflowDefinition",
        # MCP servers and MCP tools
        "MCPServer",
        "Server",
        "Tool",
        "NativeTool",
        "ToolMetadata",
        # agent prompts
        "Prompt",
        "SystemPrompt",
        # LLM models
        "LanguageModel",
        "EmbeddingModel",
    }
)

#: The ``resource_type`` value that makes a ``CallableResource`` node
#: catalog-shareable (GOC-60 E7's runnable/AGENT_SKILL distinction). A
#: ``CallableResource`` with any other (or missing) ``resource_type`` is NOT
#: shareable merely by having that label.
_CALLABLE_RESOURCE_SHAREABLE_TYPE = "AGENT_SKILL"


def check_system_graph_write(
    graph_name: str | None,
    node_type: str | None,
    actor: ActorContext | None = None,
) -> None:
    """Enforce the GOC-61 phase-1 system-graph WRITE gate (W04 + 2026-08-09 owner ruling).

    A no-op unless ``graph_name`` is a system graph
    (:func:`~agent_utilities.knowledge_graph.core.shard_topology.is_system_graph`
    — ``__commons__``, ``__control__``, ``__secrets__``, ...). When it is, TWO
    independent conditions are enforced, neither implies the other, and they
    are scoped DIFFERENTLY on purpose (see the note below):

    1. **CONTENT — deny-list, unconditional, COMMONS ONLY.** ``node_type``
       must NOT be in :data:`COMMONS_PRIVATE_NODE_TYPES` (the six confirmed
       private-class types). Checked first; **never** bypassed for the
       commons graph specifically — not for a ``kg:admin`` caller, not for a
       write already authorized through the ``graph_share`` verb
       (:func:`promote_to_commons`). A privileged caller writing a private-
       class node into commons is refused exactly like an unprivileged one —
       this is the case a coarser, allowlist-shaped gate got wrong (see the
       module-level history note on :data:`COMMONS_PRIVATE_NODE_TYPES`: an
       allowlist here would have refused legitimate ``WorkItem``/
       ``RuntimeSignal`` writes from real tenant-bound MCP sessions).
       Everything NOT on this denylist — including operational/provenance/
       ontology types — is WRITABLE to commons; what a cross-tenant reader
       later SEES there is restricted separately, on the read path (see
       :func:`filter_commons_catalog`/:func:`apply_commons_catalog_restriction`).
    2. **AUTHORITY — ``__control__``/``__secrets__`` only, NOT commons.**
       The caller holds ``kg:admin`` (reused per the GOC-61 design;
       superseded by the narrower ``kg:share``/``kg:share-admin`` scopes in
       phase 2), OR the write is happening inside an already-authorized
       sharing verb (:data:`_SHARE_VERB_ACTIVE` — currently only
       :func:`promote_to_commons`, which already checked owner-or-admin on
       the source node). A write with no bound/authenticated actor at all (a
       genuine system/background path) is exempted from this condition ONLY
       — the same best-effort exemption
       :func:`stamp_ownership`/:func:`stamp_classification` already grant at
       this exact chokepoint (``GraphComputeEngine.add_node``) — never from
       condition 1.

       **``__commons__`` is deliberately EXEMPT from condition 2** (2026-08-09
       owner ruling, second correction): the original design required
       ``kg:admin`` for every system graph, including commons, on the same
       "commons writes need admin" premise that motivated the original
       CONTENT allowlist — and the same evidence that disproved that premise
       for CONTENT (live, tenant-bound ``WorkItem``/``RuntimeSignal`` MCP
       traffic) disproves it for AUTHORITY too. Requiring ``kg:admin`` here
       would refuse the exact ordinary traffic condition 1 was just widened
       to keep working.

       ⚠️ **THIS PERMISSIVENESS ON WRITE IS SAFE ONLY BECAUSE THE READ SIDE
       IS RESTRICTIVE.** A ``WorkItem`` any authenticated tenant-bound actor
       writes into commons is not a cross-tenant leak, because
       :func:`filter_commons_catalog`/:func:`apply_commons_catalog_restriction`
       (below) ensure a DIFFERENT tenant reading commons sees only the
       catalog allowlist — never that ``WorkItem``. **The two halves of this
       split depend on each other: if the read-side catalog restriction is
       ever relaxed, weakened, or bypassed, this write-side permissiveness
       becomes an active cross-tenant data leak, not merely a stale
       comment.** Do not loosen one half without re-auditing the other.

    **Why condition 1 is commons-only, not every system graph:** the
    program owner's 2026-08-09 ruling is specifically about what may enter a
    graph "readable by every tenant" — that is ``__commons__``'s defining
    property, not ``__control__``'s or ``__secrets__``'s (control-plane and
    encrypted-secret internal stores, privileged-access-only, never fanned
    to every tenant, and never covered by the read-side catalog restriction
    either — condition 2 is what protects them instead). Applying the
    content check to those two would reject already-legitimate
    control-plane writes never part of this ruling's concern (e.g.
    ``ingest_profile.py``'s ``ProfileSpan`` nodes into ``__control__``, or
    ``secrets_client.py``'s ``Secret`` nodes into ``__secrets__``).

    Raises :class:`PermissionError` on denial; this must NOT be swallowed by
    a caller's best-effort ``except PermissionError: pass`` — unlike the
    ownership/classification stamps, a denial here must block the write.
    """
    from .shard_topology import default_graph_name, is_system_graph

    if not is_system_graph(graph_name):
        return

    # Mirrors graph_ownership.is_structurally_public's own commons identity
    # check (configured default OR the literal "__commons__" name). Computed
    # once, consumed by BOTH conditions below — condition 1 applies ONLY to
    # commons; condition 2 applies to every system graph EXCEPT commons.
    is_commons_graph = graph_name == default_graph_name() or graph_name == "__commons__"

    # Condition 1 (content) — checked first, never bypassed, COMMONS ONLY.
    if is_commons_graph:
        type_name = str(node_type or "").strip()
        if type_name in COMMONS_PRIVATE_NODE_TYPES:
            raise PermissionError(
                f"{graph_name!r} is the shared commons graph; node type "
                f"{type_name!r} is a confirmed private-class type and may "
                "not be written there"
            )
        # Commons is EXEMPT from condition 2 (authority) — see the docstring
        # warning above: this is safe ONLY because the read side restricts
        # cross-tenant visibility to the catalog allowlist. Do not add an
        # authority check here without re-auditing that dependency.
        return

    # Condition 2 (authority) — every OTHER system graph (__control__,
    # __secrets__, ...); bypassed only by an already-authorized share verb,
    # or a genuinely unauthenticated/system write path.
    if _SHARE_VERB_ACTIVE.get():
        return
    if actor is not None:
        resolved = actor
    else:
        try:
            # current_actor() raises (PermissionError subclass) rather than
            # returning a default when NO actor is bound at all — that is the
            # genuine system/background case, exempted here exactly like
            # stamp_ownership's own best-effort exemption; it must not be
            # confused with "bound but unauthenticated", handled below.
            resolved = current_actor()
        except PermissionError:
            return
    if not resolved.authenticated or not str(resolved.actor_id or "").strip():
        return
    if not _PRIVILEGED_ROLES.intersection(resolved.roles):
        raise PermissionError(
            f"Writing directly to system graph {graph_name!r} requires "
            "kg:admin authority, or must route through the graph_share verb"
        )


# ---------------------------------------------------------------------------
# System-graph (commons) READ visibility restriction (2026-08-09 owner
# ruling) — the deny-by-default half of the write/read split. Distinct from,
# and stacks with, visibility_predicate/filter_visible above: that predicate
# already hides another tenant's OWNED-private rows; the functions below
# additionally hide UNOWNED/commons-scoped operational rows (WorkItem,
# RuntimeSignal, Concept, Evidence, ...) that visibility_predicate alone
# treats as visible to everyone once no owner is stamped — which is most of
# them, since a privileged/system writer is deliberately left unowned by
# stamp_ownership.
# ---------------------------------------------------------------------------


def _is_commons_graph_name(graph_name: str | None, config: Any = None) -> bool:
    from .shard_topology import default_graph_name

    return graph_name == default_graph_name(config) or graph_name == "__commons__"


def _is_catalog_shareable(node_type: str, resource_type: Any = None) -> bool:
    """True when ``node_type`` (with, for ``CallableResource``, its
    ``resource_type``) is on the READ-side catalog allowlist."""
    if node_type == "CallableResource":
        return str(resource_type or "") == _CALLABLE_RESOURCE_SHAREABLE_TYPE
    return node_type in COMMONS_SHAREABLE_NODE_TYPES


def filter_commons_catalog(
    rows: list[dict[str, Any]],
    actor: ActorContext | None,
    graph_name: str | None,
    config: Any = None,
) -> list[dict[str, Any]]:
    """Cross-tenant commons READ restriction (2026-08-09 owner ruling), Python-side.

    A no-op unless ``graph_name`` is the commons graph, or for a privileged
    actor (full access — mirrors :func:`filter_visible`'s own privileged
    bypass). Otherwise a row is kept only when its ``node_type`` is on
    :data:`COMMONS_SHAREABLE_NODE_TYPES` (catalog-shareable to every tenant),
    OR the row's stamped ``tenant_id`` matches the reading actor's own tenant
    (the row's writer, "the system/owner", may always see its own data even
    if it is not catalog-shareable — this is what keeps a tenant's own
    WorkItem/RuntimeSignal writes visible to that SAME tenant).

    Fails CLOSED, unlike :func:`filter_visible`: a row this function cannot
    classify (no ``node_type``, no ``tenant_id``, and not the reader's own)
    is DROPPED, not kept — this is the deny-by-default half of the 2026-08-09
    ruling, so an unclassified/novel type must not leak by omission.
    """
    if not _is_commons_graph_name(graph_name, config):
        return rows
    if is_privileged(actor):
        return rows
    resolved = _require_actor(actor)
    reader_tenant = str(resolved.tenant_id or "")
    out: list[dict[str, Any]] = []
    for row in rows:
        props = _row_props(row)
        node_type = str(props.get("node_type") or "").strip()
        if node_type and _is_catalog_shareable(node_type, props.get("resource_type")):
            out.append(row)
            continue
        row_tenant = str(props.get(TENANT_KEY) or "")
        if reader_tenant and row_tenant and row_tenant == reader_tenant:
            out.append(row)
        # else: not catalog-shareable and not the reader's own tenant's data
        # -> dropped (fail closed; this is the point of the restriction).
    return out


def commons_catalog_predicate(
    actor: ActorContext | None,
    var: str = "n",
    graph_name: str | None = None,
    config: Any = None,
) -> tuple[str, dict[str, Any]] | None:
    """A Cypher boolean fragment for the commons READ catalog restriction, or
    ``None`` when it does not apply (not the commons graph, or a privileged
    actor).

    Companion to :func:`visibility_predicate`, same shape and same bound-
    parameter discipline (D-W2T-2): the caller MUST merge the returned
    ``extra_params`` into the query's params dict. The node-type literals
    spliced into the predicate text are safe exactly as
    :func:`visibility_predicate` already reasons for ``SCOPE_ORG``/
    ``SCOPE_COMMONS`` — :data:`COMMONS_SHAREABLE_NODE_TYPES` is a fixed
    module-level constant, never actor-influenced.

    This is what closes the label/type-index half of the existence-leak this
    lane's own pre-flight check found: an aggregate query (``count(n)``,
    ...) has no per-row property to post-filter (:func:`filter_commons_catalog`
    cannot run on it), so the restriction must be injected into the query
    TEXT itself for that path — this function is what
    :func:`apply_commons_catalog_restriction` uses to do that.
    """
    if not _is_commons_graph_name(graph_name, config):
        return None
    actor = _require_actor(actor)
    if is_privileged(actor):
        return None
    types_literal = ", ".join(f"'{t}'" for t in sorted(COMMONS_SHAREABLE_NODE_TYPES))
    tenant = actor.tenant_id or ""
    if not _SAFE_ID_RE.fullmatch(tenant):
        tenant = "__no_such_tenant__"
    predicate = (
        f"({var}.node_type IN [{types_literal}] "
        f"OR ({var}.node_type = 'CallableResource' "
        f"AND {var}.resource_type = '{_CALLABLE_RESOURCE_SHAREABLE_TYPE}') "
        f"OR {var}.{TENANT_KEY} = $_commons_catalog_tenant_id)"
    )
    return predicate, {"_commons_catalog_tenant_id": tenant}


def apply_commons_catalog_restriction(
    cypher: str,
    actor: ActorContext | None,
    graph_name: str | None,
    var: str | None = None,
    config: Any = None,
) -> tuple[str, dict[str, Any]]:
    """AND the commons READ catalog-restriction predicate into a Cypher read query.

    Mirrors :func:`apply_visibility`'s injection discipline exactly (same
    WHERE/RETURN detection, same ``inject_and_predicate`` parenthesization
    safety, same bound-variable derivation). Intended to run ALONGSIDE
    ``apply_visibility`` on the aggregate-query path
    (``engine_query.QueryMixin.query_cypher``), which is the one path
    :func:`filter_commons_catalog` structurally cannot reach (no per-row
    properties on an aggregate result) — this is the label/type-index leak
    this lane's pre-flight check flagged, closed here rather than left
    unaddressed.

    Returns ``(cypher, {})`` unchanged whenever the restriction does not
    apply (not commons, privileged actor, or no WHERE/RETURN site to inject
    into — the same fail-open-to-unscoped case ``apply_visibility`` already
    accepts for a fully anonymous first pattern, since there is no bound
    variable to attach a predicate to).
    """
    if not _is_commons_graph_name(graph_name, config):
        return cypher, {}
    if is_privileged(actor):
        return cypher, {}

    where_match = re.search(r"\bWHERE\b", cypher, flags=re.IGNORECASE)
    return_match = re.search(r"\bRETURN\b", cypher, flags=re.IGNORECASE)
    if not where_match and not return_match:
        return cypher, {}

    resolved_var = var
    if resolved_var is None:
        from .cypher_scoping import first_bound_node_variable

        # Matches apply_visibility's own contract exactly: raises
        # UnscopableQueryError (not None) when no bound variable exists and
        # none was supplied explicitly. Callers on the aggregate path (this
        # restriction's primary reason to exist) always pass `var` already
        # resolved via `primary_bound_variable` — the same lenient,
        # None-returning derivation `apply_visibility`'s own aggregate caller
        # uses — so they never reach this branch.
        resolved_var = first_bound_node_variable(cypher)

    predicate = commons_catalog_predicate(
        actor, var=resolved_var, graph_name=graph_name, config=config
    )
    if predicate is None:
        return cypher, {}
    cond, extra_params = predicate

    from .cypher_scoping import inject_and_predicate

    return inject_and_predicate(cypher, cond), extra_params
