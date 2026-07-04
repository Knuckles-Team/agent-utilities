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

Visibility for a non-privileged actor under ``KG_BRAIN_ENFORCE`` (composed with
the existing tenant ``scope()`` predicate):

* ``own``      — ``n._owner_id == actor.actor_id``
* ``org``      — ``n._shared_scope IN ('org','commons')``  (inside the org graph)
* ``unowned``  — ``n._owner_id IS NULL``  (legacy/system data, never hidden)
* ``commons``  — anything in the commons graph (separate graph, read-union)

Privileged actors (``admin``/``system`` roles, matching the SYSTEM_ACTOR
defaults) are unrestricted. Everything here is a **no-op unless enforcement is
on**, so default behaviour is byte-identical to today.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from ...security.brain_context import ActorContext, current_actor
from .shard_topology import default_graph_name, tenant_graph_name

logger = logging.getLogger(__name__)

__all__ = [
    "OWNER_KEY",
    "SCOPE_COMMONS",
    "SCOPE_KEY",
    "SCOPE_ORG",
    "SCOPE_PRIVATE",
    "TENANT_KEY",
    "accessible_graphs",
    "apply_visibility",
    "commons_graph_name",
    "filter_visible",
    "is_privileged",
    "make_private",
    "org_graph_name",
    "promote_to_commons",
    "read_union",
    "share",
    "share_with_org",
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
# and the SYSTEM_ACTOR defaults so the two layers agree on "privileged").
_PRIVILEGED_ROLES: frozenset[str] = frozenset({"admin", "system"})

# Cypher-literal safety: ids come from JWT claims and must never break out of the
# quoted predicate. Same character class the KG-2.6 tenant scoper accepts.
_SAFE_ID_RE = re.compile(r"[A-Za-z0-9_:.\-@]+")


def is_privileged(actor: ActorContext | None = None) -> bool:
    """True when ``actor`` holds an admin/system role (unrestricted visibility)."""
    actor = actor or current_actor()
    return bool(_PRIVILEGED_ROLES.intersection(actor.roles))


# ---------------------------------------------------------------------------
# Graph-name resolution (the "where it is placed" axis)
# ---------------------------------------------------------------------------


def commons_graph_name(config: Any = None) -> str:
    """The shared, cross-org commons graph (= the configured default graph)."""
    return default_graph_name(config)


def org_graph_name(actor: ActorContext | None = None, config: Any = None) -> str:
    """The named graph an actor's org routes to (KG-2.58), or commons if tenantless."""
    actor = actor or current_actor()
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
    actor = actor or current_actor()
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
    actor = actor or current_actor()
    if actor.tenant_id:
        properties.setdefault(TENANT_KEY, actor.tenant_id)
    if is_privileged(actor):
        return
    if not actor.actor_id or actor.actor_id == "system":
        return
    properties.setdefault(OWNER_KEY, actor.actor_id)
    properties.setdefault(SCOPE_KEY, SCOPE_PRIVATE)


# ---------------------------------------------------------------------------
# Visibility predicate (read path) — composes with the tenant scope()
# ---------------------------------------------------------------------------


def visibility_predicate(
    actor: ActorContext | None = None, var: str = "n"
) -> str | None:
    """A Cypher boolean fragment gating owner/scope, or ``None`` for full access.

    Returns ``None`` for privileged actors (no restriction). Otherwise returns
    ``(n._owner_id = '<me>' OR n._shared_scope IN ['org','commons'] OR
    n._owner_id IS NULL)`` — own, org-shared/commons, or unowned data.
    """
    actor = actor or current_actor()
    if is_privileged(actor):
        return None
    owner = actor.actor_id or ""
    if not _SAFE_ID_RE.fullmatch(owner):
        # An unsafe id can never equal a stored owner; fall back to the
        # share-scope branches only (fail closed on the owner test).
        owner = "__no_such_owner__"
    return (
        f"({var}.{OWNER_KEY} = '{owner}' "
        f"OR {var}.{SCOPE_KEY} IN ['{SCOPE_ORG}', '{SCOPE_COMMONS}'] "
        f"OR {var}.{OWNER_KEY} IS NULL)"
    )


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
    me = (actor or current_actor()).actor_id
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
    cypher: str, actor: ActorContext | None = None, var: str = "n"
) -> str:
    """AND the owner/scope visibility predicate into a Cypher read query.

    Mirrors the injection discipline of
    :meth:`TenancyManager.scope_cypher_query`: insert after the first
    ``WHERE`` (case-insensitive) or, lacking one, before the first ``RETURN``.
    Queries with no ``RETURN`` (writes/DDL) are returned unchanged.
    """
    cond = visibility_predicate(actor, var=var)
    if cond is None:
        return cypher
    m = re.search(r"\bWHERE\b", cypher, flags=re.IGNORECASE)
    if m:
        return cypher[: m.end()] + f" {cond} AND" + cypher[m.end() :]
    m = re.search(r"\bRETURN\b", cypher, flags=re.IGNORECASE)
    if m:
        return cypher[: m.start()] + f"WHERE {cond} " + cypher[m.start() :]
    return cypher


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


def _set_scope(node_id: str, scope: str, store: Any, owner: str | None = None) -> None:
    sets = [f"n.{SCOPE_KEY} = $scope"]
    params: dict[str, Any] = {"id": node_id, "scope": scope}
    if owner is not None:
        sets.append(f"n.{OWNER_KEY} = $owner")
        params["owner"] = owner
    store.execute(f"MATCH (n {{id: $id}}) SET {', '.join(sets)}", params)


def share_with_org(
    node_id: str, store: Any = None, actor: ActorContext | None = None
) -> None:
    """Make ``node_id`` visible to everyone in the owner's org (in-place flip)."""
    _set_scope(node_id, SCOPE_ORG, _store(store))


def make_private(
    node_id: str, store: Any = None, actor: ActorContext | None = None
) -> None:
    """Restrict ``node_id`` back to its owner (defaults the owner to the caller)."""
    actor = actor or current_actor()
    _set_scope(node_id, SCOPE_PRIVATE, _store(store), owner=actor.actor_id or None)


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
    if hasattr(dst, "add_node"):
        dst.add_node(node_id, **{k: v for k, v in props.items() if k != "id"})
    else:
        dst.execute(
            "MERGE (n {id: $id}) SET n += $props",
            {"id": node_id, "props": props},
        )
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
