# CONCEPT:AU-KG.compute.data-is-private-its - Durable tenant-hierarchy registry: the parent edge that makes accessible_graphs() a real multi-level overlay chain
"""Durable tenant → parent registry (the missing half of hierarchical stitching).

:func:`~.tenant_sharing.accessible_graphs` already returns an ORDERED precedence
chain ("org (+ancestors) then commons") and :func:`~.tenant_sharing.read_union`
already resolves conflicts first-in-chain-wins. Together that is overlay/underlay
with most-specific-wins, and it generalises to N levels for free. The only thing
missing was a place to *record* that a tenant has a parent:
:class:`~.company_brain.TenancyManager` kept the tree in a plain in-memory dict
that every process reset, so the chain was always exactly
``[tenant, __commons__]`` and hierarchy could never exist in practice.

This module is that place.

Where it persists, and why
--------------------------
One ``:TenantHierarchy`` node per registered tenant in the fixed, tenant-shared
**``__control__``** graph (:data:`~.shard_topology.CONTROL_GRAPH_NAME`).

* **Not the SQL catalog.** The engine's ``Method::Sql`` store is owner-scoped per
  ``(tenant, principal)``. The hierarchy has to be readable by *every* principal
  in *every* tenant — a reader in ``eng`` must be able to learn that its parent is
  ``acme`` — so an owner-scoped store is exactly the wrong shape: the row would be
  invisible to the readers that need it.
* **Not a per-tenant graph.** A tenant graph is the isolation boundary; a record
  stored there is unreadable from a sibling and, worse, would have to be read
  *before* we know which graphs are readable — a bootstrap cycle.
* **Not config.** Config is host-local and hand-edited; a tenant tree is
  operational state that has to be identical for every replica reading the same
  engine, and has to be writable at runtime by an authorized admin.
* **``__control__`` is the existing answer.** It is the one system ``__…__`` graph
  that is "readable by every tenant by design, never content/document/codebase
  data" — already the sole ``:WorkItem``/``:Schedule`` control-plane authority,
  already provisioned in the system RBAC admission grant
  (``security/system_rbac_admission.py``), already the documented home for exactly
  this class of tenant-shared control-plane fact. Reusing it adds no new store, no
  new grant, no engine change and no image rebuild.

Cross-graph edges are structurally impossible in the engine (rejected in
``graph.rs``, ``redb_store.rs``, and by the edge key's single graph component), so
the hierarchy is necessarily a **read-time projection**, never a storage property.
That is why the parent is a *property* on a control-plane node rather than an
edge, and why :func:`ancestor_chain` is resolved in Python at read time.

Read cost
---------
``accessible_graphs()`` is on the read hot path and each accessible graph costs a
per-graph query inside ``read_union``, on top of the engine's ~1s fixed RPC
overhead. The whole registry is therefore loaded in **one** label-indexed point
read (:meth:`EpistemicGraphBackend.nodes_by_label`) into a process-wide snapshot
with a short TTL (:data:`_CACHE_TTL_SECONDS`), so the steady-state cost of an
ancestor lookup is a dict walk — zero RPCs. A load failure is cached as "flat"
for the same TTL so an unreachable engine cannot add a failing RPC to every read.

Bounds and authority
--------------------
* **Depth is bounded** by :data:`MAX_TENANT_DEPTH` (4): a tenant plus at most 3
  ancestors, so ``accessible_graphs()`` returns at most 5 graphs including
  commons. That keeps the whole fan-out inside a single
  ``read_union`` thread-pool wave (``_READ_UNION_MAX_WORKERS`` = 8), i.e. one
  RPC round-trip of wall clock rather than N. The bound is enforced on **both**
  sides: :func:`set_parent` refuses an edit that would overflow it, and
  :func:`ancestor_chain` truncates regardless of what is in the store.
* **Cycles are refused** at write time (a tenant may not be its own ancestor) and
  defended again at read time (the walk carries a ``seen`` set).
* **Only ``kg:admin`` may register a tenant or set its parent.** The engine has
  ``row_visibility``/``can_see_row`` for READS and *no* ``can_write_row`` — there
  is no write-side row-ownership check to lean on — so this authorization is
  enforced here, at the single chokepoint every mutation goes through, not at a
  caller's entrypoint.
* **Commons stays last.** This module never touches ordering;
  ``accessible_graphs`` appends the commons graph after the ancestor chain, per
  the GOC-61 (2026-08-09) ruling that the commons/tenant split is intentional.
"""

from __future__ import annotations

import contextvars
import logging
import re
import threading
import time
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "MAX_TENANT_DEPTH",
    "TENANT_HIERARCHY_LABEL",
    "TenantParent",
    "ancestor_chain",
    "clear_parent",
    "invalidate_cache",
    "parent_of",
    "registry_node_id",
    "set_parent",
]


@dataclass(frozen=True, slots=True)
class TenantParent:
    """One registry record: ``tenant_id``'s parent, and the chain depth it makes.

    A typed result rather than a free-form dict on purpose — this crosses the
    MCP/REST boundary, and an untyped payload there is exactly where producer
    and consumer key contracts drift apart.
    """

    tenant_id: str
    parent_tenant_id: str
    depth: int


#: Node type/label of a registry record in ``__control__``.
TENANT_HIERARCHY_LABEL = "TenantHierarchy"

#: Maximum number of tenants in one ancestor chain, INCLUDING the tenant itself.
#: 4 → at most 3 ancestors → ``accessible_graphs()`` returns at most 5 graphs
#: (tenant + 3 ancestors + commons), which still fits one ``read_union``
#: thread-pool wave (``_READ_UNION_MAX_WORKERS`` = 8) and therefore costs one
#: RPC round-trip of wall clock, not five.
MAX_TENANT_DEPTH = 4

#: How long a loaded snapshot (including an empty/failed one) is trusted.
_CACHE_TTL_SECONDS = 30.0

#: Tenant ids come from JWT claims; keep them to the character class the KG-2.6
#: tenant scoper and ``shard_topology`` already accept.
_SAFE_TENANT_RE = re.compile(r"\A[A-Za-z0-9_:.\-@]{1,128}\Z")

_lock = threading.Lock()
_snapshot: dict[str, str] | None = None
_snapshot_at: float = 0.0

# Re-entrancy guard: loading the registry itself performs a graph read, and a
# graph read resolves a session. If any layer under us ever calls back into
# ``accessible_graphs`` we must NOT recurse into another load — report "flat"
# for the inner call instead of deadlocking or blowing the stack.
_loading: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "tenant_registry_loading", default=False
)


def registry_node_id(tenant_id: str) -> str:
    """The stable ``__control__`` node id holding ``tenant_id``'s parent."""
    return f"tenant-hierarchy:{tenant_id}"


def _valid(tenant_id: Any) -> str:
    text = str(tenant_id or "").strip()
    if not text or not _SAFE_TENANT_RE.match(text):
        raise ValueError(f"invalid tenant id: {tenant_id!r}")
    return text


# ---------------------------------------------------------------------------
# Snapshot load / cache
# ---------------------------------------------------------------------------


def _control_backend() -> Any:
    from ..backends.epistemic_graph_backend import EpistemicGraphBackend
    from .shard_topology import CONTROL_GRAPH_NAME

    return EpistemicGraphBackend().for_graph(CONTROL_GRAPH_NAME)


def _load_snapshot() -> dict[str, str]:
    """One label-indexed read of every registry record → ``{tenant: parent}``.

    ``_control_backend()`` is a *graph-scoped view* pinned to ``__control__``.
    The ambient session a caller runs under is bound to whatever graph it
    actually operates on, not necessarily ``__control__`` — calling the view
    without first retargeting the session raises ``PermissionError: "A
    graph-scoped view cannot retarget the verified GraphSession"`` for every
    caller regardless of privilege (BUG-295; see
    ``core.schedule_engine._control_session_scope`` /
    ``knowledge_graph.core.session.control_session_scope``, the shared fix).
    """
    from .session import control_session_scope

    backend = _control_backend()
    with control_session_scope(backend):
        rows = backend.nodes_by_label(TENANT_HIERARCHY_LABEL) or []
    mapping: dict[str, str] = {}
    for node_id, props in rows:
        props = props if isinstance(props, dict) else {}
        tenant = str(props.get("tenant_id") or "").strip()
        if not tenant:
            # Fall back to the id convention when the property is missing.
            tenant = str(node_id or "").removeprefix("tenant-hierarchy:").strip()
        parent = str(props.get("parent_tenant_id") or "").strip()
        if tenant and parent and tenant != parent:
            mapping[tenant] = parent
    return mapping


def _hierarchy_snapshot(*, refresh: bool = False) -> dict[str, str]:
    """The cached ``{tenant_id: parent_tenant_id}`` map (never ``None``).

    A read failure (engine down, ``__control__`` unreachable, no session) is
    reported as an EMPTY map — flat tenancy, today's behaviour — and cached for
    the same TTL so a broken engine costs one attempt per window, not one per
    read. Degrading is logged at WARNING (not DEBUG): silent degradation here
    is exactly what let ``_control_backend()`` fail its ``PermissionError`` on
    every single caller, including ``kg:admin``, for the entire time the
    registry has existed (BUG-295-class retarget bug) without a single record
    surfacing anywhere — zero ``:TenantHierarchy`` nodes were ever written and
    nobody noticed because reads just quietly fell back to flat tenancy.
    """
    global _snapshot, _snapshot_at
    now = time.monotonic()
    if not refresh:
        cached = _snapshot
        if cached is not None and (now - _snapshot_at) < _CACHE_TTL_SECONDS:
            return cached
    if _loading.get():
        return _snapshot if _snapshot is not None else {}
    token = _loading.set(True)
    try:
        loaded = _load_snapshot()
    except Exception as exc:  # noqa: BLE001 — flat tenancy is a correct degrade
        logger.warning(
            "tenant hierarchy registry unavailable, degrading to flat tenancy "
            "for %.0fs: %s",
            _CACHE_TTL_SECONDS,
            exc,
            exc_info=True,
        )
        loaded = {}
    finally:
        _loading.reset(token)
    with _lock:
        _snapshot = loaded
        _snapshot_at = time.monotonic()
    return loaded


def invalidate_cache() -> None:
    """Drop the cached snapshot (called after every mutation; test helper)."""
    global _snapshot, _snapshot_at
    with _lock:
        _snapshot = None
        _snapshot_at = 0.0


# ---------------------------------------------------------------------------
# Read side — bounded, cycle-safe ancestor projection
# ---------------------------------------------------------------------------


def parent_of(tenant_id: str) -> str | None:
    """The registered parent of ``tenant_id``, or ``None`` when it has none."""
    try:
        tenant = _valid(tenant_id)
    except ValueError:
        return None
    return _hierarchy_snapshot().get(tenant) or None


def _walk(tenant: str, mapping: dict[str, str]) -> list[str]:
    """Ancestors of ``tenant``, nearest first, bounded and cycle-safe."""
    chain: list[str] = []
    seen = {tenant}
    current = tenant
    # MAX_TENANT_DEPTH counts the tenant itself, so at most MAX-1 ancestors.
    while len(chain) < MAX_TENANT_DEPTH - 1:
        parent = mapping.get(current)
        if not parent or parent in seen:
            # No parent, or a cycle written directly into the store behind our
            # back — stop. Never loop, never emit a graph twice.
            break
        chain.append(parent)
        seen.add(parent)
        current = parent
    return chain


def ancestor_chain(tenant_id: str) -> list[str]:
    """Ordered ancestors of ``tenant_id`` (nearest parent first, bounded).

    Never raises: an invalid id or an unreachable registry yields ``[]`` (flat
    tenancy), which is exactly the pre-registry behaviour.
    """
    try:
        tenant = _valid(tenant_id)
    except ValueError:
        return []
    return _walk(tenant, _hierarchy_snapshot())


def _subtree_height(tenant: str, mapping: dict[str, str]) -> int:
    """Longest descendant chain below ``tenant`` (0 when it has no children)."""
    children: dict[str, list[str]] = {}
    for child, parent in mapping.items():
        children.setdefault(parent, []).append(child)
    height = 0
    frontier = [tenant]
    seen = {tenant}
    while frontier and height < MAX_TENANT_DEPTH + 1:
        nxt = [c for node in frontier for c in children.get(node, ()) if c not in seen]
        if not nxt:
            break
        seen.update(nxt)
        frontier = nxt
        height += 1
    return height


# ---------------------------------------------------------------------------
# Write side — kg:admin only, cycle-refusing, depth-bounded
# ---------------------------------------------------------------------------


def _require_admin(actor: Any) -> Any:
    """Only an explicitly graph-administrative actor may shape the tenant tree.

    The engine has ``row_visibility``/``can_see_row`` for reads and NO
    ``can_write_row``, so storage enforces nothing on the write side. This is
    the chokepoint: every mutation in this module goes through it.
    """
    from ...security.brain_context import current_actor
    from .tenant_sharing import is_privileged

    resolved = actor if actor is not None else current_actor()
    # ``is_privileged`` itself raises PermissionError for an unverified or
    # tenantless actor before it ever answers the capability question.
    if not is_privileged(resolved):
        raise PermissionError(
            "Setting a tenant's parent requires the explicit 'kg:admin' capability"
        )
    return resolved


def set_parent(
    tenant_id: str,
    parent_tenant_id: str,
    actor: Any = None,
) -> TenantParent:
    """Durably record ``tenant_id``'s parent. Requires ``kg:admin``.

    Refuses, in order: an invalid id, a self-parent, a cycle (``tenant_id``
    already reachable from ``parent_tenant_id``), and an edit that would push
    any chain through :data:`MAX_TENANT_DEPTH`.
    """
    resolved = _require_admin(actor)
    tenant = _valid(tenant_id)
    parent = _valid(parent_tenant_id)
    if tenant == parent:
        raise ValueError(f"tenant {tenant!r} cannot be its own parent")

    mapping = dict(_hierarchy_snapshot(refresh=True))
    # Cycle: is `tenant` already an ancestor of `parent`?
    probe = parent
    seen = {parent}
    while probe:
        if probe == tenant:
            raise ValueError(
                f"refusing tenant hierarchy cycle: {tenant!r} is already an "
                f"ancestor of {parent!r}"
            )
        probe = mapping.get(probe, "")
        if probe in seen:
            break
        seen.add(probe)

    # Depth: ancestors above `parent`, + `parent`, + `tenant`, + `tenant`'s own
    # deepest descendant chain, must all fit inside MAX_TENANT_DEPTH.
    mapping[tenant] = parent
    depth = 1 + len(_walk(parent, mapping)) + 1 + _subtree_height(tenant, mapping)
    if depth > MAX_TENANT_DEPTH:
        raise ValueError(
            f"refusing tenant hierarchy depth {depth} > MAX_TENANT_DEPTH "
            f"({MAX_TENANT_DEPTH}): every level multiplies the per-read graph "
            f"fan-out in read_union"
        )

    from .session import control_session_scope

    backend = _control_backend()
    with control_session_scope(backend):
        backend.add_node(
            registry_node_id(tenant),
            node_type=TENANT_HIERARCHY_LABEL,
            tenant_id=tenant,
            parent_tenant_id=parent,
            registered_by=str(getattr(resolved, "actor_id", "") or ""),
            registered_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )
    invalidate_cache()
    logger.info(
        "tenant hierarchy: %s -> parent %s (by %s)",
        tenant,
        parent,
        getattr(resolved, "actor_id", "?"),
    )
    return TenantParent(tenant_id=tenant, parent_tenant_id=parent, depth=depth)


def clear_parent(tenant_id: str, actor: Any = None) -> TenantParent:
    """Detach ``tenant_id`` from its parent (it becomes a root). ``kg:admin``."""
    from .session import control_session_scope

    _require_admin(actor)
    tenant = _valid(tenant_id)
    backend = _control_backend()
    with control_session_scope(backend):
        backend.add_node(
            registry_node_id(tenant),
            node_type=TENANT_HIERARCHY_LABEL,
            tenant_id=tenant,
            parent_tenant_id="",
            registered_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )
    invalidate_cache()
    return TenantParent(tenant_id=tenant, parent_tenant_id="", depth=1)
