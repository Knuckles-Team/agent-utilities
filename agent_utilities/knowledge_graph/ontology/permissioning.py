#!/usr/bin/python
from __future__ import annotations

"""Fine-grained object permissioning for the Company Brain (CONCEPT:AU-KG.ontology.redact-object-materialize-restricted).

Provenance: Palantir Foundry "object-permissioning/overview" — schema- and
instance-level access control where, beyond row-level (which *objects* an actor
may see), property/column-level redaction governs which *fields* are visible,
and *marking-based mandatory controls* propagate along links/derivations so a
sensitive marking on one object cannot be laundered through a derived object.
This module extends the EXISTING company-brain read path (``secured_reads.py``
+ ``company_brain_runtime.py`` + ``security/brain_context.py``), adding the
three capabilities that read path lacked:

* **Property/column-level access** — :func:`redact_object` drops or masks the
  individual properties an actor may not read, based on per-property
  classification/markings, returning a *filtered copy* of the object.
* **Marking-based mandatory control** — :class:`Marking` + :func:`propagate_markings`
  push a node's markings onto linked/derived/inferred nodes (the mandatory
  control that *cannot* be relaxed downstream), generalising
  ``secured_reads.inherit_inferred_acl`` beyond the 4-level classification to
  arbitrary named markings carried in :data:`MARKING_REGISTRY`.
* **Restricted view materialization** — :func:`restricted_view` composes the
  existing ``permit()`` / ``filter_rows()`` row gate with column redaction to
  produce a permission-filtered VIEW of an object set for one actor.

A single :func:`enforce` entry point is the default-on seam: it applies a
*safe* policy — allow-by-default, but enforce whenever a node carries an ACL or
markings — so it is correct to call on the live read path **regardless of**
``KG_BRAIN_ENFORCE``. (The legacy ``secured_reads`` helpers stay gated on that
flag for backward compatibility; the marking-mandatory layer here does not.)

With ``KG_BRAIN_ENFORCE`` on, the gate additionally fails **closed**
(CONCEPT:AU-OS.identity.authenticated-identity-enforcement): an ACL-check exception denies, and nodes without an ACL are
denied by policy default (escape hatch: ``KG_ACL_DEFAULT_ALLOW``). Mandatory
markings are durably persisted as ``mandatory_marking`` graph nodes (loaded on
first use, written through on registration) so separate processes agree; the
in-process ``MARKING_REGISTRY`` dict is a cache of that store.

Reuses :class:`PermissionsKernel` semantics, :class:`ActorContext`, and
:class:`DataLevelPermissions` (``DataClassification``) — no new permission
engine is introduced.
"""

import json
import logging
from typing import Any

from agent_utilities.core.config import setting

from ...models.company_brain import (
    ActorType,
    DataClassification,
    NodeACL,
)
from ...security.brain_context import ActorContext, current_actor
from ..core.company_brain_runtime import (
    brain_enforcement_enabled,
    get_company_brain,
)

logger = logging.getLogger(__name__)

# Ordering of the regulatory classification ladder (mirrors secured_reads so the
# two layers agree on "most restrictive"). Higher == more secret.
_CLASS_ORDER: dict[str, int] = {
    DataClassification.PUBLIC: 0,
    DataClassification.INTERNAL: 1,
    DataClassification.CONFIDENTIAL: 2,
    DataClassification.RESTRICTED: 3,
}

# Roles that are never redacted/blocked regardless of marking (matches the
# privileged SYSTEM_ACTOR defaults in brain_context).
_PRIVILEGED_ROLES: frozenset[str] = frozenset({"admin", "system"})

# Property keys that identify a row/object and must survive redaction so the
# caller can still correlate the filtered object back to its source.
_IDENTITY_KEYS: tuple[str, ...] = ("id", "node_id", "_id")

# Sentinel returned in place of a masked (rather than dropped) property value.
MASK_TOKEN = "***"


# ---------------------------------------------------------------------------
# Marking model + registry — mandatory controls that propagate
# ---------------------------------------------------------------------------


class Marking:
    """A mandatory access-control marking attached to one or more nodes.

    Unlike the 4-level :class:`DataClassification` (a total order), markings are
    an *unordered set* of named compartments (Palantir: "markings"). To read a
    marked object an actor must hold the marking in
    :attr:`ActorContext.roles` (markings are carried as roles of the form
    ``marking:<name>``) — this is the mandatory control: it cannot be granted by
    a per-object ACL, only by membership.

    Markings propagate: any node *derived from* or *linked to* a marked node
    inherits the marking (:func:`propagate_markings`), so a RESTRICTED parent's
    secrecy cannot be laundered through a synthesized child.
    """

    __slots__ = ("name", "description", "requires_audit")

    def __init__(
        self, name: str, description: str = "", requires_audit: bool = False
    ) -> None:
        if not name:
            raise ValueError("Marking.name must be non-empty")
        self.name = name
        self.description = description
        self.requires_audit = requires_audit

    @property
    def role_token(self) -> str:
        """The role string an actor must hold to clear this marking."""
        return f"marking:{self.name}"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Marking) and other.name == self.name

    def __hash__(self) -> int:
        return hash(self.name)

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"Marking({self.name!r})"


# Process-wide CACHE of (tenant, node_id) -> set of marking names, durably
# backed by ``mandatory_marking`` graph nodes (CONCEPT:AU-OS.identity.authenticated-identity-enforcement) so separate
# processes agree on mandatory controls: hydrated from the store on first use,
# written through on every registration. Markings live here (not on the ACL)
# because they are mandatory controls applied across the graph;
# DataLevelPermissions keeps the per-object discretionary ACL/classification.
#
# Keyed by ``(tenant, node_id)`` rather than bare ``node_id`` (AU-P0-5): two
# tenants can mint the same node id (e.g. a generic "config" or a connector's
# own numbering scheme is not guaranteed globally unique), so a bare
# ``node_id`` key would let tenant A's marking silently apply to — or a cache
# hit leak onto — tenant B's unrelated node of the same id. The tenant
# component defaults to ``""`` (see :func:`_ambient_tenant`), so a caller that
# never scopes a tenant (today's behaviour) keeps hitting the exact same
# single ``("", node_id)`` bucket as before this fix — nothing to migrate.
MARKING_REGISTRY: dict[tuple[str, str], set[str]] = {}

# Durable node type for persisted markings (one node per marked graph node).
MARKING_NODE_TYPE = "mandatory_marking"

_marking_store: Any = None
_marking_store_resolved = False
_markings_hydrated = False


def _resolve_marking_store() -> Any:
    """Resolve (once) the durable graph store backing the marking registry.

    Mirrors the EditLedger's lazy, degrade-cleanly probe: a missing/unreachable
    backend leaves the in-process cache authoritative. Under
    ``AGENT_UTILITIES_TESTING`` no live store is probed (tests inject one via
    :func:`set_marking_store`).
    """
    global _marking_store, _marking_store_resolved  # noqa: PLW0603
    if _marking_store_resolved:
        return _marking_store
    _marking_store_resolved = True
    if setting("AGENT_UTILITIES_TESTING"):
        return None
    try:
        from ..facade import KnowledgeGraph

        _marking_store = KnowledgeGraph().store
    except Exception as exc:  # noqa: BLE001 — degrade to in-process cache
        logger.debug("marking store unavailable: %s", exc)
        _marking_store = None
    return _marking_store


def set_marking_store(store: Any) -> None:
    """Inject the durable marking store (DI/test seam); forces re-hydration."""
    global _marking_store, _marking_store_resolved, _markings_hydrated  # noqa: PLW0603
    _marking_store = store
    _marking_store_resolved = True
    _markings_hydrated = False


def _ambient_tenant() -> str:
    """The tenant a marking write/read defaults to when not passed explicitly.

    Prefers the ambient :class:`~agent_utilities.knowledge_graph.core.session.GraphSession`
    (AU-P0-5's one currency), falling back to the ambient actor's
    ``tenant_id``. Returns ``""`` (the unscoped/system bucket) when neither is
    set — the exact behaviour every caller saw before markings were
    tenant-keyed.
    """
    try:
        from ..core.session import current_session

        session = current_session()
        if session is not None and session.tenant:
            return session.tenant
    except Exception:  # noqa: BLE001 — session currency is best-effort here
        pass
    try:
        return current_actor().tenant_id or ""
    except Exception:  # noqa: BLE001
        return ""


def _mkey(node_id: str, tenant: str | None) -> tuple[str, str]:
    """Build the ``(tenant, node_id)`` :data:`MARKING_REGISTRY` key."""
    return (tenant if tenant is not None else _ambient_tenant()) or "", node_id


def _hydrate_markings() -> None:
    """Load persisted markings into the cache (once per process, best-effort)."""
    global _markings_hydrated  # noqa: PLW0603
    if _markings_hydrated:
        return
    _markings_hydrated = True
    store = _resolve_marking_store()
    if store is None:
        return
    try:
        rows = (
            store.execute(
                "MATCH (m) WHERE m.type = $t "
                "RETURN m.node_id AS node_id, m.tenant_id AS tenant_id, "
                "m.markings AS markings",
                {"t": MARKING_NODE_TYPE},
            )
            or []
        )
        for row in rows:
            nid = row.get("node_id")
            tenant = row.get("tenant_id") or ""
            raw = row.get("markings")
            names = json.loads(raw) if isinstance(raw, str) else (raw or [])
            if isinstance(nid, str) and nid:
                key = (str(tenant), nid)
                MARKING_REGISTRY.setdefault(key, set()).update(
                    str(n) for n in names if n
                )
    except Exception as exc:  # noqa: BLE001 — cache stays authoritative
        logger.debug("marking hydration skipped: %s", exc)


def _persist_markings(key: tuple[str, str]) -> None:
    """Write-through ``key``'s (tenant, node_id) markings as a durable graph node."""
    store = _resolve_marking_store()
    if store is None:
        return
    tenant, node_id = key
    # Back-compat storage id for the unscoped ("") bucket keeps the exact
    # pre-P0-5 node id shape; a tenant-scoped marking gets its own namespaced
    # id so two tenants' markings for the "same" node_id never collide.
    storage_id = (
        f"marking::{node_id}" if not tenant else f"marking::{tenant}::{node_id}"
    )
    try:
        store.execute(
            "MERGE (m {id: $id}) SET m.type = $t, m.node_id = $n, "
            "m.tenant_id = $tenant, m.markings = $marks",
            {
                "id": storage_id,
                "t": MARKING_NODE_TYPE,
                "n": node_id,
                "tenant": tenant,
                "marks": json.dumps(sorted(MARKING_REGISTRY.get(key, set()))),
            },
        )
    except Exception as exc:  # noqa: BLE001 — cache stays authoritative
        logger.debug("marking persist failed for %s: %s", key, exc)


def apply_marking(
    node_id: str, marking: Marking | str, *, tenant: str | None = None
) -> None:
    """Attach a marking to a node (idempotent; written through to the graph).

    ``tenant`` (AU-P0-5) scopes the marking to one tenant's node namespace;
    omitted, it defaults to the ambient :class:`GraphSession`/actor tenant
    (:func:`_ambient_tenant`) — an unscoped caller keeps writing to the same
    ``""`` bucket as before markings were tenant-keyed.
    """
    name = marking.name if isinstance(marking, Marking) else str(marking)
    if not name:
        return
    _hydrate_markings()
    key = _mkey(node_id, tenant)
    MARKING_REGISTRY.setdefault(key, set()).add(name)
    _persist_markings(key)


def markings_for(node_id: str, *, tenant: str | None = None) -> set[str]:
    """Return the set of marking names carried by ``node_id`` under ``tenant``.

    ``tenant`` defaults to the ambient session/actor tenant (AU-P0-5), same
    fallback as :func:`apply_marking` — see :func:`_ambient_tenant`.
    """
    _hydrate_markings()
    return set(MARKING_REGISTRY.get(_mkey(node_id, tenant), ()))


def clear_markings() -> None:
    """Drop all in-process marking state (test helper; not a production path)."""
    global _marking_store, _marking_store_resolved, _markings_hydrated  # noqa: PLW0603
    MARKING_REGISTRY.clear()
    _marking_store = None
    _marking_store_resolved = False
    _markings_hydrated = False


def _actor_marking_tokens(actor: ActorContext) -> set[str]:
    """The marking names an actor clears (from ``marking:<name>`` roles)."""
    return {
        r.split(":", 1)[1]
        for r in actor.roles
        if isinstance(r, str) and r.startswith("marking:")
    }


def _is_privileged(actor: ActorContext) -> bool:
    return bool(_PRIVILEGED_ROLES & set(actor.roles))


# ---------------------------------------------------------------------------
# Property / column-level access
# ---------------------------------------------------------------------------


def _property_classification(
    obj: dict[str, Any], prop: str
) -> DataClassification | None:
    """Resolve a per-property classification from an object's metadata.

    A property is classified either by a sibling ``__classification__`` map
    (``{prop: "confidential"}``) carried on the object, or — failing that — by a
    convention key ``<prop>__classification``. Returns ``None`` (public) when no
    per-property classification is declared.
    """
    cmap = obj.get("__classification__")
    if isinstance(cmap, dict) and prop in cmap:
        raw = cmap[prop]
    else:
        raw = obj.get(f"{prop}__classification")
    if raw is None:
        return None
    try:
        return DataClassification(str(raw).lower())
    except ValueError:
        return None


def _property_markings(obj: dict[str, Any], prop: str) -> set[str]:
    """Resolve per-property markings from a sibling ``__markings__`` map."""
    mmap = obj.get("__markings__")
    if isinstance(mmap, dict):
        raw = mmap.get(prop)
        if isinstance(raw, list | tuple | set):
            return {str(x) for x in raw}
        if isinstance(raw, str) and raw:
            return {raw}
    return set()


def _actor_clearance(actor: ActorContext) -> int:
    """Map an actor's roles to a numeric classification clearance.

    Privileged (admin/system) actors clear RESTRICTED; a ``confidential`` role
    clears CONFIDENTIAL; otherwise an authenticated actor clears INTERNAL; an
    anonymous actor (no roles, no id) clears only PUBLIC.
    """
    roles = set(actor.roles)
    if _PRIVILEGED_ROLES & roles:
        return _CLASS_ORDER[DataClassification.RESTRICTED]
    if "confidential" in roles or "data_owner" in roles:
        return _CLASS_ORDER[DataClassification.CONFIDENTIAL]
    if roles or (actor.actor_id and actor.actor_id != "anonymous"):
        return _CLASS_ORDER[DataClassification.INTERNAL]
    return _CLASS_ORDER[DataClassification.PUBLIC]


def redact_object(
    obj: dict[str, Any],
    actor: ActorContext | None = None,
    *,
    mask: bool = False,
) -> dict[str, Any]:
    """Return a copy of ``obj`` with properties the actor may not read removed.

    Column/property-level access (Palantir object-permissioning): each property
    may carry its own classification and/or markings. A property is withheld
    when the actor's clearance is below the property's classification, OR the
    property carries a marking the actor does not hold. Withheld properties are
    dropped by default, or replaced with :data:`MASK_TOKEN` when ``mask`` is set
    (a "restricted view" that preserves shape). Identity keys and the metadata
    maps are always stripped of secrets but identity survives so the object
    stays correlatable.

    This never mutates the input; it returns a filtered dict.
    """
    actor = actor or current_actor()
    privileged = _is_privileged(actor)
    clearance = _actor_clearance(actor)
    actor_marks = _actor_marking_tokens(actor)

    out: dict[str, Any] = {}
    for key, value in obj.items():
        # Drop the metadata side-channels from the materialized view.
        if key in ("__classification__", "__markings__") or key.endswith(
            "__classification"
        ):
            continue
        if key in _IDENTITY_KEYS:
            out[key] = value
            continue
        if privileged:
            out[key] = value
            continue

        prop_class = _property_classification(obj, key)
        prop_marks = _property_markings(obj, key)

        denied = False
        if prop_class is not None and _CLASS_ORDER.get(prop_class, 0) > clearance:
            denied = True
        if prop_marks and not prop_marks.issubset(actor_marks):
            denied = True

        if denied:
            if mask:
                out[key] = MASK_TOKEN
            # else: drop entirely
            continue
        out[key] = value
    return out


# ---------------------------------------------------------------------------
# Marking-based mandatory control — propagation over edges
# ---------------------------------------------------------------------------


def propagate_markings(
    source_id: str,
    target_id: str,
    *,
    propagate_classification: bool = True,
    tenant: str | None = None,
) -> set[str]:
    """Propagate ``source_id``'s mandatory controls onto ``target_id``.

    Generalises ``secured_reads.inherit_inferred_acl`` beyond the 4-level
    classification: every marking on the source (and, by default, the *more
    restrictive* of the two classifications) is inherited by the target, because
    a derived/linked object must not be readable by anyone who could not read
    its source. Returns the target's marking set after propagation.

    ``tenant`` (AU-P0-5) scopes both endpoints to the same tenant's node
    namespace (defaults to the ambient session/actor tenant, matching
    :func:`apply_marking`) so propagation never crosses tenant boundaries via a
    same-named node in a different tenant's graph.

    Unlike the legacy helper this is **not** gated on ``KG_BRAIN_ENFORCE`` —
    mandatory controls must always propagate so default-on enforcement is safe.
    """
    resolved_tenant = tenant if tenant is not None else _ambient_tenant()
    src_key = _mkey(source_id, resolved_tenant)
    tgt_key = _mkey(target_id, resolved_tenant)
    _hydrate_markings()
    src_marks = set(MARKING_REGISTRY.get(src_key, ()))
    if src_marks:
        MARKING_REGISTRY.setdefault(tgt_key, set()).update(src_marks)
        _persist_markings(tgt_key)

    if propagate_classification:
        try:
            perms = get_company_brain().permissions
            levels: list[DataClassification] = []
            for nid in (source_id, target_id):
                acl = perms.get_acl(nid)
                if acl is not None:
                    levels.append(acl.classification)
            if levels:
                strictest = max(levels, key=lambda c: _CLASS_ORDER.get(c, 0))
                tgt_acl = perms.get_acl(target_id)
                if tgt_acl is None or _CLASS_ORDER.get(
                    tgt_acl.classification, 0
                ) < _CLASS_ORDER.get(strictest, 0):
                    perms.classify_node(target_id, strictest)
        except Exception as exc:  # pragma: no cover - best-effort propagation
            logger.debug(
                "classification propagation failed %s->%s: %s",
                source_id,
                target_id,
                exc,
            )

    return set(MARKING_REGISTRY.get(tgt_key, ()))


def propagate_over_edges(
    edges: list[tuple[str, str]], *, tenant: str | None = None
) -> dict[str, set[str]]:
    """Propagate markings along a list of ``(source, target)`` edges.

    A single forward pass over the provided edges (callers ordered topologically
    for transitive closure get full inheritance). Returns the resulting
    node_id -> markings map for the touched targets. ``tenant`` (AU-P0-5) is
    forwarded to :func:`propagate_markings` for every edge — all edges in one
    call share the same tenant scope (defaults to the ambient session/actor).
    """
    resolved_tenant = tenant if tenant is not None else _ambient_tenant()
    touched: dict[str, set[str]] = {}
    for source_id, target_id in edges:
        touched[target_id] = propagate_markings(
            source_id, target_id, tenant=resolved_tenant
        )
    return touched


# ---------------------------------------------------------------------------
# Mandatory marking read-gate
# ---------------------------------------------------------------------------


def _marking_permits(node_id: str, actor: ActorContext) -> bool:
    """Whether ``actor`` clears every marking carried by ``node_id``.

    Markings are looked up under ``actor.tenant_id`` (AU-P0-5) — the read
    path always has an actor in hand, so this is exact rather than ambient.
    """
    marks = markings_for(node_id, tenant=actor.tenant_id or "")
    if not marks:
        return True
    if _is_privileged(actor):
        return True
    return marks.issubset(_actor_marking_tokens(actor))


def _node_id_of(obj: Any) -> str | None:
    """Best-effort node id extraction from a row/object (dict or model)."""
    if isinstance(obj, dict):
        for key in (*_IDENTITY_KEYS, "n.id"):
            val = obj.get(key)
            if isinstance(val, str):
                return val
        for val in obj.values():
            if isinstance(val, dict):
                inner = val.get("id") or val.get("node_id")
                if isinstance(inner, str):
                    return inner
        return None
    for attr in _IDENTITY_KEYS:
        val = getattr(obj, attr, None)
        if isinstance(val, str):
            return val
    return None


def _acl_default_allow() -> bool:
    """The enforced-mode policy for ACL-less nodes (``KG_ACL_DEFAULT_ALLOW``).

    Read fresh from a typed :class:`AgentConfig` field (Configuration
    discipline — no bare env reads); any failure resolves to deny.
    """
    try:
        from agent_utilities.core.config import AgentConfig

        return bool(AgentConfig().kg_acl_default_allow)
    except Exception:  # noqa: BLE001 — fail closed
        return False


def _acl_permits(node_id: str, actor: ActorContext) -> bool:
    """Discretionary ACL read-check via the existing DataLevelPermissions.

    Legacy mode (``KG_BRAIN_ENFORCE`` off): default-allow when no ACL exists
    and allow on infra error — byte-identical to historic behaviour.

    Enforced mode (CONCEPT:AU-OS.identity.authenticated-identity-enforcement, fail CLOSED): an infra exception denies,
    and a node WITHOUT an ACL is denied by policy default unless the
    ``KG_ACL_DEFAULT_ALLOW`` escape hatch is set.
    """
    enforced = brain_enforcement_enabled()
    try:
        perms = get_company_brain().permissions
        if perms.get_acl(node_id) is None:
            return _acl_default_allow() if enforced else True
        return perms.check_permission(
            node_id,
            actor.actor_id,
            actor.actor_type,
            action="read",
            actor_roles=list(actor.roles),
        ).allowed
    except Exception as exc:
        if enforced:
            logger.warning(
                "ACL check failed for %s — denying (fail-closed): %s",
                node_id,
                exc,
            )
            return False
        logger.debug("acl check failed for %s: %s", node_id, exc)
        return True


# ---------------------------------------------------------------------------
# Restricted view materialization
# ---------------------------------------------------------------------------


def restricted_view(
    objects: list[dict[str, Any]],
    actor: ActorContext | None = None,
    *,
    mask: bool = False,
) -> list[dict[str, Any]]:
    """Materialize a permission-filtered VIEW of ``objects`` for ``actor``.

    Composes the two gates Palantir's restricted views require:
      1. **Row-level** — objects the actor may not read (mandatory markings or a
         denying ACL) are dropped entirely.
      2. **Column-level** — surviving objects are passed through
         :func:`redact_object` so properties the actor may not read are removed
         (or masked when ``mask`` is set).

    Objects whose id cannot be determined are kept and still column-redacted (we
    never silently lose data we cannot classify). Returns filtered *copies*.
    """
    actor = actor or current_actor()
    view: list[dict[str, Any]] = []
    audited: list[str] = []
    for obj in objects:
        nid = _node_id_of(obj)
        if nid is not None:
            if not _marking_permits(nid, actor) or not _acl_permits(nid, actor):
                continue
            if markings_for(nid) or (
                get_company_brain().permissions.get_acl(nid) is not None
            ):
                audited.append(nid)
        view.append(redact_object(obj, actor, mask=mask))
    if audited:
        _audit(audited, actor, summary="restricted_view")
    return view


def _audit(node_ids: list[str], actor: ActorContext, summary: str) -> None:
    """Record a read audit for marked/ACL'd nodes (mandatory-control trail)."""
    try:
        get_company_brain().provenance.record_read(
            actor_id=actor.actor_id,
            actor_type=actor.actor_type,
            nodes_accessed=list(node_ids),
            query_summary=summary,
            tenant_id=actor.tenant_id,
        )
    except Exception as exc:  # pragma: no cover - audit best-effort
        logger.debug("permissioning audit failed: %s", exc)


# ---------------------------------------------------------------------------
# Single default-on entry point
# ---------------------------------------------------------------------------


def enforce(
    objects: list[dict[str, Any]],
    actor: ActorContext | None = None,
    *,
    mask: bool = False,
) -> list[dict[str, Any]]:
    """Default-on fine-grained enforcement for a result set.

    The single seam the live read path calls. Policy is **allow-by-default but
    enforce-when-marked**: an object with neither a marking nor an ACL passes
    through unchanged, so turning this on cannot break unmarked data; an object
    that carries a mandatory marking or a discretionary ACL is row-filtered and
    column-redacted for ``actor``. Because the gate is *driven by the data's own
    controls*, it is correct to call unconditionally — it does **not** depend on
    ``KG_BRAIN_ENFORCE`` being on. (With ``KG_BRAIN_ENFORCE`` on the row gate
    tightens to fail-closed: see :func:`_acl_permits`.)

    This is the property+row composition of :func:`restricted_view`; kept as a
    named entry point so the facade read path has one stable call.
    """
    if not objects:
        return objects
    return restricted_view(objects, actor or current_actor(), mask=mask)


def build_acl(
    node_id: str,
    classification: DataClassification = DataClassification.INTERNAL,
    *,
    read_roles: list[str] | None = None,
    data_owner: str = "",
) -> NodeACL:
    """Convenience constructor that registers an ACL on the live brain.

    Returns the stored :class:`NodeACL` so callers can assert on it. Reuses
    :class:`DataLevelPermissions` rather than introducing a parallel store.
    """
    acl = NodeACL(
        node_id=node_id,
        classification=classification,
        read_roles=list(read_roles or []),
        data_owner=data_owner,
        data_owner_type=ActorType.SYSTEM,
    )
    get_company_brain().permissions.set_acl(acl)
    return acl


__all__ = [
    "MASK_TOKEN",
    "MARKING_NODE_TYPE",
    "MARKING_REGISTRY",
    "Marking",
    "apply_marking",
    "markings_for",
    "clear_markings",
    "set_marking_store",
    "redact_object",
    "propagate_markings",
    "propagate_over_edges",
    "restricted_view",
    "enforce",
    "build_acl",
]
