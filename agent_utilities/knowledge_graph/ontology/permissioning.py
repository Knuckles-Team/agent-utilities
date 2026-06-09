#!/usr/bin/python
from __future__ import annotations

"""Fine-grained object permissioning for the Company Brain (CONCEPT:KG-2.46).

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

Reuses :class:`PermissionsKernel` semantics, :class:`ActorContext`, and
:class:`DataLevelPermissions` (``DataClassification``) — no new permission
engine is introduced.
"""

import logging
from typing import Any

from ...models.company_brain import (
    ActorType,
    DataClassification,
    NodeACL,
)
from ...security.brain_context import ActorContext, current_actor
from ..core.company_brain_runtime import get_company_brain

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


# Process-wide registry of node_id -> set of marking names. Markings live here
# (not on the ACL) because they are mandatory controls applied across the graph;
# DataLevelPermissions keeps the per-object discretionary ACL/classification.
MARKING_REGISTRY: dict[str, set[str]] = {}


def apply_marking(node_id: str, marking: Marking | str) -> None:
    """Attach a marking to a node (idempotent)."""
    name = marking.name if isinstance(marking, Marking) else str(marking)
    if not name:
        return
    MARKING_REGISTRY.setdefault(node_id, set()).add(name)


def markings_for(node_id: str) -> set[str]:
    """Return the set of marking names carried by ``node_id``."""
    return set(MARKING_REGISTRY.get(node_id, ()))


def clear_markings() -> None:
    """Drop all markings (test helper; not used on production paths)."""
    MARKING_REGISTRY.clear()


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
) -> set[str]:
    """Propagate ``source_id``'s mandatory controls onto ``target_id``.

    Generalises ``secured_reads.inherit_inferred_acl`` beyond the 4-level
    classification: every marking on the source (and, by default, the *more
    restrictive* of the two classifications) is inherited by the target, because
    a derived/linked object must not be readable by anyone who could not read
    its source. Returns the target's marking set after propagation.

    Unlike the legacy helper this is **not** gated on ``KG_BRAIN_ENFORCE`` —
    mandatory controls must always propagate so default-on enforcement is safe.
    """
    src_marks = markings_for(source_id)
    if src_marks:
        MARKING_REGISTRY.setdefault(target_id, set()).update(src_marks)

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

    return markings_for(target_id)


def propagate_over_edges(edges: list[tuple[str, str]]) -> dict[str, set[str]]:
    """Propagate markings along a list of ``(source, target)`` edges.

    A single forward pass over the provided edges (callers ordered topologically
    for transitive closure get full inheritance). Returns the resulting
    node_id -> markings map for the touched targets.
    """
    touched: dict[str, set[str]] = {}
    for source_id, target_id in edges:
        touched[target_id] = propagate_markings(source_id, target_id)
    return touched


# ---------------------------------------------------------------------------
# Mandatory marking read-gate
# ---------------------------------------------------------------------------


def _marking_permits(node_id: str, actor: ActorContext) -> bool:
    """Whether ``actor`` clears every marking carried by ``node_id``."""
    marks = markings_for(node_id)
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


def _acl_permits(node_id: str, actor: ActorContext) -> bool:
    """Discretionary ACL read-check via the existing DataLevelPermissions.

    Default-allow when no ACL exists (matches DataLevelPermissions semantics),
    so this is safe to call on every read regardless of KG_BRAIN_ENFORCE.
    """
    try:
        perms = get_company_brain().permissions
        if perms.get_acl(node_id) is None:
            return True
        return perms.check_permission(
            node_id,
            actor.actor_id,
            actor.actor_type,
            action="read",
            actor_roles=list(actor.roles),
        ).allowed
    except Exception as exc:  # pragma: no cover - fail-open on infra error
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
    ``KG_BRAIN_ENFORCE`` being on.

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
    "MARKING_REGISTRY",
    "Marking",
    "apply_marking",
    "markings_for",
    "clear_markings",
    "redact_object",
    "propagate_markings",
    "propagate_over_edges",
    "restricted_view",
    "enforce",
    "build_acl",
]
