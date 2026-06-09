#!/usr/bin/python
from __future__ import annotations

"""Durable Edit Ledger — edits as structured, replayable transactions.

Provenance (Palantir Foundry doc: *object-edits/overview*): in Foundry every
mutation a user makes to an object — a property set, a link add/remove, an
object create/delete — is captured as a structured **edit** with a full
before/after snapshot, actor, and timestamp. The set of edits forms a per-object
**edit history / audit trail** that drives undo/revert and (separately) is
written back to the source datasource. This module brings that primitive to the
epistemic KG.

Where ``KGVersionEngine`` (``core/kg_versioning.py``) gives us git-like
commit/rollback over a graph_state dict but keeps its history *in memory only*,
the :class:`EditLedger` makes each edit a **durable ``object_edit`` KG node**
(via the live store on the :class:`KnowledgeGraph` facade) with
``EDITED_BY`` → actor and ``EDITS`` → target-object edges. When no backend is
reachable it degrades to an in-memory + file-serializable store so the
persistence logic is always exercised, never faked.

CONCEPT:KG-2.43
"""

import json
import logging
import os
import time
import uuid
from copy import deepcopy
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from agent_utilities.knowledge_graph.core.kg_versioning import (
    KGMutation,
    KGTransaction,
    KGVersionEngine,
    MutationType,
)
from agent_utilities.observability.audit_logger import AuditLogger

logger = logging.getLogger(__name__)

__all__ = [
    "EditType",
    "Edit",
    "EditLedger",
    "EDIT_NODE_TYPE",
]

# The KG node ``type`` under which every durable edit is materialised.
EDIT_NODE_TYPE = "object_edit"

# AuditLogger action/resource constants for the edit ledger.
ACTION_OBJECT_EDIT = "object.edit"
RESOURCE_OBJECT_EDIT = "object_edit"


class EditType(StrEnum):
    """The kinds of structured edit Foundry records over an object.

    Mirrors *object-edits/overview*: property edits, link add/remove, and
    object create/delete.
    """

    PROPERTY_SET = "property_set"
    LINK_ADD = "link_add"
    LINK_REMOVE = "link_remove"
    OBJECT_CREATE = "object_create"
    OBJECT_DELETE = "object_delete"


class Edit(BaseModel):
    """One durable, structured edit over an object. CONCEPT:KG-2.43.

    An ``Edit`` is the atomic unit of the ledger. It records *who* changed
    *what* on *which object*, with a complete ``before``/``after`` snapshot so
    the change is both reversible (revert) and reconstructable (as_of). For
    link edits the ``link`` triple (source, label, target) identifies the edge;
    for property/object edits the ``before``/``after`` carry the property maps.

    Persisted as a KG ``object_edit`` node with ``EDITED_BY`` → actor and
    ``EDITS`` → target-object edges.
    """

    id: str = ""
    actor: str = "system"
    edit_type: EditType
    object_id: str
    # Property-level snapshots (for PROPERTY_SET / OBJECT_CREATE / OBJECT_DELETE).
    before: dict[str, Any] = Field(default_factory=dict)
    after: dict[str, Any] = Field(default_factory=dict)
    # Edge identity (for LINK_ADD / LINK_REMOVE): (source, label, target).
    link_source: str = ""
    link_label: str = ""
    link_target: str = ""
    provenance: str = ""
    # Optional ref to the ActionInvocation that produced this edit.
    invocation_ref: str = ""
    timestamp: float = Field(default_factory=time.time)
    persisted: bool = False

    def model_post_init(self, __context: Any) -> None:
        if not self.id:
            self.id = f"edit:{self.edit_type}:{uuid.uuid4().hex[:12]}"

    @property
    def iso_timestamp(self) -> str:
        """ISO-8601 rendering of the edit time (UTC)."""
        return datetime.fromtimestamp(self.timestamp, tz=UTC).isoformat()

    def to_mutation(self) -> KGMutation:
        """Project this edit onto the equivalent forward ``KGMutation``.

        This is the bridge to :class:`KGVersionEngine`: applying the returned
        mutation to a graph_state reproduces the edit's *after* state. Revert
        (``revert.py``) computes the inverse of this mutation.
        """
        if self.edit_type == EditType.PROPERTY_SET:
            return KGMutation(
                mutation_type=MutationType.UPDATE_NODE,
                node_id=self.object_id,
                data=dict(self.after),
                previous_data=dict(self.before),
            )
        if self.edit_type == EditType.OBJECT_CREATE:
            return KGMutation(
                mutation_type=MutationType.ADD_NODE,
                node_id=self.object_id,
                data=dict(self.after),
            )
        if self.edit_type == EditType.OBJECT_DELETE:
            return KGMutation(
                mutation_type=MutationType.DELETE_NODE,
                node_id=self.object_id,
                previous_data=dict(self.before),
            )
        if self.edit_type == EditType.LINK_ADD:
            return KGMutation(
                mutation_type=MutationType.ADD_EDGE,
                edge_source=self.link_source,
                edge_target=self.link_target,
                edge_label=self.link_label,
            )
        # LINK_REMOVE
        return KGMutation(
            mutation_type=MutationType.DELETE_EDGE,
            edge_source=self.link_source,
            edge_target=self.link_target,
            edge_label=self.link_label,
        )


def _probe_store(facade: Any) -> Any:
    """Resolve a live store off a KnowledgeGraph facade, or ``None`` if offline.

    Mirrors the lazy, degrade-cleanly probe used by the actions executor so the
    ledger never requires a running engine.
    """
    if facade is None:
        try:
            from agent_utilities.knowledge_graph.facade import KnowledgeGraph

            facade = KnowledgeGraph()
        except Exception as exc:  # noqa: BLE001 — degrade gracefully
            logger.debug("EditLedger: facade unavailable: %s", exc)
            return None
    try:
        return facade.store
    except Exception as exc:  # noqa: BLE001
        logger.debug("EditLedger: store unavailable: %s", exc)
        return None


class EditLedger:
    """Durable, per-object edit history backed by KG ``object_edit`` nodes.

    CONCEPT:KG-2.43 — the Foundry *object-edits* primitive over the epistemic KG.

    The ledger:

    * **applies** an :class:`Edit` to a versioned graph_state through
      :class:`KGVersionEngine` (so commit/rollback semantics hold), and
    * **persists** the edit as a durable ``object_edit`` node with
      ``EDITED_BY`` → actor and ``EDITS`` → target edges on the live store, and
    * keeps an **in-memory mirror** that is file-serializable, so per-object
      ``history`` and point-in-time ``as_of`` reconstruction work identically
      whether or not a backend is reachable.

    Args:
        graph: Optional live :class:`KnowledgeGraph` facade. When omitted, one
            is probed lazily; if no store is reachable the ledger runs in the
            in-memory + file-serializable mode.
        version_engine: Optional shared :class:`KGVersionEngine` (so the ledger
            participates in an existing commit chain). A fresh one is created
            when omitted.
        audit: Optional shared :class:`AuditLogger`. A fresh one is created when
            omitted.
        graph_state: Mutable graph_state dict (``{"nodes": ..., "edges": ...}``)
            the version engine mutates. A fresh empty state is created when
            omitted.
    """

    def __init__(
        self,
        graph: Any = None,
        *,
        version_engine: KGVersionEngine | None = None,
        audit: AuditLogger | None = None,
        graph_state: dict[str, Any] | None = None,
    ) -> None:
        self._graph = graph
        self.version_engine = version_engine or KGVersionEngine()
        self.audit = audit or AuditLogger()
        self.graph_state: dict[str, Any] = graph_state or {"nodes": {}, "edges": []}
        # In-memory durable mirror keyed by object_id, insertion-ordered.
        self._edits: list[Edit] = []
        # Optional write-back router; when attached, every recorded edit is
        # pushed to the registered source datasources (CONCEPT:KG-2.43).
        self._writeback: Any = None

    def attach_writeback(self, router: Any) -> None:
        """Attach a :class:`WriteBackRouter` so recorded edits write back live.

        Wires the write-back leg into the hot ``record`` path: once attached,
        every committed edit is fanned out to the router's registered sinks
        (the source 'edits datasource'), not just mirrored/persisted.
        """
        self._writeback = router

    # ── Apply / record ────────────────────────────────────────────────────────
    def record(self, edit: Edit) -> Edit:
        """Apply an edit to the versioned graph_state and durably persist it.

        The forward mutation is committed via :class:`KGVersionEngine` (giving us
        rollback data), the edit is mirrored in memory, persisted as a KG node,
        and audited. Returns the same ``Edit`` (with ``persisted`` updated).
        """
        tx = KGTransaction(
            description=f"{edit.edit_type} on {edit.object_id} by {edit.actor}"
        )
        tx.mutations.append(edit.to_mutation())
        self.version_engine.commit(tx, self.graph_state)

        self._edits.append(edit)
        self._persist(edit)
        try:
            self.audit.log(
                actor=edit.actor,
                action=ACTION_OBJECT_EDIT,
                resource_type=RESOURCE_OBJECT_EDIT,
                resource_id=edit.object_id,
                details={
                    "edit_id": edit.id,
                    "edit_type": str(edit.edit_type),
                    "invocation_ref": edit.invocation_ref,
                    "provenance": edit.provenance,
                },
            )
        except Exception as exc:  # noqa: BLE001 — audit never blocks the write
            logger.debug("EditLedger: audit failed for %s: %s", edit.id, exc)
        if self._writeback is not None:
            try:
                self._writeback.write_back(edit)
            except Exception as exc:  # noqa: BLE001 — write-back never blocks
                logger.debug("EditLedger: write-back failed for %s: %s", edit.id, exc)
        return edit

    # ── Convenience constructors that capture before-snapshots from live state ─
    def set_property(
        self,
        object_id: str,
        properties: dict[str, Any],
        *,
        actor: str = "system",
        provenance: str = "",
        invocation_ref: str = "",
    ) -> Edit:
        """Record a PROPERTY_SET edit, snapshotting the prior values it overwrites."""
        current = self.graph_state.get("nodes", {}).get(object_id, {})
        before = {k: deepcopy(current.get(k)) for k in properties}
        edit = Edit(
            actor=actor,
            edit_type=EditType.PROPERTY_SET,
            object_id=object_id,
            before=before,
            after=dict(properties),
            provenance=provenance,
            invocation_ref=invocation_ref,
        )
        return self.record(edit)

    def create_object(
        self,
        object_id: str,
        properties: dict[str, Any] | None = None,
        *,
        actor: str = "system",
        provenance: str = "",
        invocation_ref: str = "",
    ) -> Edit:
        """Record an OBJECT_CREATE edit."""
        edit = Edit(
            actor=actor,
            edit_type=EditType.OBJECT_CREATE,
            object_id=object_id,
            after=dict(properties or {}),
            provenance=provenance,
            invocation_ref=invocation_ref,
        )
        return self.record(edit)

    def delete_object(
        self,
        object_id: str,
        *,
        actor: str = "system",
        provenance: str = "",
        invocation_ref: str = "",
    ) -> Edit:
        """Record an OBJECT_DELETE edit, snapshotting the object being removed."""
        before = deepcopy(self.graph_state.get("nodes", {}).get(object_id, {}))
        edit = Edit(
            actor=actor,
            edit_type=EditType.OBJECT_DELETE,
            object_id=object_id,
            before=before,
            provenance=provenance,
            invocation_ref=invocation_ref,
        )
        return self.record(edit)

    def add_link(
        self,
        source: str,
        target: str,
        label: str = "related",
        *,
        actor: str = "system",
        provenance: str = "",
        invocation_ref: str = "",
    ) -> Edit:
        """Record a LINK_ADD edit. The edit is keyed to the link source object."""
        edit = Edit(
            actor=actor,
            edit_type=EditType.LINK_ADD,
            object_id=source,
            link_source=source,
            link_label=label,
            link_target=target,
            provenance=provenance,
            invocation_ref=invocation_ref,
        )
        return self.record(edit)

    def remove_link(
        self,
        source: str,
        target: str,
        label: str = "related",
        *,
        actor: str = "system",
        provenance: str = "",
        invocation_ref: str = "",
    ) -> Edit:
        """Record a LINK_REMOVE edit. The edit is keyed to the link source object."""
        edit = Edit(
            actor=actor,
            edit_type=EditType.LINK_REMOVE,
            object_id=source,
            link_source=source,
            link_label=label,
            link_target=target,
            provenance=provenance,
            invocation_ref=invocation_ref,
        )
        return self.record(edit)

    # ── History / point-in-time reconstruction ────────────────────────────────
    def history(self, object_id: str) -> list[Edit]:
        """Return this object's edits in applied order (oldest first).

        CONCEPT:KG-2.43 — the per-object *edit history / audit trail*.
        """
        edits = [e for e in self._edits if e.object_id == object_id]
        return sorted(edits, key=lambda e: e.timestamp)

    def as_of(self, object_id: str, timestamp: float) -> dict[str, Any] | None:
        """Reconstruct an object's property snapshot as of ``timestamp``.

        Replays this object's PROPERTY_SET / OBJECT_CREATE / OBJECT_DELETE edits
        in order up to and including ``timestamp``, applying each edit's ``after``
        (and removing on delete). Returns the reconstructed property dict, or
        ``None`` if the object did not exist / was deleted at that time.

        This is real reconstruction from before/after snapshots, not a live read.
        """
        snapshot: dict[str, Any] | None = None
        for edit in self.history(object_id):
            if edit.timestamp > timestamp:
                break
            if edit.edit_type == EditType.OBJECT_CREATE:
                snapshot = dict(edit.after)
            elif edit.edit_type == EditType.OBJECT_DELETE:
                snapshot = None
            elif edit.edit_type == EditType.PROPERTY_SET:
                if snapshot is None:
                    snapshot = {}
                snapshot.update(edit.after)
            # LINK_* edits do not alter the property snapshot.
        return snapshot

    def all_edits(self) -> list[Edit]:
        """Every recorded edit across all objects, in applied order."""
        return list(self._edits)

    def rehydrate(self, edits: list[Edit] | Edit) -> list[Edit]:
        """Restore durable edits into the in-memory index without re-applying them.

        The in-process mirror (``_edits``) does not survive across stateless
        callers (e.g. separate HTTP workers), so an edit that was persisted as a
        durable ``object_edit`` node by one process is invisible to
        :meth:`history` / :meth:`get` / revert in another until it is loaded
        back. This is the **public** restore path for that: it indexes already-
        durable :class:`Edit` records into the mirror so ``history``/``as_of``/
        revert see them, **without** committing a forward mutation through the
        version engine (the change was already applied when first recorded) and
        without re-persisting/re-auditing.

        Idempotent: an edit whose ``id`` is already indexed is skipped, so
        repeated rehydration of the same durable set is safe. Returns the edits
        that were newly added to the mirror.
        """
        if isinstance(edits, Edit):
            edits = [edits]
        known = {e.id for e in self._edits}
        added: list[Edit] = []
        for edit in edits:
            if edit.id in known:
                continue
            self._edits.append(edit)
            known.add(edit.id)
            added.append(edit)
        return added

    def get(self, edit_id: str) -> Edit | None:
        """Look up a single recorded edit by id."""
        for e in self._edits:
            if e.id == edit_id:
                return e
        return None

    # ── Durable persistence (KG node) ──────────────────────────────────────────
    def _persist(self, edit: Edit) -> None:
        """Persist an edit as a durable ``object_edit`` KG node + edges.

        Materialises the edit through the live store with ``EDITED_BY`` → actor
        and ``EDITS`` → target-object relationships. Best-effort: a missing/
        unreachable backend leaves ``edit.persisted = False`` and the in-memory
        mirror remains authoritative.
        """
        store = _probe_store(self._graph)
        if store is None:
            return
        try:
            store.execute(
                "MERGE (n {id: $id}) SET n.type = $node_type, "
                "n.edit_type = $edit_type, n.actor = $actor, "
                "n.object_id = $object_id, n.provenance = $provenance, "
                "n.invocation_ref = $invocation_ref, n.timestamp = $timestamp, "
                "n.before = $before, n.after = $after, "
                "n.link_source = $link_source, n.link_label = $link_label, "
                "n.link_target = $link_target",
                {
                    "id": edit.id,
                    "node_type": EDIT_NODE_TYPE,
                    "edit_type": str(edit.edit_type),
                    "actor": edit.actor,
                    "object_id": edit.object_id,
                    "provenance": edit.provenance,
                    "invocation_ref": edit.invocation_ref,
                    "timestamp": edit.timestamp,
                    "before": json.dumps(edit.before),
                    "after": json.dumps(edit.after),
                    "link_source": edit.link_source,
                    "link_label": edit.link_label,
                    "link_target": edit.link_target,
                },
            )
            store.execute(
                "MATCH (n {id: $id}) MERGE (a {id: $actor}) "
                "MERGE (n)-[:EDITED_BY]->(a)",
                {"id": edit.id, "actor": edit.actor},
            )
            store.execute(
                "MATCH (n {id: $id}) MERGE (t {id: $object_id}) "
                "MERGE (n)-[:EDITS]->(t)",
                {"id": edit.id, "object_id": edit.object_id},
            )
            edit.persisted = True
        except Exception as exc:  # noqa: BLE001 — persistence is best-effort
            logger.debug("EditLedger: failed to persist edit %s: %s", edit.id, exc)

    # ── File serialization (degraded-mode durability) ──────────────────────────
    def save(self, path: str | os.PathLike[str]) -> int:
        """Serialize the in-memory ledger to a JSON file. Returns edits written."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "edits": [e.model_dump() for e in self._edits],
            "graph_state": _jsonable_state(self.graph_state),
        }
        p.write_text(json.dumps(payload, default=str), encoding="utf-8")
        return len(self._edits)

    @classmethod
    def load(
        cls,
        path: str | os.PathLike[str],
        graph: Any = None,
    ) -> EditLedger:
        """Load a ledger previously written by :meth:`save`.

        The graph_state is restored too, so ``as_of``/``history`` and further
        revert operations are immediately consistent with the persisted edits.
        """
        p = Path(path)
        payload = json.loads(p.read_text(encoding="utf-8"))
        state = payload.get("graph_state") or {"nodes": {}, "edges": []}
        # Edges round-trip through JSON as lists; restore tuple identity.
        state["edges"] = [tuple(e) for e in state.get("edges", [])]
        ledger = cls(graph=graph, graph_state=state)
        ledger._edits = [Edit(**raw) for raw in payload.get("edits", [])]
        return ledger


def _jsonable_state(state: dict[str, Any]) -> dict[str, Any]:
    """Render a graph_state dict (which holds edge tuples) as JSON-safe data."""
    return {
        "nodes": deepcopy(state.get("nodes", {})),
        "edges": [list(e) for e in state.get("edges", [])],
    }
