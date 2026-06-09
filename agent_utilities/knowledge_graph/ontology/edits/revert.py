#!/usr/bin/python
from __future__ import annotations

"""Edit revert / undo — inverse-mutation reversal of recorded edits.

Provenance (Palantir Foundry doc: *object-edits/overview*): every edit a user
makes is reversible. Foundry exposes *undo* on a single edit and *revert* over a
set of edits, restoring the object to its prior state. This module realises that
by computing the **inverse mutation** of an :class:`Edit` and applying it through
:class:`KGVersionEngine` (the same commit/rollback engine the ledger writes
forward edits with), then recording the reversal as its own durable
compensating edit so the audit trail stays append-only.

CONCEPT:KG-2.43
"""

import logging
from copy import deepcopy

from agent_utilities.knowledge_graph.core.kg_versioning import (
    KGMutation,
    MutationType,
)

from .ledger import Edit, EditLedger, EditType

logger = logging.getLogger(__name__)

__all__ = ["invert_edit", "revert_edit", "revert_edits"]


def invert_edit(edit: Edit) -> KGMutation:
    """Compute the inverse :class:`KGMutation` that undoes ``edit``.

    Real inverse computation from the edit's before/after snapshot:

    * ``PROPERTY_SET`` → UPDATE_NODE restoring the ``before`` values.
    * ``OBJECT_CREATE`` → DELETE_NODE.
    * ``OBJECT_DELETE`` → ADD_NODE restoring the ``before`` properties.
    * ``LINK_ADD`` → DELETE_EDGE on the same triple.
    * ``LINK_REMOVE`` → ADD_EDGE on the same triple.
    """
    if edit.edit_type == EditType.PROPERTY_SET:
        # Restore the exact prior values of the keys this edit overwrote.
        return KGMutation(
            mutation_type=MutationType.UPDATE_NODE,
            node_id=edit.object_id,
            data=deepcopy(edit.before),
            previous_data=deepcopy(edit.after),
        )
    if edit.edit_type == EditType.OBJECT_CREATE:
        return KGMutation(
            mutation_type=MutationType.DELETE_NODE,
            node_id=edit.object_id,
            previous_data=deepcopy(edit.after),
        )
    if edit.edit_type == EditType.OBJECT_DELETE:
        return KGMutation(
            mutation_type=MutationType.ADD_NODE,
            node_id=edit.object_id,
            data=deepcopy(edit.before),
        )
    if edit.edit_type == EditType.LINK_ADD:
        return KGMutation(
            mutation_type=MutationType.DELETE_EDGE,
            edge_source=edit.link_source,
            edge_target=edit.link_target,
            edge_label=edit.link_label,
        )
    # LINK_REMOVE
    return KGMutation(
        mutation_type=MutationType.ADD_EDGE,
        edge_source=edit.link_source,
        edge_target=edit.link_target,
        edge_label=edit.link_label,
    )


def _compensating_edit(edit: Edit, actor: str) -> Edit:
    """Build the durable compensating edit that records a reversal.

    The reversal is itself an :class:`Edit` (append-only trail): a PROPERTY_SET
    reversal flips before/after; a create-reversal becomes an OBJECT_DELETE; a
    delete-reversal becomes an OBJECT_CREATE; link reversals flip add/remove.
    """
    prov = f"revert of {edit.id}"
    if edit.edit_type == EditType.PROPERTY_SET:
        return Edit(
            actor=actor,
            edit_type=EditType.PROPERTY_SET,
            object_id=edit.object_id,
            before=deepcopy(edit.after),
            after=deepcopy(edit.before),
            provenance=prov,
            invocation_ref=edit.invocation_ref,
        )
    if edit.edit_type == EditType.OBJECT_CREATE:
        return Edit(
            actor=actor,
            edit_type=EditType.OBJECT_DELETE,
            object_id=edit.object_id,
            before=deepcopy(edit.after),
            provenance=prov,
            invocation_ref=edit.invocation_ref,
        )
    if edit.edit_type == EditType.OBJECT_DELETE:
        return Edit(
            actor=actor,
            edit_type=EditType.OBJECT_CREATE,
            object_id=edit.object_id,
            after=deepcopy(edit.before),
            provenance=prov,
            invocation_ref=edit.invocation_ref,
        )
    if edit.edit_type == EditType.LINK_ADD:
        return Edit(
            actor=actor,
            edit_type=EditType.LINK_REMOVE,
            object_id=edit.object_id,
            link_source=edit.link_source,
            link_label=edit.link_label,
            link_target=edit.link_target,
            provenance=prov,
            invocation_ref=edit.invocation_ref,
        )
    # LINK_REMOVE → LINK_ADD
    return Edit(
        actor=actor,
        edit_type=EditType.LINK_ADD,
        object_id=edit.object_id,
        link_source=edit.link_source,
        link_label=edit.link_label,
        link_target=edit.link_target,
        provenance=prov,
        invocation_ref=edit.invocation_ref,
    )


def revert_edit(
    ledger: EditLedger,
    edit_id: str,
    *,
    actor: str = "system",
) -> Edit:
    """Undo a single recorded edit, restoring the object's prior state.

    Applies the inverse mutation through the ledger's :class:`KGVersionEngine`
    against the live graph_state, then records a durable **compensating edit**
    so the ledger stays append-only and the reversal is itself auditable and
    re-reversible.

    Args:
        ledger: The ledger that recorded the edit.
        edit_id: Id of the edit to undo.
        actor: Who is performing the reversal.

    Returns:
        The durable compensating :class:`Edit` recorded for the reversal.

    Raises:
        KeyError: if no edit with ``edit_id`` is in the ledger.
    """
    edit = ledger.get(edit_id)
    if edit is None:
        raise KeyError(f"no edit {edit_id!r} in ledger")
    # ``record`` applies the compensating edit's forward mutation, which IS the
    # inverse of the original — and persists/audits it.
    compensating = _compensating_edit(edit, actor)
    return ledger.record(compensating)


def revert_edits(
    ledger: EditLedger,
    edit_ids: list[str],
    *,
    actor: str = "system",
) -> list[Edit]:
    """Revert a whole edit-set, newest-first, restoring the prior state.

    Edits are inverted in reverse application order so dependent changes unwind
    correctly (e.g. a later property set on an object created earlier in the set
    is undone before the create is undone). Each reversal is a durable
    compensating edit.

    Args:
        ledger: The ledger holding the edits.
        edit_ids: Ids of the edits to revert (any order; sequenced internally).
        actor: Who is performing the reversal.

    Returns:
        The compensating edits, in the order they were applied.
    """
    selected = [ledger.get(eid) for eid in edit_ids]
    missing = [eid for eid, e in zip(edit_ids, selected, strict=False) if e is None]
    if missing:
        raise KeyError(f"edits not in ledger: {missing}")
    ordered: list[Edit] = sorted(
        (e for e in selected if e is not None),
        key=lambda e: e.timestamp,
        reverse=True,
    )
    return [revert_edit(ledger, e.id, actor=actor) for e in ordered]


def revert_object(
    ledger: EditLedger,
    object_id: str,
    *,
    actor: str = "system",
) -> list[Edit]:
    """Revert every edit recorded against one object (full per-object undo).

    A convenience over :func:`revert_edits` that targets an object's whole
    history — the Foundry "revert all my edits to this object" path.
    """
    history = ledger.history(object_id)
    if not history:
        return []
    return revert_edits(ledger, [e.id for e in history], actor=actor)
