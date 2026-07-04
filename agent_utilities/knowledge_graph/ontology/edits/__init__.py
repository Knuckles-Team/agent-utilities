#!/usr/bin/python
from __future__ import annotations

"""Durable Edit Ledger package — Foundry *object-edits* over the epistemic KG.

Provenance (Palantir Foundry doc: *object-edits/overview*): structured edits
(property set, link add/remove, object create/delete) with before/after
snapshots form a per-object audit trail that drives undo/revert and write-back to
the source datasource. This package realises that primitive:

* :mod:`ledger` — :class:`Edit` (durable record) + :class:`EditLedger`
  (KG ``object_edit`` nodes via the live store, with in-memory + file-
  serializable degradation; per-object ``history`` and point-in-time ``as_of``).
* :mod:`revert` — inverse-mutation undo/revert of an edit or an edit-set via
  :class:`KGVersionEngine` semantics, recorded as compensating edits.
* :mod:`writeback` — registerable :class:`EditSink` + append-only JSONL sink
  keyed by object type + :class:`WriteBackRouter`.

CONCEPT:AU-KG.ontology.edit-ledger-writeback
"""

from .ledger import EDIT_NODE_TYPE, Edit, EditLedger, EditType
from .revert import invert_edit, revert_edit, revert_edits, revert_object
from .writeback import (
    EditSink,
    JsonlEditSink,
    WriteBackRouter,
    object_type_of,
)

__all__ = [
    # ledger
    "Edit",
    "EditType",
    "EditLedger",
    "EDIT_NODE_TYPE",
    # revert
    "invert_edit",
    "revert_edit",
    "revert_edits",
    "revert_object",
    # writeback
    "EditSink",
    "JsonlEditSink",
    "WriteBackRouter",
    "object_type_of",
]
