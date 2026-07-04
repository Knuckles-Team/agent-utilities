#!/usr/bin/python
from __future__ import annotations

"""Tests for the durable Edit Ledger (CONCEPT:AU-KG.ontology.edit-ledger-writeback).

Provenance: Palantir Foundry *object-edits/overview* — structured edits with
before/after snapshots, per-object history, undo/revert, write-back to source.

Self-contained against stable code (kg_versioning, audit_logger): no live
backend is required; the ledger degrades to its in-memory + file-serializable
durable mode, which is exactly what these assertions exercise.
"""

import time

from agent_utilities.knowledge_graph.ontology.edits import (
    Edit,
    EditLedger,
    EditType,
    JsonlEditSink,
    WriteBackRouter,
    object_type_of,
    revert_edit,
    revert_edits,
)


def _new_ledger() -> EditLedger:
    # No graph facade => degraded durable mode (in-memory + file-serializable).
    return EditLedger(graph=None)


def test_property_edit_and_link_add_produce_durable_records() -> None:
    ledger = _new_ledger()
    ledger.create_object("paper:001", {"type": "paper", "title": "Initial"})
    prop_edit = ledger.set_property(
        "paper:001", {"title": "Revised", "score": 11.2}, actor="alice"
    )
    link_edit = ledger.add_link(
        "paper:001", "concept:KG-2.43", label="enhances", actor="alice"
    )

    # Durable ledger records exist for both.
    assert ledger.get(prop_edit.id) is prop_edit
    assert ledger.get(link_edit.id) is link_edit
    assert prop_edit.edit_type == EditType.PROPERTY_SET
    assert link_edit.edit_type == EditType.LINK_ADD

    # Forward edits were applied to the versioned graph_state.
    node = ledger.graph_state["nodes"]["paper:001"]
    assert node["title"] == "Revised"
    assert node["score"] == 11.2
    assert (
        "paper:001",
        "concept:KG-2.43",
        "enhances",
    ) in ledger.graph_state["edges"]

    # Before-snapshot captured the value the property edit overwrote.
    assert prop_edit.before == {"title": "Initial", "score": None}


def test_per_object_history_is_ordered() -> None:
    ledger = _new_ledger()
    ledger.create_object("user:7", {"type": "user", "name": "a"})
    time.sleep(0.001)
    second = ledger.set_property("user:7", {"name": "b"}, actor="bob")
    time.sleep(0.001)
    third = ledger.set_property("user:7", {"name": "c"}, actor="bob")

    history = ledger.history("user:7")
    assert [e.edit_type for e in history] == [
        EditType.OBJECT_CREATE,
        EditType.PROPERTY_SET,
        EditType.PROPERTY_SET,
    ]
    # Oldest first, monotonic timestamps.
    assert history[-1].id == third.id
    assert history[1].id == second.id
    assert history[0].timestamp <= history[1].timestamp <= history[2].timestamp


def test_revert_restores_prior_state() -> None:
    ledger = _new_ledger()
    ledger.create_object("doc:1", {"type": "doc", "status": "draft"})
    edit = ledger.set_property("doc:1", {"status": "published"}, actor="carol")
    assert ledger.graph_state["nodes"]["doc:1"]["status"] == "published"

    comp = revert_edit(ledger, edit.id, actor="carol")
    # Inverse applied: prior value restored.
    assert ledger.graph_state["nodes"]["doc:1"]["status"] == "draft"
    # Reversal is itself a durable, append-only compensating edit.
    assert comp.provenance == f"revert of {edit.id}"
    assert comp.edit_type == EditType.PROPERTY_SET
    assert ledger.get(comp.id) is comp


def test_revert_link_add_removes_edge() -> None:
    ledger = _new_ledger()
    ledger.create_object("paper:9", {"type": "paper"})
    link = ledger.add_link("paper:9", "concept:X", label="cites")
    triple = ("paper:9", "concept:X", "cites")
    assert triple in ledger.graph_state["edges"]

    revert_edit(ledger, link.id)
    assert triple not in ledger.graph_state["edges"]


def test_revert_set_unwinds_newest_first() -> None:
    ledger = _new_ledger()
    ledger.create_object("acct:1", {"type": "acct", "balance": 0})
    e1 = ledger.set_property("acct:1", {"balance": 100})
    time.sleep(0.001)
    e2 = ledger.set_property("acct:1", {"balance": 250})
    assert ledger.graph_state["nodes"]["acct:1"]["balance"] == 250

    revert_edits(ledger, [e1.id, e2.id])
    # Both reverted, newest-first, leaving the original create value.
    assert ledger.graph_state["nodes"]["acct:1"]["balance"] == 0


def test_as_of_returns_pre_edit_snapshot() -> None:
    ledger = _new_ledger()
    ledger.create_object("paper:42", {"type": "paper", "title": "v1"})
    time.sleep(0.002)
    checkpoint = time.time()
    time.sleep(0.002)
    ledger.set_property("paper:42", {"title": "v2"})

    # As of the checkpoint (before the v2 edit) the title is still v1.
    snap = ledger.as_of("paper:42", checkpoint)
    assert snap is not None
    assert snap["title"] == "v1"

    # As of now, it is v2.
    now_snap = ledger.as_of("paper:42", time.time())
    assert now_snap is not None
    assert now_snap["title"] == "v2"


def test_writeback_emits_a_record(tmp_path) -> None:
    sink = JsonlEditSink(root=tmp_path)
    router = WriteBackRouter()
    router.register_sink("source", sink)

    ledger = _new_ledger()
    ledger.attach_writeback(router)

    edit = ledger.create_object("paper:77", {"type": "paper", "title": "WB"})
    # Live write-back leg fired on record(): a JSONL record was emitted, keyed
    # by object type.
    records = sink.read_all("paper")
    assert len(records) == 1
    assert records[0]["id"] == edit.id
    assert records[0]["object_id"] == "paper:77"
    assert object_type_of(edit) == "paper"

    # The on-disk file is the append-only per-type datasource.
    assert sink.path_for("paper").exists()


def test_object_type_falls_back_to_id_namespace() -> None:
    edit = Edit(edit_type=EditType.LINK_ADD, object_id="widget:5")
    assert object_type_of(edit) == "widget"


def test_save_and_load_round_trip(tmp_path) -> None:
    ledger = _new_ledger()
    ledger.create_object("k:1", {"type": "k", "v": 1})
    ledger.set_property("k:1", {"v": 2})
    path = tmp_path / "ledger.json"
    assert ledger.save(path) == 2

    restored = EditLedger.load(path)
    assert len(restored.all_edits()) == 2
    assert restored.graph_state["nodes"]["k:1"]["v"] == 2
    # History reconstruction works after load.
    assert len(restored.history("k:1")) == 2


def test_rehydrate_restores_durable_edits_into_index() -> None:
    # An edit recorded by one ledger (e.g. a different HTTP worker)...
    source = _new_ledger()
    recorded = source.set_property("obj:1", {"status": "active"}, actor="alice")

    # ...is invisible to a fresh ledger's index until rehydrated.
    fresh = _new_ledger()
    assert fresh.get(recorded.id) is None
    assert fresh.history("obj:1") == []

    added = fresh.rehydrate([recorded])
    assert [e.id for e in added] == [recorded.id]
    assert fresh.get(recorded.id) is recorded
    assert [e.id for e in fresh.history("obj:1")] == [recorded.id]


def test_rehydrate_is_idempotent_and_accepts_single_edit() -> None:
    ledger = _new_ledger()
    edit = Edit(edit_type=EditType.PROPERTY_SET, object_id="obj:2", after={"v": 1})

    # Accepts a single Edit (not just a list).
    first = ledger.rehydrate(edit)
    assert [e.id for e in first] == [edit.id]
    # Re-rehydrating the same edit is a no-op (no duplicate in the mirror).
    assert ledger.rehydrate(edit) == []
    assert ledger.rehydrate([edit]) == []
    assert len([e for e in ledger.all_edits() if e.id == edit.id]) == 1
