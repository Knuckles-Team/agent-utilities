"""NE-036 (au half) acceptance — native WorkItem metadata CAS (`93c139ac`).

BUG-111's fix replaced checkpoint/input/priority scheduling-metadata CAS with
the engine-native ``CasWorkItemMetadata`` RPC
(:func:`agent_utilities.orchestration.work_item._cas_work_item_metadata`),
because the OLD generic path
(:func:`agent_utilities.orchestration.work_item._cas`, which calls the
engine's generic ``compare_and_set_node_fields``) is unconditionally refused
by epistemic-graph's native-WorkItem-authority guard
(``work_item_capability::validate_generic_method``, RMDD-29) the moment the
row is claimed.

The existing suite (``tests/unit/orchestration/test_work_item.py``) proves the
NEW path works (CAS success, deterministic conflict, native_calls contains
"cas_metadata", ...), but nothing proves the OLD collision path is actually
*unreachable* for these four call sites -- a regression that silently
resurrected a call to the generic CAS would only be caught the moment it hit
a real engine and RMDD-29 refused it, not by this in-memory suite.

These tests use a ``NativeEngine`` subclass whose ``compare_and_set_node_fields``
hard-fails the test if it is ever invoked with a scheduling-metadata field
(``checkpoint_id``/``metadata``/``prio_bucket``) -- the exact three fields
``_cas_work_item_metadata`` owns. Submission's own use of ``_cas`` for the
``downstream_ids`` reverse-index (an unrelated, still-legitimate use of the
generic path -- see ``_index_downstream``'s docstring) is deliberately left
alone so this stays a precise regression guard, not a blanket ban.
"""

from __future__ import annotations

import pytest

import agent_utilities.orchestration.work_item as wi
from tests.unit.orchestration.test_work_item import NativeEngine

_METADATA_CAS_FIELDS = {"checkpoint_id", "metadata", "prio_bucket"}


class _NoLegacyMetadataCasEngine(NativeEngine):
    """``NativeEngine``, but any CAS write touching a scheduling-metadata
    field through the GENERIC ``compare_and_set_node_fields`` is a hard test
    failure -- proving checkpoint/input/priority CAS route exclusively
    through the native ``CasWorkItemMetadata`` RPC (``cas_work_item_metadata``
    below), never the pre-BUG-111 metadata-flattening collision path.
    """

    def compare_and_set_node_fields(
        self, node_id: str, conditions: dict, updates: dict
    ) -> bool:
        touched = _METADATA_CAS_FIELDS & set(updates)
        if touched:
            raise AssertionError(
                f"generic compare_and_set_node_fields was called with "
                f"scheduling-metadata field(s) {sorted(touched)} on {node_id!r} "
                "-- the former metadata-flattening collision path (BUG-111) "
                "must be unreachable for checkpoint/input/priority CAS"
            )
        return super().compare_and_set_node_fields(node_id, conditions, updates)


@pytest.fixture
def engine() -> _NoLegacyMetadataCasEngine:
    return _NoLegacyMetadataCasEngine()


def test_checkpoint_never_falls_through_to_the_generic_cas_collision_path(
    engine: _NoLegacyMetadataCasEngine,
) -> None:
    item_id = wi.submit_work_item(
        engine, kind="goal_loop", payload_ref="loop:opaque", tenant="tenant-a"
    )
    claim = wi.claim_and_start(engine, item_id, token="worker", now=10.0)
    assert claim is not None
    assert wi.checkpoint_work_item(engine, item_id, claim, "checkpoint:1", now=11.0)
    assert wi.get_work_item(engine, item_id)["checkpoint_id"] == "checkpoint:1"
    assert "cas_metadata" in engine.native_calls


def test_input_round_trip_never_falls_through_to_the_generic_cas_collision_path(
    engine: _NoLegacyMetadataCasEngine,
) -> None:
    item_id = wi.submit_work_item(
        engine, kind="generic", payload_ref="p", tenant="tenant-a"
    )
    claim = wi.claim_and_start(engine, item_id, token="worker", now=10.0)
    assert claim is not None
    assert wi.request_work_item_input(
        engine, item_id, claim, request={"prompt": "confirm?"}, now=11.0
    )
    assert wi.submit_work_item_input(
        engine, item_id, tenant="tenant-a", response={"confirmed": True}, now=12.0
    )
    item = wi.get_work_item(engine, item_id)
    assert item["metadata"]["pending_input_response"] == {"confirmed": True}
    assert engine.native_calls.count("cas_metadata") >= 2


def test_set_priority_never_falls_through_to_the_generic_cas_collision_path(
    engine: _NoLegacyMetadataCasEngine,
) -> None:
    item_id = wi.submit_work_item(
        engine, kind="generic", payload_ref="p", tenant="tenant-a"
    )
    assert wi.set_work_item_priority(engine, item_id, 3, now=10.0)
    assert wi.get_work_item(engine, item_id)["prio_bucket"] == 3
    assert "cas_metadata" in engine.native_calls


def test_submission_dependency_indexing_still_legitimately_uses_the_generic_cas(
    engine: _NoLegacyMetadataCasEngine,
) -> None:
    """Sanity check that the guard above is precise, not a blanket ban: the
    downstream-index write (``downstream_ids``, an unrelated field) is a
    legitimate, still-current use of the generic path and must keep working.
    """
    parent = wi.submit_work_item(
        engine, kind="generic", payload_ref="parent", tenant="tenant-a"
    )
    child = wi.submit_work_item(
        engine,
        kind="generic",
        payload_ref="child",
        tenant="tenant-a",
        depends_on=[parent],
    )
    assert wi.get_work_item(engine, parent)["downstream_ids"] == [child]
