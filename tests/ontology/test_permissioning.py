#!/usr/bin/python
from __future__ import annotations

"""Tests for fine-grained object permissioning (CONCEPT:KG-2.46).

Self-contained against the stable company-brain fabric; no Wave-1 ontology
primitive is imported, so this passes standalone.
"""

import pytest

from agent_utilities.knowledge_graph.core.company_brain_runtime import (
    reset_company_brain,
)
from agent_utilities.knowledge_graph.ontology.permissioning import (
    MASK_TOKEN,
    Marking,
    apply_marking,
    build_acl,
    clear_markings,
    enforce,
    markings_for,
    propagate_markings,
    redact_object,
    restricted_view,
)
from agent_utilities.models.company_brain import ActorType, DataClassification
from agent_utilities.security.brain_context import ActorContext


@pytest.fixture(autouse=True)
def _clean_state():
    reset_company_brain()
    clear_markings()
    yield
    reset_company_brain()
    clear_markings()


def _low_actor() -> ActorContext:
    # Authenticated but only clears INTERNAL; holds no markings.
    return ActorContext(actor_id="analyst:1", actor_type=ActorType.HUMAN, roles=("analyst",))


def _cleared_actor() -> ActorContext:
    # Holds the "pii" marking and confidential clearance.
    return ActorContext(
        actor_id="owner:1",
        actor_type=ActorType.HUMAN,
        roles=("confidential", "marking:pii"),
    )


# --- property/column-level redaction for a low-clearance actor --------------


def test_property_level_redaction_drops_unreadable_columns():
    obj = {
        "id": "person:1",
        "name": "Ada",
        "salary": 200000,
        "ssn": "123-45-6789",
        "__classification__": {"salary": "confidential"},
        "__markings__": {"ssn": ["pii"]},
    }
    redacted = redact_object(obj, _low_actor())

    assert redacted["id"] == "person:1"
    assert redacted["name"] == "Ada"  # public property survives
    assert "salary" not in redacted  # above INTERNAL clearance -> dropped
    assert "ssn" not in redacted  # carries pii marking actor lacks -> dropped
    # metadata side-channels are stripped from the materialized view
    assert "__classification__" not in redacted
    assert "__markings__" not in redacted
    # input is not mutated
    assert obj["salary"] == 200000


def test_property_level_mask_preserves_shape():
    obj = {"id": "p2", "name": "Bo", "salary": 90000, "__classification__": {"salary": "confidential"}}
    masked = redact_object(obj, _low_actor(), mask=True)
    assert masked["salary"] == MASK_TOKEN
    assert masked["name"] == "Bo"


def test_cleared_actor_sees_marked_property():
    obj = {"id": "p3", "ssn": "x", "__markings__": {"ssn": ["pii"]}}
    out = redact_object(obj, _cleared_actor())
    assert out["ssn"] == "x"


# --- marking propagation across an edge -------------------------------------


def test_marking_propagates_across_edge():
    apply_marking("parent", Marking("pii"))
    assert markings_for("child") == set()

    propagate_markings("parent", "child")

    assert "pii" in markings_for("child")


def test_classification_propagates_to_most_restrictive():
    build_acl("src", DataClassification.RESTRICTED)
    build_acl("dst", DataClassification.PUBLIC)

    propagate_markings("src", "dst")

    from agent_utilities.knowledge_graph.core.company_brain_runtime import (
        get_company_brain,
    )

    acl = get_company_brain().permissions.get_acl("dst")
    assert acl is not None
    assert acl.classification == DataClassification.RESTRICTED


# --- row + column restricted view -------------------------------------------


def test_restricted_view_filters_rows_and_columns():
    # public row passes through; restricted row dropped by row-level gate.
    apply_marking("secret-row", Marking("pii"))
    objects = [
        {
            "id": "public-row",
            "title": "Quarterly Summary",
            "internal_note": "draft",
            "__classification__": {"internal_note": "confidential"},
        },
        {"id": "secret-row", "title": "Layoff Plan"},
    ]

    view = restricted_view(objects, _low_actor())

    # row-level: marked row removed
    assert [o["id"] for o in view] == ["public-row"]
    # column-level: confidential note redacted on the surviving row
    assert view[0]["title"] == "Quarterly Summary"
    assert "internal_note" not in view[0]


# --- marked RESTRICTED node filtered while public passes (default-on) -------


def test_enforce_default_on_filters_marked_node_keeps_public():
    apply_marking("restricted:1", Marking("pii", requires_audit=True))
    objects = [
        {"id": "public:1", "v": 1},
        {"id": "restricted:1", "v": 2},
    ]

    # No KG_BRAIN_ENFORCE set — enforce() is mandatory-control driven, so the
    # marked node is filtered for a low actor while the unmarked one passes.
    result = enforce(objects, _low_actor())
    assert [o["id"] for o in result] == ["public:1"]

    # A cleared actor sees both.
    result2 = enforce(objects, _cleared_actor())
    assert {o["id"] for o in result2} == {"public:1", "restricted:1"}


def test_enforce_passes_unmarked_data_unchanged():
    objects = [{"id": "a", "x": 1}, {"id": "b", "x": 2}]
    out = enforce(objects, _low_actor())
    assert out == objects


def test_acl_denying_node_filtered_by_enforce():
    # Discretionary ACL with a role the low actor lacks.
    build_acl("conf:1", DataClassification.CONFIDENTIAL, read_roles=["manager"])
    objects = [{"id": "conf:1", "x": 1}, {"id": "open:1", "x": 2}]
    out = enforce(objects, _low_actor())
    assert [o["id"] for o in out] == ["open:1"]
