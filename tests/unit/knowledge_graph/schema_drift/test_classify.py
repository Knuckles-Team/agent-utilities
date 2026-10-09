"""AU-SEC requirement 004: the deterministic drift classifier and the declared evolution policy."""

from __future__ import annotations

import pytest

from agent_utilities.knowledge_graph.schema_drift import (
    DriftClass,
    FieldShape,
    RecordShape,
    classify,
    declared_policy,
    infer_shape,
    parse_policy,
    shape_digest,
)
from agent_utilities.knowledge_graph.schema_drift.classify import is_compatible
from agent_utilities.knowledge_graph.schema_drift.policy import ContractPolicyError

APPROVED = infer_shape(
    [
        {"id": "a", "name": "web", "replicas": 2, "image": "x"},
        {"id": "b", "name": "db", "replicas": 1, "image": None},
    ]
)


def kinds(approved: RecordShape, records: list[dict], **kw) -> dict[str, DriftClass]:
    return {c.field: c.kind for c in classify(approved, infer_shape(records), **kw)}


@pytest.mark.spec("AU-SEC-R004")
def test_an_unchanged_delta_has_no_drift_and_a_stable_digest() -> None:
    same = infer_shape([{"id": "c", "name": "q", "replicas": 3, "image": "y"}])
    assert classify(APPROVED, same) == ()
    assert shape_digest(APPROVED) == shape_digest(
        RecordShape.from_json(APPROVED.to_json())
    )


def test_the_six_classes_are_named_from_the_readers_side() -> None:
    records = [
        {
            "id": "a",
            "title": "web",
            "replicas": 2.5,
            "image": "x",
            "zone": 1,
            "note": None,
        },
        {"id": "b", "title": "db", "replicas": 1, "image": "y", "zone": 2},
    ]
    assert kinds(APPROVED, records) == {
        "title": DriftClass.RENAME_CANDIDATE,  # name(string, required) -> title
        "replicas": DriftClass.TYPE_WIDEN,  # integer -> number
        "zone": DriftClass.ADDITIVE_REQUIRED,
        "note": DriftClass.ADDITIVE_NULLABLE,
    }
    declared = RecordShape.of(
        {
            "id": FieldShape(frozenset({"string"}), True),
            "image": FieldShape(frozenset({"string"}), True),
        }
    )
    assert kinds(APPROVED, [], sampled=False) == {
        name: DriftClass.REMOVAL for name in ("id", "image", "name", "replicas")
    }
    narrowed = classify(APPROVED, declared, sampled=False)
    assert {c.field: c.kind for c in narrowed}["image"] is DriftClass.TYPE_NARROW


def test_an_ambiguous_rename_is_two_removals_and_additions_not_a_guess() -> None:
    approved = infer_shape([{"id": "a", "first": "x", "second": "y"}])
    changed = kinds(approved, [{"id": "a", "one": "x", "two": "y"}])
    assert changed == {
        "first": DriftClass.REMOVAL,
        "second": DriftClass.REMOVAL,
        "one": DriftClass.ADDITIVE_REQUIRED,
        "two": DriftClass.ADDITIVE_REQUIRED,
    }


def test_absence_in_a_sample_is_not_a_removal_of_an_optional_field() -> None:
    approved = infer_shape([{"id": "a", "opt": 1}, {"id": "b"}])
    assert classify(approved, infer_shape([{"id": "c"}])) == ()
    assert kinds(approved, [{"id": "c"}], sampled=False) == {"opt": DriftClass.REMOVAL}


def test_null_and_absence_widen_a_required_field() -> None:
    changes = classify(
        APPROVED, infer_shape([{"id": None, "name": "n", "replicas": 1, "image": "x"}])
    )
    widened = {c.field: c.detail for c in changes if c.kind is DriftClass.TYPE_WIDEN}
    assert widened == {"id": "null"}
    partial = classify(
        APPROVED,
        infer_shape(
            [
                {"id": "a", "name": "n", "replicas": 1, "image": "x"},
                {"id": "b", "replicas": 2, "image": "y"},
            ]
        ),
    )
    assert {c.field: c.detail for c in partial} == {"name": "absent"}


def test_classification_is_order_independent() -> None:
    records = [{"id": "a", "x": 1}, {"id": "b", "y": "s", "x": None}]
    assert classify(APPROVED, infer_shape(records)) == classify(
        APPROVED, infer_shape(list(reversed(records)))
    )


def test_only_compatible_classes_may_be_declared_continuable() -> None:
    policy = parse_policy(
        "Container-Manager-MCP=additive_nullable+additive_required, systems-manager=type_narrow"
    )
    additive = classify(
        APPROVED,
        infer_shape(
            [{"id": "a", "name": "n", "replicas": 1, "image": "i", "zone": "z"}]
        ),
    )
    assert [c.kind for c in additive] == [DriftClass.ADDITIVE_REQUIRED]
    assert is_compatible(additive)
    assert policy.allows("container-manager-mcp", additive)
    assert not policy.allows("systems-manager", additive)
    for breaking in ("rename_candidate", "type_widen", "removal", "whatever"):
        with pytest.raises(ContractPolicyError):
            parse_policy(f"src={breaking}")
    with pytest.raises(ContractPolicyError):
        parse_policy("src=additive_nullable,src=additive_required")


def test_an_invalid_declaration_continues_nothing_and_says_why() -> None:
    policy = declared_policy("src=additive_nullable,other=removal")
    assert "removal" in policy.refusal
    assert policy.auto_continue == {}
