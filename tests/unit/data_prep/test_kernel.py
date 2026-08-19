"""Focused NE-108 tests for the bounded Arrow preparation kernel."""

from __future__ import annotations

import json

import pyarrow as pa
import pytest
from pydantic import ConfigDict, ValidationError, create_model

from agent_utilities.data_prep import (
    ArrowAdapter,
    CleanPipeline,
    CleanPlan,
    InvalidRowsError,
    PlanExecutionError,
    PrepEvidence,
    ProfileLimitError,
    RowModelRegistry,
    row_model_digest,
)


def _plan(*steps: dict[str, object], **overrides: object) -> CleanPlan:
    payload: dict[str, object] = {
        "schema_version": "1",
        "steps": list(steps) or [{"verb": "canonical_names"}],
        "profile": {
            "max_rows": 20,
            "max_columns": 8,
            "max_steps": 8,
        },
        "invalid_row_disposition": "fail",
        "plan_ref": "plan:data-prep:v1",
        "policy_ref": "policy:data-prep:v1",
        "model_ref": "model:test:v1",
        "source_ref": "source:test",
        "artifact_ref": "artifact:test",
    }
    payload.update(overrides)
    return CleanPlan.model_validate(payload)


def _row_model(model_name: str, **fields: object) -> type:
    """Build a strict test model with no executable import-path lookup."""

    return create_model(
        model_name,
        __config__=ConfigDict(strict=True, extra="forbid"),
        **{field: (annotation, ...) for field, annotation in fields.items()},
    )


def _registry(row_model: type) -> RowModelRegistry:
    return RowModelRegistry({"model:test:v1": row_model})


def test_plan_gate_is_strict_and_allowlisted() -> None:
    with pytest.raises(ValidationError):
        CleanPipeline(
            {
                "schema_version": "1",
                "steps": [{"verb": "execute_python", "code": "secret"}],
                "profile": {"max_rows": 2, "max_columns": 2, "max_steps": 2},
                "invalid_row_disposition": "fail",
                "policy_ref": "policy:test",
            }
        )
    with pytest.raises(ValidationError):
        CleanPipeline(
            {
                "schema_version": "1",
                "steps": [{"verb": "dedupe", "keys": []}],
                "profile": {"max_rows": 2, "max_columns": 2, "max_steps": 2},
                "invalid_row_disposition": "fail",
                "policy_ref": "policy:test",
            }
        )
    with pytest.raises(ValidationError):
        CleanPipeline(
            {
                "schema_version": "1",
                "steps": [{"verb": "canonical_names", "unexpected": True}],
                "profile": {"max_rows": 2, "max_columns": 2, "max_steps": 2},
                "invalid_row_disposition": "fail",
                "policy_ref": "policy:test",
            }
        )
    with pytest.raises(ValidationError):
        CleanPipeline(
            {
                "schema_version": "1",
                "steps": [{"verb": "canonical_names"}],
                "profile": {"max_rows": "2", "max_columns": 2, "max_steps": 2},
                "invalid_row_disposition": "fail",
                "policy_ref": "policy:test",
            }
        )


def test_invalid_row_disposition_is_required() -> None:
    with pytest.raises(ValidationError):
        CleanPipeline(
            {
                "schema_version": "1",
                "steps": [{"verb": "canonical_names"}],
                "profile": {"max_rows": 2, "max_columns": 2, "max_steps": 2},
                "policy_ref": "policy:test",
            }
        )


@pytest.mark.parametrize(
    ("profile", "table"),
    [
        ({"max_rows": 1, "max_columns": 2, "max_steps": 2}, pa.table({"x": [1, 2]})),
        (
            {"max_rows": 2, "max_columns": 1, "max_steps": 2},
            pa.table({"x": [1, 2], "y": [3, 4]}),
        ),
        (
            {"max_rows": 2, "max_columns": 2, "max_steps": 2, "max_bytes": 1},
            pa.table({"x": [1, 2]}),
        ),
    ],
)
def test_profile_limits_are_enforced_before_execution(
    profile: dict[str, int], table: pa.Table
) -> None:
    plan = _plan(profile=profile)
    with pytest.raises(ProfileLimitError):
        CleanPipeline(
            plan,
            model_registry=_registry(
                _row_model("ProfileRow", **{name: int for name in table.column_names})
            ),
        ).run(table)


def test_lossy_casts_are_denied_and_lossless_widening_uses_safe_cast() -> None:
    narrowing = _plan(
        {
            "verb": "safe_cast",
            "column": "value",
            "source_type": "int64",
            "target_type": "int32",
        }
    )
    with pytest.raises(PlanExecutionError):
        CleanPipeline(
            narrowing,
            model_registry=_registry(_row_model("NarrowingRow", value=int)),
        ).run(pa.table({"value": [1, 2]}))

    cross_family = _plan(
        {
            "verb": "safe_cast",
            "column": "value",
            "source_type": "int64",
            "target_type": "float64",
        }
    )
    with pytest.raises(PlanExecutionError):
        CleanPipeline(
            cross_family,
            model_registry=_registry(_row_model("CrossFamilyRow", value=int)),
        ).run(pa.table({"value": [1, 2]}))

    widening = _plan(
        {
            "verb": "safe_cast",
            "column": "value",
            "source_type": "int8",
            "target_type": "int64",
        }
    )
    result = CleanPipeline(
        widening,
        model_registry=_registry(_row_model("WideningRow", value=int)),
    ).run(pa.table({"value": pa.array([1, 2], type=pa.int8())}))
    assert result.table.schema.field("value").type == pa.int64()
    assert result.evidence.checkpoint_eligible


def test_fill_values_must_fit_the_arrow_type_without_loss() -> None:
    plan = _plan({"verb": "fill_nulls", "fills": {"value": 128}})
    with pytest.raises(PlanExecutionError):
        CleanPipeline(
            plan,
            model_registry=_registry(_row_model("FillRow", value=int)),
        ).run(pa.table({"value": pa.array([None], type=pa.int8())}))


def test_canonical_name_collision_fails_closed() -> None:
    plan = _plan({"verb": "canonical_names"})
    with pytest.raises(PlanExecutionError):
        CleanPipeline(
            plan,
            model_registry=_registry(_row_model("CollisionRow", value=int)),
        ).run(pa.table({"User ID": [1], "user_id": [2]}))


def test_quarantine_is_redacted_and_cannot_claim_checkpoint_success() -> None:
    plan = _plan(
        {"verb": "null_policy", "columns": ["email"], "action": "reject"},
        {"verb": "fill_nulls", "fills": {"email": "super-secret-fill"}},
        invalid_row_disposition="quarantine",
    )
    table = pa.table({"email": ["kept@example.test", None], "value": [1, 2]})
    result = CleanPipeline(
        plan,
        model_registry=_registry(_row_model("QuarantineRow", email=str, value=int)),
    ).run(table)

    assert result.table.num_rows == 1
    assert result.evidence.outcome == "quarantined"
    assert not result.evidence.checkpoint_eligible
    assert result.evidence.quarantined_rows == 1
    evidence_text = json.dumps(result.evidence.model_dump(), sort_keys=True)
    assert "super-secret-fill" not in evidence_text
    assert "kept@example.test" not in evidence_text
    assert "email" not in evidence_text

    with pytest.raises(ValidationError):
        PrepEvidence(
            **{
                **result.evidence.model_dump(),
                "checkpoint_eligible": True,
            }
        )


def test_strict_row_model_gate_emits_redacted_per_row_outcomes() -> None:
    plan = _plan(
        {"verb": "canonical_names"},
        invalid_row_disposition="quarantine",
    )
    table = pa.table({"value": pa.array(["not-an-int", "also-not-an-int"])})
    result = CleanPipeline(
        plan,
        model_registry=_registry(_row_model("StrictIntRow", value=int)),
    ).run(table)

    assert result.table.num_rows == 0
    assert result.evidence.model_rejected_rows == 2
    assert [item.status for item in result.evidence.row_outcomes] == [
        "quarantined",
        "quarantined",
    ]
    evidence_text = json.dumps(result.evidence.model_dump(), sort_keys=True)
    assert "not-an-int" not in evidence_text
    assert "also-not-an-int" not in evidence_text


def test_row_outcomes_retain_original_ordinals_across_quarantine_steps() -> None:
    plan = _plan(
        {"verb": "null_policy", "columns": ["value"], "action": "reject"},
        invalid_row_disposition="quarantine",
    )
    table = pa.table(
        {"value": pa.array(["1", None, "3", "not-an-int"], type=pa.string())}
    )
    result = CleanPipeline(
        plan,
        model_registry=_registry(_row_model("OriginalOrdinalRow", value=int)),
    ).run(table)

    assert [item.row_index for item in result.evidence.row_outcomes] == [0, 1, 2, 3]
    assert [item.reason_code for item in result.evidence.row_outcomes] == [
        "model_rejected",
        "null_rejected",
        "model_rejected",
        "model_rejected",
    ]


def test_row_model_must_be_strict_forbid_and_no_model_is_not_a_success() -> None:
    plan = _plan({"verb": "canonical_names"})
    with pytest.raises(PlanExecutionError):
        CleanPipeline(plan).run(pa.table({"value": [1]}))

    permissive = create_model(
        "PermissiveRow",
        __config__=ConfigDict(strict=False, extra="allow"),
        value=(int, ...),
    )
    with pytest.raises(PlanExecutionError):
        CleanPipeline(
            plan,
            model_registry=RowModelRegistry({"model:test:v1": permissive}),
        ).run(pa.table({"value": [1]}))


def test_model_registry_is_the_only_model_selection_authority() -> None:
    model = _row_model("ApprovedRow", value=int)
    registry = _registry(model)
    plan = _plan({"verb": "canonical_names"})
    result = CleanPipeline(plan, model_registry=registry).run(pa.table({"value": [1]}))
    assert result.evidence.model_ref == "model:test:v1"
    assert result.evidence.model_digest == row_model_digest(model)

    unknown_ref = _plan(
        {"verb": "canonical_names"},
        model_ref="model:unapproved:v1",
    )
    with pytest.raises(PlanExecutionError):
        CleanPipeline(unknown_ref, model_registry=registry).run(
            pa.table({"value": [1]})
        )

    mismatched_digest = _plan(
        {"verb": "canonical_names"},
        model_digest="sha256:" + "0" * 64,
    )
    with pytest.raises(PlanExecutionError):
        CleanPipeline(mismatched_digest, model_registry=registry).run(
            pa.table({"value": [1]})
        )


def test_model_registry_rejects_mutated_model_schema() -> None:
    model = _row_model("MutableRow", value=int)
    registry = _registry(model)
    model.model_fields["value"].annotation = str
    model.model_rebuild(force=True)

    with pytest.raises(PlanExecutionError):
        registry.resolve("model:test:v1")


def test_fail_disposition_does_not_return_partial_success() -> None:
    plan = _plan({"verb": "null_policy", "columns": ["value"], "action": "reject"})
    with pytest.raises(InvalidRowsError) as raised:
        CleanPipeline(
            plan,
            model_registry=_registry(_row_model("NullRow", value=int)),
        ).run(pa.table({"value": [1, None]}))
    assert raised.value.row_count == 1


def test_dedupe_requires_explicit_keys_and_preserves_first_input_row() -> None:
    plan = _plan({"verb": "dedupe", "keys": ["key"]})
    result = CleanPipeline(
        plan,
        model_registry=_registry(_row_model("DedupeRow", key=int, value=str)),
    ).run(
        pa.table(
            {"key": [2, 1, 2, 1], "value": ["first-2", "first-1", "later-2", "later-1"]}
        )
    )
    assert result.table.to_pydict() == {
        "key": [2, 1],
        "value": ["first-2", "first-1"],
    }
    assert result.evidence.rows_out == 2
    assert result.evidence.dropped_rows == 2
    assert result.evidence.quarantined_rows == 0
    assert result.evidence.checkpoint_eligible
    assert [item.row_index for item in result.evidence.row_outcomes] == [0, 1, 2, 3]
    assert [item.status for item in result.evidence.row_outcomes] == [
        "accepted",
        "accepted",
        "dropped",
        "dropped",
    ]
    assert [item.reason_code for item in result.evidence.row_outcomes] == [
        "accepted",
        "accepted",
        "deduplicated",
        "deduplicated",
    ]


def test_dedupe_treats_null_keys_deterministically() -> None:
    plan = _plan({"verb": "dedupe", "keys": ["key"]})
    result = CleanPipeline(
        plan,
        model_registry=_registry(_row_model("NullKeyRow", key=int | None, value=str)),
    ).run(
        pa.table(
            {
                "key": pa.array([None, None, 1], type=pa.int64()),
                "value": ["first-null", "later-null", "one"],
            }
        )
    )
    assert result.table.to_pydict() == {
        "key": [None, 1],
        "value": ["first-null", "one"],
    }


def test_replay_is_deterministic_and_plan_is_not_mutated() -> None:
    plan = _plan(
        {"verb": "canonical_names"},
        {"verb": "dedupe", "keys": ["id"]},
        {"verb": "fill_nulls", "fills": {"name": "unknown"}},
    )
    before = plan.model_dump(mode="json")
    table = pa.table({"id": [2, 1, 2], "name": ["b", None, "later"]})
    model = _registry(_row_model("ReplayRow", id=int, name=str | None))
    first = CleanPipeline(plan, model_registry=model).run(table)
    second = CleanPipeline(plan, model_registry=model).run(table)

    assert first.table.equals(second.table)
    assert first.evidence.model_dump() == second.evidence.model_dump()
    assert plan.model_dump(mode="json") == before


def test_plan_digest_is_stable_for_fill_map_order() -> None:
    first = _plan({"verb": "fill_nulls", "fills": {"a": 1, "b": 2}})
    second = _plan({"verb": "fill_nulls", "fills": {"b": 10, "a": 9}})
    table = pa.table(
        {
            "a": pa.array([None], type=pa.int64()),
            "b": pa.array([None], type=pa.int64()),
        }
    )
    assert (
        CleanPipeline(
            first,
            model_registry=_registry(
                _row_model("DigestRowA", a=int | None, b=int | None)
            ),
        )
        .run(table)
        .evidence.plan_digest
        == CleanPipeline(
            second,
            model_registry=_registry(
                _row_model("DigestRowB", a=int | None, b=int | None)
            ),
        )
        .run(table)
        .evidence.plan_digest
    )


def test_arrow_adapter_accepts_batches_but_not_rows() -> None:
    profile = _plan().profile
    table = ArrowAdapter.from_batches(
        [pa.RecordBatch.from_arrays([pa.array([1, 2])], ["value"])],
        profile=profile,
    )
    assert table.column_names == ["value"]
    with pytest.raises(TypeError):
        ArrowAdapter.as_table([{"value": 1}], profile=profile)


def test_profile_summary_contains_only_counts_and_digests() -> None:
    summary = CleanPipeline(
        _plan(),
        model_registry=_registry(_row_model("ProfileEmailRow", email=str | None)),
    ).profile(pa.table({"email": ["secret@example.test", None]}))
    dumped = json.dumps(summary.model_dump(), sort_keys=True)
    assert "secret@example.test" not in dumped
    assert "email" not in dumped
    assert summary.rows == 2
    assert summary.null_cells == 1
    assert summary.column_profiles[0].dtype == "string"
    assert summary.column_profiles[0].null_count == 1
    assert summary.column_profiles[0].null_rate == 0.5
    assert summary.column_profiles[0].distinct_count == 1
    assert "high_null_rate" in summary.column_profiles[0].warning_codes


def test_small_secret_has_no_unkeyed_content_or_field_name_fingerprint() -> None:
    plan = _plan(
        {"verb": "fill_nulls", "fills": {"value": "one-row-secret"}},
    )
    pipeline = CleanPipeline(
        plan,
        model_registry=_registry(_row_model("SecretRow", value=str)),
    )
    table = pa.table({"value": pa.array([None], type=pa.string())})
    profile_dump = json.dumps(pipeline.profile(table).model_dump(), sort_keys=True)
    evidence_dump = json.dumps(
        pipeline.run(table).evidence.model_dump(), sort_keys=True
    )

    for dumped in (profile_dump, evidence_dump):
        assert "one-row-secret" not in dumped
        assert "value" not in dumped
        assert "input_digest" not in dumped
        assert "output_digest" not in dumped
        assert "name_digest" not in dumped
