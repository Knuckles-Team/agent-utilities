"""Certification fixtures for the typed connector preparation boundary."""

from __future__ import annotations

import json
from dataclasses import replace

import pyarrow as pa
import pytest
from pydantic import ConfigDict, ValidationError, create_model

from agent_utilities.data_prep import (
    CleanPlan,
    ConnectorArtifact,
    ConnectorArtifacts,
    ConnectorMapper,
    ConnectorPageLimits,
    ConnectorPreparation,
    ConnectorPreparationError,
    ConnectorPrepContract,
    PageNotCertifiedError,
    RowModelRegistry,
    build_native_change_envelope,
    plan_digest,
    row_model_digest,
    schema_digest,
)
from agent_utilities.protocols.source_connectors.checkpoint import (
    ConnectorCheckpoint,
)


def _row_model() -> type:
    return create_model(
        "ConnectorRow",
        __config__=ConfigDict(strict=True, extra="forbid"),
        id=(int, ...),
        updated_at=(str, ...),
        title=(str, ...),
    )


def _table() -> pa.Table:
    return pa.table(
        {
            "id": pa.array([1, 2], type=pa.int64()),
            "updated_at": pa.array(["v1", "v2"], type=pa.string()),
            "title": pa.array(["first", "second"], type=pa.string()),
        }
    )


def _plan(model: type, *, disposition: str = "fail") -> CleanPlan:
    return CleanPlan.model_validate(
        {
            "schema_version": "1",
            "steps": [{"verb": "canonical_names"}],
            "profile": {
                "max_rows": 8,
                "max_columns": 8,
                "max_steps": 4,
                "max_outcome_rows": 8,
            },
            "invalid_row_disposition": disposition,
            "plan_ref": "plan:connector:v1",
            "policy_ref": "policy:connector:v1",
            "model_ref": "model:connector-row:v1",
            "model_digest": row_model_digest(model),
            "source_ref": "source:fixture",
            "artifact_ref": "artifact:fixture",
        }
    )


def _artifact(kind: str, ref: str, digest: str) -> ConnectorArtifact:
    return ConnectorArtifact(kind=kind, ref=ref, digest=digest)


def _contract(
    table: pa.Table,
    model: type,
    *,
    disposition: str = "fail",
    max_cardinality: int = 8,
) -> tuple[ConnectorPrepContract, ConnectorArtifact]:
    plan = _plan(model, disposition=disposition)
    mapping_artifact = _artifact(
        "mapping",
        "mapping:fixture:v1",
        "sha256:" + "3" * 64,
    )
    artifacts = ConnectorArtifacts(
        raw_model=_artifact(
            "raw_model",
            plan.model_ref,
            row_model_digest(model),
        ),
        prep_plan=_artifact("prep_plan", plan.plan_ref, plan_digest(plan)),
        arrow_schema=_artifact(
            "arrow_schema",
            "schema:connector:v1",
            schema_digest(table),
        ),
        mapping=mapping_artifact,
        shacl=_artifact("shacl", "shape:connector:v1", "sha256:" + "4" * 64),
        icv=_artifact("icv", "icv:connector:v1", "sha256:" + "5" * 64),
    )
    return (
        ConnectorPrepContract(
            contract_version="connector-prep.v1",
            connector="fixture-connector",
            tenant="tenant-fixture",
            source_instance="instance-fixture",
            schema_version="source-schema-v1",
            ontology_mapping_version="mapping-v1",
            plan=plan,
            artifacts=artifacts,
            limits=ConnectorPageLimits(
                max_rows=8,
                max_columns=8,
                max_cardinality=max_cardinality,
                max_diagnostics=4,
            ),
            validation_mode=("quarantine" if disposition == "quarantine" else "strict"),
        ),
        mapping_artifact,
    )


def _mapper(
    contract: ConnectorPrepContract,
    mapping_artifact: ConnectorArtifact,
) -> ConnectorMapper:
    def map_table(table: pa.Table) -> list:
        return [
            build_native_change_envelope(
                row,
                contract=contract,
                id_field="id",
                version_field="updated_at",
            )
            for row in table.to_pylist()
        ]

    return ConnectorMapper(artifact=mapping_artifact, map_table=map_table)


def _preparer(
    table: pa.Table,
    *,
    disposition: str = "fail",
    max_cardinality: int = 8,
    mapper_factory=_mapper,
) -> tuple[ConnectorPreparation, ConnectorPrepContract]:
    model = _row_model()
    contract, mapping_artifact = _contract(
        table,
        model,
        disposition=disposition,
        max_cardinality=max_cardinality,
    )
    mapper = mapper_factory(contract, mapping_artifact)
    return (
        ConnectorPreparation(
            contract,
            model_registry=RowModelRegistry({contract.plan.model_ref: model}),
            mapper=mapper,
        ),
        contract,
    )


def test_complete_page_binds_artifacts_and_replays_deterministically() -> None:
    table = _table()
    preparer, contract = _preparer(table)
    checkpoint = ConnectorCheckpoint(cursor="page-1")

    first = preparer.prepare(table, checkpoint=checkpoint, fetch_complete=True)
    second = preparer.prepare(table, checkpoint=checkpoint, fetch_complete=True)

    assert first.certification.outcome == "complete"
    assert first.checkpoint_eligible
    assert first.checkpoint_candidate() == checkpoint
    assert first.certification.replay_digest == second.certification.replay_digest
    assert [item.idempotency_key for item in first.envelopes] == [
        item.idempotency_key for item in second.envelopes
    ]
    assert first.envelopes[0].provenance["connector_preparation"]["artifacts"] == (
        contract.artifacts.model_dump(mode="json")
    )
    assert first.envelopes[0].checkpoint is None

    marker = first.snapshot_complete(live_ids=["1", "2"])
    assert marker.operation == "snapshot_complete"
    assert marker.live_ids == ("1", "2")


def test_empty_snapshot_requires_explicit_authoritative_proof() -> None:
    table = pa.table(
        {
            "id": pa.array([], type=pa.int64()),
            "updated_at": pa.array([], type=pa.string()),
            "title": pa.array([], type=pa.string()),
        }
    )
    preparer, _ = _preparer(table)
    page = preparer.prepare(
        table,
        checkpoint=ConnectorCheckpoint(cursor="empty-page"),
        fetch_complete=True,
    )

    with pytest.raises(PageNotCertifiedError) as raised:
        page.snapshot_complete(live_ids=[])
    assert raised.value.diagnostic.code == "snapshot_not_verified"

    marker = page.snapshot_complete(live_ids=[], authoritative_empty=True)
    assert marker.operation == "snapshot_complete"
    assert marker.live_ids == ()
    assert marker.provenance["authoritative_empty"] is True


def test_strict_model_failure_is_redacted_and_cannot_return_a_page() -> None:
    table = pa.table(
        {
            "id": pa.array(["secret-invalid-id"], type=pa.string()),
            "updated_at": pa.array(["v1"], type=pa.string()),
            "title": pa.array(["private title"], type=pa.string()),
        }
    )
    preparer, _ = _preparer(table)

    with pytest.raises(ConnectorPreparationError) as raised:
        preparer.prepare(table, checkpoint=ConnectorCheckpoint(cursor="bad"))
    assert raised.value.diagnostic.code == "validation_failed"
    assert "secret-invalid-id" not in str(raised.value)
    assert "private title" not in json.dumps(
        raised.value.diagnostic.model_dump(mode="json"),
        sort_keys=True,
    )


def test_quarantine_is_explicit_and_blocks_checkpoint_and_deletion() -> None:
    table = pa.table(
        {
            "id": pa.array([1, None], type=pa.int64()),
            "updated_at": pa.array(["v1", "v2"], type=pa.string()),
            "title": pa.array(["accepted", "quarantined"], type=pa.string()),
        }
    )
    preparer, _ = _preparer(table, disposition="quarantine")
    page = preparer.prepare(
        table,
        checkpoint=ConnectorCheckpoint(cursor="quarantine"),
        fetch_complete=True,
    )

    assert page.certification.outcome == "quarantined"
    assert not page.checkpoint_eligible
    assert page.certification.diagnostics[0].code == "rows_quarantined"
    assert page.envelopes[0].source_object_id == "1"
    with pytest.raises(PageNotCertifiedError):
        page.snapshot_complete(live_ids=["1"])


def test_mapping_partial_page_cannot_become_verified_empty() -> None:
    def incomplete_mapper(contract: ConnectorPrepContract, artifact: ConnectorArtifact):
        del contract

        def map_table(table: pa.Table) -> list:
            del table
            return []

        return ConnectorMapper(artifact=artifact, map_table=map_table)

    preparer, _ = _preparer(_table(), mapper_factory=incomplete_mapper)
    page = preparer.prepare(
        _table(),
        checkpoint=ConnectorCheckpoint(cursor="partial"),
        fetch_complete=True,
    )

    assert page.certification.outcome == "partial"
    assert page.certification.diagnostics[0].code == "mapping_contract_violation"
    assert page.checkpoint_candidate() is None
    with pytest.raises(PageNotCertifiedError):
        page.snapshot_complete(live_ids=[], authoritative_empty=True)


def test_mapper_cannot_override_deterministic_idempotency() -> None:
    def non_deterministic_mapper(
        contract: ConnectorPrepContract,
        artifact: ConnectorArtifact,
    ) -> ConnectorMapper:
        def map_table(table: pa.Table) -> list:
            envelope = build_native_change_envelope(
                table.to_pylist()[0],
                contract=contract,
                id_field="id",
                version_field="updated_at",
            )
            return [replace(envelope, idempotency_key="caller-selected")]

        return ConnectorMapper(artifact=artifact, map_table=map_table)

    preparer, _ = _preparer(_table(), mapper_factory=non_deterministic_mapper)
    with pytest.raises(ConnectorPreparationError) as raised:
        preparer.prepare(_table(), fetch_complete=True)
    assert raised.value.diagnostic.code == "idempotency_key_mismatch"


def test_contract_rejects_artifact_kind_drift_and_mutation() -> None:
    table = _table()
    model = _row_model()
    plan = _plan(model)
    with pytest.raises(ValidationError):
        ConnectorArtifacts(
            raw_model=_artifact("mapping", plan.model_ref, row_model_digest(model)),
            prep_plan=_artifact("prep_plan", plan.plan_ref, plan_digest(plan)),
            arrow_schema=_artifact(
                "arrow_schema", "schema:connector:v1", schema_digest(table)
            ),
            mapping=_artifact("mapping", "mapping:fixture:v1", "sha256:" + "3" * 64),
            shacl=_artifact("shacl", "shape:connector:v1", "sha256:" + "4" * 64),
            icv=_artifact("icv", "icv:connector:v1", "sha256:" + "5" * 64),
        )

    contract, _ = _contract(table, model)
    with pytest.raises(ValidationError):
        contract.artifacts.mapping.ref = "mapping:mutated"
