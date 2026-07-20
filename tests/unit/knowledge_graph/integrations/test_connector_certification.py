"""Connector lifecycle certification contract tests (no external services)."""

from __future__ import annotations

import base64
import secrets

import pytest

from agent_utilities.knowledge_graph.integrations import (
    connector_certification as module,
)
from agent_utilities.knowledge_graph.ontology.connector_manifest import (
    ConnectorManifest,
    IntegrityInfo,
    ProvenanceSpec,
    ResourceSpec,
    SyncSpec,
)
from agent_utilities.knowledge_graph.ontology.ontology_integrity import ReleaseSigner
from agent_utilities.protocols.source_connectors.tool_schema import (
    compatibility_fingerprint,
)

_SCHEMA = {
    "type": "object",
    "properties": {"action": {"type": "string"}},
    "required": ["action"],
}
_SHAPES = """\
@prefix : <http://knuckles.team/kg#> .
@prefix shape: <http://knuckles.team/kg/fixture/shape#> .
@prefix sh: <http://www.w3.org/ns/shacl#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
shape:WidgetShape a sh:NodeShape ;
    sh:targetClass :Widget ;
    sh:property [ sh:path :sourceRecordRef ; sh:minCount 1 ; sh:datatype xsd:string ] ;
    sh:property [ sh:path :tenantReference ; sh:minCount 1 ; sh:datatype xsd:string ] ;
    sh:property [ sh:path :accessPolicyReference ; sh:minCount 1 ; sh:datatype xsd:string ] ;
    sh:property [ sh:path :provenanceReference ; sh:minCount 1 ; sh:datatype xsd:string ] .
"""


@pytest.fixture
def signer(monkeypatch: pytest.MonkeyPatch) -> ReleaseSigner:
    key = base64.urlsafe_b64encode(secrets.token_bytes(32)).decode().rstrip("=")
    monkeypatch.setenv("ONTOLOGY_RELEASE_SIGNING_TEST_MATERIAL", key)
    monkeypatch.setenv(
        "ONTOLOGY_RELEASE_SIGNING_PRIVATE_KEY_REF",
        "env://ONTOLOGY_RELEASE_SIGNING_TEST_MATERIAL",
    )
    return ReleaseSigner.from_runtime()


@pytest.fixture
def bundle() -> module.CertificationBundle:
    manifest = ConnectorManifest(
        connector="fixture-connector",
        resources=[ResourceSpec(name="Widget", id_prefix="widget")],
        sync=[
            SyncSpec(
                preset="widgets",
                server="widget-mcp",
                tool="list_widgets",
                id_field="id",
                doc_type="widget",
                tool_schema_sha256=compatibility_fingerprint("list_widgets", _SCHEMA),
            )
        ],
        provenance=ProvenanceSpec(integrity=IntegrityInfo(hash="1" * 64)),
    )
    return module.CertificationBundle(
        manifest=manifest,
        fixtures=(
            module._Fixture(
                preset="widgets",
                record={
                    "id": "fixture-record",
                    "name": "Synthetic certification record",
                },
                expected={"id": "fixture-record", "acl_state": "quarantine"},
            ),
        ),
        shapes_text=_SHAPES,
        manifest_sha256="1" * 64,
        fixtures_sha256="2" * 64,
        shapes_sha256="3" * 64,
    )


@pytest.mark.asyncio
async def test_offline_fixture_mode_proves_lifecycle_without_claiming_live_success(
    bundle: module.CertificationBundle,
    signer: ReleaseSigner,
) -> None:
    record = await module.certify_connector(
        bundle,
        mode="offline-fixture",
        signer=signer,
    )

    assert record["status"] == "offline-validated"
    assert record["live_certified"] is False
    assert record["checks"]["live_tool_schema"] == "not-run"
    assert record["checks"]["replay_idempotency"] == "passed"
    assert record["checks"]["governance_preservation"] == "passed"
    assert record["counts"] == {
        "initial": 0,
        "after_ingest": 1,
        "after_replay": 1,
        "after_update": 1,
        "after_delete": 0,
        "after_delete_replay": 0,
        "after_cleanup": 0,
    }
    assert not module.verify_certification_record(
        record, trusted_public_keys=(signer.public_key,)
    )
    assert module.verify_certification_record(
        record,
        trusted_public_keys=(signer.public_key,),
        require_live=True,
    ) == ["connector has no passing external live certification"]


class _ContractDriver:
    def __init__(self, schema: dict) -> None:
        self.reference = module.ReferenceCertificationDriver(
            tool_schemas={"list_widgets": schema}
        )

    async def invoke(self, request):
        return await self.reference.invoke(request)


@pytest.mark.asyncio
async def test_live_tool_schema_drift_signs_failure_not_success(
    bundle: module.CertificationBundle,
    signer: ReleaseSigner,
) -> None:
    record = await module.certify_connector(
        bundle,
        mode="external-live",
        signer=signer,
        driver=_ContractDriver(
            {"type": "object", "properties": {"changed": {"type": "string"}}}
        ),
    )

    assert record["status"] == "failed"
    assert record["live_certified"] is False
    assert record["checks"]["live_tool_schema"] == "failed"
    assert record["failure_class"] == "ToolSchemaContractError"
    assert not module.verify_certification_record(
        record, trusted_public_keys=(signer.public_key,)
    )


@pytest.mark.asyncio
async def test_governance_mismatch_fails_closed(
    bundle: module.CertificationBundle,
    signer: ReleaseSigner,
) -> None:
    class MismatchDriver(_ContractDriver):
        async def invoke(self, request):
            result = await super().invoke(request)
            if request.get("action") == "inspect" and result.get("governance"):
                result["governance"] = {**result["governance"], "retention": "wrong"}
            return result

    record = await module.certify_connector(
        bundle,
        mode="external-live",
        signer=signer,
        driver=MismatchDriver(_SCHEMA),
    )

    assert record["status"] == "failed"
    assert record["checks"]["governance_preservation"] == "failed"
    assert record["failure_class"] == "CertificationError"


def test_runtime_profile_rejects_literal_connection_configuration() -> None:
    with pytest.raises(ValueError, match="runtime reference"):
        module.LiveCertificationProfile(
            driver_command_ref="python driver.py",
            connector_runtime_ref="https://connector.invalid",
            engine_runtime_ref="tcp://127.0.0.1:9100",
            tenant="connector-certification",
            retention="certification-ephemeral",
        )


@pytest.mark.asyncio
async def test_tampered_signed_record_is_rejected(
    bundle: module.CertificationBundle,
    signer: ReleaseSigner,
) -> None:
    record = await module.certify_connector(
        bundle,
        mode="offline-fixture",
        signer=signer,
    )
    record["counts"]["after_ingest"] = 99

    assert (
        "certification release signature is invalid"
        in module.verify_certification_record(
            record, trusted_public_keys=(signer.public_key,)
        )
    )
