"""Release-gate tests for connector live-certification evidence."""

from __future__ import annotations

import base64
import importlib.util
import secrets
from pathlib import Path

import pytest

from agent_utilities.knowledge_graph.integrations.connector_certification import (
    REQUIRED_CHECKS,
    CertificationBundle,
    write_certification_record,
)
from agent_utilities.knowledge_graph.ontology import ontology_integrity
from agent_utilities.knowledge_graph.ontology.connector_manifest import (
    ConnectorManifest,
    IntegrityInfo,
    ProvenanceSpec,
    ResourceSpec,
    SyncSpec,
)

_SPEC = importlib.util.spec_from_file_location(
    "check_connector_live_certification",
    Path(__file__).resolve().parents[3]
    / "scripts"
    / "check_connector_live_certification.py",
)
gate = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(gate)


@pytest.fixture
def signer(monkeypatch: pytest.MonkeyPatch) -> ontology_integrity.ReleaseSigner:
    key = base64.urlsafe_b64encode(secrets.token_bytes(32)).decode().rstrip("=")
    monkeypatch.setenv("ONTOLOGY_RELEASE_SIGNING_TEST_MATERIAL", key)
    monkeypatch.setenv(
        "ONTOLOGY_RELEASE_SIGNING_PRIVATE_KEY_REF",
        "env://ONTOLOGY_RELEASE_SIGNING_TEST_MATERIAL",
    )
    return ontology_integrity.ReleaseSigner.from_runtime()


@pytest.fixture
def bundle(signer: ontology_integrity.ReleaseSigner) -> CertificationBundle:
    manifest = ConnectorManifest(
        connector="fixture-connector",
        resources=[ResourceSpec(name="Widget", id_prefix="widget")],
        sync=[
            SyncSpec(
                preset="widgets",
                server="widget-mcp",
                tool="list_widgets",
                tool_schema_sha256="4" * 64,
            )
        ],
        provenance=ProvenanceSpec(
            integrity=IntegrityInfo(hash="5" * 64),
            signer=signer.signer_id,
            signature_algorithm=signer.algorithm,
            signing_public_key=signer.public_key,
        ),
    )
    return CertificationBundle(
        manifest=manifest,
        fixtures=(),
        shapes_text="synthetic",
        manifest_sha256="1" * 64,
        fixtures_sha256="2" * 64,
        shapes_sha256="3" * 64,
    )


def _record(
    bundle: CertificationBundle,
    signer: ontology_integrity.ReleaseSigner,
    *,
    live: bool,
) -> dict:
    checks = {name: "passed" for name in REQUIRED_CHECKS}
    if not live:
        checks["live_tool_schema"] = "not-run"
    record = {
        "api_version": "graphos.io/v1",
        "kind": "ConnectorLiveCertification",
        "schema_version": "1",
        "connector": bundle.manifest.connector,
        "certified_at": "2026-07-14T00:00:00Z",
        "mode": "external-live" if live else "offline-fixture",
        "status": "certified" if live else "offline-validated",
        "live_certified": live,
        "bundle": {
            "manifest_sha256": bundle.manifest_sha256,
            "fixtures_sha256": bundle.fixtures_sha256,
            "shapes_sha256": bundle.shapes_sha256,
            "schema_version": bundle.manifest.schema_version,
        },
        "scope": {
            "sync_presets": 1,
            "fixtures_declared": 1,
            "fixtures_exercised": 1,
            "tenant_bound": True,
            "retention_bound": True,
        },
        "checks": checks,
        "counts": {"initial": 0, "after_cleanup": 0},
        "semantic_validator": "pyshacl" if live else "declared-shacl-contract",
        "evidence": {name: "6" * 64 for name in REQUIRED_CHECKS},
        "failure_class": None,
        "runtime_configuration": "externalized" if live else "none",
        "signer": signer.signer_id,
        "signature_algorithm": signer.algorithm,
        "signing_public_key": signer.public_key,
        "signature": None,
    }
    record["signature"] = signer.sign(
        ontology_integrity.canonical_signed_document_hash(record)
    )
    return record


def test_release_gate_accepts_current_signed_live_record(
    tmp_path: Path,
    bundle: CertificationBundle,
    signer: ontology_integrity.ReleaseSigner,
) -> None:
    write_certification_record(
        tmp_path / "fixture-connector.json", _record(bundle, signer, live=True)
    )

    assert gate._check_one(bundle, tmp_path, require_live=True) == []


def test_release_gate_rejects_offline_and_stale_bundle_records(
    tmp_path: Path,
    bundle: CertificationBundle,
    signer: ontology_integrity.ReleaseSigner,
) -> None:
    offline = _record(bundle, signer, live=False)
    write_certification_record(tmp_path / "fixture-connector.json", offline)
    assert "connector has no passing external live certification" in gate._check_one(
        bundle, tmp_path, require_live=True
    )

    stale = _record(bundle, signer, live=True)
    stale["bundle"]["fixtures_sha256"] = "7" * 64
    stale["signature"] = None
    stale["signature"] = signer.sign(
        ontology_integrity.canonical_signed_document_hash(stale)
    )
    write_certification_record(tmp_path / "fixture-connector.json", stale)
    assert (
        "certification does not bind the current capability bundle"
        in gate._check_one(bundle, tmp_path, require_live=True)
    )
