"""Current-only signed source attestation for connector capability bundles.

This contract proves only repository-owned bundle structure. It deliberately has
no representation for a passing live tool or executed fixture check; those belong
to :mod:`connector_certification` and its external-live evidence record.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any

from agent_utilities.knowledge_graph.ontology import ontology_integrity
from agent_utilities.knowledge_graph.ontology.connector_manifest import (
    ConnectorManifest,
)

SOURCE_ATTESTATION_KEYS = frozenset(
    {
        "api_version",
        "kind",
        "schema_version",
        "connector",
        "validated_at",
        "mode",
        "status",
        "live_certified",
        "checks",
        "compatibility",
        "artifacts",
        "signer",
        "signature_algorithm",
        "signing_public_key",
        "signature",
    }
)
SOURCE_COMPATIBILITY = {
    "agent_utilities": ">=1.27.1,<2",
    "epistemic_graph": ">=2.23.1,<3",
    "bundle_schema": "2",
}

_UTC_TIMESTAMP = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def expected_source_checks(manifest: ConnectorManifest) -> dict[str, str]:
    """Return the exact structural checks represented by the source attestation."""

    sync_result = "passed" if manifest.sync else "not-applicable"
    return {
        "declared_tool_schema": sync_result,
        "synthetic_fixture_contract": sync_result,
        "shacl_parse": "passed",
        "privacy_contract": "passed",
        "manifest_integrity": "passed",
    }


def source_attestation_violations(
    attestation: dict[str, Any], manifest: ConnectorManifest
) -> list[str]:
    """Validate the exact offline contract without inferring live execution."""

    violations: list[str] = []
    if set(attestation) != SOURCE_ATTESTATION_KEYS:
        violations.append("source attestation shape differs")
    expected_scalars = {
        "api_version": "graphos.io/v1",
        "kind": "ConnectorSourceAttestation",
        "schema_version": "2",
        "mode": "offline-source",
        "status": "source-validated",
        "live_certified": False,
    }
    if any(attestation.get(key) != value for key, value in expected_scalars.items()):
        violations.append("source attestation identity differs")
    if attestation.get("connector") != manifest.connector:
        violations.append("source attestation connector differs")
    if attestation.get("checks") != expected_source_checks(manifest):
        violations.append("source attestation checks differ")
    if any(
        not isinstance(sync.tool_schema_sha256, str)
        or not _SHA256.fullmatch(sync.tool_schema_sha256)
        for sync in manifest.sync
    ):
        violations.append("declared tool schema fingerprint is invalid")
    if attestation.get("compatibility") != SOURCE_COMPATIBILITY:
        violations.append("source attestation compatibility differs")
    validated_at = attestation.get("validated_at")
    if not isinstance(validated_at, str) or not _UTC_TIMESTAMP.fullmatch(validated_at):
        violations.append("source attestation timestamp is invalid")
    artifacts = attestation.get("artifacts")
    if (
        not isinstance(artifacts, dict)
        or not artifacts
        or any(
            not isinstance(path, str)
            or not path
            or not isinstance(digest, str)
            or not _SHA256.fullmatch(digest)
            for path, digest in artifacts.items()
        )
    ):
        violations.append("source attestation artifact ledger is invalid")
    return violations


def build_source_attestation(
    manifest: ConnectorManifest,
    *,
    artifacts: dict[str, str],
    now: datetime,
    release_signer: ontology_integrity.ReleaseSigner,
) -> dict[str, Any]:
    """Build and sign a source-only attestation with no live-pass state."""

    attestation: dict[str, Any] = {
        "api_version": "graphos.io/v1",
        "kind": "ConnectorSourceAttestation",
        "schema_version": "2",
        "connector": manifest.connector,
        "validated_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "mode": "offline-source",
        "status": "source-validated",
        "live_certified": False,
        "checks": expected_source_checks(manifest),
        "compatibility": dict(SOURCE_COMPATIBILITY),
        "artifacts": dict(sorted(artifacts.items())),
        "signer": release_signer.signer_id,
        "signature_algorithm": release_signer.algorithm,
        "signing_public_key": release_signer.public_key,
        "signature": None,
    }
    violations = source_attestation_violations(attestation, manifest)
    if violations:
        raise ValueError("source attestation inputs are invalid")
    attestation["signature"] = release_signer.sign(
        ontology_integrity.canonical_signed_document_hash(attestation)
    )
    return attestation


__all__ = [
    "SOURCE_ATTESTATION_KEYS",
    "SOURCE_COMPATIBILITY",
    "build_source_attestation",
    "expected_source_checks",
    "source_attestation_violations",
]
