"""Focused contracts for deterministic release evidence assembly."""

from __future__ import annotations

import base64
import csv
import hashlib
import json
import os
import re
import tomllib
import zipfile
from io import StringIO
from pathlib import Path

import pytest
import yaml
from jsonschema import Draft202012Validator
from packaging.requirements import Requirement

import build_backend
from agent_utilities._version import __version__ as agent_utilities_version
from agent_utilities.knowledge_graph.index_migrations import (
    index_migration_catalog,
    run_index_migration,
)
from scripts import security_contract
from scripts.release import (
    assemble_manifest,
    check_compatibility,
    connector_ledger,
    generate_oci_vulnerability_scan_evidence,
    generate_release_assembly,
    generate_release_inputs,
)
from scripts.release.generate_index_migration_catalog import render_catalog

ROOT = Path(__file__).resolve().parents[3]
CONNECTOR_COUNT = 65


def _connector_ledger_entries(count: int = CONNECTOR_COUNT) -> list[dict[str, str]]:
    return [
        {
            "connector": f"fixture-connector-{index:02d}",
            "certifiedAt": "2026-07-18T00:00:00Z",
            "recordDigest": "sha256:" + format(index + 1, "064x"),
            "bundleDigest": "sha256:" + format(index + 101, "064x"),
        }
        for index in range(count)
    ]


def _license_catalog(
    path: Path,
    licenses: dict[str, str] | None = None,
) -> Path:
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "licenses": licenses or {"fixture-package": "MIT", "pyyaml": "MIT"},
            }
        ),
        encoding="utf-8",
    )
    return path


def _project_requirement_names(pyproject: dict) -> set[str]:
    declarations = list(pyproject["project"].get("dependencies") or ())
    for values in pyproject["project"].get("optional-dependencies", {}).values():
        declarations.extend(values)
    for values in pyproject.get("dependency-groups", {}).values():
        declarations.extend(values)
    return {
        build_backend._normalized_name(Requirement(value).name)
        for value in declarations
    } | {build_backend._normalized_name(pyproject["project"]["name"])}


def _project_metadata(pyproject: dict) -> bytes:
    declarations = list(pyproject["project"].get("dependencies") or ())
    for values in pyproject["project"].get("optional-dependencies", {}).values():
        declarations.extend(values)
    lines = [
        "Metadata-Version: 2.4",
        f"Name: {pyproject['project']['name']}",
        f"Version: {agent_utilities_version}",
        *(f"Requires-Dist: {value}" for value in sorted(set(declarations))),
        "",
        "",
    ]
    return "\n".join(lines).encode()


def _minimal_wheel(
    path: Path,
    *,
    requirement: str = "PyYAML>=6",
    timestamp: tuple[int, int, int, int, int, int] = (2026, 1, 1, 0, 0, 0),
    reverse: bool = False,
    extra_members: tuple[tuple[str, bytes], ...] = (),
    executable_members: frozenset[str] = frozenset(),
) -> None:
    dist = "fixture_package-1.0.0.dist-info/"
    metadata = (
        "Metadata-Version: 2.4\n"
        "Name: fixture-package\n"
        "Version: 1.0.0\n"
        f"Requires-Dist: {requirement}\n\n"
    ).encode()
    wheel = b"Wheel-Version: 1.0\nGenerator: fixture\nRoot-Is-Purelib: true\nTag: py3-none-any\n"
    members = [
        ("fixture_package/__init__.py", b""),
        (dist + "METADATA", metadata),
        (dist + "WHEEL", wheel),
        (dist + "RECORD", b""),
        *extra_members,
    ]
    with zipfile.ZipFile(path, "w") as archive:
        for name, payload in reversed(members) if reverse else members:
            info = zipfile.ZipInfo(name, timestamp)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = (
                0o100755 if name in executable_members else 0o100644
            ) << 16
            archive.writestr(info, payload)


def test_wheel_build_backend_embeds_deterministic_sbom_and_exact_record(
    tmp_path: Path,
) -> None:
    wheel = tmp_path / "fixture_package-1.0.0-py3-none-any.whl"
    _minimal_wheel(wheel)
    catalog = _license_catalog(tmp_path / "licenses.json")
    build_backend.embed_wheel_sbom(wheel, license_catalog=catalog)
    first = hashlib.sha256(wheel.read_bytes()).digest()
    build_backend.embed_wheel_sbom(wheel, license_catalog=catalog)
    assert hashlib.sha256(wheel.read_bytes()).digest() == first
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
        sbom_name = "fixture_package-1.0.0.dist-info/sboms/package.cyclonedx.json"
        assert sbom_name in names
        sbom = json.loads(archive.read(sbom_name))
        assert sbom["bomFormat"] == "CycloneDX"
        assert sbom["metadata"]["component"]["purl"] == (
            "pkg:pypi/fixture-package@1.0.0"
        )
        assert sbom["metadata"]["component"]["licenses"] == [{"expression": "MIT"}]
        assert sbom["components"][0]["licenses"] == [{"expression": "MIT"}]
        record_name = "fixture_package-1.0.0.dist-info/RECORD"
        rows = list(csv.reader(StringIO(archive.read(record_name).decode())))
        for name, digest, size in rows:
            if name == record_name:
                assert (digest, size) == ("", "")
                continue
            payload = archive.read(name)
            expected = base64.urlsafe_b64encode(hashlib.sha256(payload).digest())
            assert digest == "sha256=" + expected.rstrip(b"=").decode()
            assert int(size) == len(payload)


def test_wheel_build_backend_normalizes_source_order_and_timestamps(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first" / "fixture_package-1.0.0-py3-none-any.whl"
    second = tmp_path / "second" / "fixture_package-1.0.0-py3-none-any.whl"
    first.parent.mkdir()
    second.parent.mkdir()
    _minimal_wheel(first, timestamp=(2025, 1, 1, 0, 0, 0))
    _minimal_wheel(second, timestamp=(2026, 6, 1, 12, 30, 0), reverse=True)
    catalog = _license_catalog(tmp_path / "licenses.json")

    build_backend.embed_wheel_sbom(first, license_catalog=catalog)
    build_backend.embed_wheel_sbom(second, license_catalog=catalog)

    assert first.read_bytes() == second.read_bytes()
    with zipfile.ZipFile(first) as archive:
        assert {info.date_time for info in archive.infolist()} == {
            build_backend._FIXED_ZIP_TIME
        }


def test_wheel_build_backend_canonicalizes_inherited_executable_bits(
    tmp_path: Path,
) -> None:
    regular = "fixture_package/__init__.py"
    script = "fixture_package-1.0.0.data/scripts/fixture-command"
    first = tmp_path / "first" / "fixture_package-1.0.0-py3-none-any.whl"
    second = tmp_path / "second" / "fixture_package-1.0.0-py3-none-any.whl"
    first.parent.mkdir()
    second.parent.mkdir()
    _minimal_wheel(
        first,
        extra_members=((script, b"#!/usr/bin/env python3\n"),),
        executable_members=frozenset({regular}),
    )
    _minimal_wheel(
        second,
        extra_members=((script, b"#!/usr/bin/env python3\n"),),
        executable_members=frozenset({script}),
    )
    catalog = _license_catalog(tmp_path / "licenses.json")

    build_backend.embed_wheel_sbom(first, license_catalog=catalog)
    build_backend.embed_wheel_sbom(second, license_catalog=catalog)

    assert first.read_bytes() == second.read_bytes()
    with zipfile.ZipFile(first) as archive:
        modes = {
            info.filename: (info.external_attr >> 16) & 0o777
            for info in archive.infolist()
            if not info.is_dir()
        }
    assert modes[regular] == 0o644
    assert modes[script] == 0o755
    assert all(
        mode == (0o755 if name == script else 0o644) for name, mode in modes.items()
    )


def test_wheel_build_backend_prunes_source_only_top_level_tooling(
    tmp_path: Path,
) -> None:
    wheel = tmp_path / "fixture_package-1.0.0-py3-none-any.whl"
    retained = {
        "scripts/__init__.py",
        "scripts/release/promote_local_release.py",
        "scripts/certification/campaign.py",
        "scripts/scale/loadgen.py",
        "deploy/__init__.py",
        "deploy/release/release-manifest.schema.json",
    }
    pruned = {
        "scripts/developer_helper.py",
        "scripts/security_sanitizer.py",
        "deploy/local_override.json",
    }
    _minimal_wheel(
        wheel,
        extra_members=tuple(
            (name, b"{}" if name.endswith(".json") else b"")
            for name in sorted(retained | pruned)
        ),
    )
    catalog = _license_catalog(tmp_path / "licenses.json")

    build_backend.embed_wheel_sbom(wheel, license_catalog=catalog)

    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
        record = archive.read("fixture_package-1.0.0.dist-info/RECORD").decode("utf-8")
    assert retained <= names
    assert names.isdisjoint(pruned)
    assert not any(name in record for name in pruned)


@pytest.mark.parametrize(
    "reference",
    [
        "file:fixture.whl",
        "https://example.invalid/fixture.whl",
        "ssh://example.invalid/fixture",
        "git+ssh://example.invalid/fixture.git",
        "git+https://example.invalid/fixture.git",
        "hg+ssh://example.invalid/fixture",
        "svn+ssh://example.invalid/fixture",
    ],
)
def test_wheel_build_backend_rejects_every_direct_reference(
    tmp_path: Path,
    reference: str,
) -> None:
    wheel = tmp_path / "fixture_package-1.0.0-py3-none-any.whl"
    _minimal_wheel(wheel, requirement=f"fixture @ {reference}")
    with pytest.raises(build_backend.WheelSbomError):
        build_backend.embed_wheel_sbom(
            wheel,
            license_catalog=_license_catalog(
                tmp_path / "licenses.json",
                {"fixture": "MIT", "fixture-package": "MIT"},
            ),
        )


def test_wheel_build_backend_rejects_missing_license_entry(tmp_path: Path) -> None:
    wheel = tmp_path / "fixture_package-1.0.0-py3-none-any.whl"
    _minimal_wheel(wheel)
    catalog = _license_catalog(tmp_path / "licenses.json", {"fixture-package": "MIT"})
    with pytest.raises(build_backend.WheelSbomError, match="does not cover"):
        build_backend.embed_wheel_sbom(wheel, license_catalog=catalog)


def test_wheel_build_backend_rejects_invalid_spdx_entry(tmp_path: Path) -> None:
    wheel = tmp_path / "fixture_package-1.0.0-py3-none-any.whl"
    _minimal_wheel(wheel)
    catalog = _license_catalog(
        tmp_path / "licenses.json",
        {"fixture-package": "MIT", "pyyaml": "MIT AND"},
    )
    with pytest.raises(build_backend.WheelSbomError, match="SPDX expression"):
        build_backend.embed_wheel_sbom(wheel, license_catalog=catalog)


def test_actual_catalog_covers_runtime_optional_and_development_declarations() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    catalog = build_backend._license_catalog(
        ROOT / "agent_utilities/dependency-license-catalog.json"
    )
    assert set(catalog) == _project_requirement_names(pyproject)


def test_actual_generated_sbom_passes_strict_license_policy(tmp_path: Path) -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    sbom = json.loads(
        build_backend._sbom(
            _project_metadata(pyproject),
            "py3-none-any",
            ROOT / "agent_utilities/dependency-license-catalog.json",
        )
    )
    assert sbom["metadata"]["component"]["licenses"] == [{"expression": "MIT"}]
    assert all(component.get("licenses") for component in sbom["components"])
    (tmp_path / "sbom.json").write_text(json.dumps(sbom), encoding="utf-8")
    contract = security_contract.load_contract(ROOT, ".security/security-contract.json")
    security_contract.check_licenses(
        tmp_path, contract, "sbom.json", "license-evidence.json"
    )
    evidence = json.loads(
        (tmp_path / "license-evidence.json").read_text(encoding="utf-8")
    )
    assert evidence["unknown"] == 0
    assert evidence["violations"] == 0


def _write_release_evidence(
    root: Path,
    name: str,
    version: str,
    kind: str,
    artifact_sha256: str,
    capabilities: list[str],
    entry_count: int | None,
    source_snapshot_digest: str,
    source_evidence_digest: str,
) -> dict[str, str]:
    evidence = root / "evidence" / name
    evidence.mkdir(parents=True)
    artifact_digest = "sha256:" + artifact_sha256
    source = {
        "apiVersion": "graphos.io/v1",
        "kind": "ComponentSourceEvidence",
        "component": name,
        "version": version,
        "artifactFormat": "oci-layout-archive" if kind == "oci" else "opaque-catalog",
        "artifactDigest": artifact_digest,
        "artifactInputDigest": artifact_digest,
        "sourceSnapshotDigest": source_snapshot_digest,
        "sourceEvidenceDigest": source_evidence_digest,
    }
    source_path = evidence / "source.json"
    source_path.write_text(
        json.dumps(source, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    inventory = (
        [
            {
                "type": "library",
                "bom-ref": "pkg:pypi/fixture-runtime@1.0.0",
                "name": "fixture-runtime",
                "version": "1.0.0",
                "purl": "pkg:pypi/fixture-runtime@1.0.0",
                "hashes": [{"alg": "SHA-256", "content": "c" * 64}],
            }
        ]
        if kind == "oci"
        else []
    )
    (evidence / "sbom.json").write_text(
        json.dumps(
            {
                "bomFormat": "CycloneDX",
                "specVersion": "1.6",
                "version": 1,
                "metadata": {
                    "component": {
                        "type": "library",
                        "bom-ref": (
                            f"pkg:{'pypi' if kind == 'oci' else 'generic'}/"
                            f"{name}@{version}"
                        ),
                        "name": name,
                        "version": version,
                        "purl": (
                            f"pkg:{'pypi' if kind == 'oci' else 'generic'}/"
                            f"{name}@{version}"
                        ),
                        "hashes": [{"alg": "SHA-256", "content": artifact_sha256}],
                    }
                },
                "components": inventory,
            }
        ),
        encoding="utf-8",
    )
    (evidence / "provenance.json").write_text(
        json.dumps(
            {
                "_type": "https://in-toto.io/Statement/v1",
                "subject": [{"name": name, "digest": {"sha256": artifact_sha256}}],
                "predicateType": "https://slsa.dev/provenance/v1",
                "predicate": {
                    "buildDefinition": {
                        "buildType": "https://graphos.invalid/build/exact-local/v1",
                        "externalParameters": {},
                        "internalParameters": {},
                        "resolvedDependencies": [
                            {
                                "uri": "urn:graphos:source-freeze",
                                "digest": {
                                    "sha256": source_snapshot_digest.removeprefix(
                                        "sha256:"
                                    )
                                },
                            }
                        ],
                    },
                    "runDetails": {
                        "builder": {
                            "id": "https://graphos.invalid/builders/exact-local/v1"
                        },
                        "byproducts": [],
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    references = {
        "source": f"evidence/{name}/source.json",
        "sbom": f"evidence/{name}/sbom.json",
        "provenance": f"evidence/{name}/provenance.json",
        "signatureBundle": f"evidence/{name}/signature.json",
    }
    component: dict[str, object] = {
        "version": version,
        "kind": kind,
        "artifact": f"{kind}:{name}@{artifact_digest}",
        "digest": artifact_digest,
        "sourceDigest": check_compatibility.file_digest(source_path),
        "sbomDigest": check_compatibility.file_digest(evidence / "sbom.json"),
        "provenanceDigest": check_compatibility.file_digest(
            evidence / "provenance.json"
        ),
        "signatureVerifierEnv": "COMPONENT_SIGNATURE_VERIFIER",
        "capabilities": sorted(capabilities),
        "evidence": references,
    }
    if entry_count is not None:
        component["entryCount"] = entry_count
    subject = check_compatibility.component_signing_subject(name, component)
    (evidence / "signature.json").write_text(
        json.dumps(
            {
                "schema": "graphos-external-signature/2",
                "scheme": "fixture-signature",
                "subjectDigest": "sha256:" + hashlib.sha256(subject).hexdigest(),
                "artifactDigest": artifact_digest,
                "signature": "fixture-signature-value",
                "verificationMaterialDigest": "sha256:" + "d" * 64,
                "signerIdentityDigest": "sha256:" + "e" * 64,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    return references


def _source_freeze_evidence(path: Path) -> dict[str, str]:
    manifest_payload = (ROOT / "deploy/release/source-freeze-gates.json").read_bytes()
    manifest = json.loads(manifest_payload)
    repository_ids = [item["id"] for item in manifest["repositories"]]
    repository_digests = {
        identifier: format(index + 5, "064x")
        for index, identifier in enumerate(repository_ids)
    }

    def aggregate(values: dict[str, str]) -> str:
        payload = json.dumps(values, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(payload).hexdigest()

    repository_token = re.compile(r"^\{repo:([a-z][a-z0-9-]{2,63})\}(.*)$")
    commands: list[dict[str, object]] = []
    for command in manifest["commands"]:
        identifiers = {command["repository"]}
        identifiers.update(
            match.group(1)
            for value in command["argv"]
            if (match := repository_token.fullmatch(value)) is not None
        )
        digest = aggregate(
            {
                identifier: repository_digests[identifier]
                for identifier in repository_ids
                if identifier in identifiers
            }
        )
        commands.append(
            {
                "id": command["id"],
                "status": "passed",
                "exit_code": 0,
                "termination": "exited",
                "source_digest_before": digest,
                "source_digest_after": digest,
            }
        )
    source_digest = aggregate(repository_digests)
    evidence = {
        "schema": "source-freeze-evidence/1",
        "status": "passed",
        "manifest_sha256": hashlib.sha256(manifest_payload).hexdigest(),
        "source_digest_before": source_digest,
        "source_digest_after": source_digest,
        "tools": [
            {"id": "git", "sha256": "3" * 64},
            {"id": "rg", "sha256": "4" * 64},
        ],
        "repositories": [
            {
                "id": identifier,
                "sha256_before": repository_digests[identifier],
                "sha256_after": repository_digests[identifier],
            }
            for identifier in repository_ids
        ],
        "commands": commands,
        "gates": [
            {
                "id": gate["id"],
                "required_evidence": gate["evidence_classes"],
                "source_status": (
                    "passed"
                    if "local-source" in gate["evidence_classes"]
                    else "not-applicable"
                ),
                "remaining_evidence": [
                    value
                    for value in gate["evidence_classes"]
                    if value != "local-source"
                ],
            }
            for gate in manifest["gates"]
        ],
    }
    path.write_text(
        json.dumps(evidence, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return check_compatibility.validate_source_freeze_evidence(path.read_bytes())


def _skill_validation_evidence(
    *,
    release_id: str,
    release_specification_digest: str,
    promotion_evidence_digest: str,
    graph_os_digest: str,
    engine_digest: str,
    configuration_digest: str,
    profile_digest: str,
    model_registry_digest: str,
    skill_catalog_digest: str,
) -> dict[str, object]:
    expected = check_compatibility._expected_skill_validation_contract()
    cases: list[dict[str, object]] = []
    for index, case_id in enumerate(check_compatibility._SKILL_CASE_IDS, start=1):
        contract = expected["cases"][case_id]
        opaque = format(index, "064x")
        cases.append(
            {
                "caseId": case_id,
                "caseDigest": contract["caseDigest"],
                "skill": contract["skill"],
                "mode": contract["mode"],
                "modelClass": contract["modelClass"],
                "status": "pass",
                "checks": {
                    "structural": "pass",
                    "modelSelection": "pass",
                    "skillBinding": "pass",
                    "semantic": "pass",
                    "delegation": (
                        "pass" if contract["mode"] == "delegated" else "not-applicable"
                    ),
                    "trace": "pass",
                    "parentKnowledgeGraph": "pass",
                },
                "skillRef": f"pref_skill_{opaque}",
                "skillBodyRef": f"pref_skill_body_{opaque}",
                "runRef": f"pref_run_{opaque}",
                "traceRef": f"pref_trace_{opaque}",
                "langfuse": {
                    "lookupMethod": "exact-name",
                    "metadataOnly": True,
                    "traceName": f"graph_run:pref_run_{opaque}",
                    "matchCount": 1,
                    "linkage": "run-evidence",
                },
                "parentKnowledgeGraph": {
                    "readbackMethod": "exact-trace-name",
                    "matchCount": 1,
                },
                "errorCodes": [],
            }
        )
    unsigned: dict[str, object] = {
        "apiVersion": "graphos.io/v2",
        "kind": "PrebundledSkillValidationEvidence",
        "evidenceVersion": 2,
        "generatedAt": "2026-07-18T00:00:00Z",
        "release": {
            "id": release_id,
            "specificationDigest": release_specification_digest,
            "promotionEvidenceDigest": promotion_evidence_digest,
            "graphOsDigest": graph_os_digest,
            "engineDigest": engine_digest,
        },
        "runtime": {
            "configurationDigest": configuration_digest,
            "profileDigest": profile_digest,
            "modelRegistryDigest": model_registry_digest,
            "sequential": True,
            "metadataOnlyObservability": True,
        },
        "catalog": {
            "skillCount": 10,
            "skillCatalogDigest": skill_catalog_digest,
            "testCaseCount": 20,
            "testCatalogDigest": expected["testCatalogDigest"],
            "caseCatalogDigest": expected["caseCatalogDigest"],
        },
        "cases": cases,
        "result": {
            "status": "pass",
            "passedCases": 20,
            "totalCases": 20,
            "fullyPassedSkills": 10,
            "totalSkills": 10,
        },
        "privacy": {
            "containsPrompts": False,
            "containsModelOutput": False,
            "containsEndpoints": False,
            "containsCredentials": False,
            "containsIdentities": False,
            "containsFilesystemLocations": False,
            "containsRawTraceIdentifiers": False,
        },
    }
    return {
        **unsigned,
        "signature": {
            "algorithm": "ed25519",
            "keyId": "key:" + "7" * 64,
            "signature": "s" * 43,
            "subjectDigest": check_compatibility.canonical_digest(unsigned),
        },
    }


def _skill_validation_deployment(
    *,
    release_id: str,
    release_specification_digest: str,
    promotion_evidence_digest: str,
    graph_os_digest: str,
    engine_digest: str,
    configuration_digest: str,
    profile_digest: str,
    model_registry_digest: str,
) -> dict[str, object]:
    return {
        "apiVersion": "graphos.io/v2",
        "kind": "SkillValidationDeployment",
        "identityAuthority": {
            "mode": "ephemeral-https-loopback",
            "tokenTtlSeconds": 300,
            "tlsVerificationRequired": True,
            "lifecycleOwned": True,
            "renewableCredentialsRequired": True,
        },
        "release": {
            "id": release_id,
            "specificationReference": "FIXTURE_RELEASE_SPECIFICATION",
            "specificationDigest": release_specification_digest,
            "promotionEvidenceReference": "FIXTURE_PROMOTION_EVIDENCE",
            "promotionEvidenceDigest": promotion_evidence_digest,
            "agentUtilitiesSha256": graph_os_digest,
            "agentUtilitiesFileCount": 100,
            "distributionClosureSha256": graph_os_digest,
            "releasePythonSha256": graph_os_digest,
            "graphOsDigest": graph_os_digest,
            "engineDigest": engine_digest,
            "startCommandReference": "FIXTURE_GRAPHOS_START_COMMAND",
        },
        "runtime": {
            "configurationReference": "FIXTURE_AGENT_CONFIGURATION",
            "configurationDigest": configuration_digest,
            "profileReference": "FIXTURE_RUNTIME_PROFILE",
            "profileDigest": profile_digest,
            "endpointReference": "FIXTURE_GRAPHOS_ENDPOINT",
            "modelRegistry": {
                "digest": model_registry_digest,
                "modelCount": 2,
                "lightCount": 1,
                "normalCount": 1,
                "localPrivateTransportOnly": True,
                "referenceBackedCredentialsOnly": True,
                "literalPrivateModelCount": 2,
                "privateDnsModelCount": 0,
                "runtimePrivateResolutionRequired": True,
            },
        },
        "readiness": {
            "timeoutSeconds": 30,
            "pollIntervalMilliseconds": 100,
        },
        "validation": {
            "caseTimeoutSeconds": 60,
            "signerCommandReference": "FIXTURE_SKILL_SIGNER",
            "verifierCommandReference": "FIXTURE_SKILL_VERIFIER",
        },
        "shutdown": {"graceSeconds": 5},
    }


def _skill_validation_lifecycle_evidence(
    *,
    release_id: str,
    release_specification_digest: str,
    promotion_evidence_digest: str,
    graph_os_digest: str,
    engine_digest: str,
    configuration_digest: str,
    profile_digest: str,
    model_registry_digest: str,
    validation_evidence_digest: str,
) -> dict[str, object]:
    counts = {"before": 0, "running": 1, "after": 0}
    unsigned: dict[str, object] = {
        "apiVersion": "graphos.io/v2",
        "kind": "SkillValidationLifecycleEvidence",
        "evidenceVersion": 2,
        "release": {
            "id": release_id,
            "specificationDigest": release_specification_digest,
            "promotionEvidenceDigest": promotion_evidence_digest,
            "agentUtilitiesSha256": graph_os_digest,
            "agentUtilitiesFileCount": 100,
            "distributionClosureSha256": graph_os_digest,
            "releasePythonSha256": graph_os_digest,
            "graphOsDigest": graph_os_digest,
            "engineDigest": engine_digest,
        },
        "runtime": {
            "configurationDigest": configuration_digest,
            "profileDigest": profile_digest,
            "modelRegistryDigest": model_registry_digest,
        },
        "identityAuthority": {
            "mode": "ephemeral-https-loopback",
            "lifecycleCounts": counts,
            "tlsVerified": True,
            "renewableCredentialsProven": True,
            "tokenMintCount": 3,
            "reaped": True,
        },
        "modelTransportProof": {
            "modelCount": 2,
            "literalPrivateModelCount": 2,
            "privateDnsModelCount": 0,
            "privateDnsUniqueResolutionProven": True,
            "privateBoundaryProven": True,
            "dnsRebindingGuarded": True,
        },
        "processGate": {
            "globalGraphOs": counts,
            "candidateGraphOs": counts,
            "candidateEngine": counts,
            "terminalProcessCounts": {
                "langfuseMcpChildren": 0,
                "loopbackOidcFixtures": 0,
            },
            "engineExecutableDigest": engine_digest,
            "installedReleaseAttested": True,
            "reaped": True,
        },
        "validation": {
            "exitCode": 0,
            "evidenceDigest": validation_evidence_digest,
            "caseCount": 20,
        },
        "result": "pass",
        "errorCode": None,
        "privacy": {
            "containsEndpoints": False,
            "containsCredentials": False,
            "containsProfiles": False,
            "containsFilesystemLocations": False,
            "containsIdentities": False,
            "containsContent": False,
        },
    }
    return {
        **unsigned,
        "signature": {
            "algorithm": "ed25519",
            "keyId": "key:" + "4" * 64,
            "signature": "l" * 43,
            "subjectDigest": check_compatibility.canonical_digest(unsigned),
        },
    }


def _exact_artifact_closure_evidence(release_id: str) -> dict[str, object]:
    digest = "sha256:" + "6" * 64
    unsigned: dict[str, object] = {
        "apiVersion": "graphos.io/v1",
        "kind": "ExactArtifactClosureEvidence",
        "schemaVersion": 1,
        "releaseId": release_id,
        "status": "passed",
        "privacySafe": True,
        "release": {
            field: digest
            for field in (
                "promotionEvidenceSha256",
                "releaseSpecSha256",
                "campaignManifestSha256",
                "agentUtilitiesSha256",
                "distributionClosureSha256",
                "releasePythonSha256",
                "graphosSha256",
                "engineSha256",
                "harnessSha256",
                "testCatalogSha256",
            )
        },
        "campaigns": {
            "faultRestart": {
                "evidenceSha256": digest,
                "matrix_cases": 60,
                "mutation_families": 15,
            },
            "protocolAuthorization": {
                "evidenceSha256": digest,
                "data_path_cases": 14,
                "protocol_cases": 10,
            },
            "workItemAgentBus": {
                "evidenceSha256": digest,
                "work_item_cases": 8,
                "agent_bus_cases": 2,
            },
            "performance": {
                "evidenceSha256": digest,
                "scenario_families": 30,
                "ledger_rows": 54,
            },
            "multimodal": {
                "evidenceSha256": digest,
                "performanceEvidenceSha256": digest,
                "modalities": 4,
                "behavior_dimensions": 12,
                "fault_cases": 16,
            },
            "knowledgeBatch": {
                "evidenceSha256": digest,
                "families": 7,
                "requirements": 7,
                "snapshot_cases": 7,
            },
            "reasoningRepair": {"evidenceSha256": digest, "cases": 9},
            "exactLocal": {
                "evidenceSha256": digest,
                "campaignManifestSha256": digest,
                "gates": 7,
                "optimizer_families": 13,
                "optimizer_modalities": 14,
            },
            "permissionGovernance": {"evidenceSha256": digest, "cases": 8},
        },
        "gates": {gate: "passed" for gate in check_compatibility._EXACT_ARTIFACT_GATES},
    }
    return {
        **unsigned,
        "signature": {
            "algorithm": "ed25519",
            "keyId": "key:" + "5" * 64,
            "signature": "A" * 43,
            "subjectDigest": check_compatibility.canonical_digest(unsigned),
        },
    }


def _oci_vulnerability_scan_evidence(
    release_id: str, components: dict[str, dict[str, object]]
) -> dict[str, object]:
    finding_counts = {
        "unknown": 0,
        "low": 0,
        "medium": 0,
        "high": 0,
        "critical": 0,
    }
    subjects = {
        name: {
            "artifactFormat": "oci-layout-archive",
            "artifactDigest": components[name]["digest"],
            "archiveDigest": components[name]["digest"],
            "reportDigest": "sha256:" + format(index + 20, "064x"),
            "scanStartedAt": "2026-07-19T00:00:00Z",
            "scanCompletedAt": "2026-07-19T00:01:00Z",
            "archiveVerified": True,
            "offline": True,
            "telemetryDisabled": True,
            "scannerExitCode": 0,
            "findingCounts": dict(finding_counts),
            "status": "passed",
        }
        for index, name in enumerate(
            ("epistemic-graph", "agent-utilities", "langfuse-agent")
        )
    }
    database = {
        "manifestDigest": "sha256:" + "5" * 64,
        "archiveDigest": "sha256:" + "6" * 64,
        "databaseDigest": "sha256:" + "7" * 64,
        "metadataDigest": "sha256:" + "8" * 64,
        "attestationDigest": "sha256:" + "9" * 64,
        "updatedAt": "2026-07-18T00:00:00Z",
        "nextUpdate": "2026-07-20T00:00:00Z",
        "manifestDigestVerified": True,
        "archiveDigestVerified": True,
        "attestationSignatureVerified": True,
        "freshAtScan": True,
    }
    unsigned = {
        "apiVersion": "graphos.io/v1",
        "kind": "OciVulnerabilityScanEvidence",
        "schemaVersion": 1,
        "releaseId": release_id,
        "generatedAt": "2026-07-19T00:01:00Z",
        "status": "passed",
        "execution": {
            "mode": "offline-private-layout",
            "sequential": True,
            "maxParallelism": 1,
            "networkAccess": False,
            "tlsProfileRef": "pref_tls_" + "d" * 64,
        },
        "scanner": {
            "name": "trivy",
            "version": "0.72.0",
            "binaryDigest": "sha256:" + "1" * 64,
            "releaseBundleDigest": "sha256:" + "2" * 64,
            "verifierDigest": "sha256:" + "3" * 64,
            "attestationDigest": "sha256:" + "4" * 64,
            "releaseSignatureVerified": True,
            "attestationSignatureVerified": True,
        },
        "databases": {
            "vulnerability": {**database, "schemaVersion": 2},
            "java": {**database, "schemaVersion": 1},
        },
        "policy": generate_oci_vulnerability_scan_evidence._policy(),
        "subjects": subjects,
        "privacy": {
            "containsRawFindings": False,
            "containsPackageInventory": False,
            "containsVulnerabilityIdentifiers": False,
            "containsEndpoints": False,
            "containsFilesystemLocations": False,
            "containsCredentials": False,
            "containsIdentities": False,
        },
    }
    return {
        **unsigned,
        "signature": {
            "algorithm": "ed25519",
            "keyId": "key:" + "a" * 64,
            "signature": "S" * 43,
            "subjectDigest": check_compatibility.canonical_digest(unsigned),
        },
    }


def _release_fixture(tmp_path: Path) -> tuple[dict, dict, Path, Path]:
    matrix_path = (
        Path(__file__).resolve().parents[3] / "deploy/release/compatibility-matrix.yml"
    )
    matrix = yaml.safe_load(matrix_path.read_text(encoding="utf-8"))
    ledger_entries = _connector_ledger_entries()
    unsigned_ledger = {
        "apiVersion": "graphos.io/v1",
        "kind": "ConnectorLiveCertificationLedger",
        "ledgerVersion": 1,
        "entryCount": len(ledger_entries),
        "entries": ledger_entries,
    }
    (tmp_path / "connector-ledger.json").write_text(
        json.dumps(
            {
                **unsigned_ledger,
                "signature": {
                    "scheme": "fixture-signature",
                    "subjectDigest": check_compatibility.canonical_digest(
                        unsigned_ledger
                    ),
                    "bundleDigest": "sha256:" + "e" * 64,
                    "signerIdentityDigest": "sha256:" + "f" * 64,
                    "value": "fixture-signature-value",
                    "verifierEnv": "CONNECTOR_LEDGER_VERIFIER",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    source_freeze_authority = _source_freeze_evidence(tmp_path / "source-freeze.json")
    components: dict[str, dict] = {}
    for index, name in enumerate(matrix["releaseTrain"]["assemblyOrder"], start=1):
        expected = matrix["components"][name]
        digest = "sha256:" + format(index, "064x")
        kind = expected["artifactKind"]
        capabilities = list(expected.get("requiredCapabilities") or ())
        entry_count = expected.get("exactEntries")
        artifact = (
            f"oci:{name}@{digest}" if kind == "oci" else f"catalog:{name}@{digest}"
        )
        component = {
            "version": str(expected["version"]).removeprefix("=="),
            "kind": kind,
            "artifact": artifact,
            "digest": digest,
            "evidence": _write_release_evidence(
                tmp_path,
                name,
                str(expected["version"]).removeprefix("=="),
                kind,
                digest.removeprefix("sha256:"),
                capabilities,
                int(entry_count) if entry_count is not None else None,
                source_freeze_authority["snapshotDigest"],
                source_freeze_authority["evidenceDigest"],
            ),
            "signatureVerifierEnv": "COMPONENT_SIGNATURE_VERIFIER",
            "capabilities": capabilities,
        }
        if entry_count is not None:
            component["entryCount"] = entry_count
        components[name] = component
    matrix_digest = check_compatibility.file_digest(matrix_path)
    configuration = check_compatibility.release_configuration_document(
        release_id="release-fixture-1",
        matrix=matrix,
        matrix_digest=matrix_digest,
    )
    migration = check_compatibility.release_migration_plan_document(
        release_id="release-fixture-1",
        matrix=matrix,
        matrix_digest=matrix_digest,
        index_migration_catalog_digest=components["index-migrations"]["digest"],
        index_migration_count=components["index-migrations"]["entryCount"],
    )
    (tmp_path / "configuration.json").write_text(
        json.dumps(configuration, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    (tmp_path / "migration.json").write_text(
        json.dumps(migration, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    configuration_digest = check_compatibility.file_digest(
        tmp_path / "configuration.json"
    )
    release_binding_digest = "sha256:" + "6" * 64
    profile_digest = "sha256:" + "8" * 64
    model_registry_digest = "sha256:" + "4" * 64
    (tmp_path / "skill-matrix.json").write_text(
        json.dumps(
            _skill_validation_evidence(
                release_id="release-fixture-1",
                release_specification_digest=release_binding_digest,
                promotion_evidence_digest=release_binding_digest,
                graph_os_digest=release_binding_digest,
                engine_digest=release_binding_digest,
                configuration_digest=configuration_digest,
                profile_digest=profile_digest,
                model_registry_digest=model_registry_digest,
                skill_catalog_digest=components["prebundled-skills"]["digest"],
            ),
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "skill-deployment.json").write_text(
        json.dumps(
            _skill_validation_deployment(
                release_id="release-fixture-1",
                release_specification_digest=release_binding_digest,
                promotion_evidence_digest=release_binding_digest,
                graph_os_digest=release_binding_digest,
                engine_digest=release_binding_digest,
                configuration_digest=configuration_digest,
                profile_digest=profile_digest,
                model_registry_digest=model_registry_digest,
            ),
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    skill_matrix_digest = check_compatibility.file_digest(
        tmp_path / "skill-matrix.json"
    )
    (tmp_path / "skill-lifecycle.json").write_text(
        json.dumps(
            _skill_validation_lifecycle_evidence(
                release_id="release-fixture-1",
                release_specification_digest=release_binding_digest,
                promotion_evidence_digest=release_binding_digest,
                graph_os_digest=release_binding_digest,
                engine_digest=release_binding_digest,
                configuration_digest=configuration_digest,
                profile_digest=profile_digest,
                model_registry_digest=model_registry_digest,
                validation_evidence_digest=skill_matrix_digest,
            ),
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "exact-artifact-closure.json").write_text(
        json.dumps(
            _exact_artifact_closure_evidence("release-fixture-1"), sort_keys=True
        )
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "oci-vulnerability-scan.json").write_text(
        json.dumps(
            _oci_vulnerability_scan_evidence("release-fixture-1", components),
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    assembly = {
        "apiVersion": "graphos.io/v1",
        "kind": "ReleaseAssembly",
        "releaseId": "release-fixture-1",
        "sourceFreezeEvidence": "source-freeze.json",
        "configuration": "configuration.json",
        "migrationPlan": "migration.json",
        "certifications": {
            "connectorLiveCertificationLedger": "connector-ledger.json",
            "prebundledSkillValidationMatrix": "skill-matrix.json",
            "skillValidationDeployment": "skill-deployment.json",
            "skillValidationLifecycleEvidence": "skill-lifecycle.json",
            "exactArtifactClosureEvidence": "exact-artifact-closure.json",
            "ociVulnerabilityScanEvidence": "oci-vulnerability-scan.json",
        },
        "components": components,
    }
    output = tmp_path / "release.json"
    return assembly, matrix, matrix_path, output


def test_current_release_matrix_schema_is_exact_and_current_only() -> None:
    matrix_path = ROOT / "deploy/release/compatibility-matrix.yml"
    schema_path = ROOT / "deploy/release/compatibility-matrix.schema.json"
    matrix = yaml.safe_load(matrix_path.read_text(encoding="utf-8"))
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema)

    validator.validate(matrix)
    assert matrix["matrixVersion"] == 2
    assert matrix["runtime"] == {
        "pythonVersion": "3.12",
        "baseImage": (
            "python:3.12-slim@sha256:"
            "57cd7c3a7a273101a6485ba99423ee568157882804b1124b4dd04266317710de"
        ),
        "pythonDependencyMode": "offline-hash-locked-wheelhouse",
        "offlineTargets": {
            "epistemic-graph": "release-local",
            "agent-utilities": "agent-local",
            "langfuse-agent": "mcp-local",
        },
    }
    assert tuple(matrix["releaseTrain"]["assemblyOrder"]) == (
        "epistemic-operations-protocol",
        "epistemic-graph",
        "agent-utilities",
        "langfuse-agent",
        "connector-bundles",
        "prebundled-skills",
        "ontology-lock",
        "index-migrations",
    )
    assert matrix["components"]["connector-bundles"]["exactEntries"] == 65
    assert matrix["components"]["index-migrations"]["exactEntries"] == 1

    runtime_drift = json.loads(json.dumps(matrix))
    runtime_drift["runtime"]["pythonVersion"] = "3.11"
    assert not validator.is_valid(runtime_drift)
    with pytest.raises(
        check_compatibility.CompatibilityError,
        match="runtime contract is not exact",
    ):
        check_compatibility.validate_compatibility_matrix(runtime_drift)

    target_drift = json.loads(json.dumps(matrix))
    target_drift["runtime"]["offlineTargets"]["epistemic-graph"] = "runtime"
    assert not validator.is_valid(target_drift)
    with pytest.raises(
        check_compatibility.CompatibilityError,
        match="runtime contract is not exact",
    ):
        check_compatibility.validate_compatibility_matrix(target_drift)

    minimum = json.loads(json.dumps(matrix))
    minimum["components"]["epistemic-graph"]["version"] = ">=2.23.1"
    assert not validator.is_valid(minimum)
    with pytest.raises(check_compatibility.CompatibilityError, match="pin exactly"):
        check_compatibility.validate_compatibility_matrix(minimum)

    wrong_count = json.loads(json.dumps(matrix))
    wrong_count["components"]["connector-bundles"]["exactEntries"] = 64
    assert not validator.is_valid(wrong_count)
    with pytest.raises(check_compatibility.CompatibilityError, match="exactly"):
        check_compatibility.validate_compatibility_matrix(wrong_count)

    alias = json.loads(json.dumps(matrix))
    alias["components"]["epistemic-graph"]["version"] = "==02.23.1"
    assert not validator.is_valid(alias)
    with pytest.raises(
        check_compatibility.CompatibilityError, match="canonical current spelling"
    ):
        check_compatibility.validate_compatibility_matrix(alias)

    unknown = json.loads(json.dumps(matrix))
    unknown["components"]["epistemic-graph"]["compatibilityAlias"] = "current"
    assert not validator.is_valid(unknown)
    with pytest.raises(
        check_compatibility.CompatibilityError, match="keys are not exact"
    ):
        check_compatibility.validate_compatibility_matrix(unknown)

    reordered = json.loads(json.dumps(matrix))
    reordered["releaseTrain"]["assemblyOrder"][0:2] = reversed(
        reordered["releaseTrain"]["assemblyOrder"][0:2]
    )
    assert not validator.is_valid(reordered)
    with pytest.raises(
        check_compatibility.CompatibilityError, match="order is not exact"
    ):
        check_compatibility.validate_compatibility_matrix(reordered)

    extra_dependency = json.loads(json.dumps(matrix))
    extra_dependency["components"]["epistemic-graph"]["dependsOn"] = {
        "epistemic-operations-protocol": "==1"
    }
    assert not validator.is_valid(extra_dependency)
    with pytest.raises(
        check_compatibility.CompatibilityError, match="dependency topology is not exact"
    ):
        check_compatibility.validate_compatibility_matrix(extra_dependency)


def test_typed_release_inputs_are_deterministic_schema_valid_and_path_free() -> None:
    matrix_path = ROOT / "deploy/release/compatibility-matrix.yml"
    migration_catalog = ROOT / "deploy/release/index-migrations.catalog.json"
    configuration = generate_release_inputs.generate_configuration(
        release_id="release-fixture-1",
        matrix_path=matrix_path,
    )
    migration = generate_release_inputs.generate_migration_plan(
        release_id="release-fixture-1",
        matrix_path=matrix_path,
        index_migration_catalog_path=migration_catalog,
    )

    assert configuration == generate_release_inputs.generate_configuration(
        release_id="release-fixture-1",
        matrix_path=matrix_path,
    )
    assert migration == generate_release_inputs.generate_migration_plan(
        release_id="release-fixture-1",
        matrix_path=matrix_path,
        index_migration_catalog_path=migration_catalog,
    )
    for name, value in (
        ("release-configuration.schema.json", configuration),
        ("release-migration-plan.schema.json", migration),
    ):
        schema = json.loads((ROOT / "deploy/release" / name).read_text())
        Draft202012Validator.check_schema(schema)
        Draft202012Validator(schema).validate(value)
    retained = json.dumps([configuration, migration], sort_keys=True)
    assert str(ROOT) not in retained
    assert "://" not in retained.replace(
        "python:3.12-slim@sha256:", "python-image@sha256:"
    )


def test_release_inputs_reject_empty_or_mixed_source_authority() -> None:
    matrix_path = ROOT / "deploy/release/compatibility-matrix.yml"
    matrix = yaml.safe_load(matrix_path.read_text(encoding="utf-8"))
    with pytest.raises(check_compatibility.CompatibilityError):
        check_compatibility.validate_release_configuration(
            {},
            release_id="release-fixture-1",
            matrix=matrix,
            matrix_digest=check_compatibility.file_digest(matrix_path),
        )
    repeated = {
        name: {
            "sourceSnapshotDigest": "sha256:" + "a" * 64,
            "sourceEvidenceDigest": "sha256:" + "b" * 64,
        }
        for name in check_compatibility._RELEASE_ORDER
    }
    with pytest.raises(
        check_compatibility.CompatibilityError,
        match="authority is absent or invalid",
    ):
        check_compatibility._validate_single_source_freeze(repeated, None)
    with pytest.raises(
        check_compatibility.CompatibilityError,
        match="one source-freeze authority",
    ):
        check_compatibility._validate_single_source_freeze(
            repeated,
            {
                "snapshotDigest": "sha256:" + "c" * 64,
                "evidenceDigest": "sha256:" + "d" * 64,
            },
        )


def test_release_and_operational_schemas_separate_local_and_external_gates() -> None:
    manifest_schema = json.loads(
        (ROOT / "deploy/release/release-manifest.schema.json").read_text(
            encoding="utf-8"
        )
    )
    operational_schema = json.loads(
        (ROOT / "deploy/release/operational-evidence.schema.json").read_text(
            encoding="utf-8"
        )
    )
    Draft202012Validator.check_schema(manifest_schema)
    Draft202012Validator.check_schema(operational_schema)
    components = {
        "epistemic-operations-protocol",
        "epistemic-graph",
        "agent-utilities",
        "langfuse-agent",
        "connector-bundles",
        "prebundled-skills",
        "ontology-lock",
        "index-migrations",
    }
    certifications = {
        "connectorLiveCertificationLedger",
        "prebundledSkillValidationMatrix",
        "skillValidationDeployment",
        "skillValidationLifecycleEvidence",
        "exactArtifactClosureEvidence",
        "ociVulnerabilityScanEvidence",
    }

    assert "manifestState" in manifest_schema["required"]
    assert "sourceFreezeEvidenceDigest" in manifest_schema["required"]
    assert "signature" not in manifest_schema["required"]
    assert manifest_schema["properties"]["manifestState"]["enum"] == [
        "unsigned-local-binder",
        "signed-release",
    ]
    manifest_certifications = manifest_schema["properties"]["certificationDigests"]
    assert set(manifest_certifications["required"]) == certifications
    assert manifest_certifications["additionalProperties"] is False
    manifest_evidence = manifest_schema["properties"]["evidence"]
    assert "sourceFreezeEvidence" in manifest_evidence["required"]
    gate_authorities = manifest_schema["$defs"]["gateAuthority"]["properties"][
        "authority"
    ]["enum"]
    assert "certification:skillValidationLifecycleEvidence" in gate_authorities
    assert "certification:skillValidationDeployment" not in gate_authorities
    release_components = operational_schema["properties"]["release"]["properties"][
        "componentDigests"
    ]
    assert set(release_components["required"]) == components
    assert release_components["additionalProperties"] is False
    operational_certifications = operational_schema["properties"]["release"][
        "properties"
    ]["certificationDigests"]
    assert set(operational_certifications["required"]) == certifications
    assert operational_certifications["additionalProperties"] is False
    assert "signature" in operational_schema["required"]
    privacy = operational_schema["properties"]["privacy"]["properties"]
    assert privacy["containsDirectIdentifiers"]["const"] is False
    assert privacy["containsEndpoints"]["const"] is False
    assert privacy["containsFilesystemLocations"]["const"] is False


def test_skill_lifecycle_digest_is_authoritative_for_every_required_gate() -> None:
    authority = "certification:skillValidationLifecycleEvidence"

    assert {
        gate
        for gate, authorities in check_compatibility._EXACT_GATE_AUTHORITIES.items()
        if authority in authorities
    } == {"G-07", "G-12", "G-18", "G-27", "G-29", "G-30", "G-36", "G-38"}


def test_oci_scan_digest_is_authoritative_for_supply_chain_and_security() -> None:
    authority = "certification:ociVulnerabilityScanEvidence"

    assert {
        gate
        for gate, authorities in check_compatibility._EXACT_GATE_AUTHORITIES.items()
        if authority in authorities
    } == {"G-22", "G-38"}


def test_release_assembler_rejects_repeated_freeze_digests_without_evidence(
    tmp_path: Path,
) -> None:
    assembly, matrix, matrix_path, output = _release_fixture(tmp_path)
    assembly.pop("sourceFreezeEvidence")

    with pytest.raises(
        assemble_manifest.AssemblyError,
        match="release assembly keys are not exact",
    ):
        assemble_manifest.assemble(
            assembly,
            matrix,
            matrix_path=matrix_path,
            output_path=output,
        )


def test_release_assembler_rejects_tampered_source_freeze_gate(
    tmp_path: Path,
) -> None:
    assembly, matrix, matrix_path, output = _release_fixture(tmp_path)
    source_freeze = tmp_path / assembly["sourceFreezeEvidence"]
    evidence = json.loads(source_freeze.read_text(encoding="utf-8"))
    evidence["gates"][0]["source_status"] = "not-applicable"
    source_freeze.write_text(
        json.dumps(evidence, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        check_compatibility.CompatibilityError,
        match="source-freeze gate evidence is not exact",
    ):
        assemble_manifest.assemble(
            assembly,
            matrix,
            matrix_path=matrix_path,
            output_path=output,
        )


def test_release_manifest_rejects_source_freeze_digest_drift(tmp_path: Path) -> None:
    assembly, matrix, matrix_path, output = _release_fixture(tmp_path)
    manifest = assemble_manifest.assemble(
        assembly,
        matrix,
        matrix_path=matrix_path,
        output_path=output,
    )
    manifest["sourceFreezeEvidenceDigest"] = "sha256:" + "f" * 64

    with pytest.raises(
        check_compatibility.CompatibilityError,
        match="source-freeze digest differs from referenced evidence",
    ):
        check_compatibility.verify_release_manifest(
            manifest,
            matrix,
            matrix_path=matrix_path,
            manifest_path=output,
            verify_signatures=False,
            require_manifest_signature=False,
        )


def test_release_manifest_independently_rejects_tampered_source_freeze_gate(
    tmp_path: Path,
) -> None:
    assembly, matrix, matrix_path, output = _release_fixture(tmp_path)
    manifest = assemble_manifest.assemble(
        assembly,
        matrix,
        matrix_path=matrix_path,
        output_path=output,
    )
    source_freeze = tmp_path / manifest["evidence"]["sourceFreezeEvidence"]
    evidence = json.loads(source_freeze.read_text(encoding="utf-8"))
    evidence["gates"][0]["source_status"] = "not-applicable"
    source_freeze.write_text(
        json.dumps(evidence, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    manifest["sourceFreezeEvidenceDigest"] = check_compatibility.file_digest(
        source_freeze
    )

    with pytest.raises(
        check_compatibility.CompatibilityError,
        match="source-freeze gate evidence is not exact",
    ):
        check_compatibility.verify_release_manifest(
            manifest,
            matrix,
            matrix_path=matrix_path,
            manifest_path=output,
            verify_signatures=False,
            require_manifest_signature=False,
        )


def test_release_assembler_rejects_oci_scan_archive_binding_drift(
    tmp_path: Path,
) -> None:
    assembly, matrix, matrix_path, output = _release_fixture(tmp_path)
    path = tmp_path / "oci-vulnerability-scan.json"
    evidence = json.loads(path.read_text(encoding="utf-8"))
    evidence["subjects"]["agent-utilities"]["archiveDigest"] = "sha256:" + "f" * 64
    unsigned = {key: value for key, value in evidence.items() if key != "signature"}
    evidence["signature"]["subjectDigest"] = check_compatibility.canonical_digest(
        unsigned
    )
    path.write_text(json.dumps(evidence, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(
        check_compatibility.CompatibilityError,
        match="OCI vulnerability scan binding differs for agent-utilities",
    ):
        assemble_manifest.assemble(
            assembly, matrix, matrix_path=matrix_path, output_path=output
        )


def test_release_assembler_rejects_normalized_version_alias(tmp_path: Path) -> None:
    assembly, matrix, matrix_path, output = _release_fixture(tmp_path)
    assembly["components"]["epistemic-graph"]["version"] = "2.23.1+local"

    with pytest.raises(assemble_manifest.AssemblyError, match="current matrix version"):
        assemble_manifest.assemble(
            assembly, matrix, matrix_path=matrix_path, output_path=output
        )


def test_release_gate_requires_three_distinct_oci_subjects(tmp_path: Path) -> None:
    assembly, _matrix, _matrix_path, _output = _release_fixture(tmp_path)
    components = assembly["components"]
    shared_digest = components["epistemic-graph"]["digest"]
    components["agent-utilities"]["digest"] = shared_digest
    components["agent-utilities"]["artifact"] = f"oci:agent-utilities@{shared_digest}"

    with pytest.raises(
        check_compatibility.CompatibilityError,
        match="three distinct OCI subject digests",
    ):
        check_compatibility._validate_distinct_oci_subjects(components)


@pytest.mark.parametrize(
    "evidence_kind",
    ("source", "sbom", "provenance", "signatureBundle"),
)
def test_release_assembler_semantically_rejects_each_component_evidence_kind(
    tmp_path: Path,
    evidence_kind: str,
) -> None:
    assembly, matrix, matrix_path, output = _release_fixture(tmp_path)
    reference = assembly["components"]["agent-utilities"]["evidence"][evidence_kind]
    (tmp_path / reference).write_text("{}\n", encoding="utf-8")

    with pytest.raises(check_compatibility.CompatibilityError):
        assemble_manifest.assemble(
            assembly, matrix, matrix_path=matrix_path, output_path=output
        )


def test_release_assembler_rejects_semantically_tampered_skill_matrix(
    tmp_path: Path,
) -> None:
    assembly, matrix, matrix_path, output = _release_fixture(tmp_path)
    path = tmp_path / "skill-matrix.json"
    evidence = json.loads(path.read_text(encoding="utf-8"))
    evidence["result"]["passedCases"] = 19
    unsigned = {key: value for key, value in evidence.items() if key != "signature"}
    evidence["signature"]["subjectDigest"] = check_compatibility.canonical_digest(
        unsigned
    )
    path.write_text(json.dumps(evidence, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(
        check_compatibility.CompatibilityError,
        match="current schema",
    ):
        assemble_manifest.assemble(
            assembly, matrix, matrix_path=matrix_path, output_path=output
        )


@pytest.mark.parametrize(
    "missing",
    ("skillValidationDeployment", "skillValidationLifecycleEvidence"),
)
def test_release_assembler_requires_current_skill_lifecycle_inputs(
    tmp_path: Path,
    missing: str,
) -> None:
    assembly, matrix, matrix_path, output = _release_fixture(tmp_path)
    assembly["certifications"].pop(missing)
    assembly["certifications"][f"legacy{missing[0].upper()}{missing[1:]}"] = (
        "skill-lifecycle.json"
    )

    with pytest.raises(
        assemble_manifest.AssemblyError,
        match="certification evidence catalog is not exact",
    ):
        assemble_manifest.assemble(
            assembly,
            matrix,
            matrix_path=matrix_path,
            output_path=output,
        )


@pytest.mark.parametrize(
    ("target", "field", "value", "message"),
    (
        (
            "skill-lifecycle.json",
            ("processGate", "candidateEngine", "running"),
            0,
            "current schema",
        ),
        (
            "skill-lifecycle.json",
            ("processGate", "terminalProcessCounts", "langfuseMcpChildren"),
            1,
            "current schema",
        ),
        (
            "skill-lifecycle.json",
            ("validation", "evidenceDigest"),
            "sha256:" + "5" * 64,
            "lifecycle evidence binding",
        ),
        (
            "skill-deployment.json",
            ("runtime", "modelRegistry", "localPrivateTransportOnly"),
            False,
            "current schema",
        ),
        (
            "skill-matrix.json",
            ("release", "engineDigest"),
            "sha256:" + "5" * 64,
            "release binding differs",
        ),
    ),
)
def test_release_assembler_rejects_unbound_skill_lifecycle_authority(
    tmp_path: Path,
    target: str,
    field: tuple[str, ...],
    value: object,
    message: str,
) -> None:
    assembly, matrix, matrix_path, output = _release_fixture(tmp_path)
    path = tmp_path / target
    evidence = json.loads(path.read_text(encoding="utf-8"))
    nested = evidence
    for name in field[:-1]:
        nested = nested[name]
    nested[field[-1]] = value
    if "signature" in evidence:
        unsigned = {key: item for key, item in evidence.items() if key != "signature"}
        evidence["signature"]["subjectDigest"] = check_compatibility.canonical_digest(
            unsigned
        )
    path.write_text(json.dumps(evidence, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(check_compatibility.CompatibilityError, match=message):
        assemble_manifest.assemble(
            assembly,
            matrix,
            matrix_path=matrix_path,
            output_path=output,
        )


def test_skill_lifecycle_verifier_uses_the_bound_deployment_selector(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _release_fixture(tmp_path)
    deployment = json.loads(
        (tmp_path / "skill-deployment.json").read_text(encoding="utf-8")
    )
    evidence = json.loads(
        (tmp_path / "skill-lifecycle.json").read_text(encoding="utf-8")
    )
    observed: list[list[str]] = []

    def adapter(
        command: list[str],
        payload: bytes,
        *,
        maximum: int,
        timeout: int = 120,
    ) -> tuple[int, bytes, bytes]:
        del maximum, timeout
        observed.append(command)
        assert json.loads(payload) == evidence
        response = {
            "verified": True,
            "subjectDigest": evidence["signature"]["subjectDigest"],
            "keyId": evidence["signature"]["keyId"],
        }
        return 0, json.dumps(response).encode(), b""

    verifier = tmp_path / "fixture-verifier"
    verifier.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    verifier.chmod(0o700)
    monkeypatch.setenv("FIXTURE_SKILL_VERIFIER", json.dumps([str(verifier)]))
    monkeypatch.setattr(check_compatibility, "_bounded_adapter", adapter)

    check_compatibility._verify_skill_validation_evidence(
        evidence,
        deployment=deployment,
        field="skillValidationLifecycleEvidence",
    )
    assert observed == [[str(verifier.resolve())]]


def test_skill_lifecycle_verifier_rejects_shell_adapter_before_launch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _release_fixture(tmp_path)
    deployment = json.loads(
        (tmp_path / "skill-deployment.json").read_text(encoding="utf-8")
    )
    evidence = json.loads(
        (tmp_path / "skill-lifecycle.json").read_text(encoding="utf-8")
    )
    monkeypatch.setenv(
        "FIXTURE_SKILL_VERIFIER", json.dumps(["/bin/sh", "-c", "exit 0"])
    )

    def not_launched(*_args: object, **_kwargs: object) -> tuple[int, bytes, bytes]:
        raise AssertionError("untrusted verifier must not be launched")

    monkeypatch.setattr(check_compatibility, "_bounded_adapter", not_launched)

    with pytest.raises(
        check_compatibility.CompatibilityError,
        match="skill validation verifier is unavailable",
    ):
        check_compatibility._verify_skill_validation_evidence(
            evidence,
            deployment=deployment,
            field="skillValidationLifecycleEvidence",
        )


def test_release_assembler_rejects_semantically_tampered_exact_artifact_closure(
    tmp_path: Path,
) -> None:
    assembly, matrix, matrix_path, output = _release_fixture(tmp_path)
    path = tmp_path / "exact-artifact-closure.json"
    evidence = json.loads(path.read_text(encoding="utf-8"))
    evidence["campaigns"]["reasoningRepair"]["cases"] = 8
    unsigned = {key: value for key, value in evidence.items() if key != "signature"}
    evidence["signature"]["subjectDigest"] = check_compatibility.canonical_digest(
        unsigned
    )
    path.write_text(json.dumps(evidence, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(
        check_compatibility.CompatibilityError,
        match="reasoningRepair",
    ):
        assemble_manifest.assemble(
            assembly, matrix, matrix_path=matrix_path, output_path=output
        )


def test_release_gate_rejects_non_authoritative_exact_gate_mapping(
    tmp_path: Path,
) -> None:
    assembly, matrix, matrix_path, output = _release_fixture(tmp_path)
    manifest = assemble_manifest.assemble(
        assembly, matrix, matrix_path=matrix_path, output_path=output
    )
    manifest["exactGateEvidence"]["G-03"][0]["digest"] = "sha256:" + "f" * 64

    with pytest.raises(
        check_compatibility.CompatibilityError,
        match="not authoritative",
    ):
        check_compatibility.verify_release_manifest(
            manifest,
            matrix,
            matrix_path=matrix_path,
            manifest_path=output,
            verify_signatures=False,
            require_manifest_signature=False,
        )


def test_release_assembler_opens_evidence_and_signs_only_via_external_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assembly, matrix, matrix_path, output = _release_fixture(tmp_path)
    unsigned = assemble_manifest.assemble(
        assembly, matrix, matrix_path=matrix_path, output_path=output
    )
    manifest_schema = json.loads(
        (ROOT / "deploy/release/release-manifest.schema.json").read_text(
            encoding="utf-8"
        )
    )
    Draft202012Validator(manifest_schema).validate(unsigned)
    assert unsigned["manifestState"] == "unsigned-local-binder"
    assert set(unsigned["exactGateEvidence"]) == set(
        check_compatibility._EXACT_GATE_AUTHORITIES
    )
    assert unsigned["exactGateEvidence"]["G-01"] == [
        {
            "authority": "certification:exactArtifactClosureEvidence",
            "digest": unsigned["certificationDigests"]["exactArtifactClosureEvidence"],
        }
    ]

    def signer(_: str, payload: bytes) -> dict[str, str]:
        subject = check_compatibility.canonical_digest(json.loads(payload))
        return {
            "scheme": "fixture-signature",
            "subjectDigest": subject,
            "bundleDigest": "sha256:" + "a" * 64,
            "signerIdentityDigest": "sha256:" + "b" * 64,
            "signature": "fixture-signature-value",
        }

    monkeypatch.setattr(assemble_manifest, "_external_command", signer)
    monkeypatch.setattr(check_compatibility, "_verify_signature", lambda *args: None)
    monkeypatch.setattr(
        check_compatibility,
        "_verify_exact_artifact_closure",
        lambda *args: None,
    )
    monkeypatch.setattr(
        check_compatibility,
        "_verify_skill_validation_evidence",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        check_compatibility,
        "_verify_oci_vulnerability_scan_evidence",
        lambda *args: None,
    )
    signed = assemble_manifest.sign(
        unsigned,
        matrix,
        matrix_path=matrix_path,
        manifest_path=output,
        signer_env="RELEASE_SIGNER_COMMAND",
        verifier_env="RELEASE_VERIFIER_COMMAND",
    )
    Draft202012Validator(manifest_schema).validate(signed)
    assert signed["manifestState"] == "signed-release"
    report = check_compatibility.verify_release_manifest(
        signed,
        matrix,
        matrix_path=matrix_path,
        manifest_path=output,
        verify_signatures=False,
    )
    assert report["ok"] is True
    assert report["signaturesVerified"] is False

    (
        tmp_path / assembly["components"]["agent-utilities"]["evidence"]["sbom"]
    ).write_text("{}\n", encoding="utf-8")
    with pytest.raises(check_compatibility.CompatibilityError):
        check_compatibility.verify_release_manifest(
            signed,
            matrix,
            manifest_path=output,
            verify_signatures=False,
        )


def test_release_assembly_generator_requires_exact_eight_component_set(
    tmp_path: Path,
) -> None:
    assembly, matrix, matrix_path, output = _release_fixture(tmp_path)
    declarations = tmp_path / "components"
    declarations.mkdir()
    component_files: dict[str, Path] = {}
    for name, declaration in assembly["components"].items():
        path = declarations / f"{name}.json"
        path.write_text(json.dumps(declaration), encoding="utf-8")
        component_files[name] = path

    generated = generate_release_assembly.generate(
        release_id=assembly["releaseId"],
        matrix_path=matrix_path,
        output_path=output,
        source_freeze_evidence=assembly["sourceFreezeEvidence"],
        configuration=assembly["configuration"],
        migration_plan=assembly["migrationPlan"],
        connector_ledger=assembly["certifications"]["connectorLiveCertificationLedger"],
        skill_validation_matrix=assembly["certifications"][
            "prebundledSkillValidationMatrix"
        ],
        skill_validation_deployment=assembly["certifications"][
            "skillValidationDeployment"
        ],
        skill_validation_lifecycle_evidence=assembly["certifications"][
            "skillValidationLifecycleEvidence"
        ],
        exact_artifact_closure=assembly["certifications"][
            "exactArtifactClosureEvidence"
        ],
        oci_vulnerability_scan=assembly["certifications"][
            "ociVulnerabilityScanEvidence"
        ],
        component_files=component_files,
    )
    schema = json.loads(
        (ROOT / "deploy/release/release-assembly.schema.json").read_text(
            encoding="utf-8"
        )
    )
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(generated)
    assert generated == assembly
    for unsafe_reference in (
        "evidence/../secret.json",
        "./secret.json",
        "evidence//secret.json",
        "evidence/.secret.json",
    ):
        traversal = json.loads(json.dumps(generated))
        traversal["configuration"] = unsafe_reference
        assert not Draft202012Validator(schema).is_valid(traversal)

    with pytest.raises(
        generate_release_assembly.ReleaseAssemblyError,
        match="release-relative",
    ):
        generate_release_assembly.generate(
            release_id=assembly["releaseId"],
            matrix_path=matrix_path,
            output_path=output,
            source_freeze_evidence=assembly["sourceFreezeEvidence"],
            configuration="/environment/configuration.json",
            migration_plan=assembly["migrationPlan"],
            connector_ledger=assembly["certifications"][
                "connectorLiveCertificationLedger"
            ],
            skill_validation_matrix=assembly["certifications"][
                "prebundledSkillValidationMatrix"
            ],
            skill_validation_deployment=assembly["certifications"][
                "skillValidationDeployment"
            ],
            skill_validation_lifecycle_evidence=assembly["certifications"][
                "skillValidationLifecycleEvidence"
            ],
            exact_artifact_closure=assembly["certifications"][
                "exactArtifactClosureEvidence"
            ],
            oci_vulnerability_scan=assembly["certifications"][
                "ociVulnerabilityScanEvidence"
            ],
            component_files=component_files,
        )

    component_files.pop("ontology-lock")
    with pytest.raises(
        generate_release_assembly.ReleaseAssemblyError,
        match="differs from the matrix",
    ):
        generate_release_assembly.generate(
            release_id=assembly["releaseId"],
            matrix_path=matrix_path,
            output_path=output,
            source_freeze_evidence=assembly["sourceFreezeEvidence"],
            configuration=assembly["configuration"],
            migration_plan=assembly["migrationPlan"],
            connector_ledger=assembly["certifications"][
                "connectorLiveCertificationLedger"
            ],
            skill_validation_matrix=assembly["certifications"][
                "prebundledSkillValidationMatrix"
            ],
            skill_validation_deployment=assembly["certifications"][
                "skillValidationDeployment"
            ],
            skill_validation_lifecycle_evidence=assembly["certifications"][
                "skillValidationLifecycleEvidence"
            ],
            exact_artifact_closure=assembly["certifications"][
                "exactArtifactClosureEvidence"
            ],
            oci_vulnerability_scan=assembly["certifications"][
                "ociVulnerabilityScanEvidence"
            ],
            component_files=component_files,
        )


def test_release_assembly_generator_rejects_output_alias_and_hardlink(
    tmp_path: Path,
) -> None:
    assembly, _matrix, matrix_path, _output = _release_fixture(tmp_path)
    declarations = tmp_path / "components"
    declarations.mkdir()
    component_files: dict[str, Path] = {}
    for name, declaration in assembly["components"].items():
        path = declarations / f"{name}.json"
        path.write_text(json.dumps(declaration), encoding="utf-8")
        component_files[name] = path

    with pytest.raises(
        generate_release_assembly.ReleaseAssemblyError,
        match="must not alias an input",
    ):
        generate_release_assembly.generate(
            release_id=assembly["releaseId"],
            matrix_path=matrix_path,
            output_path=tmp_path / "configuration.json",
            source_freeze_evidence=assembly["sourceFreezeEvidence"],
            configuration=assembly["configuration"],
            migration_plan=assembly["migrationPlan"],
            connector_ledger=assembly["certifications"][
                "connectorLiveCertificationLedger"
            ],
            skill_validation_matrix=assembly["certifications"][
                "prebundledSkillValidationMatrix"
            ],
            skill_validation_deployment=assembly["certifications"][
                "skillValidationDeployment"
            ],
            skill_validation_lifecycle_evidence=assembly["certifications"][
                "skillValidationLifecycleEvidence"
            ],
            exact_artifact_closure=assembly["certifications"][
                "exactArtifactClosureEvidence"
            ],
            oci_vulnerability_scan=assembly["certifications"][
                "ociVulnerabilityScanEvidence"
            ],
            component_files=component_files,
        )

    linked = tmp_path / "linked-output.json"
    source = tmp_path / "hardlink-source.json"
    source.write_text("{}\n", encoding="utf-8")
    os.link(source, linked)
    with pytest.raises(
        generate_release_assembly.ReleaseAssemblyError,
        match="unaliased regular file",
    ):
        generate_release_assembly.write(linked, assembly)


def test_unsigned_local_binder_precedes_external_live_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assembly, matrix, matrix_path, output = _release_fixture(tmp_path)
    (tmp_path / "connector-ledger.json").write_text("{}\n", encoding="utf-8")

    unsigned = assemble_manifest.assemble(
        assembly, matrix, matrix_path=matrix_path, output_path=output
    )

    schema = json.loads(
        (ROOT / "deploy/release/release-manifest.schema.json").read_text(
            encoding="utf-8"
        )
    )
    Draft202012Validator(schema).validate(unsigned)
    monkeypatch.setattr(check_compatibility, "_verify_signature", lambda *args: None)
    monkeypatch.setattr(
        check_compatibility,
        "_verify_exact_artifact_closure",
        lambda *args: None,
    )
    monkeypatch.setattr(
        check_compatibility,
        "_verify_skill_validation_evidence",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        check_compatibility,
        "_verify_oci_vulnerability_scan_evidence",
        lambda *args: None,
    )
    with pytest.raises(
        check_compatibility.CompatibilityError,
        match="connector live-certification ledger",
    ):
        assemble_manifest.sign(
            unsigned,
            matrix,
            matrix_path=matrix_path,
            manifest_path=output,
            signer_env="RELEASE_SIGNER_COMMAND",
            verifier_env="RELEASE_VERIFIER_COMMAND",
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("name", "unrelated-component"),
        ("version", "9.9.9"),
        ("purl", "pkg:pypi/unrelated-component@1.27.0"),
        ("bom-ref", "pkg:pypi/unrelated-component@1.27.0"),
        ("hashes", [{"alg": "SHA-256", "content": "f" * 64}]),
    ],
)
def test_release_gate_rejects_sbom_root_not_bound_to_component(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    assembly, matrix, matrix_path, output = _release_fixture(tmp_path)
    manifest = assemble_manifest.assemble(
        assembly, matrix, matrix_path=matrix_path, output_path=output
    )
    component = manifest["components"]["agent-utilities"]
    sbom_path = tmp_path / component["evidence"]["sbom"]
    sbom = json.loads(sbom_path.read_text(encoding="utf-8"))
    sbom["metadata"]["component"][field] = value
    sbom_path.write_text(json.dumps(sbom), encoding="utf-8")
    component["sbomDigest"] = check_compatibility.file_digest(sbom_path)

    with pytest.raises(check_compatibility.CompatibilityError, match="sbom root"):
        check_compatibility.verify_release_manifest(
            manifest,
            matrix,
            matrix_path=matrix_path,
            manifest_path=output,
            verify_signatures=False,
            require_manifest_signature=False,
        )


def test_release_gate_rejects_empty_cyclonedx_document(tmp_path: Path) -> None:
    assembly, matrix, matrix_path, output = _release_fixture(tmp_path)
    manifest = assemble_manifest.assemble(
        assembly, matrix, matrix_path=matrix_path, output_path=output
    )
    component = manifest["components"]["agent-utilities"]
    sbom_path = tmp_path / component["evidence"]["sbom"]
    sbom_path.write_text(
        json.dumps(
            {
                "bomFormat": "CycloneDX",
                "specVersion": "1.6",
                "version": 1,
                "components": [],
            }
        ),
        encoding="utf-8",
    )
    component["sbomDigest"] = check_compatibility.file_digest(sbom_path)

    with pytest.raises(check_compatibility.CompatibilityError, match="root component"):
        check_compatibility.verify_release_manifest(
            manifest,
            matrix,
            matrix_path=matrix_path,
            manifest_path=output,
            verify_signatures=False,
            require_manifest_signature=False,
        )


def test_connector_ledger_signature_binds_the_exact_unsigned_ledger(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entries = _connector_ledger_entries()
    unsigned = {
        "apiVersion": "graphos.io/v1",
        "kind": "ConnectorLiveCertificationLedger",
        "ledgerVersion": 1,
        "entryCount": len(entries),
        "entries": entries,
    }

    def signer(_: str, payload: bytes) -> dict[str, str]:
        return {
            "scheme": "fixture-signature",
            "subjectDigest": check_compatibility.canonical_digest(json.loads(payload)),
            "bundleDigest": "sha256:" + "3" * 64,
            "signerIdentityDigest": "sha256:" + "4" * 64,
            "signature": "fixture-signature-value",
        }

    monkeypatch.setattr(connector_ledger, "_external_command", signer)
    signed = connector_ledger.sign_ledger(
        unsigned,
        signer_env="CONNECTOR_LEDGER_SIGNER",
        verifier_env="CONNECTOR_LEDGER_VERIFIER",
    )
    connector_ledger._validate_signed(signed)
    signed["entryCount"] = 2
    with pytest.raises(assemble_manifest.AssemblyError):
        connector_ledger._validate_signed(signed)


def test_connector_ledger_requires_the_exact_current_fleet() -> None:
    entries = _connector_ledger_entries(CONNECTOR_COUNT - 1)
    unsigned = {
        "apiVersion": "graphos.io/v1",
        "kind": "ConnectorLiveCertificationLedger",
        "ledgerVersion": 1,
        "entryCount": len(entries),
        "entries": entries,
    }

    with pytest.raises(assemble_manifest.AssemblyError, match="entry count"):
        connector_ledger._validate_unsigned(unsigned)

    ledger_schema = json.loads(
        (
            ROOT / "deploy/release/connector-live-certification-ledger.schema.json"
        ).read_text(encoding="utf-8")
    )
    assert not Draft202012Validator(ledger_schema).is_valid(
        {
            **unsigned,
            "signature": {
                "scheme": "fixture-signature",
                "subjectDigest": check_compatibility.canonical_digest(unsigned),
                "bundleDigest": "sha256:" + "3" * 64,
                "signerIdentityDigest": "sha256:" + "4" * 64,
                "value": "fixture-signature-value",
                "verifierEnv": "CONNECTOR_LEDGER_VERIFIER",
            },
        }
    )


def test_index_migration_catalog_is_deterministic_and_executor_is_live() -> None:
    catalog = index_migration_catalog()
    assert json.loads(render_catalog()) == catalog
    assert catalog["entryCount"] == 1

    class Backend:
        interval = 0

        def hydrate_engine_embeddings(self, batch_log_every: int = 5000) -> int:
            self.interval = batch_log_every
            return 7

    backend = Backend()
    assert (
        run_index_migration("embedding-authority-ann-v1", backend, batch_log_every=100)
        == 7
    )
    assert backend.interval == 100
