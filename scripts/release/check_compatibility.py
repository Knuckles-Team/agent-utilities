#!/usr/bin/python
"""Fail-closed compatibility and signature gate for an exact GraphOS release."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import stat
import subprocess
import threading
from pathlib import Path, PurePosixPath
from typing import Any

import yaml
from jsonschema import Draft202012Validator
from packaging.specifiers import SpecifierSet
from packaging.version import InvalidVersion, Version

_DIGEST = re.compile(r"^sha256:([a-f0-9]{64})$")
_ENV_NAME = re.compile(r"^[A-Z][A-Z0-9_]{2,63}$")
_REFERENCE = re.compile(
    r"^(?=.{1,256}$)[A-Za-z0-9][A-Za-z0-9._-]{0,127}"
    r"(?:/[A-Za-z0-9][A-Za-z0-9._-]{0,127}){0,15}$"
)
_SIGNATURE_SCHEME = re.compile(r"^[a-z0-9][a-z0-9+._-]{1,63}$")
_SIGNATURE_VALUE = re.compile(r"^[A-Za-z0-9+/_=-]{16,16384}$")
_MAX_EVIDENCE_BYTES = 64 * 1024 * 1024
_MAX_COMPONENT_SOURCE_BYTES = 1024 * 1024
_MAX_COMPONENT_SBOM_BYTES = 16 * 1024 * 1024
_MAX_COMPONENT_PROVENANCE_BYTES = 1024 * 1024
_MAX_COMPONENT_SIGNATURE_BYTES = 1024 * 1024
_MAX_ADAPTER_OUTPUT_BYTES = 1024 * 1024
_COMPONENT_NAME = re.compile(r"^[a-z0-9][a-z0-9.-]{1,127}$")
_COMPONENT_VERSION = re.compile(r"^(?:0|[1-9][0-9]*)(?:\.(?:0|[1-9][0-9]*))*$")
_OPAQUE_REFERENCE = re.compile(r"^pref_[a-z_]+_[a-f0-9]{64}$")
_TRACE_NAME = re.compile(r"^graph_run:pref_run_[a-f0-9]{64}$")
_SOURCE_FREEZE_DIGEST = re.compile(r"^(?!0{64}$)[a-f0-9]{64}$")
_COMPONENT_BUILD_TYPE = "https://graphos.invalid/build/exact-local/v1"
_COMPONENT_BUILDER_ID = "https://graphos.invalid/builders/exact-local/v1"
_SKILL_NAMES = (
    "deployment",
    "development",
    "engine",
    "evolution",
    "ingestion",
    "modeling",
    "orchestration",
    "query",
    "research",
    "runtime",
)
_SKILL_CASE_IDS = tuple(
    f"{skill}-{mode}" for skill in _SKILL_NAMES for mode in ("delegated", "direct")
)
_REQUIRED_SCHEMAS = {
    "requestContext",
    "mutationBatch",
    "changeEnvelope",
    "workItem",
    "artifact",
    "knowledgeBatch",
    "analyticsJob",
    "traceOutcome",
    "placementRoute",
    "claimWorkItem",
    "evidenceBundle",
    "operationResult",
}
_RELEASE_ORDER = (
    "epistemic-operations-protocol",
    "epistemic-graph",
    "agent-utilities",
    "langfuse-agent",
    "connector-bundles",
    "prebundled-skills",
    "ontology-lock",
    "index-migrations",
)
_OCI_COMPONENTS = (
    "epistemic-graph",
    "agent-utilities",
    "langfuse-agent",
)
_CURRENT_COMPONENT_VERSIONS = {
    "epistemic-operations-protocol": "1",
    "epistemic-graph": "2.23.1",
    "agent-utilities": "1.27.1",
    "langfuse-agent": "1.0.3",
    "connector-bundles": "1",
    "prebundled-skills": "1",
    "ontology-lock": "1",
    "index-migrations": "1",
}
_CURRENT_CONNECTOR_ENTRIES = 65
_CURRENT_RUNTIME_CONTRACT = {
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
_CURRENT_COMPONENT_DEPENDENCIES = {
    "epistemic-operations-protocol": {},
    "epistemic-graph": {},
    "agent-utilities": {"epistemic-graph": "==2.23.1"},
    "langfuse-agent": {"agent-utilities": "==1.27.1"},
    "connector-bundles": {
        "agent-utilities": "==1.27.1",
        "epistemic-graph": "==2.23.1",
    },
    "prebundled-skills": {"agent-utilities": "==1.27.1"},
    "ontology-lock": {},
    "index-migrations": {},
}
_CERTIFICATION_DIGESTS = {
    "connectorLiveCertificationLedger",
    "prebundledSkillValidationMatrix",
    "skillValidationDeployment",
    "skillValidationLifecycleEvidence",
    "exactArtifactClosureEvidence",
    "ociVulnerabilityScanEvidence",
}
_RELEASE_SCHEMA_ROOT = Path(__file__).resolve().parents[2] / "deploy" / "release"
_SKILL_VALIDATION_MATRIX_SCHEMA = (
    _RELEASE_SCHEMA_ROOT / "prebundled-skill-validation-evidence.schema.json"
)
_SKILL_VALIDATION_DEPLOYMENT_SCHEMA = (
    _RELEASE_SCHEMA_ROOT / "skill-validation-deployment.schema.json"
)
_SKILL_VALIDATION_LIFECYCLE_SCHEMA = (
    _RELEASE_SCHEMA_ROOT / "skill-validation-deployment-evidence.schema.json"
)
_OCI_VULNERABILITY_SCAN_SCHEMA = (
    _RELEASE_SCHEMA_ROOT / "oci-vulnerability-scan-evidence.schema.json"
)
_RELEASE_CONFIGURATION_SCHEMA = (
    _RELEASE_SCHEMA_ROOT / "release-configuration.schema.json"
)
_RELEASE_MIGRATION_PLAN_SCHEMA = (
    _RELEASE_SCHEMA_ROOT / "release-migration-plan.schema.json"
)
_SOURCE_FREEZE_EVIDENCE_SCHEMA = (
    _RELEASE_SCHEMA_ROOT / "source-freeze-evidence.schema.json"
)
_SOURCE_FREEZE_MANIFEST_SCHEMA = (
    _RELEASE_SCHEMA_ROOT / "source-freeze-gates.schema.json"
)
_SOURCE_FREEZE_MANIFEST = _RELEASE_SCHEMA_ROOT / "source-freeze-gates.json"
_SOURCE_FREEZE_REPOSITORIES = (
    "agent-utilities",
    "epistemic-graph",
    "langfuse-agent",
    "provider-fleet",
)
_OCI_VULNERABILITY_SCAN_VERIFIER_ENV = "OCI_SCAN_EVIDENCE_VERIFIER_COMMAND"
_EXACT_ARTIFACT_CLOSURE_VERIFIER_ENV = "EXACT_ARTIFACT_CLOSURE_VERIFIER_COMMAND"
_EXACT_ARTIFACT_GATES = (
    "G-01",
    "G-02",
    "G-04",
    "G-05",
    "G-08",
    "G-09",
    "G-14",
    "G-15",
    "G-17",
    "G-26",
    "G-30",
    "G-32",
    "G-34",
    "G-35",
    "G-37",
)
_EXACT_GATE_AUTHORITIES = {
    "G-01": ("certification:exactArtifactClosureEvidence",),
    "G-02": ("certification:exactArtifactClosureEvidence",),
    "G-03": ("component:epistemic-graph",),
    "G-04": ("certification:exactArtifactClosureEvidence",),
    "G-05": ("certification:exactArtifactClosureEvidence",),
    "G-06": (
        "component:connector-bundles",
        "certification:connectorLiveCertificationLedger",
    ),
    "G-07": (
        "component:agent-utilities",
        "component:epistemic-graph",
        "certification:skillValidationLifecycleEvidence",
    ),
    "G-08": ("certification:exactArtifactClosureEvidence",),
    "G-09": ("certification:exactArtifactClosureEvidence",),
    "G-11": ("certification:exactArtifactClosureEvidence",),
    "G-12": (
        "component:epistemic-operations-protocol",
        "certification:prebundledSkillValidationMatrix",
        "certification:skillValidationLifecycleEvidence",
    ),
    "G-13": (
        "component:epistemic-operations-protocol",
        "component:epistemic-graph",
    ),
    "G-14": ("certification:exactArtifactClosureEvidence",),
    "G-15": ("certification:exactArtifactClosureEvidence",),
    "G-17": ("certification:exactArtifactClosureEvidence",),
    "G-18": (
        "component:agent-utilities",
        "certification:skillValidationLifecycleEvidence",
    ),
    "G-22": (
        "component:epistemic-graph",
        "component:agent-utilities",
        "component:langfuse-agent",
        "certification:ociVulnerabilityScanEvidence",
    ),
    "G-25": ("component:epistemic-operations-protocol",),
    "G-26": ("certification:exactArtifactClosureEvidence",),
    "G-27": (
        "certification:prebundledSkillValidationMatrix",
        "certification:skillValidationLifecycleEvidence",
    ),
    "G-29": (
        "component:langfuse-agent",
        "certification:prebundledSkillValidationMatrix",
        "certification:skillValidationLifecycleEvidence",
    ),
    "G-30": (
        "certification:exactArtifactClosureEvidence",
        "certification:skillValidationLifecycleEvidence",
    ),
    "G-31": (
        "component:epistemic-graph",
        "certification:exactArtifactClosureEvidence",
    ),
    "G-32": ("certification:exactArtifactClosureEvidence",),
    "G-33": ("component:agent-utilities",),
    "G-34": ("certification:exactArtifactClosureEvidence",),
    "G-35": ("certification:exactArtifactClosureEvidence",),
    "G-36": (
        "component:agent-utilities",
        "component:epistemic-graph",
        "certification:exactArtifactClosureEvidence",
        "certification:skillValidationLifecycleEvidence",
    ),
    "G-37": ("certification:exactArtifactClosureEvidence",),
    "G-38": (
        "component:epistemic-graph",
        "component:agent-utilities",
        "component:langfuse-agent",
        "certification:connectorLiveCertificationLedger",
        "certification:prebundledSkillValidationMatrix",
        "certification:skillValidationLifecycleEvidence",
        "certification:exactArtifactClosureEvidence",
        "certification:ociVulnerabilityScanEvidence",
    ),
}


class CompatibilityError(ValueError):
    """The exact release is not compatible or is not verifiably signed."""


def _exact_keys(
    value: dict[str, Any],
    *,
    required: set[str],
    optional: set[str] | None = None,
    field: str,
) -> None:
    missing = required - set(value)
    unknown = set(value) - required - (optional or set())
    if missing or unknown:
        raise CompatibilityError(
            f"{field} keys are not exact; missing={sorted(missing)}, unknown={sorted(unknown)}"
        )


def _input_bytes(path: Path, *, maximum: int = _MAX_EVIDENCE_BYTES) -> bytes:
    """Open one bounded input once and reject aliases or path replacement."""

    if path.is_symlink():
        raise CompatibilityError("release input must be an unaliased regular file")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise CompatibilityError("release input is unavailable") from exc
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or metadata.st_size > maximum
        ):
            raise CompatibilityError("release input violates its size boundary")
        before = (
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_size,
            metadata.st_mtime_ns,
            metadata.st_ctime_ns,
        )
        payload = bytearray()
        while len(payload) <= maximum:
            chunk = os.read(descriptor, min(65_536, maximum + 1 - len(payload)))
            if not chunk:
                break
            payload.extend(chunk)
        after = os.fstat(descriptor)
        if (
            before
            != (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            )
            or len(payload) != metadata.st_size
        ):
            raise CompatibilityError("release input changed while it was read")
        try:
            path_metadata = path.stat(follow_symlinks=False)
        except OSError:
            raise CompatibilityError(
                "release input changed while it was read"
            ) from None
        if (path_metadata.st_dev, path_metadata.st_ino) != (
            metadata.st_dev,
            metadata.st_ino,
        ):
            raise CompatibilityError("release input changed while it was read")
        return bytes(payload)
    finally:
        os.close(descriptor)


def _load(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(_input_bytes(path))
    if not isinstance(value, dict):
        raise CompatibilityError("manifest root must be a mapping")
    return value


def canonical_digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def file_digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(_input_bytes(path)).hexdigest()


def _evidence_bytes(
    manifest_path: Path,
    reference: Any,
    field: str,
    *,
    maximum: int = _MAX_EVIDENCE_BYTES,
) -> bytes:
    text = str(reference or "")
    relative = PurePosixPath(text)
    if (
        not _REFERENCE.fullmatch(text)
        or relative.is_absolute()
        or ".." in relative.parts
    ):
        raise CompatibilityError(
            f"{field} must be a release-relative evidence reference"
        )
    base = manifest_path.parent.resolve()
    candidate = manifest_path.parent
    for part in relative.parts:
        candidate /= part
        if candidate.is_symlink():
            raise CompatibilityError(f"{field} evidence symlinks are not accepted")
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(base)
    except (OSError, ValueError):
        raise CompatibilityError(f"{field} evidence is unavailable") from None
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(resolved, flags)
    except OSError:
        raise CompatibilityError(f"{field} evidence is unavailable") from None
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or metadata.st_size > maximum
        ):
            raise CompatibilityError(f"{field} evidence is not a bounded regular file")
        before = (
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_size,
            metadata.st_mtime_ns,
            metadata.st_ctime_ns,
        )
        payload = bytearray()
        remaining = maximum + 1
        while remaining:
            chunk = os.read(descriptor, min(65_536, remaining))
            if not chunk:
                break
            payload.extend(chunk)
            remaining -= len(chunk)
        after_metadata = os.fstat(descriptor)
        after = (
            after_metadata.st_dev,
            after_metadata.st_ino,
            after_metadata.st_size,
            after_metadata.st_mtime_ns,
            after_metadata.st_ctime_ns,
        )
        if before != after or len(payload) != metadata.st_size:
            raise CompatibilityError(f"{field} evidence changed while it was read")
        if len(payload) > maximum:
            raise CompatibilityError(f"{field} evidence exceeds its size boundary")
        try:
            path_metadata = resolved.stat(follow_symlinks=False)
        except OSError:
            raise CompatibilityError(
                f"{field} evidence changed while it was read"
            ) from None
        if (path_metadata.st_dev, path_metadata.st_ino) != (
            metadata.st_dev,
            metadata.st_ino,
        ):
            raise CompatibilityError(f"{field} evidence changed while it was read")
        descriptor_link = Path(f"/proc/self/fd/{descriptor}")
        if descriptor_link.exists():
            try:
                descriptor_link.resolve().relative_to(base)
            except ValueError:
                raise CompatibilityError(
                    f"{field} evidence escaped its release root"
                ) from None
        return bytes(payload)
    finally:
        os.close(descriptor)


def _json_evidence(payload: bytes, field: str) -> dict[str, Any]:
    def exact_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise CompatibilityError(f"{field} evidence has duplicate keys")
            value[key] = item
        return value

    try:
        value = json.loads(payload, object_pairs_hook=exact_pairs)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CompatibilityError(f"{field} evidence must be a JSON object") from exc
    if not isinstance(value, dict):
        raise CompatibilityError(f"{field} evidence must be a JSON object")
    return value


def _validate_release_schema(
    value: dict[str, Any],
    *,
    schema_path: Path,
    field: str,
) -> None:
    try:
        schema = json.loads(
            _input_bytes(schema_path, maximum=_MAX_COMPONENT_SOURCE_BYTES)
        )
        if not isinstance(schema, dict):
            raise TypeError
        Draft202012Validator.check_schema(schema)
        error = next(Draft202012Validator(schema).iter_errors(value), None)
    except Exception as exc:
        raise CompatibilityError(f"{field} schema is unavailable") from exc
    if error is not None:
        raise CompatibilityError(f"{field} does not satisfy its current schema")


def _source_freeze_aggregate(values: dict[str, str]) -> str:
    canonical = json.dumps(values, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


def validate_source_freeze_evidence(payload: bytes) -> dict[str, str]:
    """Validate and content-address the sole canonical source-freeze authority."""

    evidence = _json_evidence(payload, "source-freeze")
    _validate_release_schema(
        evidence,
        schema_path=_SOURCE_FREEZE_EVIDENCE_SCHEMA,
        field="source-freeze evidence",
    )
    canonical = (
        json.dumps(evidence, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    if payload != canonical:
        raise CompatibilityError("source-freeze evidence is not canonical JSON")
    manifest_payload = _input_bytes(
        _SOURCE_FREEZE_MANIFEST,
        maximum=_MAX_COMPONENT_SOURCE_BYTES,
    )
    manifest = _json_evidence(manifest_payload, "source-freeze manifest")
    _validate_release_schema(
        manifest,
        schema_path=_SOURCE_FREEZE_MANIFEST_SCHEMA,
        field="source-freeze manifest",
    )
    if evidence.get("manifest_sha256") != hashlib.sha256(manifest_payload).hexdigest():
        raise CompatibilityError("source-freeze manifest binding is not exact")
    if (
        evidence.get("status") != "passed"
        or evidence.get("source_digest_before")
        != evidence.get("source_digest_after")
    ):
        raise CompatibilityError("source-freeze evidence did not pass exactly")

    tools = evidence.get("tools")
    if not isinstance(tools, list) or [tool.get("id") for tool in tools] != [
        "git",
        "rg",
    ]:
        raise CompatibilityError("source-freeze tool authority is not exact")

    manifest_repositories = manifest.get("repositories")
    repository_ids = tuple(
        str(item.get("id") or "")
        for item in manifest_repositories
        if isinstance(item, dict)
    ) if isinstance(manifest_repositories, list) else ()
    if repository_ids != _SOURCE_FREEZE_REPOSITORIES:
        raise CompatibilityError("source-freeze repository authority is not exact")
    repositories = evidence.get("repositories")
    if not isinstance(repositories, list) or tuple(
        str(item.get("id") or "") for item in repositories if isinstance(item, dict)
    ) != repository_ids:
        raise CompatibilityError("source-freeze repository evidence is not exact")
    before: dict[str, str] = {}
    after: dict[str, str] = {}
    for repository in repositories:
        if not isinstance(repository, dict):
            raise CompatibilityError("source-freeze repository evidence is not exact")
        identifier = str(repository["id"])
        before_digest = str(repository["sha256_before"])
        after_digest = str(repository["sha256_after"])
        if (
            _SOURCE_FREEZE_DIGEST.fullmatch(before_digest) is None
            or _SOURCE_FREEZE_DIGEST.fullmatch(after_digest) is None
            or before_digest != after_digest
        ):
            raise CompatibilityError("source-freeze repository evidence is not exact")
        before[identifier] = before_digest
        after[identifier] = after_digest
    if (
        evidence.get("source_digest_before") != _source_freeze_aggregate(before)
        or evidence.get("source_digest_after") != _source_freeze_aggregate(after)
    ):
        raise CompatibilityError("source-freeze aggregate digest is not exact")

    commands = evidence.get("commands")
    manifest_commands = manifest.get("commands")
    if (
        not isinstance(commands, list)
        or not isinstance(manifest_commands, list)
        or len(commands) != len(manifest_commands)
    ):
        raise CompatibilityError("source-freeze command evidence is not exact")
    repository_token = re.compile(r"^\{repo:([a-z][a-z0-9-]{2,63})\}(.*)$")
    for command, expected in zip(commands, manifest_commands, strict=True):
        if not isinstance(command, dict) or not isinstance(expected, dict):
            raise CompatibilityError("source-freeze command evidence is not exact")
        identifiers = {str(expected.get("repository") or "")}
        argv = expected.get("argv")
        if not isinstance(argv, list):
            raise CompatibilityError("source-freeze command authority is invalid")
        identifiers.update(
            match.group(1)
            for token in argv
            if isinstance(token, str)
            and (match := repository_token.fullmatch(token)) is not None
        )
        command_before = {
            identifier: before[identifier]
            for identifier in repository_ids
            if identifier in identifiers
        }
        command_after = {
            identifier: after[identifier]
            for identifier in repository_ids
            if identifier in identifiers
        }
        if command != {
            "id": expected.get("id"),
            "status": "passed",
            "exit_code": 0,
            "termination": "exited",
            "source_digest_before": _source_freeze_aggregate(command_before),
            "source_digest_after": _source_freeze_aggregate(command_after),
        }:
            raise CompatibilityError("source-freeze command evidence is not exact")

    gates = evidence.get("gates")
    manifest_gates = manifest.get("gates")
    if (
        not isinstance(gates, list)
        or not isinstance(manifest_gates, list)
        or len(gates) != 39
        or len(gates) != len(manifest_gates)
    ):
        raise CompatibilityError("source-freeze gate evidence is not exact")
    for gate, expected in zip(gates, manifest_gates, strict=True):
        if not isinstance(gate, dict) or not isinstance(expected, dict):
            raise CompatibilityError("source-freeze gate evidence is not exact")
        required_evidence = expected.get("evidence_classes")
        if not isinstance(required_evidence, list):
            raise CompatibilityError("source-freeze gate authority is invalid")
        if gate != {
            "id": expected.get("id"),
            "required_evidence": required_evidence,
            "source_status": (
                "passed" if "local-source" in required_evidence else "not-applicable"
            ),
            "remaining_evidence": [
                value for value in required_evidence if value != "local-source"
            ],
        }:
            raise CompatibilityError("source-freeze gate evidence is not exact")
    return {
        "evidenceDigest": "sha256:" + hashlib.sha256(payload).hexdigest(),
        "snapshotDigest": "sha256:" + str(evidence["source_digest_after"]),
    }


def _validate_sbom_binding(
    name: str,
    component: dict[str, Any],
    sbom: dict[str, Any],
) -> None:
    """Bind a CycloneDX root component to this exact release artifact."""

    if (
        sbom.get("bomFormat") != "CycloneDX"
        or str(sbom.get("specVersion") or "") not in {"1.5", "1.6"}
        or not isinstance(sbom.get("components"), list)
    ):
        raise CompatibilityError(f"{name}.sbom is not a supported CycloneDX document")
    metadata = sbom.get("metadata")
    root = metadata.get("component") if isinstance(metadata, dict) else None
    if not isinstance(root, dict):
        raise CompatibilityError(f"{name}.sbom has no root component identity")
    version = str(component.get("version") or "")
    package_type = "pypi" if component.get("kind") == "oci" else "generic"
    expected_purl = f"pkg:{package_type}/{name}@{version}"
    if (
        root.get("name") != name
        or str(root.get("version") or "") != version
        or root.get("purl") != expected_purl
        or root.get("bom-ref") != expected_purl
    ):
        raise CompatibilityError(
            f"{name}.sbom root identity differs from the release component"
        )
    expected_sha256 = str(component.get("digest") or "").removeprefix("sha256:")
    hashes = root.get("hashes")
    if (
        not isinstance(hashes, list)
        or len(hashes) != 1
        or not isinstance(hashes[0], dict)
        or set(hashes[0]) != {"alg", "content"}
        or hashes[0].get("alg") != "SHA-256"
        or hashes[0].get("content") != expected_sha256
    ):
        raise CompatibilityError(
            f"{name}.sbom root hash differs from the release artifact digest"
        )
    inventory = sbom["components"]
    if component.get("kind") == "oci" and not inventory:
        raise CompatibilityError(f"{name}.sbom has no closed Python wheel inventory")
    references: set[str] = set()
    for item in inventory:
        if not isinstance(item, dict) or set(item) != {
            "type",
            "bom-ref",
            "name",
            "version",
            "purl",
            "hashes",
        }:
            raise CompatibilityError(f"{name}.sbom component inventory is not exact")
        reference = str(item.get("bom-ref") or "")
        if (
            item.get("type") != "library"
            or item.get("purl") != reference
            or not reference.startswith("pkg:pypi/")
            or reference in references
            or not _COMPONENT_NAME.fullmatch(str(item.get("name") or ""))
            or not str(item.get("version") or "")
        ):
            raise CompatibilityError(f"{name}.sbom component inventory is invalid")
        references.add(reference)
        item_hashes = item.get("hashes")
        if (
            not isinstance(item_hashes, list)
            or len(item_hashes) != 1
            or not isinstance(item_hashes[0], dict)
            or set(item_hashes[0]) != {"alg", "content"}
            or item_hashes[0].get("alg") != "SHA-256"
            or re.fullmatch(
                r"(?!0{64}$)[a-f0-9]{64}", str(item_hashes[0].get("content") or "")
            )
            is None
        ):
            raise CompatibilityError(f"{name}.sbom component hash is invalid")


def _validate_component_source(
    name: str,
    component: dict[str, Any],
    payload: bytes,
) -> dict[str, Any]:
    source = _json_evidence(payload, f"{name}.source")
    _exact_keys(
        source,
        required={
            "apiVersion",
            "kind",
            "component",
            "version",
            "artifactFormat",
            "artifactDigest",
            "artifactInputDigest",
            "sourceSnapshotDigest",
            "sourceEvidenceDigest",
        },
        field=f"{name}.source",
    )
    expected_format = (
        "oci-layout-archive" if component.get("kind") == "oci" else "opaque-catalog"
    )
    if (
        source.get("apiVersion") != "graphos.io/v1"
        or source.get("kind") != "ComponentSourceEvidence"
        or source.get("component") != name
        or source.get("version") != component.get("version")
        or source.get("artifactFormat") != expected_format
        or source.get("artifactDigest") != component.get("digest")
    ):
        raise CompatibilityError(f"{name}.source identity differs from the component")
    for field in (
        "artifactDigest",
        "artifactInputDigest",
        "sourceSnapshotDigest",
        "sourceEvidenceDigest",
    ):
        _digest(source.get(field), f"{name}.source.{field}")
    if component.get("kind") == "catalog" and source.get(
        "artifactInputDigest"
    ) != source.get("artifactDigest"):
        raise CompatibilityError(f"{name}.source catalog input digest differs")
    return source


def _validate_single_source_freeze(
    source_documents: dict[str, dict[str, Any]],
    authority: dict[str, str] | None,
) -> None:
    """Bind all eight release components to the validated source-freeze record."""

    if set(source_documents) != set(_RELEASE_ORDER):
        raise CompatibilityError(
            "release components do not share one source-freeze authority"
        )
    if not isinstance(authority, dict) or set(authority) != {
        "evidenceDigest",
        "snapshotDigest",
    }:
        raise CompatibilityError(
            "release source-freeze authority is absent or invalid"
        )
    evidence_digest = _digest(
        authority.get("evidenceDigest"), "sourceFreezeEvidenceDigest"
    )
    snapshot_digest = _digest(
        authority.get("snapshotDigest"), "sourceFreezeSnapshotDigest"
    )
    if any(
        document.get("sourceSnapshotDigest") != snapshot_digest
        or document.get("sourceEvidenceDigest") != evidence_digest
        for document in source_documents.values()
    ):
        raise CompatibilityError(
            "release components do not share one source-freeze authority"
        )


def _validate_component_provenance(
    name: str,
    component: dict[str, Any],
    source: dict[str, Any],
    payload: bytes,
) -> None:
    provenance = _json_evidence(payload, f"{name}.provenance")
    _exact_keys(
        provenance,
        required={"_type", "subject", "predicateType", "predicate"},
        field=f"{name}.provenance",
    )
    artifact_sha256 = str(component.get("digest") or "").removeprefix("sha256:")
    expected_subject = [{"name": name, "digest": {"sha256": artifact_sha256}}]
    if (
        provenance.get("_type") != "https://in-toto.io/Statement/v1"
        or provenance.get("predicateType") != "https://slsa.dev/provenance/v1"
        or provenance.get("subject") != expected_subject
    ):
        raise CompatibilityError(f"{name}.provenance subject is not exact")
    predicate = provenance.get("predicate")
    if not isinstance(predicate, dict):
        raise CompatibilityError(f"{name}.provenance predicate is required")
    _exact_keys(
        predicate,
        required={"buildDefinition", "runDetails"},
        field=f"{name}.provenance.predicate",
    )
    definition = predicate.get("buildDefinition")
    if not isinstance(definition, dict):
        raise CompatibilityError(f"{name}.provenance buildDefinition is required")
    _exact_keys(
        definition,
        required={
            "buildType",
            "externalParameters",
            "internalParameters",
            "resolvedDependencies",
        },
        field=f"{name}.provenance.buildDefinition",
    )
    expected_dependency = [
        {
            "uri": "urn:graphos:source-freeze",
            "digest": {
                "sha256": str(source["sourceSnapshotDigest"]).removeprefix("sha256:")
            },
        }
    ]
    if (
        definition.get("buildType") != _COMPONENT_BUILD_TYPE
        or definition.get("externalParameters") != {}
        or definition.get("internalParameters") != {}
        or definition.get("resolvedDependencies") != expected_dependency
    ):
        raise CompatibilityError(f"{name}.provenance source binding is not exact")
    run_details = predicate.get("runDetails")
    if run_details != {
        "builder": {"id": _COMPONENT_BUILDER_ID},
        "byproducts": [],
    }:
        raise CompatibilityError(f"{name}.provenance builder identity is not exact")


def component_signing_subject(name: str, component: dict[str, Any]) -> bytes:
    """Return the sole canonical byte subject accepted by signer and verifier."""

    signature = component.get("signature")
    verifier_env = (
        signature.get("verifierEnv")
        if isinstance(signature, dict)
        else component.get("signatureVerifierEnv")
    )
    declaration: dict[str, Any] = {
        "version": component.get("version"),
        "kind": component.get("kind"),
        "artifact": component.get("artifact"),
        "digest": component.get("digest"),
        "sourceDigest": component.get("sourceDigest"),
        "sbomDigest": component.get("sbomDigest"),
        "provenanceDigest": component.get("provenanceDigest"),
        "evidence": component.get("evidence"),
        "verifierEnv": verifier_env,
        "capabilities": sorted(component.get("capabilities") or ()),
    }
    if "entryCount" in component:
        declaration["entryCount"] = component["entryCount"]
    return _canonical_bytes(
        {
            "schema": "graphos-component-signing-subject/1",
            "component": name,
            "declaration": declaration,
        }
    )


def _validate_component_signature_bundle(
    name: str,
    component: dict[str, Any],
    payload: bytes,
) -> tuple[dict[str, Any], bytes]:
    bundle = _json_evidence(payload, f"{name}.signatureBundle")
    _exact_keys(
        bundle,
        required={
            "schema",
            "scheme",
            "subjectDigest",
            "artifactDigest",
            "signature",
            "verificationMaterialDigest",
            "signerIdentityDigest",
        },
        field=f"{name}.signatureBundle",
    )
    subject = component_signing_subject(name, component)
    subject_digest = "sha256:" + hashlib.sha256(subject).hexdigest()
    if (
        bundle.get("schema") != "graphos-external-signature/2"
        or bundle.get("subjectDigest") != subject_digest
        or bundle.get("artifactDigest") != component.get("digest")
        or not _SIGNATURE_SCHEME.fullmatch(str(bundle.get("scheme") or ""))
        or not _SIGNATURE_VALUE.fullmatch(str(bundle.get("signature") or ""))
    ):
        raise CompatibilityError(f"{name}.signatureBundle is not bound to its subject")
    for field in (
        "subjectDigest",
        "artifactDigest",
        "verificationMaterialDigest",
        "signerIdentityDigest",
    ):
        _digest(bundle.get(field), f"{name}.signatureBundle.{field}")
    return bundle, subject


def _inspect_component_evidence(
    name: str,
    component: dict[str, Any],
    manifest_path: Path,
) -> dict[str, Any]:
    evidence = component.get("evidence")
    if not isinstance(evidence, dict):
        raise CompatibilityError(f"{name}.evidence is required")
    _exact_keys(
        evidence,
        required={"source", "sbom", "provenance", "signatureBundle"},
        field=f"{name}.evidence",
    )
    source_raw = _evidence_bytes(
        manifest_path,
        evidence["source"],
        f"{name}.source",
        maximum=_MAX_COMPONENT_SOURCE_BYTES,
    )
    source = _validate_component_source(name, component, source_raw)
    source_digest = "sha256:" + hashlib.sha256(source_raw).hexdigest()
    sbom_raw = _evidence_bytes(
        manifest_path,
        evidence["sbom"],
        f"{name}.sbom",
        maximum=_MAX_COMPONENT_SBOM_BYTES,
    )
    _validate_sbom_binding(name, component, _json_evidence(sbom_raw, f"{name}.sbom"))
    sbom_digest = "sha256:" + hashlib.sha256(sbom_raw).hexdigest()
    provenance_raw = _evidence_bytes(
        manifest_path,
        evidence["provenance"],
        f"{name}.provenance",
        maximum=_MAX_COMPONENT_PROVENANCE_BYTES,
    )
    _validate_component_provenance(name, component, source, provenance_raw)
    provenance_digest = "sha256:" + hashlib.sha256(provenance_raw).hexdigest()
    derived = {
        **component,
        "sourceDigest": source_digest,
        "sbomDigest": sbom_digest,
        "provenanceDigest": provenance_digest,
    }
    for field, observed in (
        ("sourceDigest", source_digest),
        ("sbomDigest", sbom_digest),
        ("provenanceDigest", provenance_digest),
    ):
        if field in component and component.get(field) != observed:
            raise CompatibilityError(f"{name}.{field} differs from referenced evidence")
    bundle_raw = _evidence_bytes(
        manifest_path,
        evidence["signatureBundle"],
        f"{name}.signatureBundle",
        maximum=_MAX_COMPONENT_SIGNATURE_BYTES,
    )
    bundle_digest = "sha256:" + hashlib.sha256(bundle_raw).hexdigest()
    signature = component.get("signature")
    if isinstance(signature, dict) and signature.get("bundleDigest") != bundle_digest:
        raise CompatibilityError(
            f"{name} signature bundle digest differs from evidence"
        )
    _bundle, subject = _validate_component_signature_bundle(name, derived, bundle_raw)
    subject_digest = "sha256:" + hashlib.sha256(subject).hexdigest()
    request = {
        "schema": "graphos-component-verification-request/1",
        "subject": {
            "encoding": "base64",
            "digest": subject_digest,
            "value": base64.b64encode(subject).decode("ascii"),
        },
        "signatureBundle": {
            "encoding": "base64",
            "digest": bundle_digest,
            "value": base64.b64encode(bundle_raw).decode("ascii"),
        },
    }
    return {
        "sourceDigest": source_digest,
        "sourceEvidence": source,
        "sbomDigest": sbom_digest,
        "provenanceDigest": provenance_digest,
        "bundleDigest": bundle_digest,
        "verificationRequest": request,
    }


def _validate_component_evidence(
    name: str,
    component: dict[str, Any],
    manifest_path: Path,
) -> dict[str, Any]:
    return _inspect_component_evidence(name, component, manifest_path)[
        "verificationRequest"
    ]


def _digest(value: Any, field: str) -> str:
    text = str(value or "")
    match = _DIGEST.fullmatch(text)
    if not match or set(match.group(1)) == {"0"}:
        raise CompatibilityError(f"{field} must be a non-sentinel sha256 digest")
    return text


def _component_names(matrix: dict[str, Any]) -> list[str]:
    order = matrix.get("releaseTrain", {}).get("assemblyOrder")
    components = matrix.get("components")
    if not isinstance(order, list) or not isinstance(components, dict):
        raise CompatibilityError("compatibility matrix has no component release order")
    if set(order) != set(components):
        raise CompatibilityError("release order and component catalog differ")
    return [str(name) for name in order]


def _validate_distinct_oci_subjects(components: dict[str, Any]) -> None:
    """Require one independently materialized OCI subject per runtime component."""

    digests: dict[str, str] = {}
    for name in _OCI_COMPONENTS:
        component = components.get(name)
        if not isinstance(component, dict) or component.get("kind") != "oci":
            raise CompatibilityError(f"component {name} must be an OCI subject")
        digest = _digest(component.get("digest"), f"{name}.digest")
        if component.get("artifact") != f"oci:{name}@{digest}":
            raise CompatibilityError(
                f"component {name} OCI subject is not pinned to its declared digest"
            )
        digests[name] = digest
    if len(set(digests.values())) != len(digests):
        raise CompatibilityError(
            "runtime components must use three distinct OCI subject digests"
        )


def exact_gate_evidence(
    components: dict[str, Any],
    certification_digests: dict[str, Any],
) -> dict[str, list[dict[str, str]]]:
    """Materialize every exact-class gate as authoritative signed digest records."""

    records: dict[str, list[dict[str, str]]] = {}
    for gate, authorities in _EXACT_GATE_AUTHORITIES.items():
        gate_records: list[dict[str, str]] = []
        for authority in authorities:
            category, separator, name = authority.partition(":")
            if not separator:
                raise CompatibilityError("exact-gate authority is invalid")
            if category == "component":
                component = components.get(name)
                digest = (
                    component.get("digest") if isinstance(component, dict) else None
                )
            elif category == "certification":
                digest = certification_digests.get(name)
            else:
                raise CompatibilityError("exact-gate authority is invalid")
            gate_records.append(
                {
                    "authority": authority,
                    "digest": _digest(digest, f"exactGateEvidence.{gate}.{authority}"),
                }
            )
        records[gate] = gate_records
    return records


def _exact_version_spec(value: Any, field: str) -> SpecifierSet:
    text = str(value or "")
    if not re.fullmatch(r"==[0-9]+(?:\.[0-9]+)*(?:[a-z0-9.-]+)?", text, re.I):
        raise CompatibilityError(f"{field} must pin exactly one version")
    return SpecifierSet(text)


def _exact_version(value: Any, field: str) -> str:
    """Return the sole current version text without accepting normalized aliases."""

    text = str(value or "")
    _exact_version_spec(text, field)
    version = text.removeprefix("==")
    try:
        parsed = Version(version)
    except InvalidVersion as exc:
        raise CompatibilityError(f"{field} must pin exactly one version") from exc
    if str(parsed) != version:
        raise CompatibilityError(f"{field} must use the canonical current spelling")
    return version


def validate_compatibility_matrix(matrix: dict[str, Any]) -> dict[str, Any]:
    """Validate the exact source release train without requiring a release."""
    if (
        matrix.get("apiVersion") != "graphos.io/v1"
        or matrix.get("kind") != "CompatibilityMatrix"
    ):
        raise CompatibilityError("unsupported compatibility matrix apiVersion/kind")
    _exact_keys(
        matrix,
        required={
            "apiVersion",
            "kind",
            "matrixVersion",
            "runtime",
            "protocol",
            "components",
            "releaseTrain",
        },
        field="compatibility matrix",
    )
    if matrix.get("matrixVersion") != 2:
        raise CompatibilityError("unsupported compatibility matrix version")
    runtime = matrix.get("runtime")
    if not isinstance(runtime, dict):
        raise CompatibilityError("compatibility runtime must be a mapping")
    _exact_keys(
        runtime,
        required={
            "pythonVersion",
            "baseImage",
            "pythonDependencyMode",
            "offlineTargets",
        },
        field="compatibility runtime",
    )
    offline_targets = runtime.get("offlineTargets")
    if not isinstance(offline_targets, dict):
        raise CompatibilityError("compatibility offline targets must be a mapping")
    _exact_keys(
        offline_targets,
        required={"epistemic-graph", "agent-utilities", "langfuse-agent"},
        field="compatibility offline targets",
    )
    if runtime != _CURRENT_RUNTIME_CONTRACT:
        raise CompatibilityError("compatibility runtime contract is not exact")
    protocol = matrix.get("protocol")
    if not isinstance(protocol, dict):
        raise CompatibilityError("compatibility protocol must be a mapping")
    _exact_keys(
        protocol,
        required={"name", "version", "schemas"},
        field="compatibility protocol",
    )
    schemas = protocol.get("schemas")
    if (
        protocol.get("name") != "epistemic-operations"
        or protocol.get("version") != "1"
        or not isinstance(schemas, dict)
        or set(schemas) != _REQUIRED_SCHEMAS
        or any(
            not re.fullmatch(r"[1-9][0-9]*", str(version))
            for version in schemas.values()
        )
    ):
        raise CompatibilityError("epistemic protocol schema catalog is not exact")
    components = matrix.get("components")
    if not isinstance(components, dict) or set(components) != set(_RELEASE_ORDER):
        raise CompatibilityError("compatibility matrix component set is not exact")
    release_train = matrix.get("releaseTrain")
    if not isinstance(release_train, dict):
        raise CompatibilityError("release train must be a mapping")
    _exact_keys(
        release_train,
        required={
            "assemblyOrder",
            "activationMode",
            "stateMigrationMode",
            "rollbackRequires",
        },
        field="release train",
    )
    if tuple(release_train.get("assemblyOrder") or ()) != _RELEASE_ORDER:
        raise CompatibilityError("release assembly order is not exact")
    if release_train.get("activationMode") != "atomic-exact-release":
        raise CompatibilityError("release activation must be one atomic exact cutover")
    if release_train.get("stateMigrationMode") != "one-time-persisted-state":
        raise CompatibilityError("only one-time persisted-state migration is allowed")
    required_rollback = {
        "signed-pre-cutover-snapshot",
        "verified-state-migration",
        "prior-exact-release-manifest",
        "prior-ontology-lock",
        "prior-connector-catalog",
    }
    if set(release_train.get("rollbackRequires") or ()) != required_rollback:
        raise CompatibilityError("release rollback prerequisites are not exact")
    for name, expected in components.items():
        if not isinstance(expected, dict):
            raise CompatibilityError(f"matrix component {name} must be a mapping")
        _exact_keys(
            expected,
            required={"version", "artifactKind"},
            optional={
                "dependsOn",
                "requiredCapabilities",
                "exactEntries",
                "canonicalization",
                "migrationMode",
            },
            field=f"matrix component {name}",
        )
        component_version = _exact_version(expected["version"], f"{name}.version")
        if component_version != _CURRENT_COMPONENT_VERSIONS[name]:
            raise CompatibilityError(
                f"matrix component {name} is not the current source version"
            )
        expected_kind = (
            "oci"
            if name in {"epistemic-graph", "agent-utilities", "langfuse-agent"}
            else "catalog"
        )
        if expected.get("artifactKind") != expected_kind:
            raise CompatibilityError(
                f"matrix component {name} has the wrong artifact kind"
            )
        dependencies = expected.get("dependsOn") or {}
        if not isinstance(dependencies, dict) or not set(dependencies).issubset(
            components
        ):
            raise CompatibilityError(
                f"matrix component {name} has an unknown dependency"
            )
        if dependencies != _CURRENT_COMPONENT_DEPENDENCIES[name]:
            raise CompatibilityError(
                f"matrix component {name} dependency topology is not exact"
            )
        for dependency, specifier in dependencies.items():
            dependency_version = _exact_version(
                specifier, f"{name}.dependsOn.{dependency}"
            )
            expected_dependency_version = _exact_version(
                components[dependency]["version"], f"{dependency}.version"
            )
            if dependency_version != expected_dependency_version:
                raise CompatibilityError(
                    f"matrix component {name} dependency is not the current component"
                )
        capabilities = expected.get("requiredCapabilities") or []
        if (
            not isinstance(capabilities, list)
            or len(capabilities) != len(set(capabilities))
            or not all(
                isinstance(capability, str) and capability
                for capability in capabilities
            )
        ):
            raise CompatibilityError(
                f"matrix component {name} capabilities are invalid"
            )
    if (
        int(components["connector-bundles"].get("exactEntries") or 0)
        != _CURRENT_CONNECTOR_ENTRIES
    ):
        raise CompatibilityError(
            "connector catalog must contain exactly the configured provider fleet"
        )
    if int(components["prebundled-skills"].get("exactEntries") or 0) != 10:
        raise CompatibilityError(
            "pre-bundled skill catalog must contain exactly ten skills"
        )
    if components["ontology-lock"].get("canonicalization") != "urdna2015-sha256":
        raise CompatibilityError("ontology lock canonicalization is not exact")
    if (
        components["index-migrations"].get("migrationMode")
        != "one-time-persisted-state"
        or int(components["index-migrations"].get("exactEntries") or 0) != 1
    ):
        raise CompatibilityError(
            "index migration policy must be one-time persisted-state migration"
        )
    return {"ok": True, "components": len(components), "schemas": len(schemas)}


def release_configuration_document(
    *, release_id: str, matrix: dict[str, Any], matrix_digest: str
) -> dict[str, Any]:
    """Return the sole typed release configuration accepted by this matrix."""

    validate_compatibility_matrix(matrix)
    if re.fullmatch(r"release-[a-z0-9][a-z0-9.-]{2,63}", release_id) is None:
        raise CompatibilityError("release configuration releaseId is invalid")
    _digest(matrix_digest, "release configuration compatibilityMatrixDigest")
    runtime = matrix["runtime"]
    release_train = matrix["releaseTrain"]
    capabilities = set(matrix["components"]["agent-utilities"]["requiredCapabilities"])
    return {
        "apiVersion": "graphos.io/v1",
        "kind": "ReleaseConfiguration",
        "releaseId": release_id,
        "compatibilityMatrixDigest": matrix_digest,
        "runtime": {
            "pythonVersion": runtime["pythonVersion"],
            "baseImage": runtime["baseImage"],
            "pythonDependencyMode": runtime["pythonDependencyMode"],
            "offlineTargets": dict(runtime["offlineTargets"]),
            "activationMode": release_train["activationMode"],
            "mcpToolMode": "intent",
        },
        "features": {
            "nativeProgramEvolution": "native-program-evolution" in capabilities,
            "langfuseEvolution": "langfuse-evolution" in capabilities,
        },
        "security": {
            "agentConfigRequired": True,
            "identityRequired": True,
            "tlsVerificationRequired": True,
            "referenceBackedSecretsOnly": True,
            "metadataOnlyEvidence": True,
        },
    }


def release_migration_plan_document(
    *,
    release_id: str,
    matrix: dict[str, Any],
    matrix_digest: str,
    index_migration_catalog_digest: str,
    index_migration_count: int,
) -> dict[str, Any]:
    """Return the sole typed one-time migration plan accepted by this matrix."""

    validate_compatibility_matrix(matrix)
    if re.fullmatch(r"release-[a-z0-9][a-z0-9.-]{2,63}", release_id) is None:
        raise CompatibilityError("release migration plan releaseId is invalid")
    _digest(matrix_digest, "release migration plan compatibilityMatrixDigest")
    _digest(
        index_migration_catalog_digest,
        "release migration plan indexMigrationCatalogDigest",
    )
    if (
        not isinstance(index_migration_count, int)
        or isinstance(index_migration_count, bool)
        or index_migration_count < 1
    ):
        raise CompatibilityError("release migration plan entry count is invalid")
    release_train = matrix["releaseTrain"]
    return {
        "apiVersion": "graphos.io/v1",
        "kind": "ReleaseMigrationPlan",
        "releaseId": release_id,
        "compatibilityMatrixDigest": matrix_digest,
        "indexMigrationCatalogDigest": index_migration_catalog_digest,
        "indexMigrationCount": index_migration_count,
        "migrationMode": release_train["stateMigrationMode"],
        "execution": {
            "writeFenceRequired": True,
            "signedSnapshotRequired": True,
            "verifiedMigrationRequired": True,
            "idempotentResumeRequired": True,
            "rollbackRequires": list(release_train["rollbackRequires"]),
        },
    }


def validate_release_configuration(
    value: dict[str, Any],
    *,
    release_id: str,
    matrix: dict[str, Any],
    matrix_digest: str,
) -> None:
    _validate_release_schema(
        value,
        schema_path=_RELEASE_CONFIGURATION_SCHEMA,
        field="release configuration",
    )
    if value != release_configuration_document(
        release_id=release_id,
        matrix=matrix,
        matrix_digest=matrix_digest,
    ):
        raise CompatibilityError(
            "release configuration differs from the current matrix contract"
        )


def validate_release_migration_plan(
    value: dict[str, Any],
    *,
    release_id: str,
    matrix: dict[str, Any],
    matrix_digest: str,
    index_migration_catalog_digest: str,
) -> None:
    _validate_release_schema(
        value,
        schema_path=_RELEASE_MIGRATION_PLAN_SCHEMA,
        field="release migration plan",
    )
    count = value.get("indexMigrationCount")
    if not isinstance(count, int) or isinstance(count, bool):
        raise CompatibilityError("release migration plan entry count is invalid")
    if value != release_migration_plan_document(
        release_id=release_id,
        matrix=matrix,
        matrix_digest=matrix_digest,
        index_migration_catalog_digest=index_migration_catalog_digest,
        index_migration_count=count,
    ):
        raise CompatibilityError(
            "release migration plan differs from the current matrix contract"
        )


def _bounded_adapter(
    command: list[str],
    payload: bytes,
    *,
    maximum: int,
) -> tuple[int, bytes, bytes]:
    """Run one trusted adapter without allowing unbounded captured output."""

    try:
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            close_fds=True,
        )
    except OSError as exc:
        raise CompatibilityError("external adapter could not be launched") from exc
    stdout = bytearray()
    stderr = bytearray()
    overflow = threading.Event()

    def read_bounded(stream: Any, destination: bytearray) -> None:
        try:
            while True:
                chunk = stream.read(65_536)
                if not chunk:
                    return
                room = maximum + 1 - len(destination)
                if room > 0:
                    destination.extend(chunk[:room])
                if len(destination) > maximum or len(chunk) > room:
                    overflow.set()
                    process.kill()
                    return
        finally:
            stream.close()

    assert process.stdout is not None
    assert process.stderr is not None
    stdout_thread = threading.Thread(
        target=read_bounded,
        args=(process.stdout, stdout),
        daemon=True,
    )
    stderr_thread = threading.Thread(
        target=read_bounded,
        args=(process.stderr, stderr),
        daemon=True,
    )
    stdout_thread.start()
    stderr_thread.start()

    def write_payload() -> None:
        assert process.stdin is not None
        try:
            process.stdin.write(payload)
            process.stdin.flush()
        except (BrokenPipeError, OSError):
            pass
        finally:
            try:
                process.stdin.close()
            except (BrokenPipeError, OSError):
                pass

    stdin_thread = threading.Thread(target=write_payload, daemon=True)
    stdin_thread.start()
    try:
        try:
            returncode = process.wait(timeout=120)
        except subprocess.TimeoutExpired as exc:
            process.kill()
            process.wait()
            raise CompatibilityError("external adapter timed out") from exc
    finally:
        stdin_thread.join(timeout=5)
        stdout_thread.join(timeout=5)
        stderr_thread.join(timeout=5)
        if process.poll() is None:
            process.kill()
            process.wait()
    if overflow.is_set():
        raise CompatibilityError("external adapter output exceeds its boundary")
    return returncode, bytes(stdout), bytes(stderr)


def _verify_signature(
    name: str, component: dict[str, Any], signature_bundle: dict[str, Any]
) -> None:
    signature = component.get("signature")
    if not isinstance(signature, dict):
        raise CompatibilityError(f"{name}.signature is required")
    _exact_keys(
        signature,
        required={"bundleDigest", "verifierEnv"},
        field=f"{name}.signature",
    )
    _digest(signature.get("bundleDigest"), f"{name}.signature.bundleDigest")
    env_name = str(signature.get("verifierEnv") or "")
    if not _ENV_NAME.fullmatch(env_name):
        raise CompatibilityError(f"{name}.signature.verifierEnv is invalid")
    raw_command = os.environ.get(env_name, "")
    if not raw_command:
        raise CompatibilityError(f"signature verifier command {env_name} is absent")
    try:
        command = json.loads(raw_command)
    except json.JSONDecodeError as exc:
        raise CompatibilityError(f"{env_name} must contain a JSON argv array") from exc
    if (
        not isinstance(command, list)
        or not command
        or not all(isinstance(part, str) and part for part in command)
    ):
        raise CompatibilityError(f"{env_name} must contain a non-empty JSON argv array")
    payload = _canonical_bytes(signature_bundle)
    returncode, stdout, stderr = _bounded_adapter(
        command,
        payload,
        maximum=_MAX_ADAPTER_OUTPUT_BYTES,
    )
    if returncode != 0:
        raise CompatibilityError(
            f"signature verification failed for {name}; output_digest="
            + hashlib.sha256(stdout + stderr).hexdigest()
        )
    try:
        verified = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise CompatibilityError(
            f"signature verifier returned non-JSON for {name}"
        ) from exc
    expected = {
        "verified": True,
        "subjectDigest": signature_bundle["subject"]["digest"],
        "artifactDigest": component["digest"],
    }
    if verified != expected:
        raise CompatibilityError(
            f"signature verifier did not bind {name} to its digest"
        )


def _validate_manifest_signature(manifest: dict[str, Any]) -> dict[str, Any]:
    signature = manifest.get("signature")
    if not isinstance(signature, dict):
        raise CompatibilityError("release manifest signature is required")
    _exact_keys(
        signature,
        required={
            "scheme",
            "subjectDigest",
            "bundleDigest",
            "signerIdentityDigest",
            "value",
            "verifierEnv",
        },
        field="release manifest signature",
    )
    for field in ("subjectDigest", "bundleDigest", "signerIdentityDigest"):
        _digest(signature.get(field), f"release manifest signature.{field}")
    if not _SIGNATURE_SCHEME.fullmatch(str(signature.get("scheme") or "")):
        raise CompatibilityError("release manifest signature scheme is invalid")
    if not _SIGNATURE_VALUE.fullmatch(str(signature.get("value") or "")):
        raise CompatibilityError("release manifest signature value is invalid")
    verifier_env = str(signature.get("verifierEnv") or "")
    if not _ENV_NAME.fullmatch(verifier_env):
        raise CompatibilityError("release manifest signature verifierEnv is invalid")
    unsigned = {key: value for key, value in manifest.items() if key != "signature"}
    if signature["subjectDigest"] != canonical_digest(unsigned):
        raise CompatibilityError(
            "release manifest signature does not bind the manifest"
        )
    return signature


def _expected_skill_validation_contract() -> dict[str, Any]:
    matrix_path = (
        Path(__file__).resolve().parents[2]
        / "agent_utilities/skills/runtime_validation.yaml"
    )
    try:
        payload = matrix_path.read_bytes()
    except OSError as exc:
        raise CompatibilityError("skill validation catalog is unavailable") from exc
    if not payload or len(payload) > _MAX_COMPONENT_SOURCE_BYTES:
        raise CompatibilityError("skill validation catalog violates its size boundary")
    try:
        matrix = yaml.safe_load(payload)
    except yaml.YAMLError as exc:
        raise CompatibilityError("skill validation catalog is invalid") from exc
    if not isinstance(matrix, dict) or not isinstance(matrix.get("cases"), list):
        raise CompatibilityError("skill validation catalog is invalid")
    cases: dict[str, dict[str, Any]] = {}
    for item in matrix["cases"]:
        if not isinstance(item, dict):
            raise CompatibilityError("skill validation catalog is invalid")
        contract = {
            "id": str(item.get("id") or ""),
            "skill": str(item.get("skill") or ""),
            "mode": str(item.get("mode") or ""),
            "modelClass": str(item.get("model_class") or ""),
            "taskDigest": "sha256:"
            + hashlib.sha256(str(item.get("task") or "").encode("utf-8")).hexdigest(),
            "expectedRoutes": list(item.get("expected_routes") or ()),
            "allowedTools": list(item.get("allowed_tools") or ()),
            "readOnly": item.get("read_only"),
        }
        case_id = contract["id"]
        if not isinstance(case_id, str) or case_id in cases:
            raise CompatibilityError("skill validation catalog is invalid")
        cases[case_id] = {
            **contract,
            "caseDigest": canonical_digest(contract),
        }
    if tuple(sorted(cases)) != _SKILL_CASE_IDS:
        raise CompatibilityError("skill validation catalog case set is not exact")
    case_catalog = [
        {"caseId": case_id, "caseDigest": cases[case_id]["caseDigest"]}
        for case_id in sorted(cases)
    ]
    return {
        "testCatalogDigest": canonical_digest(matrix),
        "caseCatalogDigest": canonical_digest(case_catalog),
        "cases": cases,
    }


def validate_prebundled_skill_matrix(
    payload: bytes,
    *,
    release_id: str,
    release_specification_digest: str,
    promotion_evidence_digest: str,
    graph_os_digest: str,
    engine_digest: str,
    configuration_digest: str,
    skill_catalog_digest: str,
) -> dict[str, Any]:
    """Require the exact signed, passing current 20-case runtime evidence."""

    evidence = _json_evidence(payload, "prebundledSkillValidationMatrix")
    _validate_release_schema(
        evidence,
        schema_path=_SKILL_VALIDATION_MATRIX_SCHEMA,
        field="prebundledSkillValidationMatrix",
    )
    _exact_keys(
        evidence,
        required={
            "apiVersion",
            "kind",
            "evidenceVersion",
            "generatedAt",
            "release",
            "runtime",
            "catalog",
            "cases",
            "result",
            "privacy",
            "signature",
        },
        field="prebundledSkillValidationMatrix",
    )
    if (
        evidence.get("apiVersion") != "graphos.io/v2"
        or evidence.get("kind") != "PrebundledSkillValidationEvidence"
        or evidence.get("evidenceVersion") != 2
        or not re.fullmatch(
            r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9:.+-]+Z?",
            str(evidence.get("generatedAt") or ""),
        )
    ):
        raise CompatibilityError("skill validation evidence identity is invalid")
    release = evidence.get("release")
    if not isinstance(release, dict):
        raise CompatibilityError("skill validation release binding is required")
    expected_release = {
        "id": release_id,
        "specificationDigest": release_specification_digest,
        "promotionEvidenceDigest": promotion_evidence_digest,
        "graphOsDigest": graph_os_digest,
        "engineDigest": engine_digest,
    }
    if release != expected_release:
        raise CompatibilityError("skill validation release binding differs")
    runtime = evidence.get("runtime")
    if not isinstance(runtime, dict):
        raise CompatibilityError("skill validation runtime binding is required")
    _exact_keys(
        runtime,
        required={
            "configurationDigest",
            "profileDigest",
            "modelRegistryDigest",
            "sequential",
            "metadataOnlyObservability",
        },
        field="skill validation runtime",
    )
    if (
        runtime.get("configurationDigest") != configuration_digest
        or runtime.get("sequential") is not True
        or runtime.get("metadataOnlyObservability") is not True
    ):
        raise CompatibilityError("skill validation runtime binding differs")
    _digest(runtime.get("profileDigest"), "skill validation runtime.profileDigest")
    _digest(
        runtime.get("modelRegistryDigest"),
        "skill validation runtime.modelRegistryDigest",
    )
    expected = _expected_skill_validation_contract()
    catalog = evidence.get("catalog")
    if not isinstance(catalog, dict):
        raise CompatibilityError("skill validation catalog binding is required")
    _exact_keys(
        catalog,
        required={
            "skillCount",
            "skillCatalogDigest",
            "testCaseCount",
            "testCatalogDigest",
            "caseCatalogDigest",
        },
        field="skill validation catalog",
    )
    if catalog != {
        "skillCount": 10,
        "skillCatalogDigest": skill_catalog_digest,
        "testCaseCount": 20,
        "testCatalogDigest": expected["testCatalogDigest"],
        "caseCatalogDigest": expected["caseCatalogDigest"],
    }:
        raise CompatibilityError("skill validation catalog binding differs")
    cases = evidence.get("cases")
    if (
        not isinstance(cases, list)
        or len(cases) != 20
        or tuple(
            str(case.get("caseId") or "") for case in cases if isinstance(case, dict)
        )
        != _SKILL_CASE_IDS
    ):
        raise CompatibilityError("skill validation case set is not exact")
    for case in cases:
        if not isinstance(case, dict):
            raise CompatibilityError("skill validation case is invalid")
        _exact_keys(
            case,
            required={
                "caseId",
                "caseDigest",
                "skill",
                "mode",
                "modelClass",
                "status",
                "checks",
                "skillRef",
                "skillBodyRef",
                "runRef",
                "traceRef",
                "langfuse",
                "parentKnowledgeGraph",
                "errorCodes",
            },
            field="skill validation case",
        )
        expected_case = expected["cases"][case["caseId"]]
        if (
            any(
                case.get(field) != expected_case[field]
                for field in ("caseDigest", "skill", "mode", "modelClass")
            )
            or case.get("status") != "pass"
        ):
            raise CompatibilityError("skill validation case binding differs")
        checks = case.get("checks")
        required_checks = {
            "structural",
            "modelSelection",
            "skillBinding",
            "semantic",
            "delegation",
            "trace",
            "parentKnowledgeGraph",
        }
        if not isinstance(checks, dict) or set(checks) != required_checks:
            raise CompatibilityError("skill validation checks are not exact")
        expected_delegation = (
            "pass" if case["mode"] == "delegated" else "not-applicable"
        )
        if checks.get("delegation") != expected_delegation or any(
            checks.get(field) != "pass" for field in required_checks - {"delegation"}
        ):
            raise CompatibilityError("skill validation case did not fully pass")
        for field in ("skillRef", "skillBodyRef", "runRef", "traceRef"):
            if not _OPAQUE_REFERENCE.fullmatch(str(case.get(field) or "")):
                raise CompatibilityError("skill validation reference is invalid")
        langfuse = case.get("langfuse")
        if not isinstance(langfuse, dict) or set(langfuse) != {
            "lookupMethod",
            "metadataOnly",
            "traceName",
            "matchCount",
            "linkage",
        }:
            raise CompatibilityError("skill validation Langfuse evidence is not exact")
        if (
            langfuse.get("lookupMethod") != "exact-name"
            or langfuse.get("metadataOnly") is not True
            or not _TRACE_NAME.fullmatch(str(langfuse.get("traceName") or ""))
            or langfuse.get("matchCount") != 1
            or langfuse.get("linkage") != "run-evidence"
        ):
            raise CompatibilityError("skill validation Langfuse linkage did not pass")
        parent = case.get("parentKnowledgeGraph")
        if parent != {"readbackMethod": "exact-trace-name", "matchCount": 1}:
            raise CompatibilityError(
                "skill validation parent graph linkage did not pass"
            )
        if case.get("errorCodes") != []:
            raise CompatibilityError("skill validation case retained errors")
    result = evidence.get("result")
    if result != {
        "status": "pass",
        "passedCases": 20,
        "totalCases": 20,
        "fullyPassedSkills": 10,
        "totalSkills": 10,
    }:
        raise CompatibilityError("skill validation result is not an exact pass")
    privacy = evidence.get("privacy")
    expected_privacy = {
        "containsPrompts",
        "containsModelOutput",
        "containsEndpoints",
        "containsCredentials",
        "containsIdentities",
        "containsFilesystemLocations",
        "containsRawTraceIdentifiers",
    }
    if (
        not isinstance(privacy, dict)
        or set(privacy) != expected_privacy
        or any(value is not False for value in privacy.values())
    ):
        raise CompatibilityError("skill validation privacy evidence is not exact")
    signature = evidence.get("signature")
    if not isinstance(signature, dict):
        raise CompatibilityError("skill validation signature is required")
    _exact_keys(
        signature,
        required={"algorithm", "keyId", "signature", "subjectDigest"},
        field="skill validation signature",
    )
    if (
        signature.get("algorithm")
        not in {"ed25519", "ecdsa-p256-sha256", "rsa-pss-sha256"}
        or re.fullmatch(r"key:[a-f0-9]{64}", str(signature.get("keyId") or "")) is None
        or re.fullmatch(
            r"[A-Za-z0-9_-]{43,4096}", str(signature.get("signature") or "")
        )
        is None
    ):
        raise CompatibilityError("skill validation signature is invalid")
    unsigned = {key: value for key, value in evidence.items() if key != "signature"}
    if signature.get("subjectDigest") != canonical_digest(unsigned):
        raise CompatibilityError(
            "skill validation signature does not bind the evidence"
        )
    return evidence


def validate_skill_validation_deployment(
    payload: bytes,
    *,
    release_id: str,
    configuration_digest: str,
    release_binding: dict[str, Any],
    validation_evidence: dict[str, Any],
) -> dict[str, Any]:
    """Require the exact current deployment authority for skill certification."""

    deployment = _json_evidence(payload, "skillValidationDeployment")
    _validate_release_schema(
        deployment,
        schema_path=_SKILL_VALIDATION_DEPLOYMENT_SCHEMA,
        field="skillValidationDeployment",
    )
    if (
        deployment.get("apiVersion") != "graphos.io/v2"
        or deployment.get("kind") != "SkillValidationDeployment"
    ):
        raise CompatibilityError("skill validation deployment identity is invalid")
    identity_authority = deployment.get("identityAuthority")
    if (
        not isinstance(identity_authority, dict)
        or identity_authority.get("mode") != "ephemeral-https-loopback"
        or isinstance(identity_authority.get("tokenTtlSeconds"), bool)
        or not isinstance(identity_authority.get("tokenTtlSeconds"), int)
        or not 180 <= identity_authority["tokenTtlSeconds"] <= 3_600
        or identity_authority.get("tlsVerificationRequired") is not True
        or identity_authority.get("lifecycleOwned") is not True
        or identity_authority.get("renewableCredentialsRequired") is not True
    ):
        raise CompatibilityError(
            "skill validation deployment identity authority differs"
        )
    release = deployment.get("release")
    if not isinstance(release, dict):
        raise CompatibilityError("skill validation deployment release is required")
    expected_release = {
        "id": release_id,
        "specificationDigest": release_binding["releaseSpecSha256"],
        "promotionEvidenceDigest": release_binding["promotionEvidenceSha256"],
        "agentUtilitiesSha256": release_binding["agentUtilitiesSha256"],
        "distributionClosureSha256": release_binding["distributionClosureSha256"],
        "releasePythonSha256": release_binding["releasePythonSha256"],
        "graphOsDigest": release_binding["graphosSha256"],
        "engineDigest": release_binding["engineSha256"],
    }
    agent_file_count = release.get("agentUtilitiesFileCount")
    if (
        any(release.get(key) != value for key, value in expected_release.items())
        or isinstance(agent_file_count, bool)
        or not isinstance(agent_file_count, int)
        or agent_file_count < 10
    ):
        raise CompatibilityError("skill validation deployment release binding differs")
    runtime = deployment.get("runtime")
    validation_runtime = validation_evidence.get("runtime")
    if not isinstance(runtime, dict) or not isinstance(validation_runtime, dict):
        raise CompatibilityError("skill validation deployment runtime is required")
    model_registry = runtime.get("modelRegistry")
    literal_private_count = (
        model_registry.get("literalPrivateModelCount")
        if isinstance(model_registry, dict)
        else None
    )
    private_dns_count = (
        model_registry.get("privateDnsModelCount")
        if isinstance(model_registry, dict)
        else None
    )
    if (
        runtime.get("configurationDigest") != configuration_digest
        or runtime.get("configurationDigest")
        != validation_runtime.get("configurationDigest")
        or runtime.get("profileDigest") != validation_runtime.get("profileDigest")
        or not isinstance(model_registry, dict)
        or any(
            model_registry.get(key) != value
            for key, value in {
                "digest": validation_runtime.get("modelRegistryDigest"),
                "modelCount": 2,
                "lightCount": 1,
                "normalCount": 1,
                "localPrivateTransportOnly": True,
                "referenceBackedCredentialsOnly": True,
                "runtimePrivateResolutionRequired": True,
            }.items()
        )
        or isinstance(literal_private_count, bool)
        or not isinstance(literal_private_count, int)
        or isinstance(private_dns_count, bool)
        or not isinstance(private_dns_count, int)
        or literal_private_count + private_dns_count != 2
    ):
        raise CompatibilityError("skill validation deployment runtime binding differs")
    return deployment


def validate_skill_validation_lifecycle(
    payload: bytes,
    *,
    release_id: str,
    configuration_digest: str,
    validation_evidence_digest: str,
    release_binding: dict[str, Any],
    validation_evidence: dict[str, Any],
    deployment: dict[str, Any],
) -> dict[str, Any]:
    """Require the signed passing v2 lifecycle for the exact 20-case evidence."""

    evidence = _json_evidence(payload, "skillValidationLifecycleEvidence")
    _validate_release_schema(
        evidence,
        schema_path=_SKILL_VALIDATION_LIFECYCLE_SCHEMA,
        field="skillValidationLifecycleEvidence",
    )
    if (
        evidence.get("apiVersion") != "graphos.io/v2"
        or evidence.get("kind") != "SkillValidationLifecycleEvidence"
        or evidence.get("evidenceVersion") != 2
        or evidence.get("result") != "pass"
        or evidence.get("errorCode") is not None
    ):
        raise CompatibilityError("skill validation lifecycle identity is not passing")
    validation_expected_release = {
        "id": release_id,
        "specificationDigest": release_binding["releaseSpecSha256"],
        "promotionEvidenceDigest": release_binding["promotionEvidenceSha256"],
        "graphOsDigest": release_binding["graphosSha256"],
        "engineDigest": release_binding["engineSha256"],
    }
    release = evidence.get("release")
    validation_release = validation_evidence.get("release")
    deployment_release = deployment.get("release")
    if (
        not isinstance(deployment_release, dict)
        or validation_release != validation_expected_release
    ):
        raise CompatibilityError("skill validation lifecycle release binding differs")
    expected_release = {
        **validation_expected_release,
        "agentUtilitiesSha256": release_binding["agentUtilitiesSha256"],
        "agentUtilitiesFileCount": deployment_release["agentUtilitiesFileCount"],
        "distributionClosureSha256": release_binding["distributionClosureSha256"],
        "releasePythonSha256": release_binding["releasePythonSha256"],
    }
    if release != expected_release or any(
        deployment_release.get(key) != value for key, value in expected_release.items()
    ):
        raise CompatibilityError("skill validation lifecycle release binding differs")
    validation_runtime = validation_evidence.get("runtime")
    deployment_runtime = deployment.get("runtime")
    model_registry = (
        deployment_runtime.get("modelRegistry")
        if isinstance(deployment_runtime, dict)
        else None
    )
    expected_runtime = {
        "configurationDigest": configuration_digest,
        "profileDigest": (
            validation_runtime.get("profileDigest")
            if isinstance(validation_runtime, dict)
            else None
        ),
        "modelRegistryDigest": (
            model_registry.get("digest") if isinstance(model_registry, dict) else None
        ),
    }
    if evidence.get("runtime") != expected_runtime or (
        not isinstance(validation_runtime, dict)
        or any(
            validation_runtime.get(key) != value
            for key, value in expected_runtime.items()
        )
    ):
        raise CompatibilityError("skill validation lifecycle runtime binding differs")
    identity_authority = evidence.get("identityAuthority")
    deployment_authority = deployment.get("identityAuthority")
    if (
        not isinstance(deployment_authority, dict)
        or not isinstance(identity_authority, dict)
        or identity_authority.get("mode") != deployment_authority.get("mode")
        or identity_authority.get("lifecycleCounts")
        != {"before": 0, "running": 1, "after": 0}
        or identity_authority.get("tlsVerified") is not True
        or identity_authority.get("renewableCredentialsProven") is not True
        or isinstance(identity_authority.get("tokenMintCount"), bool)
        or not isinstance(identity_authority.get("tokenMintCount"), int)
        or identity_authority["tokenMintCount"] < 2
        or identity_authority.get("reaped") is not True
    ):
        raise CompatibilityError("skill validation lifecycle identity differs")
    model_registry = (
        deployment.get("runtime", {}).get("modelRegistry")
        if isinstance(deployment.get("runtime"), dict)
        else None
    )
    expected_transport_proof = {
        "modelCount": 2,
        "literalPrivateModelCount": (
            model_registry.get("literalPrivateModelCount")
            if isinstance(model_registry, dict)
            else None
        ),
        "privateDnsModelCount": (
            model_registry.get("privateDnsModelCount")
            if isinstance(model_registry, dict)
            else None
        ),
        "privateDnsUniqueResolutionProven": True,
        "privateBoundaryProven": True,
        "dnsRebindingGuarded": True,
    }
    if evidence.get("modelTransportProof") != expected_transport_proof:
        raise CompatibilityError(
            "skill validation lifecycle model transport proof differs"
        )
    counts = {"before": 0, "running": 1, "after": 0}
    expected_process_gate = {
        "globalGraphOs": counts,
        "candidateGraphOs": counts,
        "candidateEngine": counts,
        "terminalProcessCounts": {
            "langfuseMcpChildren": 0,
            "loopbackOidcFixtures": 0,
        },
        "engineExecutableDigest": release_binding["engineSha256"],
        "installedReleaseAttested": True,
        "reaped": True,
    }
    if evidence.get("processGate") != expected_process_gate:
        raise CompatibilityError("skill validation lifecycle process gate differs")
    if evidence.get("validation") != {
        "exitCode": 0,
        "evidenceDigest": validation_evidence_digest,
        "caseCount": 20,
    }:
        raise CompatibilityError("skill validation lifecycle evidence binding differs")
    privacy = evidence.get("privacy")
    if (
        not isinstance(privacy, dict)
        or set(privacy)
        != {
            "containsEndpoints",
            "containsCredentials",
            "containsProfiles",
            "containsFilesystemLocations",
            "containsIdentities",
            "containsContent",
        }
        or any(value is not False for value in privacy.values())
    ):
        raise CompatibilityError("skill validation lifecycle privacy is not exact")
    signature = evidence.get("signature")
    if not isinstance(signature, dict):
        raise CompatibilityError("skill validation lifecycle signature is required")
    unsigned = {key: value for key, value in evidence.items() if key != "signature"}
    if signature.get("subjectDigest") != canonical_digest(unsigned):
        raise CompatibilityError("skill validation lifecycle signature is unbound")
    return evidence


def _verify_skill_validation_evidence(
    evidence: dict[str, Any],
    *,
    deployment: dict[str, Any],
    field: str,
) -> None:
    validation = deployment.get("validation")
    reference = (
        str(validation.get("verifierCommandReference") or "")
        if isinstance(validation, dict)
        else ""
    )
    if not _ENV_NAME.fullmatch(reference):
        raise CompatibilityError("skill validation verifier reference is invalid")
    try:
        from agent_utilities.skills.runtime_validation import _external_command

        command = _external_command(reference)
    except Exception as exc:
        raise CompatibilityError("skill validation verifier is unavailable") from exc
    returncode, stdout, stderr = _bounded_adapter(
        command,
        _canonical_bytes(evidence),
        maximum=_MAX_ADAPTER_OUTPUT_BYTES,
    )
    if returncode != 0:
        raise CompatibilityError(
            f"{field} verification failed; output_digest="
            + hashlib.sha256(stdout + stderr).hexdigest()
        )
    try:
        response = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise CompatibilityError(f"{field} verifier returned non-JSON") from exc
    signature = evidence["signature"]
    if response != {
        "verified": True,
        "subjectDigest": signature["subjectDigest"],
        "keyId": signature["keyId"],
    }:
        raise CompatibilityError(f"{field} verification is not exact")


def validate_exact_artifact_closure(
    payload: bytes,
    *,
    release_id: str,
) -> dict[str, Any]:
    """Require the signed closure for every current live exact-artifact campaign."""

    evidence = _json_evidence(payload, "exactArtifactClosureEvidence")
    _exact_keys(
        evidence,
        required={
            "apiVersion",
            "kind",
            "schemaVersion",
            "releaseId",
            "status",
            "privacySafe",
            "release",
            "campaigns",
            "gates",
            "signature",
        },
        field="exactArtifactClosureEvidence",
    )
    if (
        evidence.get("apiVersion") != "graphos.io/v1"
        or evidence.get("kind") != "ExactArtifactClosureEvidence"
        or evidence.get("schemaVersion") != 1
        or evidence.get("releaseId") != release_id
        or evidence.get("status") != "passed"
        or evidence.get("privacySafe") is not True
    ):
        raise CompatibilityError("exact-artifact closure identity is not exact")
    release = evidence.get("release")
    release_fields = {
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
    }
    if not isinstance(release, dict) or set(release) != release_fields:
        raise CompatibilityError("exact-artifact closure release binding is not exact")
    for field, digest in release.items():
        _digest(digest, f"exactArtifactClosureEvidence.release.{field}")
    campaign_contract = {
        "faultRestart": {"matrix_cases": 60, "mutation_families": 15},
        "protocolAuthorization": {"data_path_cases": 14, "protocol_cases": 10},
        "workItemAgentBus": {"work_item_cases": 8, "agent_bus_cases": 2},
        "performance": {"scenario_families": 30, "ledger_rows": 54},
        "multimodal": {
            "modalities": 4,
            "behavior_dimensions": 12,
            "fault_cases": 16,
        },
        "knowledgeBatch": {"families": 7, "requirements": 7, "snapshot_cases": 7},
        "reasoningRepair": {"cases": 9},
        "exactLocal": {
            "gates": 7,
            "optimizer_families": 13,
            "optimizer_modalities": 14,
        },
        "permissionGovernance": {"cases": 8},
    }
    extra_digest_fields = {
        "multimodal": {"performanceEvidenceSha256"},
        "exactLocal": {"campaignManifestSha256"},
        "workItemAgentBus": set(),
        "permissionGovernance": set(),
    }
    campaigns = evidence.get("campaigns")
    if not isinstance(campaigns, dict) or set(campaigns) != set(campaign_contract):
        raise CompatibilityError("exact-artifact campaign catalog is not exact")
    for name, constants in campaign_contract.items():
        campaign = campaigns.get(name)
        digest_fields = {"evidenceSha256", *extra_digest_fields.get(name, set())}
        if (
            not isinstance(campaign, dict)
            or set(campaign) != set(constants) | digest_fields
            or any(campaign.get(field) != value for field, value in constants.items())
        ):
            raise CompatibilityError(f"exact-artifact campaign {name} is not exact")
        for field in digest_fields:
            _digest(campaign.get(field), f"exactArtifactClosureEvidence.{name}.{field}")
    gates = evidence.get("gates")
    if gates != {gate: "passed" for gate in _EXACT_ARTIFACT_GATES}:
        raise CompatibilityError("exact-artifact closure gate catalog is not exact")
    signature = evidence.get("signature")
    if (
        not isinstance(signature, dict)
        or set(signature) != {"algorithm", "keyId", "signature", "subjectDigest"}
        or signature.get("algorithm")
        not in {"ed25519", "ecdsa-p256-sha256", "rsa-pss-sha256"}
        or re.fullmatch(r"key:[a-f0-9]{64}", str(signature.get("keyId") or "")) is None
        or re.fullmatch(
            r"[A-Za-z0-9_-]{43,4096}", str(signature.get("signature") or "")
        )
        is None
    ):
        raise CompatibilityError("exact-artifact closure signature is invalid")
    unsigned = {key: value for key, value in evidence.items() if key != "signature"}
    if signature.get("subjectDigest") != canonical_digest(unsigned):
        raise CompatibilityError("exact-artifact closure signature subject differs")
    return evidence


def validate_oci_vulnerability_scan_evidence(
    payload: bytes,
    *,
    release_id: str,
) -> dict[str, Any]:
    """Require the exact aggregate-only scan for all current OCI subjects."""

    evidence = _json_evidence(payload, "ociVulnerabilityScanEvidence")
    _validate_release_schema(
        evidence,
        schema_path=_OCI_VULNERABILITY_SCAN_SCHEMA,
        field="ociVulnerabilityScanEvidence",
    )
    if evidence.get("releaseId") != release_id:
        raise CompatibilityError("OCI vulnerability scan release binding differs")
    try:
        from scripts.release import generate_oci_vulnerability_scan_evidence

        generate_oci_vulnerability_scan_evidence.validate_evidence(
            evidence,
            verifier_env=_OCI_VULNERABILITY_SCAN_VERIFIER_ENV,
            verify_signature=False,
        )
    except Exception as exc:
        raise CompatibilityError(
            "OCI vulnerability scan evidence is not exact and passing"
        ) from exc
    return evidence


def _verify_oci_vulnerability_scan_evidence(evidence: dict[str, Any]) -> None:
    try:
        from scripts.release import generate_oci_vulnerability_scan_evidence

        generate_oci_vulnerability_scan_evidence.validate_evidence(
            evidence,
            verifier_env=_OCI_VULNERABILITY_SCAN_VERIFIER_ENV,
            verify_signature=True,
        )
    except Exception as exc:
        raise CompatibilityError(
            "OCI vulnerability scan evidence signature verification failed"
        ) from exc


def _verify_exact_artifact_closure(evidence: dict[str, Any]) -> None:
    raw_command = os.environ.get(_EXACT_ARTIFACT_CLOSURE_VERIFIER_ENV, "")
    if not raw_command:
        raise CompatibilityError("exact-artifact closure verifier is absent")
    try:
        command = json.loads(raw_command)
    except json.JSONDecodeError as exc:
        raise CompatibilityError(
            "exact-artifact closure verifier must be a JSON argv array"
        ) from exc
    if (
        not isinstance(command, list)
        or not command
        or not all(isinstance(part, str) and part for part in command)
    ):
        raise CompatibilityError(
            "exact-artifact closure verifier must be a JSON argv array"
        )
    returncode, stdout, stderr = _bounded_adapter(
        command,
        _canonical_bytes(evidence),
        maximum=_MAX_ADAPTER_OUTPUT_BYTES,
    )
    if returncode != 0:
        raise CompatibilityError(
            "exact-artifact closure verification failed; output_digest="
            + hashlib.sha256(stdout + stderr).hexdigest()
        )
    try:
        response = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise CompatibilityError(
            "exact-artifact closure verifier returned non-JSON"
        ) from exc
    signature = evidence["signature"]
    if response != {
        "verified": True,
        "subjectDigest": signature["subjectDigest"],
        "keyId": signature["keyId"],
    }:
        raise CompatibilityError("exact-artifact closure verification is not exact")


def validate_connector_ledger(payload: bytes) -> dict[str, Any]:
    """Validate the signed local binding for externally certified connectors."""

    ledger = _json_evidence(payload, "connectorLiveCertificationLedger")
    _exact_keys(
        ledger,
        required={
            "apiVersion",
            "kind",
            "ledgerVersion",
            "entryCount",
            "entries",
            "signature",
        },
        field="connector live-certification ledger",
    )
    entries = ledger.get("entries")
    if (
        ledger.get("apiVersion") != "graphos.io/v1"
        or ledger.get("kind") != "ConnectorLiveCertificationLedger"
        or ledger.get("ledgerVersion") != 1
        or not isinstance(entries, list)
        or not entries
        or ledger.get("entryCount") != len(entries)
        or len(entries) != _CURRENT_CONNECTOR_ENTRIES
    ):
        raise CompatibilityError("connector live-certification ledger is invalid")
    connectors: list[str] = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise CompatibilityError(
                "connector live-certification ledger entry is invalid"
            )
        _exact_keys(
            entry,
            required={"connector", "certifiedAt", "recordDigest", "bundleDigest"},
            field="connector live-certification ledger entry",
        )
        connector = str(entry.get("connector") or "")
        if not re.fullmatch(r"[a-z0-9][a-z0-9._-]{1,127}", connector):
            raise CompatibilityError("connector live-certification identity is invalid")
        connectors.append(connector)
        _digest(entry.get("recordDigest"), "connector ledger recordDigest")
        _digest(entry.get("bundleDigest"), "connector ledger bundleDigest")
        if not re.fullmatch(
            r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z",
            str(entry.get("certifiedAt") or ""),
        ):
            raise CompatibilityError(
                "connector live-certification timestamp is invalid"
            )
    if connectors != sorted(set(connectors)):
        raise CompatibilityError("connector live-certification entries are not exact")
    signature = ledger.get("signature")
    if not isinstance(signature, dict):
        raise CompatibilityError("connector live-certification signature is required")
    _exact_keys(
        signature,
        required={
            "scheme",
            "subjectDigest",
            "bundleDigest",
            "signerIdentityDigest",
            "value",
            "verifierEnv",
        },
        field="connector live-certification signature",
    )
    for field in ("subjectDigest", "bundleDigest", "signerIdentityDigest"):
        _digest(signature.get(field), f"connector ledger signature.{field}")
    if not _SIGNATURE_SCHEME.fullmatch(str(signature.get("scheme") or "")):
        raise CompatibilityError(
            "connector live-certification signature scheme is invalid"
        )
    if not _SIGNATURE_VALUE.fullmatch(str(signature.get("value") or "")):
        raise CompatibilityError(
            "connector live-certification signature value is invalid"
        )
    if not _ENV_NAME.fullmatch(str(signature.get("verifierEnv") or "")):
        raise CompatibilityError("connector live-certification verifier is invalid")
    unsigned = {key: value for key, value in ledger.items() if key != "signature"}
    if signature.get("subjectDigest") != canonical_digest(unsigned):
        raise CompatibilityError("connector live-certification signature is unbound")
    return ledger


def _verify_manifest_signature(manifest: dict[str, Any]) -> None:
    signature = _validate_manifest_signature(manifest)
    raw_command = os.environ.get(signature["verifierEnv"], "")
    if not raw_command:
        raise CompatibilityError(
            "release manifest signature verifier command is absent"
        )
    try:
        command = json.loads(raw_command)
    except json.JSONDecodeError as exc:
        raise CompatibilityError(
            "release manifest verifier must be a JSON argv array"
        ) from exc
    if (
        not isinstance(command, list)
        or not command
        or not all(isinstance(part, str) and part for part in command)
    ):
        raise CompatibilityError("release manifest verifier must be a JSON argv array")
    result = subprocess.run(
        command,
        input=json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode(),
        capture_output=True,
        check=False,
        timeout=120,
    )
    if result.returncode != 0:
        raise CompatibilityError(
            "release manifest signature verification failed; output_digest="
            + hashlib.sha256(result.stdout + result.stderr).hexdigest()
        )
    try:
        response = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise CompatibilityError("release manifest verifier returned non-JSON") from exc
    if (
        not isinstance(response, dict)
        or response.get("verified") is not True
        or response.get("subjectDigest") != signature["subjectDigest"]
    ):
        raise CompatibilityError("release manifest verifier did not bind its subject")


def verify_release_manifest(
    manifest: dict[str, Any],
    matrix: dict[str, Any],
    *,
    matrix_path: Path | None = None,
    manifest_path: Path | None = None,
    verify_signatures: bool = True,
    require_manifest_signature: bool = True,
) -> dict[str, Any]:
    validate_compatibility_matrix(matrix)
    if (
        manifest.get("apiVersion") != "graphos.io/v1"
        or manifest.get("kind") != "ReleaseManifest"
    ):
        raise CompatibilityError("unsupported release manifest apiVersion/kind")
    required_manifest_keys = {
        "apiVersion",
        "kind",
        "manifestState",
        "releaseId",
        "matrixDigest",
        "sourceFreezeEvidenceDigest",
        "configurationDigest",
        "protocolSchemas",
        "components",
        "migrationPlanDigest",
        "certificationDigests",
        "exactGateEvidence",
        "evidence",
    }
    if require_manifest_signature:
        required_manifest_keys.add("signature")
    _exact_keys(
        manifest,
        required=required_manifest_keys,
        field="release manifest",
    )
    expected_state = (
        "signed-release" if require_manifest_signature else "unsigned-local-binder"
    )
    if manifest.get("manifestState") != expected_state:
        raise CompatibilityError(
            "release manifest state does not match its signature phase"
        )
    _exact_keys(
        matrix,
        required={
            "apiVersion",
            "kind",
            "matrixVersion",
            "runtime",
            "protocol",
            "components",
            "releaseTrain",
        },
        field="compatibility matrix",
    )
    if matrix.get("matrixVersion") != 2:
        raise CompatibilityError("unsupported compatibility matrix version")
    protocol = matrix.get("protocol")
    if not isinstance(protocol, dict):
        raise CompatibilityError("compatibility protocol must be a mapping")
    _exact_keys(
        protocol,
        required={"name", "version", "schemas"},
        field="compatibility protocol",
    )
    release_train = matrix.get("releaseTrain")
    if not isinstance(release_train, dict):
        raise CompatibilityError("release train must be a mapping")
    _exact_keys(
        release_train,
        required={
            "assemblyOrder",
            "activationMode",
            "stateMigrationMode",
            "rollbackRequires",
        },
        field="release train",
    )
    release_id = str(manifest.get("releaseId") or "")
    if not re.fullmatch(r"release-[a-z0-9][a-z0-9.-]{2,63}", release_id):
        raise CompatibilityError("releaseId must be an opaque release identifier")
    if matrix_path is not None and manifest.get("matrixDigest") != file_digest(
        matrix_path
    ):
        raise CompatibilityError(
            "release matrix digest does not match the approved matrix"
        )
    _digest(manifest.get("matrixDigest"), "matrixDigest")
    _digest(
        manifest.get("sourceFreezeEvidenceDigest"),
        "sourceFreezeEvidenceDigest",
    )
    _digest(manifest.get("configurationDigest"), "configurationDigest")
    _digest(manifest.get("migrationPlanDigest"), "migrationPlanDigest")
    if manifest_path is None:
        raise CompatibilityError("release evidence requires the manifest location")
    release_evidence = manifest.get("evidence")
    if not isinstance(release_evidence, dict):
        raise CompatibilityError("release evidence catalog is required")
    _exact_keys(
        release_evidence,
        required={
            "sourceFreezeEvidence",
            "configuration",
            "migrationPlan",
            "certifications",
        },
        field="release evidence catalog",
    )
    source_freeze_raw = _evidence_bytes(
        manifest_path,
        release_evidence["sourceFreezeEvidence"],
        "sourceFreezeEvidence",
        maximum=_MAX_COMPONENT_SOURCE_BYTES,
    )
    if manifest["sourceFreezeEvidenceDigest"] != (
        "sha256:" + hashlib.sha256(source_freeze_raw).hexdigest()
    ):
        raise CompatibilityError(
            "source-freeze digest differs from referenced evidence"
        )
    source_freeze_authority = validate_source_freeze_evidence(source_freeze_raw)
    configuration_raw = _evidence_bytes(
        manifest_path, release_evidence["configuration"], "configuration"
    )
    migration_raw = _evidence_bytes(
        manifest_path, release_evidence["migrationPlan"], "migrationPlan"
    )
    if manifest["configurationDigest"] != (
        "sha256:" + hashlib.sha256(configuration_raw).hexdigest()
    ):
        raise CompatibilityError(
            "configuration digest differs from referenced evidence"
        )
    if manifest["migrationPlanDigest"] != (
        "sha256:" + hashlib.sha256(migration_raw).hexdigest()
    ):
        raise CompatibilityError(
            "migration plan digest differs from referenced evidence"
        )
    configuration_document = _json_evidence(
        configuration_raw, "release configuration"
    )
    migration_document = _json_evidence(migration_raw, "release migration plan")
    validate_release_configuration(
        configuration_document,
        release_id=release_id,
        matrix=matrix,
        matrix_digest=str(manifest["matrixDigest"]),
    )
    certification_digests = manifest.get("certificationDigests")
    if (
        not isinstance(certification_digests, dict)
        or set(certification_digests) != _CERTIFICATION_DIGESTS
    ):
        raise CompatibilityError("release certification digest catalog is not exact")
    for name, digest in certification_digests.items():
        _digest(digest, f"certificationDigests.{name}")
    certification_evidence = release_evidence.get("certifications")
    if (
        not isinstance(certification_evidence, dict)
        or set(certification_evidence) != _CERTIFICATION_DIGESTS
    ):
        raise CompatibilityError("release certification evidence catalog is not exact")
    certification_payloads: dict[str, bytes] = {}
    for name, reference in certification_evidence.items():
        payload = _evidence_bytes(manifest_path, reference, f"certification.{name}")
        if certification_digests[name] != (
            "sha256:" + hashlib.sha256(payload).hexdigest()
        ):
            raise CompatibilityError(f"certification digest differs for {name}")
        certification_payloads[name] = payload
    if require_manifest_signature:
        validate_connector_ledger(
            certification_payloads["connectorLiveCertificationLedger"]
        )
    closure = validate_exact_artifact_closure(
        certification_payloads["exactArtifactClosureEvidence"],
        release_id=release_id,
    )
    oci_vulnerability_scan = validate_oci_vulnerability_scan_evidence(
        certification_payloads["ociVulnerabilityScanEvidence"],
        release_id=release_id,
    )
    closure_release = closure["release"]
    raw_components = manifest.get("components")
    skill_component = (
        raw_components.get("prebundled-skills")
        if isinstance(raw_components, dict)
        else None
    )
    if not isinstance(skill_component, dict):
        raise CompatibilityError("prebundled skill component is unavailable")
    skill_validation = validate_prebundled_skill_matrix(
        certification_payloads["prebundledSkillValidationMatrix"],
        release_id=release_id,
        release_specification_digest=closure_release["releaseSpecSha256"],
        promotion_evidence_digest=closure_release["promotionEvidenceSha256"],
        graph_os_digest=closure_release["graphosSha256"],
        engine_digest=closure_release["engineSha256"],
        configuration_digest=str(manifest["configurationDigest"]),
        skill_catalog_digest=str(skill_component.get("digest") or ""),
    )
    skill_deployment = validate_skill_validation_deployment(
        certification_payloads["skillValidationDeployment"],
        release_id=release_id,
        configuration_digest=str(manifest["configurationDigest"]),
        release_binding=closure_release,
        validation_evidence=skill_validation,
    )
    skill_lifecycle = validate_skill_validation_lifecycle(
        certification_payloads["skillValidationLifecycleEvidence"],
        release_id=release_id,
        configuration_digest=str(manifest["configurationDigest"]),
        validation_evidence_digest=certification_digests[
            "prebundledSkillValidationMatrix"
        ],
        release_binding=closure_release,
        validation_evidence=skill_validation,
        deployment=skill_deployment,
    )
    if verify_signatures:
        _verify_exact_artifact_closure(closure)
        _verify_skill_validation_evidence(
            skill_validation,
            deployment=skill_deployment,
            field="prebundledSkillValidationMatrix",
        )
        _verify_skill_validation_evidence(
            skill_lifecycle,
            deployment=skill_deployment,
            field="skillValidationLifecycleEvidence",
        )
        _verify_oci_vulnerability_scan_evidence(oci_vulnerability_scan)
    expected_schemas = matrix.get("protocol", {}).get("schemas")
    if manifest.get("protocolSchemas") != expected_schemas:
        raise CompatibilityError(
            "protocol schema versions do not match the matrix exactly"
        )
    expected_names = _component_names(matrix)
    components = manifest.get("components")
    if not isinstance(components, dict) or set(components) != set(expected_names):
        raise CompatibilityError(
            "release component set does not match the matrix exactly"
        )
    _validate_distinct_oci_subjects(components)
    versions: dict[str, Version] = {}
    component_source_evidence: dict[str, dict[str, Any]] = {}
    for name in expected_names:
        component = components[name]
        if not isinstance(component, dict):
            raise CompatibilityError(f"component {name} must be a mapping")
        _exact_keys(
            component,
            required={
                "version",
                "kind",
                "artifact",
                "digest",
                "sourceDigest",
                "sbomDigest",
                "provenanceDigest",
                "signature",
                "capabilities",
                "evidence",
            },
            optional={"entryCount"},
            field=f"component {name}",
        )
        version_text = str(component.get("version") or "")
        try:
            version = Version(version_text)
        except InvalidVersion as exc:
            raise CompatibilityError(
                f"component {name} has an invalid version"
            ) from exc
        expected = matrix["components"][name]
        expected_version = _exact_version(expected["version"], f"{name}.version")
        if version_text != expected_version or str(version) != version_text:
            raise CompatibilityError(
                f"component {name} version is not the current matrix version"
            )
        if component.get("kind") != expected.get("artifactKind"):
            raise CompatibilityError(f"component {name} artifact kind does not match")
        versions[name] = version
        for field in ("digest", "sourceDigest", "sbomDigest", "provenanceDigest"):
            _digest(component.get(field), f"{name}.{field}")
        artifact = str(component.get("artifact") or "")
        if not artifact or "latest" in artifact.casefold():
            raise CompatibilityError(
                f"component {name} artifact must be exact and not latest"
            )
        if component.get("kind") == "oci":
            expected_artifact = f"oci:{name}@{component['digest']}"
            if artifact != expected_artifact:
                raise CompatibilityError(
                    f"component {name} OCI subject is not pinned to its declared digest"
                )
        elif not re.fullmatch(
            r"catalog:[a-z0-9][a-z0-9.-]{1,127}@sha256:[a-f0-9]{64}", artifact
        ) or not artifact.endswith("@" + str(component["digest"])):
            raise CompatibilityError(
                f"component {name} catalog is not an opaque digest-pinned reference"
            )
        capability_values = component.get("capabilities")
        if (
            not isinstance(capability_values, list)
            or len(capability_values) != len(set(capability_values))
            or not all(isinstance(value, str) and value for value in capability_values)
        ):
            raise CompatibilityError(f"component {name} capabilities are invalid")
        capabilities = set(capability_values)
        missing = set(expected.get("requiredCapabilities") or ()) - capabilities
        if missing:
            raise CompatibilityError(
                f"component {name} lacks required capabilities: {sorted(missing)}"
            )
        exact = expected.get("exactEntries")
        if exact is not None and int(component.get("entryCount") or 0) != int(exact):
            raise CompatibilityError(
                f"component {name} entry count does not match the matrix"
            )
        inspected_evidence = _inspect_component_evidence(name, component, manifest_path)
        source_document = inspected_evidence["sourceEvidence"]
        if not isinstance(source_document, dict):
            raise CompatibilityError(f"{name}.source evidence is unavailable")
        component_source_evidence[name] = source_document
        signature_bundle = inspected_evidence["verificationRequest"]
        if verify_signatures:
            _verify_signature(name, component, signature_bundle)
    index_migrations = components["index-migrations"]
    validate_release_migration_plan(
        migration_document,
        release_id=release_id,
        matrix=matrix,
        matrix_digest=str(manifest["matrixDigest"]),
        index_migration_catalog_digest=str(index_migrations["digest"]),
    )
    if migration_document["indexMigrationCount"] != index_migrations.get(
        "entryCount"
    ):
        raise CompatibilityError(
            "release migration plan entry count differs from the index catalog"
        )
    _validate_single_source_freeze(
        component_source_evidence,
        source_freeze_authority,
    )
    scan_subjects = oci_vulnerability_scan.get("subjects")
    if not isinstance(scan_subjects, dict):
        raise CompatibilityError("OCI vulnerability scan subjects are unavailable")
    for name in _OCI_COMPONENTS:
        subject = scan_subjects.get(name)
        source_document = component_source_evidence.get(name)
        component = components[name]
        if (
            not isinstance(subject, dict)
            or not isinstance(source_document, dict)
            or subject.get("artifactDigest") != component.get("digest")
            or subject.get("artifactDigest") != source_document.get("artifactDigest")
            or subject.get("archiveDigest")
            != source_document.get("artifactInputDigest")
        ):
            raise CompatibilityError(
                f"OCI vulnerability scan binding differs for {name}"
            )
    if manifest.get("exactGateEvidence") != exact_gate_evidence(
        components,
        certification_digests,
    ):
        raise CompatibilityError("exact-gate evidence mapping is not authoritative")
    for name, expected in matrix["components"].items():
        if not isinstance(expected, dict):
            raise CompatibilityError(f"matrix component {name} must be a mapping")
        _exact_keys(
            expected,
            required={"version", "artifactKind"},
            optional={
                "dependsOn",
                "requiredCapabilities",
                "exactEntries",
                "canonicalization",
                "migrationMode",
            },
            field=f"matrix component {name}",
        )
        for dependency, specifier in (expected.get("dependsOn") or {}).items():
            if versions[dependency] not in SpecifierSet(str(specifier)):
                raise CompatibilityError(
                    f"{name} dependency {dependency} is incompatible"
                )
    if require_manifest_signature:
        _validate_manifest_signature(manifest)
        if verify_signatures:
            _verify_manifest_signature(manifest)
    return {
        "ok": True,
        "releaseId": release_id,
        "releaseDigest": canonical_digest(manifest),
        "componentDigests": {
            name: components[name]["digest"] for name in expected_names
        },
        "certificationDigests": dict(certification_digests),
        "signaturesVerified": verify_signatures and require_manifest_signature,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="check-graphos-compatibility")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument(
        "--matrix",
        type=Path,
        default=Path("deploy/release/compatibility-matrix.yml"),
    )
    parser.add_argument(
        "--matrix-only",
        action="store_true",
        help="Validate the exact compatibility/release-train source contract only.",
    )
    parser.add_argument(
        "--structure-only",
        action="store_true",
        help="Validate structure/compatibility without invoking external signature verifiers.",
    )
    args = parser.parse_args(argv)
    try:
        matrix = _load(args.matrix)
        if args.matrix_only:
            report = {
                **validate_compatibility_matrix(matrix),
                "matrixDigest": file_digest(args.matrix),
            }
        else:
            if args.manifest is None:
                raise CompatibilityError(
                    "--manifest is required unless --matrix-only is used"
                )
            manifest = _load(args.manifest)
            report = verify_release_manifest(
                manifest,
                matrix,
                matrix_path=args.matrix,
                manifest_path=args.manifest,
                verify_signatures=not args.structure_only,
            )
    except Exception as exc:  # noqa: BLE001 - one fail-closed CLI boundary
        print(json.dumps({"ok": False, "error": type(exc).__name__}, sort_keys=True))
        return 1
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
