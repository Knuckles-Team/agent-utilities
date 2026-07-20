"""Governed connector lifecycle certification.

The connector capability bundle is the test input; upstream records are never
copied into certification evidence.  ``offline-fixture`` mode exercises the
contract against a bounded in-memory reference driver.  ``external-live`` mode
uses an operator-supplied runtime driver, resolved only through secret
references, to start the real connector and apply the same synthetic fixtures
to an isolated engine scope.

The resulting record contains aggregate counts, booleans, and cryptographic
digests only.  It never contains endpoints, credentials, runtime references,
host/user names, local paths, fixture content, or source identities.  A live
success is impossible unless every external operation actually completed.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import re
import stat
import subprocess
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, Protocol

import yaml

from ...models.company_brain import DataClassification
from ...protocols.source_connectors.base import ExternalAccess
from ...protocols.source_connectors.tool_schema import validate_live_tool_contract
from ...security.persistence_privacy import PersistencePrivacyGuard
from ..ingestion.change_envelope import ChangeEnvelope
from ..ontology import ontology_integrity
from ..ontology.connector_manifest import ConnectorManifest, SyncSpec
from ..ontology.connector_manifest_gate import check_manifest_bytes
from .connector_source_attestation import source_attestation_violations

__all__ = [
    "CertificationBundle",
    "CertificationDriver",
    "CertificationLimits",
    "CertificationPolicy",
    "LiveCertificationProfile",
    "ReferenceCertificationDriver",
    "RuntimeCommandCertificationDriver",
    "certify_connector",
    "load_live_profile",
    "load_certification_record",
    "verify_certification_record",
    "write_certification_record",
]

_RUNTIME_REF = re.compile(r"^(?:vault|secret|env)://[A-Za-z0-9_./#-]+$")
_SAFE_CONNECTOR = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")
_SAFE_ALIAS = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$")
_HEX_DIGEST = re.compile(r"^[a-f0-9]{64}$")
_MAX_ARTIFACT_BYTES = 4 * 1024 * 1024
_MAX_JSON_NODES = 32_768
_BASE_CHILD_ENV = frozenset(
    {
        "APPDATA",
        "COMSPEC",
        "HOME",
        "LANG",
        "LC_ALL",
        "LOCALAPPDATA",
        "PATH",
        "PATHEXT",
        "REQUESTS_CA_BUNDLE",
        "SSL_CERT_DIR",
        "SSL_CERT_FILE",
        "SYSTEMROOT",
        "TEMP",
        "TMP",
        "USERPROFILE",
        "UV_NATIVE_TLS",
        "WINDIR",
        "XDG_CACHE_HOME",
        "XDG_CONFIG_HOME",
        "XDG_DATA_HOME",
        "XDG_RUNTIME_DIR",
        "XDG_STATE_HOME",
    }
)

REQUIRED_CHECKS = (
    "bundle_integrity",
    "live_tool_schema",
    "fixture_ingest",
    "update",
    "delete",
    "replay_idempotency",
    "governance_preservation",
    "semantic_validation",
    "count_reconciliation",
    "cleanup",
)


class CertificationError(RuntimeError):
    """The connector could not satisfy the certification contract."""


@dataclass(frozen=True, slots=True)
class CertificationLimits:
    """Resource ceilings for one sequential connector certification."""

    max_records: int = 64
    max_payload_bytes: int = 256 * 1024
    max_response_bytes: int = 1024 * 1024
    timeout_seconds: float = 120.0

    def __post_init__(self) -> None:
        if not 1 <= self.max_records <= 256:
            raise ValueError("certification max_records must be in [1, 256]")
        if not 1024 <= self.max_payload_bytes <= 1024 * 1024:
            raise ValueError("certification payload limit is invalid")
        if not 4096 <= self.max_response_bytes <= 4 * 1024 * 1024:
            raise ValueError("certification response limit is invalid")
        if not 1.0 <= self.timeout_seconds <= 600.0:
            raise ValueError("certification timeout is invalid")


@dataclass(frozen=True, slots=True)
class CertificationPolicy:
    """Runtime-only governance applied to synthetic certification objects."""

    tenant: str = "connector-certification"
    retention: str = "certification-ephemeral"
    legal_hold: bool = False

    def __post_init__(self) -> None:
        for label, value in (("tenant", self.tenant), ("retention", self.retention)):
            if not _SAFE_ALIAS.fullmatch(str(value or "")):
                raise ValueError(f"certification {label} alias is invalid")
        if self.legal_hold:
            raise ValueError(
                "synthetic certification data cannot be placed on legal hold"
            )


@dataclass(frozen=True, slots=True)
class _Fixture:
    preset: str
    record: dict[str, Any]
    expected: dict[str, Any]


@dataclass(frozen=True, slots=True)
class CertificationBundle:
    """A verified connector-owned capability bundle, without host-local labels."""

    manifest: ConnectorManifest
    fixtures: tuple[_Fixture, ...]
    shapes_text: str
    manifest_sha256: str
    fixtures_sha256: str
    shapes_sha256: str

    @classmethod
    def load(cls, connector_root: Path) -> CertificationBundle:
        root = connector_root.resolve()
        manifest_path = root / "connector_manifest.yml"
        manifest_bytes = _read_regular(manifest_path, _MAX_ARTIFACT_BYTES)
        try:
            manifest_data = yaml.safe_load(manifest_bytes.decode("utf-8"))
            manifest = ConnectorManifest.model_validate(manifest_data)
        except Exception as exc:
            raise CertificationError("connector manifest is invalid") from exc
        if (
            not _SAFE_CONNECTOR.fullmatch(manifest.connector)
            or root.name != manifest.connector
        ):
            raise CertificationError("connector bundle identity is invalid")
        violations = check_manifest_bytes(manifest_path, require_signature=True)
        if violations:
            raise CertificationError("connector manifest integrity verification failed")

        fixture_matches = sorted(root.glob("*/ontology/fixtures/records.json"))
        shape_matches = sorted(root.glob("*/ontology/shapes/connector.shacl.ttl"))
        cert_matches = sorted(root.glob("*/ontology/certification.json"))
        if not (len(fixture_matches) == len(shape_matches) == len(cert_matches) == 1):
            raise CertificationError("connector capability bundle is incomplete")
        fixture_path, shapes_path, certification_path = (
            fixture_matches[0],
            shape_matches[0],
            cert_matches[0],
        )
        fixture_bytes = _read_regular(fixture_path, _MAX_ARTIFACT_BYTES)
        shapes_bytes = _read_regular(shapes_path, _MAX_ARTIFACT_BYTES)
        certification_bytes = _read_regular(certification_path, _MAX_ARTIFACT_BYTES)
        try:
            fixture_doc = _bounded_json(fixture_bytes, max_bytes=_MAX_ARTIFACT_BYTES)
            certification = _bounded_json(
                certification_bytes, max_bytes=_MAX_ARTIFACT_BYTES
            )
        except (ValueError, UnicodeDecodeError) as exc:
            raise CertificationError("connector capability JSON is invalid") from exc
        if not isinstance(fixture_doc, dict) or not isinstance(certification, dict):
            raise CertificationError("connector capability JSON must be an object")
        if fixture_doc.get("connector") != manifest.connector:
            raise CertificationError("connector fixture identity differs from manifest")
        if source_attestation_violations(certification, manifest):
            raise CertificationError("connector source attestation is invalid")

        public_key = str(manifest.provenance.signing_public_key or "")
        digest = ontology_integrity.canonical_signed_document_hash(certification)
        if not ontology_integrity.verify_release_signature(
            digest,
            certification.get("signature"),
            signer_id=certification.get("signer"),
            algorithm=certification.get("signature_algorithm"),
            public_key=certification.get("signing_public_key"),
            trusted_public_keys=(public_key,) if public_key else (),
        ):
            raise CertificationError(
                "connector bundle certification signature is invalid"
            )

        ledger = certification.get("artifacts")
        if not isinstance(ledger, dict):
            raise CertificationError("connector bundle artifact ledger is absent")
        expected_artifacts = {
            manifest_path.relative_to(root).as_posix(): _sha256(manifest_bytes),
            fixture_path.relative_to(root).as_posix(): _sha256(fixture_bytes),
            shapes_path.relative_to(root).as_posix(): _sha256(shapes_bytes),
        }
        if any(ledger.get(name) != value for name, value in expected_artifacts.items()):
            raise CertificationError(
                "connector bundle differs from its artifact ledger"
            )

        rows = fixture_doc.get("fixtures")
        if not isinstance(rows, list) or len(rows) > 256:
            raise CertificationError("connector fixture collection is invalid")
        fixtures: list[_Fixture] = []
        for row in rows:
            if not isinstance(row, dict):
                raise CertificationError("connector fixture entry is invalid")
            preset = str(row.get("preset") or "")
            record = row.get("record")
            expected = row.get("expected")
            if (
                not preset
                or not isinstance(record, dict)
                or not isinstance(expected, dict)
            ):
                raise CertificationError("connector fixture contract is incomplete")
            fixtures.append(_Fixture(preset, dict(record), dict(expected)))
        signed_presets = {sync.preset for sync in manifest.sync}
        if signed_presets and {item.preset for item in fixtures} != signed_presets:
            raise CertificationError("fixtures do not cover every signed source preset")
        try:
            shapes_text = shapes_bytes.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise CertificationError("connector SHACL artifact is not UTF-8") from exc
        return cls(
            manifest=manifest,
            fixtures=tuple(fixtures),
            shapes_text=shapes_text,
            manifest_sha256=_sha256(manifest_bytes),
            fixtures_sha256=_sha256(fixture_bytes),
            shapes_sha256=_sha256(shapes_bytes),
        )


class CertificationDriver(Protocol):
    """Minimal live/runtime adapter; every response must be JSON-compatible."""

    async def invoke(self, request: Mapping[str, Any]) -> dict[str, Any]: ...


@dataclass(slots=True)
class ReferenceCertificationDriver:
    """Bounded, non-persistent fixture driver used only by offline mode/tests."""

    tool_schemas: dict[str, dict[str, Any]] = field(default_factory=dict)
    _seen: set[str] = field(default_factory=set)
    _records: dict[tuple[str, str, str, str], dict[str, Any]] = field(
        default_factory=dict
    )
    _tombstones: dict[tuple[str, str, str, str], dict[str, Any]] = field(
        default_factory=dict
    )

    async def invoke(self, request: Mapping[str, Any]) -> dict[str, Any]:
        action = request.get("action")
        if action == "list_tools":
            return {
                "tools": [
                    {"name": name, "inputSchema": schema}
                    for name, schema in sorted(self.tool_schemas.items())
                ]
            }
        if action == "apply":
            envelope = request.get("envelope")
            if not isinstance(envelope, dict):
                raise CertificationError("driver apply request has no envelope")
            dedup = str(envelope.get("idempotency_key") or "")
            if dedup in self._seen:
                return {"applied": False, "replayed": True}
            self._seen.add(dedup)
            key = _scope_key(envelope)
            governance = _governance(envelope)
            if envelope.get("operation") == "delete":
                self._records.pop(key, None)
                self._tombstones[key] = governance
            elif envelope.get("operation") == "upsert":
                self._records[key] = {
                    "governance": governance,
                    "payload": envelope.get("typed_payload"),
                }
                self._tombstones.pop(key, None)
            else:
                raise CertificationError("reference driver operation is unsupported")
            return {"applied": True, "replayed": False}
        if action == "count":
            scope = request.get("scope")
            if not isinstance(scope, dict):
                raise CertificationError("driver count request has no scope")
            prefix = (
                str(scope.get("tenant") or ""),
                str(scope.get("connector") or ""),
                str(scope.get("source_instance") or ""),
            )
            return {"count": sum(key[:3] == prefix for key in self._records)}
        if action == "inspect":
            scope = request.get("scope")
            if not isinstance(scope, dict):
                raise CertificationError("driver inspect request has no scope")
            key = (
                str(scope.get("tenant") or ""),
                str(scope.get("connector") or ""),
                str(scope.get("source_instance") or ""),
                str(scope.get("source_object_id") or ""),
            )
            record = self._records.get(key)
            return {
                "exists": record is not None,
                "governance": record.get("governance") if record else None,
                "tombstone_governance": self._tombstones.get(key),
            }
        raise CertificationError("certification driver action is unsupported")


@dataclass(frozen=True, slots=True)
class LiveCertificationProfile:
    """External live configuration; durable values remain secret references."""

    driver_command_ref: str
    connector_runtime_ref: str
    engine_runtime_ref: str
    tenant: str
    retention: str
    tls_profile_ref: str | None = None

    def __post_init__(self) -> None:
        for label, value in (
            ("driver_command_ref", self.driver_command_ref),
            ("connector_runtime_ref", self.connector_runtime_ref),
            ("engine_runtime_ref", self.engine_runtime_ref),
        ):
            if not _RUNTIME_REF.fullmatch(value):
                raise ValueError(
                    f"live certification {label} must be a runtime reference"
                )
        if self.tls_profile_ref and not _RUNTIME_REF.fullmatch(self.tls_profile_ref):
            raise ValueError(
                "live certification TLS profile must be a runtime reference"
            )
        CertificationPolicy(tenant=self.tenant, retention=self.retention)

    def runtime_descriptor(self) -> dict[str, Any]:
        return {
            "connector_runtime_ref": self.connector_runtime_ref,
            "engine_runtime_ref": self.engine_runtime_ref,
            "tls_profile_ref": self.tls_profile_ref,
        }


def load_live_profile(reference: str) -> LiveCertificationProfile:
    """Resolve one externalized live profile without retaining its reference."""

    if not _RUNTIME_REF.fullmatch(str(reference or "")):
        raise ValueError("live certification profile must be a runtime reference")
    from ...security.secrets_client import create_secrets_client

    try:
        raw = create_secrets_client().resolve_ref(reference)
        parsed = _bounded_json(str(raw or "").encode("utf-8"), max_bytes=64 * 1024)
    except Exception as exc:
        raise CertificationError("live certification profile is unavailable") from exc
    required = {
        "driver_command_ref",
        "connector_runtime_ref",
        "engine_runtime_ref",
        "tenant",
        "retention",
    }
    optional = {"tls_profile_ref"}
    if (
        not isinstance(parsed, dict)
        or set(parsed) - required - optional
        or required - set(parsed)
    ):
        raise CertificationError("live certification profile fields are not exact")
    return LiveCertificationProfile(
        driver_command_ref=str(parsed["driver_command_ref"]),
        connector_runtime_ref=str(parsed["connector_runtime_ref"]),
        engine_runtime_ref=str(parsed["engine_runtime_ref"]),
        tenant=str(parsed["tenant"]),
        retention=str(parsed["retention"]),
        tls_profile_ref=str(parsed.get("tls_profile_ref") or "") or None,
    )


@dataclass(slots=True)
class RuntimeCommandCertificationDriver:
    """JSON-stdin driver resolved from an operator-owned secret reference.

    The child receives only a small runtime environment and environment secrets
    explicitly named by ``env://`` references in the command/profile.  Stdout is
    bounded and stderr is discarded so source content and credentials cannot
    enter a certification record or terminal log.
    """

    profile: LiveCertificationProfile
    limits: CertificationLimits = field(default_factory=CertificationLimits)

    async def invoke(self, request: Mapping[str, Any]) -> dict[str, Any]:
        command = self._command()
        payload = dict(request)
        payload["runtime"] = self.profile.runtime_descriptor()
        raw = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
        if len(raw) > self.limits.max_payload_bytes:
            raise CertificationError(
                "certification driver request exceeds its boundary"
            )
        child_env = _delegated_environment(command, payload)
        process = await asyncio.wait_for(
            asyncio.create_subprocess_exec(
                *command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                env=child_env,
            ),
            timeout=self.limits.timeout_seconds,
        )
        assert process.stdin is not None and process.stdout is not None

        async def exchange() -> tuple[bytes, int]:
            process.stdin.write(raw)
            await process.stdin.drain()
            process.stdin.close()
            chunks: list[bytes] = []
            total = 0
            while True:
                chunk = await process.stdout.read(
                    min(65_536, self.limits.max_response_bytes + 1 - total)
                )
                if not chunk:
                    break
                chunks.append(chunk)
                total += len(chunk)
                if total > self.limits.max_response_bytes:
                    process.kill()
                    await process.wait()
                    raise CertificationError(
                        "certification driver response exceeds its boundary"
                    )
            return b"".join(chunks), await process.wait()

        try:
            response, code = await asyncio.wait_for(
                exchange(), timeout=self.limits.timeout_seconds
            )
        except TimeoutError as exc:
            process.kill()
            await process.wait()
            raise CertificationError("certification driver timed out") from exc
        except Exception:
            if process.returncode is None:
                process.kill()
                await process.wait()
            raise
        if code != 0:
            raise CertificationError("certification driver rejected the operation")
        try:
            parsed = _bounded_json(response, max_bytes=self.limits.max_response_bytes)
        except Exception as exc:
            raise CertificationError(
                "certification driver returned invalid JSON"
            ) from exc
        if not isinstance(parsed, dict):
            raise CertificationError("certification driver response must be an object")
        return parsed

    def _command(self) -> tuple[str, ...]:
        from ...security.secrets_client import create_secrets_client

        try:
            raw = create_secrets_client().resolve_ref(self.profile.driver_command_ref)
            value = _bounded_json(str(raw or "").encode("utf-8"), max_bytes=32 * 1024)
        except Exception as exc:
            raise CertificationError(
                "certification driver command is unavailable"
            ) from exc
        if (
            not isinstance(value, list)
            or not 1 <= len(value) <= 64
            or not all(
                isinstance(part, str)
                and 1 <= len(part) <= 4096
                and "\x00" not in part
                and "\r" not in part
                and "\n" not in part
                for part in value
            )
        ):
            raise CertificationError("certification driver command is invalid")
        return tuple(value)


async def certify_connector(
    bundle: CertificationBundle,
    *,
    mode: str,
    signer: ontology_integrity.ReleaseSigner,
    driver: CertificationDriver | None = None,
    policy: CertificationPolicy | None = None,
    limits: CertificationLimits | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Exercise one bundle and return a signed, content-free certification record."""

    if mode not in {"offline-fixture", "external-live"}:
        raise ValueError("certification mode is invalid")
    limits = limits or CertificationLimits()
    policy = policy or CertificationPolicy()
    if mode == "offline-fixture":
        if driver is not None and not isinstance(driver, ReferenceCertificationDriver):
            raise ValueError("offline certification accepts only the reference driver")
        driver = driver or ReferenceCertificationDriver()
    elif driver is None or isinstance(driver, ReferenceCertificationDriver):
        raise ValueError(
            "external-live certification requires an external runtime driver"
        )

    started = (now or datetime.now(UTC)).astimezone(UTC)
    checks = {name: "not-run" for name in REQUIRED_CHECKS}
    checks["bundle_integrity"] = "passed"
    counts: dict[str, int] = {}
    evidence: dict[str, str] = {}
    semantic_validator = "not-run"
    failure_class: str | None = None
    fixtures = bundle.fixtures[: limits.max_records]
    if bundle.manifest.sync and len(fixtures) < len(bundle.manifest.sync):
        failure_class = "FixtureBoundaryError"

    run_key = uuid.uuid4().hex
    source_instance = f"certification-{run_key}"
    scope = {
        "tenant": policy.tenant,
        "connector": bundle.manifest.connector,
        "source_instance": source_instance,
    }
    envelopes: list[ChangeEnvelope] = []
    try:
        if failure_class:
            raise CertificationError("fixture boundary omits a signed preset")
        if not bundle.manifest.sync:
            raise CertificationError("connector declares no certifiable source presets")
        if mode == "external-live":
            result = await driver.invoke(
                {
                    "api_version": "graphos.connector-certification/v1",
                    "action": "list_tools",
                    "connector": bundle.manifest.connector,
                    "required_tools": sorted(
                        {sync.tool for sync in bundle.manifest.sync}
                    ),
                }
            )
            _verify_live_tools(result, bundle.manifest.sync)
            checks["live_tool_schema"] = "passed"
            evidence["live_tool_schema"] = _evidence_digest(
                {"tools": sorted({sync.tool for sync in bundle.manifest.sync})}
            )

        counts["initial"] = await _count(driver, scope)
        if counts["initial"] != 0:
            raise CertificationError("isolated certification scope is not empty")
        for index, fixture in enumerate(fixtures):
            envelope = _fixture_envelope(
                bundle,
                fixture,
                index=index,
                run_key=run_key,
                source_instance=source_instance,
                policy=policy,
                version="1",
            )
            envelopes.append(envelope)
            await _apply(driver, envelope, expect_replay=False)
            await _assert_governance(driver, envelope)
        counts["after_ingest"] = await _count(driver, scope)
        if counts["after_ingest"] != len(envelopes):
            raise CertificationError("fixture ingest count does not reconcile")
        checks["fixture_ingest"] = "passed"

        for envelope in envelopes:
            await _apply(driver, envelope, expect_replay=True)
        counts["after_replay"] = await _count(driver, scope)
        if counts["after_replay"] != len(envelopes):
            raise CertificationError("replay changed the live record count")
        checks["replay_idempotency"] = "passed"

        first = envelopes[0]
        updated = _next_envelope(first, version="2", operation="upsert")
        await _apply(driver, updated, expect_replay=False)
        await _assert_governance(driver, updated)
        counts["after_update"] = await _count(driver, scope)
        if counts["after_update"] != len(envelopes):
            raise CertificationError("update changed the live record count")
        checks["update"] = "passed"

        deleted = _next_envelope(updated, version="3", operation="delete")
        await _apply(driver, deleted, expect_replay=False)
        counts["after_delete"] = await _count(driver, scope)
        if counts["after_delete"] != len(envelopes) - 1:
            raise CertificationError("delete count does not reconcile")
        await _assert_tombstone_governance(driver, deleted)
        await _apply(driver, deleted, expect_replay=True)
        counts["after_delete_replay"] = await _count(driver, scope)
        if counts["after_delete_replay"] != len(envelopes) - 1:
            raise CertificationError("delete replay changed the live record count")
        checks["delete"] = "passed"
        checks["governance_preservation"] = "passed"

        semantic_validator = _semantic_validation(
            bundle, envelopes, require_pyshacl=mode == "external-live"
        )
        checks["semantic_validation"] = "passed"
        checks["count_reconciliation"] = "passed"

        for envelope in envelopes[1:]:
            cleanup = _next_envelope(envelope, version="cleanup", operation="delete")
            await _apply(driver, cleanup, expect_replay=False)
        counts["after_cleanup"] = await _count(driver, scope)
        if counts["after_cleanup"] != 0:
            raise CertificationError("certification cleanup count does not reconcile")
        checks["cleanup"] = "passed"
    except Exception as exc:  # signed aggregate failure; no source/error text retained
        failure_class = type(exc).__name__
        try:
            for envelope in envelopes:
                cleanup = _next_envelope(
                    envelope, version=f"cleanup-{uuid.uuid4().hex}", operation="delete"
                )
                await driver.invoke(
                    {
                        "api_version": "graphos.connector-certification/v1",
                        "action": "apply",
                        "envelope": cleanup.as_dict(),
                    }
                )
            if await _count(driver, scope) == 0:
                checks["cleanup"] = "passed"
        except Exception:
            checks["cleanup"] = "failed"

    if mode == "external-live" and checks["live_tool_schema"] == "not-run":
        checks["live_tool_schema"] = "failed"
    for name, status in tuple(checks.items()):
        if status == "not-run" and name != "live_tool_schema":
            checks[name] = "failed"
    if mode == "offline-fixture" and failure_class is None:
        checks["live_tool_schema"] = "not-run"
        status = "offline-validated"
        live_certified = False
    elif failure_class is None and all(value == "passed" for value in checks.values()):
        status = "certified"
        live_certified = True
    else:
        status = "failed"
        live_certified = False
    for name in REQUIRED_CHECKS:
        evidence.setdefault(
            name,
            _evidence_digest({"check": name, "status": checks[name], "mode": mode}),
        )

    record: dict[str, Any] = {
        "api_version": "graphos.io/v1",
        "kind": "ConnectorLiveCertification",
        "schema_version": "1",
        "connector": bundle.manifest.connector,
        "certified_at": started.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "mode": mode,
        "status": status,
        "live_certified": live_certified,
        "bundle": {
            "manifest_sha256": bundle.manifest_sha256,
            "fixtures_sha256": bundle.fixtures_sha256,
            "shapes_sha256": bundle.shapes_sha256,
            "schema_version": bundle.manifest.schema_version,
        },
        "scope": {
            "sync_presets": len(bundle.manifest.sync),
            "fixtures_declared": len(bundle.fixtures),
            "fixtures_exercised": len(fixtures),
            "tenant_bound": bool(policy.tenant),
            "retention_bound": bool(policy.retention),
        },
        "checks": checks,
        "counts": counts,
        "semantic_validator": semantic_validator,
        "evidence": dict(sorted(evidence.items())),
        "failure_class": failure_class,
        "runtime_configuration": "externalized" if mode == "external-live" else "none",
        "signer": signer.signer_id,
        "signature_algorithm": signer.algorithm,
        "signing_public_key": signer.public_key,
        "signature": None,
    }
    record["signature"] = signer.sign(
        ontology_integrity.canonical_signed_document_hash(record)
    )
    return record


def verify_certification_record(
    record: Mapping[str, Any],
    *,
    trusted_public_keys: Sequence[str],
    require_live: bool = False,
) -> list[str]:
    """Verify signature, exact aggregate schema, and pass/fail semantics."""

    violations: list[str] = []
    required = {
        "api_version",
        "kind",
        "schema_version",
        "connector",
        "certified_at",
        "mode",
        "status",
        "live_certified",
        "bundle",
        "scope",
        "checks",
        "counts",
        "semantic_validator",
        "evidence",
        "failure_class",
        "runtime_configuration",
        "signer",
        "signature_algorithm",
        "signing_public_key",
        "signature",
    }
    if set(record) != required:
        return ["certification record fields are not exact"]
    if (
        record.get("api_version") != "graphos.io/v1"
        or record.get("kind") != "ConnectorLiveCertification"
        or record.get("schema_version") != "1"
        or not _SAFE_CONNECTOR.fullmatch(str(record.get("connector") or ""))
    ):
        violations.append("certification record identity is invalid")
    try:
        parsed_time = datetime.strptime(
            str(record.get("certified_at") or ""), "%Y-%m-%dT%H:%M:%SZ"
        ).replace(tzinfo=UTC)
        if parsed_time.year < 2020:
            raise ValueError
    except ValueError:
        violations.append("certification timestamp is invalid")
    bundle = record.get("bundle")
    if not isinstance(bundle, dict) or set(bundle) != {
        "manifest_sha256",
        "fixtures_sha256",
        "shapes_sha256",
        "schema_version",
    }:
        violations.append("certification bundle binding is invalid")
    elif any(
        not _HEX_DIGEST.fullmatch(str(bundle.get(name) or ""))
        for name in ("manifest_sha256", "fixtures_sha256", "shapes_sha256")
    ):
        violations.append("certification bundle digest is invalid")
    elif not re.fullmatch(
        r"[A-Za-z0-9._-]{1,64}", str(bundle.get("schema_version") or "")
    ):
        violations.append("certification bundle schema version is invalid")
    scope = record.get("scope")
    if not isinstance(scope, dict) or set(scope) != {
        "sync_presets",
        "fixtures_declared",
        "fixtures_exercised",
        "tenant_bound",
        "retention_bound",
    }:
        violations.append("certification scope is invalid")
    elif (
        any(
            not isinstance(scope.get(name), int)
            or isinstance(scope.get(name), bool)
            or not 0 <= scope[name] <= 256
            for name in ("sync_presets", "fixtures_declared", "fixtures_exercised")
        )
        or scope.get("tenant_bound") is not True
        or scope.get("retention_bound") is not True
    ):
        violations.append("certification scope values are invalid")
    checks = record.get("checks")
    if not isinstance(checks, dict) or set(checks) != set(REQUIRED_CHECKS):
        violations.append("certification check catalog is invalid")
    elif any(value not in {"passed", "failed", "not-run"} for value in checks.values()):
        violations.append("certification check status is invalid")
    evidence = record.get("evidence")
    if not isinstance(evidence, dict) or set(evidence) != set(REQUIRED_CHECKS):
        violations.append("certification evidence catalog is invalid")
    elif any(
        not _HEX_DIGEST.fullmatch(str(value or "")) for value in evidence.values()
    ):
        violations.append("certification evidence digest is invalid")
    counts = record.get("counts")
    allowed_counts = {
        "initial",
        "after_ingest",
        "after_replay",
        "after_update",
        "after_delete",
        "after_delete_replay",
        "after_cleanup",
    }
    if not isinstance(counts, dict) or not set(counts).issubset(allowed_counts):
        violations.append("certification count catalog is invalid")
    elif any(
        not isinstance(value, int)
        or isinstance(value, bool)
        or not 0 <= value <= 1_000_000
        for value in counts.values()
    ):
        violations.append("certification count value is invalid")
    if record.get("semantic_validator") not in {
        "not-run",
        "declared-shacl-contract",
        "pyshacl",
    }:
        violations.append("certification semantic validator is invalid")
    failure_class = record.get("failure_class")
    if failure_class is not None and not re.fullmatch(
        r"[A-Za-z_][A-Za-z0-9_]{0,127}", str(failure_class)
    ):
        violations.append("certification failure class is invalid")
    if record.get("runtime_configuration") not in {"none", "externalized"}:
        violations.append("certification runtime configuration marker is invalid")

    digest = ontology_integrity.canonical_signed_document_hash(dict(record))
    if not ontology_integrity.verify_release_signature(
        digest,
        record.get("signature"),
        signer_id=record.get("signer"),
        algorithm=record.get("signature_algorithm"),
        public_key=record.get("signing_public_key"),
        trusted_public_keys=tuple(trusted_public_keys),
    ):
        violations.append("certification release signature is invalid")
    if require_live and (
        record.get("mode") != "external-live"
        or record.get("status") != "certified"
        or record.get("live_certified") is not True
        or not isinstance(checks, dict)
        or any(checks.get(name) != "passed" for name in REQUIRED_CHECKS)
        or record.get("runtime_configuration") != "externalized"
        or record.get("failure_class") is not None
    ):
        violations.append("connector has no passing external live certification")
    if not require_live and record.get("status") not in {
        "certified",
        "offline-validated",
        "failed",
    }:
        violations.append("certification status is invalid")
    return violations


def write_certification_record(path: Path, record: Mapping[str, Any]) -> None:
    """Write one bounded signed record without following an output symlink."""

    payload = (
        json.dumps(
            record,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    if len(payload) > 1024 * 1024:
        raise CertificationError("certification record exceeds its boundary")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise CertificationError("certification output symlinks are not accepted")
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(temporary, flags, 0o600)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise CertificationError("certification record write failed")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    try:
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_certification_record(path: Path) -> dict[str, Any]:
    """Load one bounded regular certification record."""

    value = _bounded_json(_read_regular(path, 1024 * 1024), max_bytes=1024 * 1024)
    if not isinstance(value, dict):
        raise CertificationError("certification record must be a JSON object")
    return value


def _read_regular(path: Path, max_bytes: int) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    if not nofollow and path.is_symlink():
        raise CertificationError("capability bundle symlinks are not accepted")
    descriptor = os.open(path, flags | nofollow)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > max_bytes:
            raise CertificationError("capability bundle artifact is invalid")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, min(65_536, max_bytes + 1 - total))
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                raise CertificationError(
                    "capability bundle artifact exceeds its boundary"
                )
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _bounded_json(raw: bytes, *, max_bytes: int) -> Any:
    if len(raw) > max_bytes:
        raise ValueError("JSON payload exceeds its boundary")
    value = json.loads(raw.decode("utf-8"))
    stack: list[tuple[Any, int]] = [(value, 0)]
    nodes = 0
    while stack:
        current, depth = stack.pop()
        nodes += 1
        if nodes > _MAX_JSON_NODES or depth > 32:
            raise ValueError("JSON payload exceeds its structural boundary")
        if isinstance(current, dict):
            if len(current) > 4096 or any(not isinstance(key, str) for key in current):
                raise ValueError("JSON object is invalid")
            stack.extend((item, depth + 1) for item in current.values())
        elif isinstance(current, list):
            if len(current) > 4096:
                raise ValueError("JSON collection is invalid")
            stack.extend((item, depth + 1) for item in current)
        elif current is not None and not isinstance(current, (str, int, float, bool)):
            raise ValueError("JSON value is invalid")
        elif isinstance(current, float) and not math.isfinite(current):
            raise ValueError("JSON number is invalid")
    return value


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _evidence_digest(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(
        b"graphos:connector-certification:v1\x00" + payload
    ).hexdigest()


def _scope_key(envelope: Mapping[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(envelope.get("tenant") or ""),
        str(envelope.get("connector") or ""),
        str(envelope.get("source_instance") or ""),
        str(envelope.get("source_object_id") or ""),
    )


def _governance(envelope: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "tenant": str(envelope.get("tenant") or ""),
        "retention": str(envelope.get("retention") or ""),
        "legal_hold": bool(envelope.get("legal_hold")),
        "classification": str(envelope.get("classification") or ""),
        "source_acl": envelope.get("source_acl"),
    }


def _fixture_envelope(
    bundle: CertificationBundle,
    fixture: _Fixture,
    *,
    index: int,
    run_key: str,
    source_instance: str,
    policy: CertificationPolicy,
    version: str,
) -> ChangeEnvelope:
    raw = dict(fixture.record)
    for untrusted in (
        "external_access",
        "tenant",
        "retention",
        "legal_hold",
        "classification",
        "provenance",
    ):
        raw.pop(untrusted, None)
    guard = PersistencePrivacyGuard()
    sanitized, report = guard.sanitize(raw)
    if report.changed or not isinstance(sanitized, dict):
        raise CertificationError("certification fixture contains private content")
    encoded = json.dumps(
        sanitized,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    if len(encoded) > 256 * 1024:
        raise CertificationError("certification fixture exceeds its record boundary")
    object_id = (
        "cert-"
        + hashlib.sha256(
            f"{bundle.manifest.connector}:{run_key}:{fixture.preset}:{index}".encode()
        ).hexdigest()[:32]
    )
    sync = next(item for item in bundle.manifest.sync if item.preset == fixture.preset)
    normalized_doc_type = re.sub(r"[^a-z0-9]", "", str(sync.doc_type or "").lower())
    resource = next(
        (
            item.name
            for item in bundle.manifest.resources
            if re.sub(r"[^a-z0-9]", "", item.name.lower()) == normalized_doc_type
        ),
        bundle.manifest.resources[0].name if bundle.manifest.resources else "Document",
    )
    payload = {**sanitized, "id": object_id, "type": resource}
    acl_state = str(fixture.expected.get("acl_state") or "quarantine")
    if acl_state == "quarantine":
        access = ExternalAccess.quarantined()
        classification = DataClassification.INTERNAL
    elif acl_state == "public":
        access = ExternalAccess.public()
        classification = DataClassification.PUBLIC
    else:
        raise CertificationError("fixture ACL expectation is unsupported")
    return ChangeEnvelope(
        connector=bundle.manifest.connector,
        tenant=policy.tenant,
        source_instance=source_instance,
        source_object_id=object_id,
        source_version=version,
        schema_version=bundle.manifest.schema_version,
        ontology_mapping_version=bundle.manifest_sha256,
        typed_payload=payload,
        source_acl=access,
        classification=classification,
        retention=policy.retention,
        legal_hold=policy.legal_hold,
        provenance={"fixture_sha256": _sha256(encoded), "certification": True},
        checkpoint=version,
    )


def _next_envelope(
    envelope: ChangeEnvelope, *, version: str, operation: str
) -> ChangeEnvelope:
    payload = dict(envelope.typed_payload or {}) if operation == "upsert" else None
    if payload is not None:
        payload["certification_revision"] = version
    return ChangeEnvelope(
        connector=envelope.connector,
        operation=operation,  # type: ignore[arg-type]
        tenant=envelope.tenant,
        source_instance=envelope.source_instance,
        source_object_id=envelope.source_object_id,
        source_version=version,
        schema_version=envelope.schema_version,
        ontology_mapping_version=envelope.ontology_mapping_version,
        typed_payload=payload,
        source_acl=envelope.source_acl,
        classification=envelope.classification,
        retention=envelope.retention,
        legal_hold=envelope.legal_hold,
        provenance=dict(envelope.provenance),
        checkpoint=version,
    )


async def _apply(
    driver: CertificationDriver, envelope: ChangeEnvelope, *, expect_replay: bool
) -> None:
    response = await driver.invoke(
        {
            "api_version": "graphos.connector-certification/v1",
            "action": "apply",
            "envelope": envelope.as_dict(),
        }
    )
    applied = response.get("applied") is True
    replayed = response.get("replayed") is True
    if expect_replay and (applied or not replayed):
        raise CertificationError("driver did not prove idempotent replay")
    if not expect_replay and (not applied or replayed):
        raise CertificationError("driver did not apply a new envelope")


async def _count(driver: CertificationDriver, scope: Mapping[str, Any]) -> int:
    response = await driver.invoke(
        {
            "api_version": "graphos.connector-certification/v1",
            "action": "count",
            "scope": dict(scope),
        }
    )
    value = response.get("count")
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or not 0 <= value <= 1_000_000
    ):
        raise CertificationError("driver count response is invalid")
    return value


async def _assert_governance(
    driver: CertificationDriver, envelope: ChangeEnvelope
) -> None:
    response = await driver.invoke(
        {
            "api_version": "graphos.connector-certification/v1",
            "action": "inspect",
            "scope": {
                "tenant": envelope.tenant,
                "connector": envelope.connector,
                "source_instance": envelope.source_instance,
                "source_object_id": envelope.source_object_id,
            },
        }
    )
    expected = _governance(envelope.as_dict())
    if response.get("exists") is not True or response.get("governance") != expected:
        raise CertificationError("driver did not preserve envelope governance")


async def _assert_tombstone_governance(
    driver: CertificationDriver, envelope: ChangeEnvelope
) -> None:
    response = await driver.invoke(
        {
            "api_version": "graphos.connector-certification/v1",
            "action": "inspect",
            "scope": {
                "tenant": envelope.tenant,
                "connector": envelope.connector,
                "source_instance": envelope.source_instance,
                "source_object_id": envelope.source_object_id,
            },
        }
    )
    expected = _governance(envelope.as_dict())
    if (
        response.get("exists") is not False
        or response.get("tombstone_governance") != expected
    ):
        raise CertificationError("driver did not preserve tombstone governance")


def _verify_live_tools(result: Mapping[str, Any], syncs: Sequence[SyncSpec]) -> None:
    tools = result.get("tools")
    if not isinstance(tools, list) or len(tools) > 4096:
        raise CertificationError("live connector returned no bounded tool catalog")
    for sync in syncs:
        validate_live_tool_contract(
            tools,
            tool_name=sync.tool,
            expected_schema_sha256=str(sync.tool_schema_sha256 or ""),
        )


def _semantic_validation(
    bundle: CertificationBundle,
    envelopes: Sequence[ChangeEnvelope],
    *,
    require_pyshacl: bool,
) -> str:
    try:
        import rdflib
    except ImportError:
        if require_pyshacl:
            raise CertificationError(
                "live certification requires the SHACL runtime"
            ) from None
        _declared_semantic_validation(bundle, envelopes)
        return "declared-shacl-contract"
    try:
        shapes = rdflib.Graph()
        shapes.parse(data=bundle.shapes_text, format="turtle")
        data = rdflib.Graph()
        kg = rdflib.Namespace("http://knuckles.team/kg#")
        for index, envelope in enumerate(envelopes):
            payload = envelope.typed_payload or {}
            resource = str(payload.get("type") or "Document")
            subject = rdflib.URIRef(f"urn:graphos:connector-certification:{index}")
            data.add((subject, rdflib.RDF.type, kg[resource]))
            data.add((subject, kg.sourceRecordRef, rdflib.Literal("opaque")))
            data.add((subject, kg.tenantReference, rdflib.Literal("bound")))
            data.add((subject, kg.accessPolicyReference, rdflib.Literal("bound")))
            data.add((subject, kg.provenanceReference, rdflib.Literal("bound")))
        try:
            from pyshacl import validate
        except ImportError:
            if require_pyshacl:
                raise CertificationError(
                    "live certification requires the SHACL runtime"
                ) from None
            target_class = rdflib.URIRef("http://www.w3.org/ns/shacl#targetClass")
            targets = {str(value) for value in shapes.objects(predicate=target_class)}
            declared = {
                str(
                    rdflib.Namespace("http://knuckles.team/kg#")[
                        str((envelope.typed_payload or {}).get("type") or "Document")
                    ]
                )
                for envelope in envelopes
            }
            if not declared.issubset(targets):
                raise CertificationError(
                    "declared semantic coverage is incomplete"
                ) from None
            return "declared-shacl-contract"
        conforms, _results_graph, _results_text = validate(
            data,
            shacl_graph=shapes,
            abort_on_first=False,
            allow_infos=False,
            allow_warnings=False,
        )
        if not bool(conforms):
            raise CertificationError("synthetic fixture does not conform to SHACL")
        return "pyshacl"
    except CertificationError:
        raise
    except Exception as exc:
        raise CertificationError("semantic validation failed") from exc


def _declared_semantic_validation(
    bundle: CertificationBundle, envelopes: Sequence[ChangeEnvelope]
) -> None:
    """Dependency-free validation of the signed generated SHACL declaration."""

    if not 1 <= len(bundle.shapes_text.encode("utf-8")) <= _MAX_ARTIFACT_BYTES:
        raise CertificationError("declared semantic artifact is invalid")
    targets = set(
        re.findall(
            r"\bsh:targetClass\s+:([A-Za-z_][A-Za-z0-9_-]{0,127})\b",
            bundle.shapes_text,
        )
    )
    declared = {
        str((envelope.typed_payload or {}).get("type") or "Document")
        for envelope in envelopes
    }
    required_paths = {
        "sourceRecordRef",
        "tenantReference",
        "accessPolicyReference",
        "provenanceReference",
    }
    paths = set(
        re.findall(
            r"\bsh:path\s+:([A-Za-z_][A-Za-z0-9_-]{0,127})\b",
            bundle.shapes_text,
        )
    )
    if not declared.issubset(targets) or not required_paths.issubset(paths):
        raise CertificationError("declared semantic coverage is incomplete")


def _delegated_environment(
    command: Sequence[str], payload: Mapping[str, Any]
) -> dict[str, str]:
    env = {key: value for key, value in os.environ.items() if key in _BASE_CHILD_ENV}
    values: list[str] = list(command)
    stack: list[Any] = [payload]
    while stack:
        item = stack.pop()
        if isinstance(item, dict):
            stack.extend(item.values())
        elif isinstance(item, list):
            stack.extend(item)
        elif isinstance(item, str):
            values.append(item)
    for value in values:
        if value.startswith("env://"):
            name = value[6:]
            if (
                re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,127}", name)
                and name in os.environ
            ):
                env[name] = os.environ[name]
    return env


def safe_record_name(connector: str) -> str:
    """Return the only accepted fleet record filename for a connector."""

    if not _SAFE_CONNECTOR.fullmatch(connector):
        raise ValueError("connector identity is invalid")
    name = PurePosixPath(f"{connector}.json")
    if len(name.parts) != 1:
        raise ValueError("connector identity is invalid")
    return name.name
