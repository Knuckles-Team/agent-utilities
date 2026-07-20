"""Deployment-owned exact-release orchestration for bundled-skill certification.

The command starts one exact candidate GraphOS process from an externally supplied
JSON argv reference, proves readiness through the packaged AgentConfig/TLS/OIDC
probe, invokes the validator from the same installed release, and always stops and
reaps the marked GraphOS and local engine. Durable configuration contains only
digests, booleans, bounds, counts, and environment-reference names.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import secrets
import signal
import stat
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from agent_utilities.core._env import setting
from agent_utilities.deployment.certification_oidc import (
    EphemeralLoopbackOidcAuthority,
)
from agent_utilities.skills.runtime_validation import (
    _digest_bytes,
    _external_command,
    load_matrix,
    minimum_campaign_authority_ttl_seconds,
    publish_report,
    render_evidence,
    sign_and_verify_evidence,
)

_DIGEST = re.compile(r"^sha256:(?!0{64}$)[a-f0-9]{64}$")
_MARKER_ENV = "GRAPHOS_SKILL_VALIDATION_INSTANCE"
# The engine launcher deliberately inherits only documented engine namespaces;
# this marker uses that trusted prefix so the exact local Rust child can prove
# descent without widening its sanitized environment.
_ENGINE_MARKER_ENV = "EPISTEMIC_GRAPH_SKILL_VALIDATION_INSTANCE"
_PROFILE_ENV = "AGENT_UTILITIES_RUNTIME_PROFILE_REF"


@dataclass(frozen=True)
class _ProcessCounts:
    """Privacy-safe process classes plus internal marked-engine handles."""

    global_graph_os: int
    candidate_graph_os: int
    candidate_engine: int
    langfuse_mcp_children: int
    loopback_oidc_fixtures: int
    marked_engines: tuple[Path, ...]


class DeploymentError(RuntimeError):
    """Controlled lifecycle failure whose message is never reported."""


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=False)


class ReleaseBinding(_StrictModel):
    id: str = Field(pattern=r"^release-[a-z0-9][a-z0-9.-]{2,63}$")
    specification_reference: str = Field(
        alias="specificationReference", pattern=r"^[A-Z][A-Z0-9_]{2,63}$"
    )
    specification_digest: str = Field(alias="specificationDigest")
    promotion_evidence_reference: str = Field(
        alias="promotionEvidenceReference", pattern=r"^[A-Z][A-Z0-9_]{2,63}$"
    )
    promotion_evidence_digest: str = Field(alias="promotionEvidenceDigest")
    agent_utilities_sha256: str = Field(alias="agentUtilitiesSha256")
    agent_utilities_file_count: int = Field(alias="agentUtilitiesFileCount", ge=10)
    distribution_closure_sha256: str = Field(alias="distributionClosureSha256")
    release_python_sha256: str = Field(alias="releasePythonSha256")
    graph_os_digest: str = Field(alias="graphOsDigest")
    engine_digest: str = Field(alias="engineDigest")
    start_command_reference: str = Field(
        alias="startCommandReference", pattern=r"^[A-Z][A-Z0-9_]{2,63}$"
    )

    @field_validator(
        "specification_digest",
        "promotion_evidence_digest",
        "agent_utilities_sha256",
        "distribution_closure_sha256",
        "release_python_sha256",
        "graph_os_digest",
        "engine_digest",
    )
    @classmethod
    def _digest(cls, value: str) -> str:
        if _DIGEST.fullmatch(value) is None:
            raise ValueError("digest_invalid")
        return value


class ModelRegistryBinding(_StrictModel):
    digest: str
    model_count: Literal[2] = Field(alias="modelCount")
    light_count: Literal[1] = Field(alias="lightCount")
    normal_count: Literal[1] = Field(alias="normalCount")
    local_private_transport_only: Literal[True] = Field(
        alias="localPrivateTransportOnly"
    )
    reference_backed_credentials_only: Literal[True] = Field(
        alias="referenceBackedCredentialsOnly"
    )
    literal_private_model_count: int = Field(
        alias="literalPrivateModelCount", ge=0, le=2
    )
    private_dns_model_count: int = Field(alias="privateDnsModelCount", ge=0, le=2)
    runtime_private_resolution_required: Literal[True] = Field(
        alias="runtimePrivateResolutionRequired"
    )

    @field_validator("digest")
    @classmethod
    def _digest(cls, value: str) -> str:
        if _DIGEST.fullmatch(value) is None:
            raise ValueError("digest_invalid")
        return value

    @model_validator(mode="after")
    def _model_transport_cardinality(self) -> ModelRegistryBinding:
        if self.literal_private_model_count + self.private_dns_model_count != 2:
            raise ValueError("model_transport_cardinality_invalid")
        return self


class RuntimeBinding(_StrictModel):
    configuration_reference: str = Field(
        alias="configurationReference", pattern=r"^[A-Z][A-Z0-9_]{2,63}$"
    )
    configuration_digest: str = Field(alias="configurationDigest")
    profile_reference: str = Field(
        alias="profileReference", pattern=r"^[A-Z][A-Z0-9_]{2,63}$"
    )
    profile_digest: str = Field(alias="profileDigest")
    endpoint_reference: str = Field(
        alias="endpointReference", pattern=r"^[A-Z][A-Z0-9_]{2,63}$"
    )
    model_registry: ModelRegistryBinding = Field(alias="modelRegistry")

    @field_validator("profile_digest", "configuration_digest")
    @classmethod
    def _digest(cls, value: str) -> str:
        if _DIGEST.fullmatch(value) is None:
            raise ValueError("digest_invalid")
        return value


class ReadinessBinding(_StrictModel):
    timeout_seconds: int = Field(alias="timeoutSeconds", ge=1, le=300)
    poll_interval_milliseconds: int = Field(
        alias="pollIntervalMilliseconds", ge=50, le=5_000
    )


class ValidationBinding(_StrictModel):
    case_timeout_seconds: int = Field(alias="caseTimeoutSeconds", ge=1, le=600)
    signer_command_reference: str = Field(
        alias="signerCommandReference", pattern=r"^[A-Z][A-Z0-9_]{2,63}$"
    )
    verifier_command_reference: str = Field(
        alias="verifierCommandReference", pattern=r"^[A-Z][A-Z0-9_]{2,63}$"
    )


class ShutdownBinding(_StrictModel):
    grace_seconds: int = Field(alias="graceSeconds", ge=1, le=60)


class IdentityAuthorityBinding(_StrictModel):
    mode: Literal["ephemeral-https-loopback"]
    token_ttl_seconds: int = Field(alias="tokenTtlSeconds", ge=180, le=3_600)
    tls_verification_required: Literal[True] = Field(alias="tlsVerificationRequired")
    lifecycle_owned: Literal[True] = Field(alias="lifecycleOwned")
    renewable_credentials_required: Literal[True] = Field(
        alias="renewableCredentialsRequired"
    )


class SkillValidationDeployment(_StrictModel):
    api_version: Literal["graphos.io/v2"] = Field(alias="apiVersion")
    kind: Literal["SkillValidationDeployment"]
    identity_authority: IdentityAuthorityBinding = Field(alias="identityAuthority")
    release: ReleaseBinding
    runtime: RuntimeBinding
    readiness: ReadinessBinding
    validation: ValidationBinding
    shutdown: ShutdownBinding

    @model_validator(mode="after")
    def _identity_lease_covers_campaign(self) -> SkillValidationDeployment:
        defaults, _cases = load_matrix()
        trace_timeout = defaults.get("trace_timeout_seconds")
        if (
            isinstance(trace_timeout, bool)
            or not isinstance(trace_timeout, int | float)
            or trace_timeout <= 0
        ):
            raise ValueError("campaign_trace_timeout_invalid")
        required_ttl = minimum_campaign_authority_ttl_seconds(
            case_timeout=self.validation.case_timeout_seconds,
            trace_timeout=float(trace_timeout),
            shutdown_grace=self.shutdown.grace_seconds,
        )
        if self.identity_authority.token_ttl_seconds < required_ttl:
            raise ValueError("identity_token_ttl_campaign_window_invalid")
        return self


def _json_without_duplicates(payload: str) -> Any:
    def exact_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise DeploymentError("configuration_duplicate_key")
            value[key] = item
        return value

    return json.loads(payload, object_pairs_hook=exact_pairs)


def load_deployment(path: Path) -> SkillValidationDeployment:
    try:
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        try:
            metadata = os.fstat(descriptor)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_nlink != 1
                or not 1 <= metadata.st_size <= 64 * 1024
            ):
                raise DeploymentError("configuration_not_regular")
            before = (
                metadata.st_dev,
                metadata.st_ino,
                metadata.st_size,
                metadata.st_mtime_ns,
                metadata.st_ctime_ns,
            )
            raw = bytearray()
            while len(raw) <= 64 * 1024:
                chunk = os.read(descriptor, min(64 * 1024 + 1 - len(raw), 65_536))
                if not chunk:
                    break
                raw.extend(chunk)
            after_metadata = os.fstat(descriptor)
            after = (
                after_metadata.st_dev,
                after_metadata.st_ino,
                after_metadata.st_size,
                after_metadata.st_mtime_ns,
                after_metadata.st_ctime_ns,
            )
            if before != after or len(raw) != metadata.st_size:
                raise DeploymentError("configuration_changed_during_read")
            path_metadata = path.stat(follow_symlinks=False)
            if (
                path_metadata.st_dev,
                path_metadata.st_ino,
                path_metadata.st_size,
                path_metadata.st_mtime_ns,
                path_metadata.st_ctime_ns,
            ) != before:
                raise DeploymentError("configuration_changed_during_read")
        finally:
            os.close(descriptor)
        payload = bytes(raw).decode("utf-8")
        return SkillValidationDeployment.model_validate(
            _json_without_duplicates(payload)
        )
    except DeploymentError:
        raise
    except Exception as exc:
        raise DeploymentError("configuration_invalid") from exc


def _runtime_reference(name: str) -> str:
    value = str(setting(name, "") or "")
    if not value or "\x00" in value or len(value) > 4_096:
        raise DeploymentError("runtime_reference_unresolved")
    return value


def _regular_executable(path: Path, *, name: str) -> Path:
    if not path.is_absolute() or path.name != name:
        raise DeploymentError("release_executable_invalid")
    try:
        original = path.lstat()
        canonical = path.resolve(strict=True)
        metadata = canonical.lstat()
    except OSError as exc:
        raise DeploymentError("release_executable_invalid") from exc
    if (
        stat.S_ISLNK(original.st_mode)
        or not stat.S_ISREG(original.st_mode)
        or canonical.name != name
        or canonical.is_symlink()
        or not stat.S_ISREG(metadata.st_mode)
        or (original.st_dev, original.st_ino) != (metadata.st_dev, metadata.st_ino)
        or not os.access(canonical, os.X_OK)
    ):
        raise DeploymentError("release_executable_invalid")
    return canonical


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        metadata = os.fstat(handle.fileno())
        if (
            not stat.S_ISREG(metadata.st_mode)
            or not 1 <= metadata.st_size <= 2 * 1024 * 1024 * 1024
        ):
            raise DeploymentError("release_executable_size_invalid")
        consumed = 0
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            consumed += len(chunk)
            if consumed > 2 * 1024 * 1024 * 1024:
                raise DeploymentError("release_executable_size_invalid")
            digest.update(chunk)
        after = os.fstat(handle.fileno())
    try:
        path_metadata = path.stat()
    except OSError as exc:
        raise DeploymentError("release_executable_changed_during_read") from exc
    if (
        consumed != metadata.st_size
        or (
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_size,
            metadata.st_mtime_ns,
            metadata.st_ctime_ns,
        )
        != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        or (
            path_metadata.st_dev,
            path_metadata.st_ino,
            path_metadata.st_size,
            path_metadata.st_mtime_ns,
            path_metadata.st_ctime_ns,
        )
        != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
    ):
        raise DeploymentError("release_executable_changed_during_read")
    return "sha256:" + digest.hexdigest()


def _validate_evidence_destinations(destinations: tuple[Path, Path, Path]) -> None:
    """Require three fresh sibling outputs before any release process can start."""

    normalized = tuple(Path(os.path.abspath(path)) for path in destinations)
    if len(set(normalized)) != len(normalized):
        raise DeploymentError("evidence_destinations_not_distinct")
    if len({path.parent for path in normalized}) != 1:
        raise DeploymentError("evidence_destinations_not_alongside")
    for path in normalized:
        try:
            path.lstat()
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise DeploymentError("evidence_destination_unavailable") from exc
        raise DeploymentError("evidence_destination_not_fresh")


def _bounded_proc_bytes(path: Path, *, limit: int) -> bytes:
    try:
        with path.open("rb") as handle:
            payload = handle.read(limit + 1)
    except (FileNotFoundError, ProcessLookupError):
        return b""
    except OSError as exc:
        raise DeploymentError("process_observation_unavailable") from exc
    if len(payload) > limit:
        raise DeploymentError("process_observation_boundary_exceeded")
    return payload


def _process_kind(entry: Path) -> str | None:
    try:
        executable = Path(os.readlink(entry / "exe")).name
    except OSError:
        executable = ""
    raw = _bounded_proc_bytes(entry / "cmdline", limit=64 * 1024)
    try:
        argv = [item.decode("utf-8", "strict") for item in raw.split(b"\x00") if item]
    except UnicodeDecodeError as exc:
        raise DeploymentError("process_observation_invalid") from exc
    candidates = {executable}
    candidates.update(Path(item).name for item in argv[:3])
    if "epistemic-graph-server" in candidates:
        return "engine"
    if "graph-os" in candidates:
        return "graph-os"
    if "langfuse-mcp" in candidates:
        return "langfuse-mcp-child"
    if "loopback_oidc.py" in candidates:
        return "loopback-oidc-fixture"
    for index, argument in enumerate(argv[:-1]):
        if argument != "-m":
            continue
        module = argv[index + 1]
        if module == "agent_utilities.mcp.kg_server":
            return "graph-os"
        if module == "langfuse_agent.mcp_server":
            return "langfuse-mcp-child"
        if module == "scripts.certification.loopback_oidc":
            return "loopback-oidc-fixture"
    return None


def _process_snapshot(marker: str) -> list[tuple[Path, str, bool]]:
    """Return a bounded internal snapshot; process identities never leave memory."""

    proc = Path("/proc")
    if os.name != "posix" or not proc.is_dir():
        raise DeploymentError("process_observation_unsupported")
    snapshots: list[tuple[Path, str, bool]] = []
    scanned = 0
    graph_marker = f"{_MARKER_ENV}={marker}".encode()
    engine_marker = f"{_ENGINE_MARKER_ENV}={marker}".encode()
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        scanned += 1
        if scanned > 262_144:
            raise DeploymentError("process_observation_boundary_exceeded")
        kind = _process_kind(entry)
        if kind is None:
            continue
        marker_present = False
        if kind in {"graph-os", "engine"}:
            variables = _bounded_proc_bytes(entry / "environ", limit=1024 * 1024).split(
                b"\x00"
            )
            marker_present = (
                graph_marker in variables
                if kind == "graph-os"
                else engine_marker in variables
            )
        snapshots.append((entry, kind, marker_present))
    return snapshots


def _process_counts(marker: str) -> _ProcessCounts:
    snapshot = _process_snapshot(marker)
    global_graph_os = sum(kind == "graph-os" for _entry, kind, _marked in snapshot)
    candidate_graph_os = sum(
        kind == "graph-os" and marked for _entry, kind, marked in snapshot
    )
    candidate_engines = tuple(
        entry for entry, kind, marked in snapshot if kind == "engine" and marked
    )
    return _ProcessCounts(
        global_graph_os=global_graph_os,
        candidate_graph_os=candidate_graph_os,
        candidate_engine=len(candidate_engines),
        langfuse_mcp_children=sum(
            kind == "langfuse-mcp-child" for _entry, kind, _marked in snapshot
        ),
        loopback_oidc_fixtures=sum(
            kind == "loopback-oidc-fixture" for _entry, kind, _marked in snapshot
        ),
        marked_engines=candidate_engines,
    )


def _wait_for_terminal_process_gate(
    marker: str, grace_seconds: int
) -> _ProcessCounts:
    """Wait for lifecycle-owned process teardown to become observable.

    MCP stdio children run in their own sessions.  GraphOS awaits their shutdown,
    but a child that has already been asked to exit can remain visible in
    ``/proc`` briefly after the GraphOS process itself is reaped.  Publishing the
    first post-shutdown snapshot therefore turns an ordinary process-reaping
    race into a false lifecycle failure.  Poll the complete zero-process gate for
    at most the deployment-owned shutdown grace and return the final observed
    counts; a real leak still fails closed in the caller.
    """

    deadline = time.monotonic() + grace_seconds
    while True:
        counts = _process_counts(marker)
        if (
            counts.global_graph_os,
            counts.candidate_graph_os,
            counts.candidate_engine,
            counts.langfuse_mcp_children,
            counts.loopback_oidc_fixtures,
        ) == (0, 0, 0, 0, 0):
            return counts
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return counts
        time.sleep(min(0.05, remaining))


def _marked_engine_digest(marker: str) -> str:
    counts = _process_counts(marker)
    if counts.candidate_engine != 1:
        raise DeploymentError("candidate_engine_count_invalid")
    try:
        return _file_digest(counts.marked_engines[0] / "exe")
    except OSError as exc:
        raise DeploymentError("candidate_engine_digest_unavailable") from exc


def _terminate_marked_engines(marker: str, grace_seconds: int) -> None:
    counts = _process_counts(marker)
    for entry in counts.marked_engines:
        try:
            os.kill(int(entry.name), signal.SIGTERM)
        except ProcessLookupError:
            continue
    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline:
        if _process_counts(marker).candidate_engine == 0:
            return
        time.sleep(0.05)
    for entry in _process_counts(marker).marked_engines:
        try:
            os.kill(int(entry.name), signal.SIGKILL)
        except ProcessLookupError:
            continue


def _stop_and_reap(process: subprocess.Popen[bytes], grace_seconds: int) -> None:
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=grace_seconds)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=grace_seconds)
    else:
        process.wait()


def _wait_until_ready(
    process: subprocess.Popen[bytes],
    *,
    readiness_executable: Path,
    deployment_path: Path,
    environment: dict[str, str],
    timeout_seconds: int,
    poll_interval_milliseconds: int,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise DeploymentError("candidate_exited_before_ready")
        remaining = max(0.1, deadline - time.monotonic())
        try:
            completed = subprocess.run(
                [
                    str(readiness_executable),
                    "--deployment",
                    str(deployment_path),
                    "--request-timeout",
                    str(min(15.0, remaining)),
                ],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
                timeout=min(20.0, remaining + 1.0),
                close_fds=True,
                env=environment,
            )
            if completed.returncode == 0:
                return
        except (OSError, subprocess.TimeoutExpired):
            pass
        time.sleep(min(poll_interval_milliseconds / 1000.0, remaining))
    raise DeploymentError("candidate_readiness_timeout")


def _lifecycle_subject(
    deployment: SkillValidationDeployment,
    *,
    global_counts: tuple[int, int, int],
    graph_os_counts: tuple[int, int, int],
    engine_counts: tuple[int, int, int],
    identity_authority_counts: tuple[int, int, int],
    terminal_process_counts: tuple[int, int],
    identity_tls_verified: bool,
    renewable_credentials_proven: bool,
    identity_token_mint_count: int,
    model_transport_proof: dict[str, Any],
    engine_executable_digest: str | None,
    installed_release_attested: bool,
    reaped: bool,
    validator_exit_code: int | None,
    validation_evidence_digest: str | None,
    validation_case_count: int,
    error_code: str | None,
) -> dict[str, Any]:
    passed = (
        global_counts == (0, 1, 0)
        and graph_os_counts == (0, 1, 0)
        and engine_counts == (0, 1, 0)
        and identity_authority_counts == (0, 1, 0)
        and terminal_process_counts == (0, 0)
        and identity_tls_verified
        and renewable_credentials_proven
        and identity_token_mint_count >= 2
        and model_transport_proof
        == {
            "modelCount": 2,
            "literalPrivateModelCount": (
                deployment.runtime.model_registry.literal_private_model_count
            ),
            "privateDnsModelCount": (
                deployment.runtime.model_registry.private_dns_model_count
            ),
            "privateDnsUniqueResolutionProven": True,
            "privateBoundaryProven": True,
            "dnsRebindingGuarded": True,
        }
        and engine_executable_digest == deployment.release.engine_digest
        and installed_release_attested
        and reaped
        and validator_exit_code == 0
        and validation_evidence_digest is not None
        and validation_case_count == 20
        and error_code is None
    )

    def counts(value: tuple[int, int, int]) -> dict[str, int]:
        return {"before": value[0], "running": value[1], "after": value[2]}

    return {
        "apiVersion": "graphos.io/v2",
        "kind": "SkillValidationLifecycleEvidence",
        "evidenceVersion": 2,
        "release": {
            "id": deployment.release.id,
            "specificationDigest": deployment.release.specification_digest,
            "promotionEvidenceDigest": deployment.release.promotion_evidence_digest,
            "agentUtilitiesSha256": deployment.release.agent_utilities_sha256,
            "agentUtilitiesFileCount": deployment.release.agent_utilities_file_count,
            "distributionClosureSha256": deployment.release.distribution_closure_sha256,
            "releasePythonSha256": deployment.release.release_python_sha256,
            "graphOsDigest": deployment.release.graph_os_digest,
            "engineDigest": deployment.release.engine_digest,
        },
        "runtime": {
            "configurationDigest": deployment.runtime.configuration_digest,
            "profileDigest": deployment.runtime.profile_digest,
            "modelRegistryDigest": deployment.runtime.model_registry.digest,
        },
        "identityAuthority": {
            "mode": deployment.identity_authority.mode,
            "lifecycleCounts": counts(identity_authority_counts),
            "tlsVerified": identity_tls_verified,
            "renewableCredentialsProven": renewable_credentials_proven,
            "tokenMintCount": identity_token_mint_count,
            "reaped": identity_authority_counts[2] == 0,
        },
        "modelTransportProof": model_transport_proof,
        "processGate": {
            "globalGraphOs": counts(global_counts),
            "candidateGraphOs": counts(graph_os_counts),
            "candidateEngine": counts(engine_counts),
            "terminalProcessCounts": {
                "langfuseMcpChildren": terminal_process_counts[0],
                "loopbackOidcFixtures": terminal_process_counts[1],
            },
            "engineExecutableDigest": engine_executable_digest,
            "installedReleaseAttested": installed_release_attested,
            "reaped": reaped,
        },
        "validation": {
            "exitCode": validator_exit_code,
            "evidenceDigest": validation_evidence_digest,
            "caseCount": validation_case_count,
        },
        "result": "pass" if passed else "fail",
        "errorCode": error_code,
        "privacy": {
            "containsEndpoints": False,
            "containsCredentials": False,
            "containsProfiles": False,
            "containsFilesystemLocations": False,
            "containsIdentities": False,
            "containsContent": False,
        },
    }


def run_deployment(
    deployment: SkillValidationDeployment,
    *,
    deployment_path: Path,
    report_path: Path,
    validation_evidence_path: Path,
    lifecycle_evidence_path: Path,
) -> int:
    """Execute one exact zero/one/zero lifecycle and publish signed evidence."""

    _validate_evidence_destinations(
        (report_path, validation_evidence_path, lifecycle_evidence_path)
    )
    from agent_utilities.deployment.skill_validation_assets import (
        attest_installed_release_binding,
        load_runtime_materials,
        prove_model_registry_runtime,
        verify_release_bindings,
    )

    # No service can start until every supplied digest has been independently
    # recomputed and the exact promotion subject has been externally verified.
    promotion_evidence = verify_release_bindings(deployment)
    runtime_materials = load_runtime_materials(
        deployment, require_active_configuration=True
    )
    start_argv = _external_command(deployment.release.start_command_reference)
    start_executable = _regular_executable(Path(start_argv[0]), name="graph-os")
    start_argv = [str(start_executable), *start_argv[1:]]
    if _file_digest(start_executable) != deployment.release.graph_os_digest:
        raise DeploymentError("graph_os_digest_mismatch")
    attest_installed_release_binding(
        deployment,
        start_executable=start_executable,
        promotion_evidence=promotion_evidence,
    )
    installed_release_attested = True
    validator = _regular_executable(
        start_executable.with_name("agent-utilities-validate-skills"),
        name="agent-utilities-validate-skills",
    )
    readiness_executable = _regular_executable(
        start_executable.with_name("graph-os-skill-readiness"),
        name="graph-os-skill-readiness",
    )
    _runtime_reference(deployment.runtime.endpoint_reference)
    # Resolve both external signature commands before a service can be started.
    _external_command(deployment.validation.signer_command_reference)
    _external_command(deployment.validation.verifier_command_reference)

    marker = secrets.token_hex(32)
    before = _process_counts(marker)
    before_global = before.global_graph_os
    before_graph_os = before.candidate_graph_os
    before_engine = before.candidate_engine
    running_global = 0
    running_graph_os = 0
    running_engine = 0
    identity_authority_before = 0
    identity_authority_running = 0
    identity_authority_after = 0
    after_global = before_global
    after_graph_os = before_graph_os
    after_engine = before_engine
    terminal_process_counts = (
        before.langfuse_mcp_children,
        before.loopback_oidc_fixtures,
    )
    reaped = False
    identity_tls_verified = False
    renewable_credentials_proven = False
    identity_token_mint_count = 0
    model_transport_proof: dict[str, Any] = {
        "modelCount": 2,
        "literalPrivateModelCount": (
            deployment.runtime.model_registry.literal_private_model_count
        ),
        "privateDnsModelCount": deployment.runtime.model_registry.private_dns_model_count,
        "privateDnsUniqueResolutionProven": False,
        "privateBoundaryProven": False,
        "dnsRebindingGuarded": False,
    }
    engine_executable_digest: str | None = None
    validator_exit_code: int | None = None
    validation_digest: str | None = None
    validation_case_count = 0
    error_code: str | None = None
    process: subprocess.Popen[bytes] | None = None
    authority = EphemeralLoopbackOidcAuthority(
        token_ttl_seconds=deployment.identity_authority.token_ttl_seconds
    )
    environment = dict(os.environ)
    environment[_MARKER_ENV] = marker
    environment[_ENGINE_MARKER_ENV] = marker
    environment[_PROFILE_ENV] = (
        "profile:" + deployment.runtime.profile_digest.removeprefix("sha256:")
    )
    try:
        if (before_global, before_graph_os, before_engine) != (
            0,
            0,
            0,
        ) or terminal_process_counts != (0, 0):
            raise DeploymentError("process_gate_preexisting")
        model_private_hosts = runtime_materials.get("modelPrivateHosts")
        models = runtime_materials.get("models")
        if (
            not isinstance(model_private_hosts, list)
            or any(not isinstance(host, str) for host in model_private_hosts)
            or not isinstance(models, list)
        ):
            raise DeploymentError("runtime_model_registry_invalid")
        model_transport_proof = prove_model_registry_runtime(
            models, model_private_hosts
        )
        authority.start()
        identity_authority_running = 1 if authority.running else 0
        identity_tls_verified = authority.tls_verified
        if identity_authority_running != 1 or not identity_tls_verified:
            raise DeploymentError("identity_authority_not_ready")
        environment = authority.child_environment(
            environment,
            model_private_hosts=model_private_hosts,
        )
        process = subprocess.Popen(
            start_argv,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            close_fds=True,
            start_new_session=True,
            env=environment,
        )
        _wait_until_ready(
            process,
            readiness_executable=readiness_executable,
            deployment_path=deployment_path,
            environment=environment,
            timeout_seconds=deployment.readiness.timeout_seconds,
            poll_interval_milliseconds=deployment.readiness.poll_interval_milliseconds,
        )
        running = _process_counts(marker)
        running_global = running.global_graph_os
        running_graph_os = running.candidate_graph_os
        running_engine = running.candidate_engine
        if (running_global, running_graph_os, running_engine) != (
            1,
            1,
            1,
        ) or process.poll() is not None:
            raise DeploymentError("candidate_process_count_invalid")
        engine_executable_digest = _marked_engine_digest(marker)
        if engine_executable_digest != deployment.release.engine_digest:
            raise DeploymentError("candidate_engine_digest_mismatch")
        completed = subprocess.run(
            [
                str(validator),
                "--mode",
                "all",
                "--case-timeout",
                str(deployment.validation.case_timeout_seconds),
                "--report",
                str(report_path),
                "--evidence",
                str(validation_evidence_path),
                "--release-id",
                deployment.release.id,
                "--release-specification-digest",
                deployment.release.specification_digest,
                "--promotion-evidence-digest",
                deployment.release.promotion_evidence_digest,
                "--graph-os-digest",
                deployment.release.graph_os_digest,
                "--engine-digest",
                deployment.release.engine_digest,
                "--runtime-config-digest",
                deployment.runtime.configuration_digest,
                "--runtime-profile-digest",
                deployment.runtime.profile_digest,
                "--model-registry-digest",
                deployment.runtime.model_registry.digest,
                "--signer-command-ref",
                deployment.validation.signer_command_reference,
                "--verifier-command-ref",
                deployment.validation.verifier_command_reference,
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=min(
                4 * 60 * 60,
                deployment.validation.case_timeout_seconds * 20 + 20 * 60 + 300,
            ),
            close_fds=True,
            env=environment,
        )
        validator_exit_code = completed.returncode
        if completed.returncode != 0:
            raise DeploymentError("skill_validator_failed")
        validation_payload = validation_evidence_path.read_bytes()
        if not 1 <= len(validation_payload) <= 8 * 1024 * 1024:
            raise DeploymentError("validation_evidence_size_invalid")
        validation_digest = _digest_bytes(validation_payload)
        try:
            validation_document = _json_without_duplicates(
                validation_payload.decode("utf-8")
            )
            validation_cases = validation_document.get("cases")
            validation_result = validation_document.get("result")
        except Exception as exc:
            raise DeploymentError("validation_evidence_invalid") from exc
        if (
            not isinstance(validation_cases, list)
            or len(validation_cases) != 20
            or not isinstance(validation_result, dict)
            or validation_result.get("status") != "pass"
        ):
            raise DeploymentError("validation_evidence_invalid")
        validation_case_count = 20
        renewable_credentials_proven = authority.prove_renewable()
        identity_token_mint_count = authority.token_mint_count
        if not renewable_credentials_proven or identity_token_mint_count < 2:
            raise DeploymentError("identity_authority_not_renewable")
    except DeploymentError as exc:
        error_code = str(exc)
    except Exception:
        error_code = "deployment_boundary_failed"
    finally:
        if process is not None:
            try:
                _stop_and_reap(process, deployment.shutdown.grace_seconds)
            except Exception:
                error_code = "candidate_reap_failed"
        try:
            _terminate_marked_engines(marker, deployment.shutdown.grace_seconds)
        except Exception:
            error_code = "candidate_engine_reap_failed"
        identity_tls_verified = identity_tls_verified or authority.tls_verified
        identity_token_mint_count = max(
            identity_token_mint_count, authority.token_mint_count
        )
        try:
            authority.stop()
        except Exception:
            error_code = "identity_authority_reap_failed"
        identity_authority_after = 1 if authority.running else 0
        after = _wait_for_terminal_process_gate(
            marker, deployment.shutdown.grace_seconds
        )
        after_global = after.global_graph_os
        after_graph_os = after.candidate_graph_os
        after_engine = after.candidate_engine
        terminal_process_counts = (
            after.langfuse_mcp_children,
            after.loopback_oidc_fixtures,
        )
        reaped = (
            (after_global, after_graph_os, after_engine) == (0, 0, 0)
            and identity_authority_after == 0
            and terminal_process_counts == (0, 0)
            and (process is None or process.poll() is not None)
        )
        if not reaped:
            error_code = (
                "identity_authority_reap_failed"
                if identity_authority_after != 0
                else (
                    "terminal_process_count_invalid"
                    if terminal_process_counts != (0, 0)
                    else "candidate_process_leaked"
                )
            )

    unsigned = _lifecycle_subject(
        deployment,
        global_counts=(before_global, running_global, after_global),
        graph_os_counts=(before_graph_os, running_graph_os, after_graph_os),
        engine_counts=(before_engine, running_engine, after_engine),
        identity_authority_counts=(
            identity_authority_before,
            identity_authority_running,
            identity_authority_after,
        ),
        terminal_process_counts=terminal_process_counts,
        identity_tls_verified=identity_tls_verified,
        renewable_credentials_proven=renewable_credentials_proven,
        identity_token_mint_count=identity_token_mint_count,
        model_transport_proof=model_transport_proof,
        engine_executable_digest=engine_executable_digest,
        installed_release_attested=installed_release_attested,
        reaped=reaped,
        validator_exit_code=validator_exit_code,
        validation_evidence_digest=validation_digest,
        validation_case_count=validation_case_count,
        error_code=error_code,
    )
    signed = sign_and_verify_evidence(
        unsigned,
        signer_reference=deployment.validation.signer_command_reference,
        verifier_reference=deployment.validation.verifier_command_reference,
    )
    publish_report(lifecycle_evidence_path, render_evidence(signed))
    return 0 if signed["result"] == "pass" else 1


def _arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="graph-os-certify-skills")
    parser.add_argument("--deployment", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--validation-evidence", type=Path, required=True)
    parser.add_argument("--lifecycle-evidence", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _arguments(argv)
    try:
        deployment = load_deployment(args.deployment)
        return run_deployment(
            deployment,
            deployment_path=args.deployment,
            report_path=args.report,
            validation_evidence_path=args.validation_evidence,
            lifecycle_evidence_path=args.lifecycle_evidence,
        )
    except Exception as exc:  # noqa: BLE001 - never expose runtime material
        print(json.dumps({"ok": False, "error": type(exc).__name__}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
