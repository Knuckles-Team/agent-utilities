"""Packaged assets for exact-release bundled-skill certification.

This module owns the current certification asset generator, the AgentConfig-aware
readiness probe, and independent signed-evidence verification.  Durable outputs
contain only release/runtime digests, aggregate counts, booleans, and environment
reference names.  Endpoint values, credentials, identities, content, commands,
profiles, and filesystem locations remain deployment-owned runtime material.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import ipaddress
import json
import math
import os
import re
import socket
import stat
from importlib.resources import files
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

from jsonschema import Draft202012Validator

from agent_utilities.core._env import setting
from agent_utilities.deployment.certification_oidc import (
    AUTHORITY_MODE,
    DEFAULT_TOKEN_TTL_SECONDS,
    validated_token_ttl_seconds,
)
from agent_utilities.release_catalogs import prebundled_skill_catalog_digest
from agent_utilities.skills.validation import SKILLS_ROOT

if TYPE_CHECKING:
    from agent_utilities.deployment.skill_validation import SkillValidationDeployment

_MAX_MATERIAL_BYTES = 4 * 1024 * 1024
_MAX_EVIDENCE_BYTES = 8 * 1024 * 1024
_REFERENCE = re.compile(r"^[A-Z][A-Z0-9_]{2,63}$")
_DIGEST = re.compile(r"^sha256:(?!0{64}$)[a-f0-9]{64}$")
_PRIVATE_MODEL_NETWORKS = (
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("fc00::/7"),
)


class CertificationAssetError(RuntimeError):
    """Path-free, content-free fail-closed certification error."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _digest(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _read_regular(path: Path, *, limit: int, code: str) -> bytes:
    if not path.is_absolute():
        raise CertificationAssetError(code)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise CertificationAssetError(code) from exc
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or not 1 <= metadata.st_size <= limit
        ):
            raise CertificationAssetError(code)
        before = (
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_size,
            metadata.st_mtime_ns,
            metadata.st_ctime_ns,
        )
        chunks: list[bytes] = []
        remaining = limit + 1
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        payload = b"".join(chunks)
        after = os.fstat(descriptor)
        if (
            len(payload) != metadata.st_size
            or len(payload) > limit
            or before
            != (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            )
        ):
            raise CertificationAssetError(code)
        try:
            path_metadata = path.stat(follow_symlinks=False)
        except OSError as exc:
            raise CertificationAssetError(code) from exc
        if (
            path_metadata.st_dev,
            path_metadata.st_ino,
            path_metadata.st_size,
            path_metadata.st_mtime_ns,
            path_metadata.st_ctime_ns,
        ) != before:
            raise CertificationAssetError(code)
        return payload
    finally:
        os.close(descriptor)


def _hash_regular(path: Path, *, limit: int, code: str) -> str:
    if not path.is_absolute():
        raise CertificationAssetError(code)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise CertificationAssetError(code) from exc
    hasher = hashlib.sha256()
    consumed = 0
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or not 1 <= metadata.st_size <= limit
        ):
            raise CertificationAssetError(code)
        before = (
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_size,
            metadata.st_mtime_ns,
            metadata.st_ctime_ns,
        )
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            consumed += len(chunk)
            if consumed > limit:
                raise CertificationAssetError(code)
            hasher.update(chunk)
        after = os.fstat(descriptor)
        if consumed != metadata.st_size or before != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise CertificationAssetError(code)
    finally:
        os.close(descriptor)
    try:
        path_metadata = path.stat(follow_symlinks=False)
    except OSError as exc:
        raise CertificationAssetError(code) from exc
    if (
        path_metadata.st_dev,
        path_metadata.st_ino,
        path_metadata.st_size,
        path_metadata.st_mtime_ns,
        path_metadata.st_ctime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ):
        raise CertificationAssetError(code)
    return "sha256:" + hasher.hexdigest()


def _json_without_duplicates(payload: bytes, *, code: str) -> Any:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in items:
            if key in value:
                raise CertificationAssetError(code)
            value[key] = item
        return value

    try:
        return json.loads(payload, object_pairs_hook=pairs)
    except CertificationAssetError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CertificationAssetError(code) from exc


def _material_path(reference: str, *, code: str) -> Path:
    if _REFERENCE.fullmatch(reference) is None:
        raise CertificationAssetError(code)
    raw = str(setting(reference, "") or "")
    if not raw or "\x00" in raw or len(raw.encode("utf-8")) > 4_096:
        raise CertificationAssetError(code)
    path = Path(raw)
    if not path.is_absolute():
        raise CertificationAssetError(code)
    return path


def _model_configuration(root: Any) -> tuple[list[Any], list[str]]:
    if not isinstance(root, dict):
        raise CertificationAssetError("runtime_configuration_invalid")
    from agent_utilities.core.config import ChatModelConfig

    model_keys = [key for key in ("CHAT_MODELS", "chat_models") if key in root]
    host_keys = [
        key
        for key in (
            "MODEL_HTTP_ALLOWED_PRIVATE_HOSTS",
            "model_http_allowed_private_hosts",
        )
        if key in root
    ]
    if len(model_keys) != 1 or len(host_keys) != 1:
        raise CertificationAssetError("runtime_model_registry_missing")
    raw_models = root[model_keys[0]]
    raw_hosts = root[host_keys[0]]
    if (
        not isinstance(raw_models, list)
        or not isinstance(raw_hosts, list)
        or any(not isinstance(host, str) for host in raw_hosts)
    ):
        raise CertificationAssetError("runtime_model_registry_invalid")
    try:
        models = [ChatModelConfig.model_validate(item) for item in raw_models]
    except Exception as exc:
        raise CertificationAssetError("runtime_model_registry_invalid") from exc
    return models, [str(host) for host in raw_hosts]


def _identity_authority_configuration(root: Any) -> dict[str, Any]:
    """Resolve the current-only lifecycle authority controls from AgentConfig."""

    if not isinstance(root, dict):
        raise CertificationAssetError("runtime_configuration_invalid")
    mode_keys = [
        key
        for key in (
            "SKILL_CERT_IDENTITY_AUTHORITY_MODE",
            "skill_cert_identity_authority_mode",
        )
        if key in root
    ]
    ttl_keys = [
        key
        for key in (
            "SKILL_CERT_IDENTITY_TOKEN_TTL_SECONDS",
            "skill_cert_identity_token_ttl_seconds",
        )
        if key in root
    ]
    if len(mode_keys) > 1 or len(ttl_keys) > 1:
        raise CertificationAssetError("runtime_identity_authority_invalid")
    mode = root[mode_keys[0]] if mode_keys else AUTHORITY_MODE
    ttl = root[ttl_keys[0]] if ttl_keys else DEFAULT_TOKEN_TTL_SECONDS
    if mode != AUTHORITY_MODE:
        raise CertificationAssetError("runtime_identity_authority_invalid")
    try:
        token_ttl_seconds = validated_token_ttl_seconds(ttl)
    except Exception as exc:
        raise CertificationAssetError("runtime_identity_authority_invalid") from exc
    return {
        "mode": AUTHORITY_MODE,
        "tokenTtlSeconds": token_ttl_seconds,
        "tlsVerificationRequired": True,
        "lifecycleOwned": True,
        "renewableCredentialsRequired": True,
    }


def derive_model_registry_proof(
    models: list[Any], allowed_private_hosts: list[str]
) -> dict[str, Any]:
    """Derive a privacy-safe proof for the exact light/normal model registry."""

    allowed = {
        str(host).strip().casefold().rstrip(".") for host in allowed_private_hosts
    }
    if len(models) != 2 or len(allowed) > 256:
        raise CertificationAssetError("runtime_model_registry_cardinality_invalid")
    canonical: list[dict[str, str]] = []
    counts = {"light": 0, "normal": 0}
    literal_private_model_count = 0
    private_dns_model_count = 0
    model_ids: set[str] = set()
    for model in models:
        model_id = str(getattr(model, "id", "") or "")
        level = str(getattr(model, "intelligence_level", "") or "").casefold()
        if not model_id or model_id in model_ids or level not in counts:
            raise CertificationAssetError("runtime_model_registry_class_invalid")
        model_ids.add(model_id)
        counts[level] += 1
        try:
            parsed = urlsplit(str(getattr(model, "base_url", "") or ""))
            host = str(parsed.hostname or "").casefold().rstrip(".")
            port = parsed.port
        except ValueError as exc:
            raise CertificationAssetError("runtime_model_transport_invalid") from exc
        if (
            parsed.scheme.casefold() not in {"http", "https"}
            or not host
            or (port is not None and not 1 <= port <= 65_535)
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
            or host not in allowed
        ):
            raise CertificationAssetError("runtime_model_transport_invalid")
        try:
            address = ipaddress.ip_address(host)
        except ValueError:
            locality = "private-dns-runtime-pinned"
            private_dns_model_count += 1
        else:
            if address.is_loopback:
                locality = "loopback"
            elif any(address in network for network in _PRIVATE_MODEL_NETWORKS):
                locality = "private"
            else:
                raise CertificationAssetError("runtime_model_locality_unproven")
            literal_private_model_count += 1
        auth_modes: list[str] = []
        if getattr(model, "api_key_ref", None):
            auth_modes.append("api-key-reference")
        if getattr(model, "oauth2", None):
            auth_modes.append("oauth2-secret-reference")
        if not auth_modes:
            raise CertificationAssetError("runtime_model_credentials_unreferenced")
        if getattr(model, "headers_ref", None):
            auth_modes.append("supplemental-header-reference")
        canonical.append(
            {
                "modelIdentityDigest": _digest(model_id.encode("utf-8")),
                "class": level,
                "transport": locality,
                "scheme": parsed.scheme.casefold(),
                "authentication": "+".join(sorted(auth_modes)),
            }
        )
    if counts != {"light": 1, "normal": 1}:
        raise CertificationAssetError("runtime_model_registry_class_invalid")
    return {
        "digest": _digest(
            _canonical_bytes(sorted(canonical, key=lambda item: item["class"]))
        ),
        "modelCount": 2,
        "lightCount": 1,
        "normalCount": 1,
        "localPrivateTransportOnly": True,
        "referenceBackedCredentialsOnly": True,
        "literalPrivateModelCount": literal_private_model_count,
        "privateDnsModelCount": private_dns_model_count,
        "runtimePrivateResolutionRequired": True,
    }


def prove_model_registry_runtime(
    models: list[Any],
    allowed_private_hosts: list[str],
    *,
    resolver: Any = None,
) -> dict[str, Any]:
    """Prove private DNS once; request transports independently pin and recheck it."""

    proof = derive_model_registry_proof(models, allowed_private_hosts)
    allowed = {
        str(host).strip().casefold().rstrip(".") for host in allowed_private_hosts
    }
    private_dns_count = 0
    literal_count = 0
    getaddrinfo = resolver or socket.getaddrinfo
    for model in models:
        try:
            parsed = urlsplit(str(getattr(model, "base_url", "") or ""))
            host = str(parsed.hostname or "").casefold().rstrip(".")
            port = parsed.port
        except ValueError as exc:
            raise CertificationAssetError("runtime_model_transport_invalid") from exc
        try:
            address = ipaddress.ip_address(host)
        except ValueError:
            if not host or host not in allowed:
                raise CertificationAssetError(
                    "runtime_model_private_dns_unproven"
                ) from None
            try:
                answers = getaddrinfo(host, port, type=socket.SOCK_STREAM)
            except OSError as exc:
                raise CertificationAssetError(
                    "runtime_model_private_dns_unproven"
                ) from exc
            resolved: set[str] = set()
            for answer_count, answer in enumerate(answers, start=1):
                if answer_count > 64:
                    raise CertificationAssetError(
                        "runtime_model_private_dns_unproven"
                    ) from None
                try:
                    resolved.add(ipaddress.ip_address(str(answer[4][0])).compressed)
                except (IndexError, TypeError, ValueError) as exc:
                    raise CertificationAssetError(
                        "runtime_model_private_dns_unproven"
                    ) from exc
            if len(resolved) != 1:
                raise CertificationAssetError(
                    "runtime_model_private_dns_unproven"
                ) from None
            resolved_address = ipaddress.ip_address(next(iter(resolved)))
            if not (
                resolved_address.is_loopback
                or any(
                    resolved_address in network for network in _PRIVATE_MODEL_NETWORKS
                )
            ):
                raise CertificationAssetError(
                    "runtime_model_private_dns_unproven"
                ) from None
            private_dns_count += 1
        else:
            if not (
                address.is_loopback
                or any(address in network for network in _PRIVATE_MODEL_NETWORKS)
            ):
                raise CertificationAssetError("runtime_model_locality_unproven")
            literal_count += 1
    if (
        literal_count != proof["literalPrivateModelCount"]
        or private_dns_count != proof["privateDnsModelCount"]
    ):
        raise CertificationAssetError("runtime_model_transport_proof_mismatch")
    return {
        "modelCount": 2,
        "literalPrivateModelCount": literal_count,
        "privateDnsModelCount": private_dns_count,
        "privateDnsUniqueResolutionProven": True,
        "privateBoundaryProven": True,
        "dnsRebindingGuarded": True,
    }


def _validate_profile(
    payload: bytes,
    *,
    configuration_digest: str,
    model_registry_digest: str,
    identity_authority: dict[str, Any],
) -> None:
    value = _json_without_duplicates(payload, code="runtime_profile_invalid")
    expected = {
        "apiVersion": "graphos.io/v2",
        "kind": "SkillValidationRuntimeProfile",
        "configurationDigest": configuration_digest,
        "modelRegistryDigest": model_registry_digest,
        "identityAuthority": identity_authority,
        "engineTopology": "local-autostart",
        "observability": "metadata-only",
        "sequential": True,
    }
    if value != expected:
        raise CertificationAssetError("runtime_profile_invalid")


def _configuration_proof(payload: bytes) -> dict[str, Any]:
    root = _json_without_duplicates(payload, code="runtime_configuration_invalid")
    models, allowed = _model_configuration(root)
    return derive_model_registry_proof(models, allowed)


def _runtime_profile_document(configuration: bytes) -> dict[str, Any]:
    proof = _configuration_proof(configuration)
    identity_authority = _identity_authority_configuration(
        _json_without_duplicates(configuration, code="runtime_configuration_invalid")
    )
    return {
        "apiVersion": "graphos.io/v2",
        "kind": "SkillValidationRuntimeProfile",
        "configurationDigest": _digest(configuration),
        "modelRegistryDigest": proof["digest"],
        "identityAuthority": identity_authority,
        "engineTopology": "local-autostart",
        "observability": "metadata-only",
        "sequential": True,
    }


def generate_runtime_profile(
    *, configuration_reference: str, profile_reference: str
) -> dict[str, Any]:
    """Publish the deterministic runtime profile resolved through AgentConfig."""

    configuration_path = _material_path(
        configuration_reference,
        code="runtime_configuration_reference_invalid",
    )
    profile_path = _material_path(
        profile_reference,
        code="runtime_profile_reference_invalid",
    )
    configuration = _read_regular(
        configuration_path,
        limit=_MAX_MATERIAL_BYTES,
        code="runtime_configuration_invalid",
    )
    try:
        if configuration_path.resolve(strict=True) == profile_path.resolve(
            strict=False
        ):
            raise CertificationAssetError("runtime_profile_destination_invalid")
    except CertificationAssetError:
        raise
    except OSError as exc:
        raise CertificationAssetError("runtime_profile_destination_invalid") from exc
    profile = _runtime_profile_document(configuration)
    rendered = json.dumps(profile, sort_keys=True, indent=2) + "\n"
    from agent_utilities.skills.runtime_validation import publish_report

    publish_report(profile_path, rendered)
    if _read_regular(
        profile_path,
        limit=_MAX_MATERIAL_BYTES,
        code="runtime_profile_invalid",
    ) != rendered.encode("utf-8"):
        raise CertificationAssetError("runtime_profile_invalid")
    return profile


def load_runtime_materials(
    deployment: SkillValidationDeployment, *, require_active_configuration: bool
) -> dict[str, Any]:
    """Recompute and cross-verify exact runtime configuration/profile material."""

    configuration_path = _material_path(
        deployment.runtime.configuration_reference,
        code="runtime_configuration_reference_invalid",
    )
    profile_path = _material_path(
        deployment.runtime.profile_reference,
        code="runtime_profile_reference_invalid",
    )
    configuration = _read_regular(
        configuration_path,
        limit=_MAX_MATERIAL_BYTES,
        code="runtime_configuration_invalid",
    )
    profile = _read_regular(
        profile_path, limit=_MAX_MATERIAL_BYTES, code="runtime_profile_invalid"
    )
    if _digest(configuration) != deployment.runtime.configuration_digest:
        raise CertificationAssetError("runtime_configuration_digest_mismatch")
    if _digest(profile) != deployment.runtime.profile_digest:
        raise CertificationAssetError("runtime_profile_digest_mismatch")
    proof = _configuration_proof(configuration)
    root = _json_without_duplicates(configuration, code="runtime_configuration_invalid")
    models, model_private_hosts = _model_configuration(root)
    identity_authority = _identity_authority_configuration(root)
    expected_proof = deployment.runtime.model_registry.model_dump(by_alias=True)
    if proof != expected_proof:
        raise CertificationAssetError("runtime_model_registry_digest_mismatch")
    _validate_profile(
        profile,
        configuration_digest=deployment.runtime.configuration_digest,
        model_registry_digest=proof["digest"],
        identity_authority=identity_authority,
    )
    if identity_authority != deployment.identity_authority.model_dump(by_alias=True):
        raise CertificationAssetError("runtime_identity_authority_mismatch")
    if require_active_configuration:
        from agent_utilities.core.paths import config_dir

        active = config_dir() / "config.json"
        try:
            if not os.path.samefile(configuration_path, active):
                raise CertificationAssetError("runtime_configuration_not_active")
        except OSError as exc:
            raise CertificationAssetError("runtime_configuration_not_active") from exc
    return {
        "modelRegistry": proof,
        "identityAuthority": identity_authority,
        "models": models,
        "modelPrivateHosts": model_private_hosts,
    }


def verify_release_bindings(
    deployment: SkillValidationDeployment,
) -> dict[str, Any]:
    """Verify spec, signed promotion evidence, and installed artifact digests."""

    specification_path = _material_path(
        deployment.release.specification_reference,
        code="release_specification_reference_invalid",
    )
    evidence_path = _material_path(
        deployment.release.promotion_evidence_reference,
        code="promotion_evidence_reference_invalid",
    )
    specification = _read_regular(
        specification_path,
        limit=_MAX_MATERIAL_BYTES,
        code="release_specification_invalid",
    )
    evidence_payload = _read_regular(
        evidence_path,
        limit=_MAX_EVIDENCE_BYTES,
        code="promotion_evidence_invalid",
    )
    if _digest(specification) != deployment.release.specification_digest:
        raise CertificationAssetError("release_specification_digest_mismatch")
    if _digest(evidence_payload) != deployment.release.promotion_evidence_digest:
        raise CertificationAssetError("promotion_evidence_digest_mismatch")
    try:
        from scripts.release.promote_local_release import verify_evidence_file

        evidence = verify_evidence_file(
            spec_path=specification_path,
            release_id=deployment.release.id,
            evidence_path=evidence_path,
        )
    except Exception as exc:
        raise CertificationAssetError("promotion_evidence_verification_failed") from exc
    certification = evidence.get("certificationArtifacts")
    if evidence.get("status") != "promoted" or not isinstance(certification, dict):
        raise CertificationAssetError("promotion_evidence_not_promoted")
    if evidence.get("specDigest") != deployment.release.specification_digest:
        raise CertificationAssetError("promotion_specification_binding_mismatch")
    if _promotion_certification_binding(certification) != {
        "agentUtilitiesSha256": deployment.release.agent_utilities_sha256,
        "agentUtilitiesFileCount": deployment.release.agent_utilities_file_count,
        "distributionClosureSha256": deployment.release.distribution_closure_sha256,
        "releasePythonSha256": deployment.release.release_python_sha256,
        "graphOsDigest": deployment.release.graph_os_digest,
        "engineDigest": deployment.release.engine_digest,
    }:
        raise CertificationAssetError("promotion_artifact_binding_mismatch")
    return evidence


def _promotion_certification_binding(
    certification: dict[str, Any],
) -> dict[str, Any]:
    count = certification.get("agentUtilitiesFileCount")
    if isinstance(count, bool) or not isinstance(count, int) or count < 10:
        raise CertificationAssetError("promotion_artifact_binding_invalid")
    raw_fields = {
        "agentUtilitiesSha256": "agentUtilitiesSha256",
        "distributionClosureSha256": "distributionClosureSha256",
        "releasePythonSha256": "releasePythonSha256",
        "graphOsDigest": "graphosSha256",
        "engineDigest": "engineSha256",
    }
    binding = {
        target: "sha256:" + str(certification.get(source) or "")
        for target, source in raw_fields.items()
    }
    if any(_DIGEST.fullmatch(value) is None for value in binding.values()):
        raise CertificationAssetError("promotion_artifact_binding_invalid")
    return {**binding, "agentUtilitiesFileCount": count}


def attest_installed_release_binding(
    deployment: SkillValidationDeployment,
    *,
    start_executable: Path,
    promotion_evidence: dict[str, Any],
) -> dict[str, Any]:
    """Recompute the exact sealed release containing the selected GraphOS."""

    try:
        original = start_executable.lstat()
        canonical = start_executable.resolve(strict=True)
        canonical_metadata = canonical.lstat()
    except OSError as exc:
        raise CertificationAssetError("installed_release_layout_invalid") from exc
    if (
        not start_executable.is_absolute()
        or stat.S_ISLNK(original.st_mode)
        or not stat.S_ISREG(original.st_mode)
        or original.st_nlink != 1
        or stat.S_ISLNK(canonical_metadata.st_mode)
        or not stat.S_ISREG(canonical_metadata.st_mode)
        or canonical_metadata.st_nlink != 1
        or (original.st_dev, original.st_ino)
        != (canonical_metadata.st_dev, canonical_metadata.st_ino)
        or canonical.name != "graph-os"
        or canonical.parent.name != "bin"
        or canonical.parent.parent.name != "runtime"
    ):
        raise CertificationAssetError("installed_release_layout_invalid")
    release_root = canonical.parent.parent.parent
    certification = promotion_evidence.get("certificationArtifacts")
    if not isinstance(certification, dict):
        raise CertificationAssetError("promotion_evidence_not_promoted")
    expected = _promotion_certification_binding(certification)
    try:
        from scripts.release.promote_local_release import attest_installed_release

        installed = attest_installed_release(release_root)
    except Exception as exc:
        raise CertificationAssetError("installed_release_attestation_failed") from exc
    try:
        after = start_executable.stat(follow_symlinks=False)
    except OSError as exc:
        raise CertificationAssetError("installed_release_attestation_failed") from exc
    if (
        original.st_dev,
        original.st_ino,
        original.st_size,
        original.st_mtime_ns,
        original.st_ctime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ):
        raise CertificationAssetError("installed_release_attestation_failed")
    actual = _promotion_certification_binding(installed)
    deployment_binding = {
        "agentUtilitiesSha256": deployment.release.agent_utilities_sha256,
        "agentUtilitiesFileCount": deployment.release.agent_utilities_file_count,
        "distributionClosureSha256": deployment.release.distribution_closure_sha256,
        "releasePythonSha256": deployment.release.release_python_sha256,
        "graphOsDigest": deployment.release.graph_os_digest,
        "engineDigest": deployment.release.engine_digest,
    }
    if actual != expected or actual != deployment_binding:
        raise CertificationAssetError("installed_release_attestation_mismatch")
    return actual


def generate_deployment(
    *,
    release_id: str,
    release_specification: Path,
    promotion_evidence: Path,
    runtime_configuration: Path,
    runtime_profile: Path,
    specification_reference: str,
    promotion_evidence_reference: str,
    configuration_reference: str,
    profile_reference: str,
    endpoint_reference: str,
    start_command_reference: str,
    signer_command_reference: str,
    verifier_command_reference: str,
    readiness_timeout_seconds: int,
    poll_interval_milliseconds: int,
    case_timeout_seconds: int,
    shutdown_grace_seconds: int,
) -> SkillValidationDeployment:
    """Generate one closed deployment from exact external release/runtime inputs."""

    from agent_utilities.deployment.skill_validation import SkillValidationDeployment
    from agent_utilities.skills.runtime_validation import _external_command

    specification = _read_regular(
        release_specification,
        limit=_MAX_MATERIAL_BYTES,
        code="release_specification_invalid",
    )
    evidence_payload = _read_regular(
        promotion_evidence,
        limit=_MAX_EVIDENCE_BYTES,
        code="promotion_evidence_invalid",
    )
    configuration = _read_regular(
        runtime_configuration,
        limit=_MAX_MATERIAL_BYTES,
        code="runtime_configuration_invalid",
    )
    profile = _read_regular(
        runtime_profile, limit=_MAX_MATERIAL_BYTES, code="runtime_profile_invalid"
    )
    configuration_digest = _digest(configuration)
    proof = _configuration_proof(configuration)
    identity_authority = _identity_authority_configuration(
        _json_without_duplicates(configuration, code="runtime_configuration_invalid")
    )
    _validate_profile(
        profile,
        configuration_digest=configuration_digest,
        model_registry_digest=proof["digest"],
        identity_authority=identity_authority,
    )
    try:
        from scripts.release.promote_local_release import verify_evidence_file

        evidence = verify_evidence_file(
            spec_path=release_specification,
            release_id=release_id,
            evidence_path=promotion_evidence,
        )
    except Exception as exc:
        raise CertificationAssetError("promotion_evidence_verification_failed") from exc
    certification = evidence.get("certificationArtifacts")
    if evidence.get("status") != "promoted" or not isinstance(certification, dict):
        raise CertificationAssetError("promotion_evidence_not_promoted")
    release_binding = _promotion_certification_binding(certification)
    graph_os_digest = release_binding["graphOsDigest"]
    engine_digest = release_binding["engineDigest"]
    start_argv = _external_command(start_command_reference)
    try:
        executable = Path(start_argv[0]).resolve(strict=True)
    except OSError as exc:
        raise CertificationAssetError("graph_os_executable_invalid") from exc
    if executable.name != "graph-os":
        raise CertificationAssetError("graph_os_executable_invalid")
    if (
        _hash_regular(
            executable,
            limit=2 * 1024 * 1024 * 1024,
            code="graph_os_executable_invalid",
        )
        != graph_os_digest
    ):
        raise CertificationAssetError("graph_os_digest_mismatch")
    deployment = SkillValidationDeployment.model_validate(
        {
            "apiVersion": "graphos.io/v2",
            "kind": "SkillValidationDeployment",
            "identityAuthority": identity_authority,
            "release": {
                "id": release_id,
                "specificationReference": specification_reference,
                "specificationDigest": _digest(specification),
                "promotionEvidenceReference": promotion_evidence_reference,
                "promotionEvidenceDigest": _digest(evidence_payload),
                "agentUtilitiesSha256": release_binding["agentUtilitiesSha256"],
                "agentUtilitiesFileCount": release_binding["agentUtilitiesFileCount"],
                "distributionClosureSha256": release_binding[
                    "distributionClosureSha256"
                ],
                "releasePythonSha256": release_binding["releasePythonSha256"],
                "graphOsDigest": graph_os_digest,
                "engineDigest": engine_digest,
                "startCommandReference": start_command_reference,
            },
            "runtime": {
                "configurationReference": configuration_reference,
                "configurationDigest": configuration_digest,
                "profileReference": profile_reference,
                "profileDigest": _digest(profile),
                "endpointReference": endpoint_reference,
                "modelRegistry": proof,
            },
            "readiness": {
                "timeoutSeconds": readiness_timeout_seconds,
                "pollIntervalMilliseconds": poll_interval_milliseconds,
            },
            "validation": {
                "caseTimeoutSeconds": case_timeout_seconds,
                "signerCommandReference": signer_command_reference,
                "verifierCommandReference": verifier_command_reference,
            },
            "shutdown": {"graceSeconds": shutdown_grace_seconds},
        }
    )
    attest_installed_release_binding(
        deployment,
        start_executable=executable,
        promotion_evidence=evidence,
    )
    return deployment


async def probe_readiness(
    deployment: SkillValidationDeployment, *, request_timeout: float = 15.0
) -> None:
    """Prove active AgentConfig, TLS/auth, GraphOS tools, and a local engine."""

    if (
        isinstance(request_timeout, bool)
        or not isinstance(request_timeout, int | float)
        or not math.isfinite(float(request_timeout))
        or not 0.1 <= float(request_timeout) <= 120.0
    ):
        raise CertificationAssetError("readiness_timeout_invalid")
    runtime_materials = load_runtime_materials(
        deployment, require_active_configuration=True
    )
    runtime_transport_proof = prove_model_registry_runtime(
        runtime_materials["models"], runtime_materials["modelPrivateHosts"]
    )
    if (
        runtime_transport_proof["literalPrivateModelCount"]
        != deployment.runtime.model_registry.literal_private_model_count
        or runtime_transport_proof["privateDnsModelCount"]
        != deployment.runtime.model_registry.private_dns_model_count
    ):
        raise CertificationAssetError("runtime_model_transport_proof_mismatch")
    endpoint = str(setting(deployment.runtime.endpoint_reference, "") or "").strip()
    try:
        parsed = urlsplit(endpoint)
        host = str(parsed.hostname or "").casefold().rstrip(".")
        port = parsed.port
    except ValueError as exc:
        raise CertificationAssetError("graph_os_endpoint_not_local") from exc
    try:
        address = ipaddress.ip_address(host)
        loopback = address.is_loopback
    except ValueError:
        loopback = host in {"localhost", "localhost.localdomain"}
    if (
        parsed.scheme.casefold() not in {"http", "https"}
        or not loopback
        or (port is not None and not 1 <= port <= 65_535)
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise CertificationAssetError("graph_os_endpoint_not_local")

    from agent_utilities.core.config import config
    from agent_utilities.core.transport_security import resolve_configured_tls_profile
    from agent_utilities.knowledge_graph.core.engine_resolver import resolve_engine
    from agent_utilities.knowledge_graph.core.shard_topology import is_local_endpoint
    from agent_utilities.mcp.client_credentials import (
        child_auth,
        outbound_auth_configuration_status,
    )
    from agent_utilities.mcp.toolset_factory import build_http_toolset
    from agent_utilities.skills.runtime_validation import _call_tool, _ensure_tool

    if str(config.mcp_url or "").strip() != endpoint:
        raise CertificationAssetError("graph_os_endpoint_not_active")
    active_proof = derive_model_registry_proof(
        list(config.chat_models), list(config.model_http_allowed_private_hosts)
    )
    if active_proof != deployment.runtime.model_registry.model_dump(by_alias=True):
        raise CertificationAssetError("runtime_model_registry_not_active")
    auth_status = outbound_auth_configuration_status()
    if auth_status.get("ready") is not True:
        raise CertificationAssetError("graph_os_identity_not_ready")
    trust = resolve_configured_tls_profile("mcp", config=config)
    trust.cleanup()
    toolset = build_http_toolset(
        endpoint,
        auth=child_auth({}),
        timeout=request_timeout,
        toolset_id="skill-certification-readiness",
    )
    async with toolset.client as client:
        for tool in ("graph_orchestrate", "graph_query", "graph_jobs"):
            await _ensure_tool(client, tool, request_timeout)
        await _call_tool(
            client,
            "graph_query",
            {
                "cypher": "MATCH (n) RETURN n LIMIT 0",
                "params": "{}",
                "scope": "local",
            },
            request_timeout,
        )
    resolved = resolve_engine(config, "skill-validation-readiness")
    if resolved.mode == "remote" or not is_local_endpoint(resolved.endpoint):
        raise CertificationAssetError("engine_topology_not_local")


def _schema(name: str) -> dict[str, Any]:
    payload = files("deploy.release").joinpath(name).read_bytes()
    value = _json_without_duplicates(payload, code="certification_schema_invalid")
    if not isinstance(value, dict):
        raise CertificationAssetError("certification_schema_invalid")
    Draft202012Validator.check_schema(value)
    return value


def _signed_document(path: Path, *, code: str) -> tuple[dict[str, Any], bytes]:
    payload = _read_regular(path, limit=_MAX_EVIDENCE_BYTES, code=code)
    value = _json_without_duplicates(payload, code=code)
    if not isinstance(value, dict):
        raise CertificationAssetError(code)
    return value, payload


def verify_certification_documents(
    *,
    deployment_path: Path,
    validation_evidence_path: Path,
    lifecycle_evidence_path: Path,
) -> None:
    """Independently verify release/runtime bindings and both signed subjects."""

    from agent_utilities.deployment.skill_validation import load_deployment
    from agent_utilities.skills.runtime_validation import (
        _external_command,
        _test_catalog_evidence,
        load_matrix,
        verify_signed_evidence,
    )

    deployment = load_deployment(deployment_path)
    promotion_evidence = verify_release_bindings(deployment)
    load_runtime_materials(deployment, require_active_configuration=False)
    start_argv = _external_command(deployment.release.start_command_reference)
    attest_installed_release_binding(
        deployment,
        start_executable=Path(start_argv[0]),
        promotion_evidence=promotion_evidence,
    )
    validation, validation_payload = _signed_document(
        validation_evidence_path, code="validation_evidence_invalid"
    )
    lifecycle, _lifecycle_payload = _signed_document(
        lifecycle_evidence_path, code="lifecycle_evidence_invalid"
    )
    try:
        Draft202012Validator(
            _schema("prebundled-skill-validation-evidence.schema.json")
        ).validate(validation)
        Draft202012Validator(
            _schema("skill-validation-deployment-evidence.schema.json")
        ).validate(lifecycle)
    except Exception as exc:
        raise CertificationAssetError("certification_evidence_schema_invalid") from exc
    verify_signed_evidence(
        validation,
        verifier_reference=deployment.validation.verifier_command_reference,
    )
    verify_signed_evidence(
        lifecycle,
        verifier_reference=deployment.validation.verifier_command_reference,
    )
    expected_release = {
        "id": deployment.release.id,
        "specificationDigest": deployment.release.specification_digest,
        "promotionEvidenceDigest": deployment.release.promotion_evidence_digest,
        "agentUtilitiesSha256": deployment.release.agent_utilities_sha256,
        "agentUtilitiesFileCount": deployment.release.agent_utilities_file_count,
        "distributionClosureSha256": deployment.release.distribution_closure_sha256,
        "releasePythonSha256": deployment.release.release_python_sha256,
        "graphOsDigest": deployment.release.graph_os_digest,
        "engineDigest": deployment.release.engine_digest,
    }
    expected_runtime = {
        "configurationDigest": deployment.runtime.configuration_digest,
        "profileDigest": deployment.runtime.profile_digest,
        "modelRegistryDigest": deployment.runtime.model_registry.digest,
    }
    validation_expected_release = {
        key: value
        for key, value in expected_release.items()
        if key
        not in {
            "agentUtilitiesSha256",
            "agentUtilitiesFileCount",
            "distributionClosureSha256",
            "releasePythonSha256",
        }
    }
    if validation.get("release") != validation_expected_release:
        raise CertificationAssetError("validation_release_binding_mismatch")
    validation_runtime = validation.get("runtime")
    if not isinstance(validation_runtime, dict) or any(
        validation_runtime.get(key) != value for key, value in expected_runtime.items()
    ):
        raise CertificationAssetError("validation_runtime_binding_mismatch")
    _defaults, cases = load_matrix()
    catalogs = _test_catalog_evidence(cases)
    expected_catalog = {
        "skillCount": 10,
        "skillCatalogDigest": prebundled_skill_catalog_digest(SKILLS_ROOT),
        "testCaseCount": 20,
        "testCatalogDigest": catalogs["testCatalogDigest"],
        "caseCatalogDigest": catalogs["caseCatalogDigest"],
    }
    if validation.get("catalog") != expected_catalog:
        raise CertificationAssetError("validation_catalog_binding_mismatch")
    evidence_cases = validation.get("cases")
    expected_cases = {case.case_id: case for case in cases}
    if not isinstance(evidence_cases, list) or len(evidence_cases) != 20:
        raise CertificationAssetError("validation_case_binding_mismatch")
    for item in evidence_cases:
        if not isinstance(item, dict):
            raise CertificationAssetError("validation_case_binding_mismatch")
        case_id = str(item.get("caseId") or "")
        expected_case = expected_cases.get(case_id)
        if expected_case is None or (
            item.get("caseDigest") != catalogs["caseDigests"].get(case_id)
            or item.get("skill") != expected_case.skill
            or item.get("mode") != expected_case.mode
            or item.get("modelClass") != expected_case.model_class
            or item.get("status") != "pass"
        ):
            raise CertificationAssetError("validation_case_binding_mismatch")
    result = validation.get("result")
    if not isinstance(result, dict) or result.get("status") != "pass":
        raise CertificationAssetError("validation_result_failed")
    if (
        lifecycle.get("release") != expected_release
        or lifecycle.get("runtime") != expected_runtime
    ):
        raise CertificationAssetError("lifecycle_binding_mismatch")
    identity_authority = lifecycle.get("identityAuthority")
    if (
        not isinstance(identity_authority, dict)
        or identity_authority.get("mode") != deployment.identity_authority.mode
        or identity_authority.get("lifecycleCounts")
        != {"before": 0, "running": 1, "after": 0}
        or identity_authority.get("tlsVerified") is not True
        or identity_authority.get("renewableCredentialsProven") is not True
        or isinstance(identity_authority.get("tokenMintCount"), bool)
        or not isinstance(identity_authority.get("tokenMintCount"), int)
        or identity_authority["tokenMintCount"] < 2
        or identity_authority.get("reaped") is not True
    ):
        raise CertificationAssetError("lifecycle_identity_authority_mismatch")
    if lifecycle.get("modelTransportProof") != {
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
    }:
        raise CertificationAssetError("lifecycle_model_transport_proof_mismatch")
    process_gate = lifecycle.get("processGate")
    if (
        not isinstance(process_gate, dict)
        or process_gate.get("engineExecutableDigest")
        != deployment.release.engine_digest
    ):
        raise CertificationAssetError("lifecycle_engine_binding_mismatch")
    if process_gate.get("terminalProcessCounts") != {
        "langfuseMcpChildren": 0,
        "loopbackOidcFixtures": 0,
    }:
        raise CertificationAssetError("lifecycle_terminal_process_count_mismatch")
    lifecycle_validation = lifecycle.get("validation")
    if (
        not isinstance(lifecycle_validation, dict)
        or lifecycle_validation.get("evidenceDigest") != _digest(validation_payload)
        or lifecycle_validation.get("caseCount") != 20
        or lifecycle.get("result") != "pass"
    ):
        raise CertificationAssetError("lifecycle_validation_binding_mismatch")


def _generator_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="graph-os-generate-skill-certification")
    parser.add_argument("--release-id", required=True)
    parser.add_argument("--release-specification", type=Path, required=True)
    parser.add_argument("--promotion-evidence", type=Path, required=True)
    parser.add_argument("--runtime-configuration", type=Path, required=True)
    parser.add_argument("--runtime-profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--specification-reference", required=True)
    parser.add_argument("--promotion-evidence-reference", required=True)
    parser.add_argument("--configuration-reference", required=True)
    parser.add_argument("--profile-reference", required=True)
    parser.add_argument("--endpoint-reference", required=True)
    parser.add_argument("--start-command-reference", required=True)
    parser.add_argument("--signer-command-reference", required=True)
    parser.add_argument("--verifier-command-reference", required=True)
    parser.add_argument("--readiness-timeout-seconds", type=int, default=120)
    parser.add_argument("--poll-interval-milliseconds", type=int, default=250)
    parser.add_argument("--case-timeout-seconds", type=int, default=120)
    parser.add_argument("--shutdown-grace-seconds", type=int, default=30)
    return parser.parse_args(argv)


def _profile_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="graph-os-generate-skill-runtime-profile")
    parser.add_argument("--configuration-reference", required=True)
    parser.add_argument("--profile-reference", required=True)
    return parser.parse_args(argv)


def profile_main(argv: list[str] | None = None) -> int:
    args = _profile_arguments(argv)
    try:
        generate_runtime_profile(
            configuration_reference=args.configuration_reference,
            profile_reference=args.profile_reference,
        )
    except Exception as exc:  # noqa: BLE001 - never expose external material
        print(json.dumps({"ok": False, "error": type(exc).__name__}, sort_keys=True))
        return 1
    print(json.dumps({"ok": True}, sort_keys=True))
    return 0


def generator_main(argv: list[str] | None = None) -> int:
    args = _generator_arguments(argv)
    try:
        deployment = generate_deployment(
            release_id=args.release_id,
            release_specification=args.release_specification,
            promotion_evidence=args.promotion_evidence,
            runtime_configuration=args.runtime_configuration,
            runtime_profile=args.runtime_profile,
            specification_reference=args.specification_reference,
            promotion_evidence_reference=args.promotion_evidence_reference,
            configuration_reference=args.configuration_reference,
            profile_reference=args.profile_reference,
            endpoint_reference=args.endpoint_reference,
            start_command_reference=args.start_command_reference,
            signer_command_reference=args.signer_command_reference,
            verifier_command_reference=args.verifier_command_reference,
            readiness_timeout_seconds=args.readiness_timeout_seconds,
            poll_interval_milliseconds=args.poll_interval_milliseconds,
            case_timeout_seconds=args.case_timeout_seconds,
            shutdown_grace_seconds=args.shutdown_grace_seconds,
        )
        from agent_utilities.skills.runtime_validation import publish_report

        publish_report(
            args.output,
            json.dumps(deployment.model_dump(by_alias=True), sort_keys=True, indent=2)
            + "\n",
        )
    except Exception as exc:  # noqa: BLE001 - never expose external material
        print(json.dumps({"ok": False, "error": type(exc).__name__}, sort_keys=True))
        return 1
    print(json.dumps({"ok": True}, sort_keys=True))
    return 0


def readiness_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="graph-os-skill-readiness")
    parser.add_argument("--deployment", type=Path, required=True)
    parser.add_argument("--request-timeout", type=float, default=15.0)
    args = parser.parse_args(argv)
    try:
        from agent_utilities.deployment.skill_validation import load_deployment

        deployment = load_deployment(args.deployment)
        asyncio.run(probe_readiness(deployment, request_timeout=args.request_timeout))
    except Exception as exc:  # noqa: BLE001 - never expose external material
        print(json.dumps({"ready": False, "error": type(exc).__name__}, sort_keys=True))
        return 1
    print(json.dumps({"ready": True}, sort_keys=True))
    return 0


def verifier_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="graph-os-verify-skill-certification")
    parser.add_argument("--deployment", type=Path, required=True)
    parser.add_argument("--validation-evidence", type=Path, required=True)
    parser.add_argument("--lifecycle-evidence", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        verify_certification_documents(
            deployment_path=args.deployment,
            validation_evidence_path=args.validation_evidence,
            lifecycle_evidence_path=args.lifecycle_evidence,
        )
    except Exception as exc:  # noqa: BLE001 - never expose external material
        print(
            json.dumps({"verified": False, "error": type(exc).__name__}, sort_keys=True)
        )
        return 1
    print(json.dumps({"verified": True}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(generator_main())
