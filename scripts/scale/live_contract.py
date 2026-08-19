"""Fail-closed identity and artifact contract for live workload runs.

The mock load path is intentionally zero-infrastructure and remains useful for CI,
but its results can never be mistaken for production certification.  A live run
must carry an explicit, immutable identity for the release, deployment topology,
loadgen image, workload contract, and the tracked source authority that produced
the deployment manifest.  It must also prove that it is a client of a configured
remote engine with an authenticated process identity.

Only opaque digests and identity digests leave this module.  Endpoint values,
credential references, and tenant/principal labels are used for validation and are
never returned in workload evidence.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from typing import Mapping

from scripts.scale.loadgen_source_authority import (
    source_authority_digest as build_source_authority_digest,
)
from scripts.scale.workload_contract import WorkloadContract

_DIGEST = re.compile(r"^sha256:(?!0{64}$)[a-f0-9]{64}$")
_MAX_TEXT_BYTES = 4_096
_AUTH_SOURCES = (
    "KG_AUTH_TOKEN_REF",
    "KG_IDENTITY_OAUTH2",
    "GRAPH_SERVICE_AUTH_SECRET",
)
_TLS_PROFILE_SOURCES = ("ENGINE_TLS_PROFILE_REF", "ENGINE_TLS_PROFILE")


class LiveRuntimeContractError(RuntimeError):
    """The live load path is missing an authority, identity, or immutable pin."""


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _digest(value: object, field: str) -> str:
    text = str(value or "").strip()
    if not _DIGEST.fullmatch(text):
        raise LiveRuntimeContractError(f"live workload {field} must be a sha256 digest")
    return text


def _text(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise LiveRuntimeContractError(f"live workload {field} is missing or invalid")
    text = value.strip()
    if (
        not text
        or len(text.encode("utf-8")) > _MAX_TEXT_BYTES
        or any(character in text for character in "\x00\r\n")
    ):
        raise LiveRuntimeContractError(f"live workload {field} is missing or invalid")
    return text


def _env_value(
    environment: Mapping[str, str],
    name: str,
    override: str | None,
) -> str:
    configured = str(environment.get(name, "") or "").strip()
    if override is not None and configured and configured != str(override).strip():
        raise LiveRuntimeContractError(f"live workload {name} disagrees with its argv pin")
    return _text(override if override is not None else configured, name)


def _endpoints(environment: Mapping[str, str]) -> tuple[str, ...]:
    raw = str(environment.get("GRAPH_SERVICE_ENDPOINTS", "") or "").strip()
    if not raw:
        raise LiveRuntimeContractError(
            "live workload requires GRAPH_SERVICE_ENDPOINTS; local engine fallback is forbidden"
        )
    if raw.startswith("["):
        try:
            values = json.loads(raw)
        except (TypeError, ValueError) as exc:
            raise LiveRuntimeContractError(
                "GRAPH_SERVICE_ENDPOINTS is not a valid endpoint list"
            ) from exc
        if not isinstance(values, list):
            raise LiveRuntimeContractError("GRAPH_SERVICE_ENDPOINTS is not a list")
        endpoints = tuple(_text(value, "GRAPH_SERVICE_ENDPOINTS") for value in values)
    else:
        endpoints = tuple(
            _text(value, "GRAPH_SERVICE_ENDPOINTS")
            for value in raw.split(",")
            if str(value).strip()
        )
    if not endpoints:
        raise LiveRuntimeContractError("GRAPH_SERVICE_ENDPOINTS is empty")
    return endpoints


def _authority_digest(
    endpoints: tuple[str, ...], environment: Mapping[str, str]
) -> str:
    auth_sources = tuple(
        name for name in _AUTH_SOURCES if str(environment.get(name, "") or "").strip()
    )
    if len(auth_sources) != 1:
        raise LiveRuntimeContractError(
            "live workload requires exactly one engine authentication source "
            "(KG_AUTH_TOKEN_REF, KG_IDENTITY_OAUTH2, or GRAPH_SERVICE_AUTH_SECRET)"
        )
    auth_source = auth_sources[0]
    _text(environment.get(auth_source), auth_source)

    tls_profiles = tuple(
        name
        for name in _TLS_PROFILE_SOURCES
        if str(environment.get(name, "") or "").strip()
    )
    if len(tls_profiles) > 1:
        raise LiveRuntimeContractError(
            "live workload must select one engine TLS profile source"
        )
    if tls_profiles:
        tls_source = tls_profiles[0]
        _text(environment.get(tls_source), tls_source)
        tls_mode = "profile"
        mtls = False
    else:
        # Production Swarm/Kubernetes cells may inject the verified CA and
        # server name directly rather than a named profile.  Treat this as a
        # complete alternative, never as permission to omit transport trust.
        for name in ("ENGINE_CA_BUNDLE", "ENGINE_TLS_SERVER_NAME"):
            _text(environment.get(name), name)
        has_cert = bool(str(environment.get("ENGINE_CLIENT_CERT", "") or "").strip())
        has_key = bool(str(environment.get("ENGINE_CLIENT_KEY", "") or "").strip())
        if has_cert != has_key:
            raise LiveRuntimeContractError(
                "live workload engine client certificate and key must be supplied together"
            )
        if has_cert:
            _text(environment.get("ENGINE_CLIENT_CERT"), "ENGINE_CLIENT_CERT")
            _text(environment.get("ENGINE_CLIENT_KEY"), "ENGINE_CLIENT_KEY")
        tls_source = "ENGINE_CA_BUNDLE"
        tls_mode = "explicit_bundle"
        mtls = has_cert
    payload = {
        "daemon_role": "client",
        "endpoints": sorted(endpoints),
        "auth_source": auth_source,
        "tls_source": tls_source,
        "tls_mode": tls_mode,
        "mtls": mtls,
    }
    return "sha256:" + hashlib.sha256(_canonical(payload)).hexdigest()


@dataclass(frozen=True)
class LiveRuntimeContract:
    """Validated live-run pins with privacy-safe evidence projection."""

    release_digest: str
    topology_digest: str
    image_digest: str
    contract_digest: str
    identity_digest: str
    engine_authority_digest: str
    source_authority_digest: str

    @classmethod
    def from_environment(
        cls,
        contract: WorkloadContract,
        *,
        release_digest: str | None = None,
        topology_digest: str | None = None,
        image_digest: str | None = None,
        contract_digest: str | None = None,
        tenant: str | None = None,
        principal: str | None = None,
        audience: str | None = None,
        environment: Mapping[str, str] | None = None,
    ) -> "LiveRuntimeContract":
        # An explicitly supplied empty mapping is a useful deterministic
        # fixture and must not accidentally inherit credentials or endpoints
        # from the invoking host.
        env = os.environ if environment is None else environment
        role = str(env.get("KG_DAEMON_ROLE", "") or "").strip().casefold()
        if role != "client":
            raise LiveRuntimeContractError(
                "live workload requires KG_DAEMON_ROLE=client; host election is forbidden"
            )
        endpoints = _endpoints(env)
        authority_digest = _authority_digest(endpoints, env)
        release = _env_value(env, "LOADGEN_RELEASE_DIGEST", release_digest)
        topology = _env_value(env, "LOADGEN_TOPOLOGY_DIGEST", topology_digest)
        image = _env_value(env, "LOADGEN_IMAGE_DIGEST", image_digest)
        workload = _env_value(
            env, "LOADGEN_WORKLOAD_CONTRACT_DIGEST", contract_digest
        )
        if workload != contract.contract_digest:
            raise LiveRuntimeContractError(
                "LOADGEN_WORKLOAD_CONTRACT_DIGEST does not match the loaded contract"
            )
        tenant_ref = _env_value(env, "LOADGEN_TENANT", tenant)
        principal_ref = _env_value(env, "LOADGEN_PRINCIPAL", principal)
        audience_ref = _env_value(env, "LOADGEN_AUDIENCE", audience)
        source_repository = _env_value(env, "LOADGEN_SOURCE_REPOSITORY", None)
        source_revision = _env_value(env, "LOADGEN_SOURCE_REVISION", None)
        source_manifest_digest = _env_value(
            env, "LOADGEN_SOURCE_MANIFEST_DIGEST", None
        )
        source_authority = _env_value(
            env, "LOADGEN_SOURCE_AUTHORITY_DIGEST", None
        )
        expected_source_authority = build_source_authority_digest(
            source_repository,
            source_revision,
            source_manifest_digest,
        )
        if source_authority != expected_source_authority:
            raise LiveRuntimeContractError(
                "LOADGEN_SOURCE_AUTHORITY_DIGEST does not match the pinned source"
            )
        identity_payload = {
            "tenant": tenant_ref,
            "principal": principal_ref,
            "audience": audience_ref,
        }
        identity_digest = "sha256:" + hashlib.sha256(
            _canonical(identity_payload)
        ).hexdigest()
        return cls(
            release_digest=_digest(release, "release digest"),
            topology_digest=_digest(topology, "topology digest"),
            image_digest=_digest(image, "image digest"),
            contract_digest=_digest(workload, "contract digest"),
            identity_digest=identity_digest,
            engine_authority_digest=authority_digest,
            source_authority_digest=_digest(
                source_authority, "source authority digest"
            ),
        )

    def as_report(self) -> dict[str, str]:
        """Return only opaque values suitable for the aggregate load report."""

        return {
            "mode": "live",
            "release_digest": self.release_digest,
            "topology_digest": self.topology_digest,
            "image_digest": self.image_digest,
            "contract_digest": self.contract_digest,
            "identity_digest": self.identity_digest,
            "engine_authority_digest": self.engine_authority_digest,
            "source_authority_digest": self.source_authority_digest,
        }


def mock_runtime(contract: WorkloadContract) -> dict[str, str]:
    """Mark the explicit CI path; it is never a production certification claim."""

    return {"mode": "mock", "contract_digest": contract.contract_digest}
