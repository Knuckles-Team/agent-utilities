#!/usr/bin/env python3
"""Fail-closed static contract for the GraphOS Docker Swarm profile.

The check is cluster-independent. It proves that the committed template uses
only external secret references, immutable-image indirection, authenticated
TLS boundaries, restricted containers, and bounded scheduling.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
from pathlib import Path
from typing import Any

import yaml


class SwarmAssetError(ValueError):
    """A required Swarm deployment invariant is absent or unsafe."""


_IMAGE_EXPRESSION = "${GRAPHOS_IMAGE_DIGEST:?set an immutable image digest}"
_RUNTIME_IMAGE = re.compile(r"^[a-z0-9][a-z0-9._/:-]*@sha256:[0-9a-f]{64}$")
_REQUIRED_POLICY = {
    "APP_PROFILE": "production",
    "AUTH_TYPE": "jwt",
    "MCP_TOOL_MODE": "intent",
    "MCP_CLIENT_AUTH": "oidc-client-credentials",
}
_REQUIRED_ENGINE_POLICY = {
    "APP_PROFILE": "production",
    "EPISTEMIC_GRAPH_REQUIRE_VERIFIED_CONTEXT": "1",
    "EPISTEMIC_GRAPH_REQUIRE_SIGNED": "1",
    "EPISTEMIC_GRAPH_RLS_DEFAULT_DENY": "1",
    "EPISTEMIC_GRAPH_PERSIST_BACKEND": "redb",
}
_FRONT_SECRET_TARGETS = {
    "FASTMCP_SERVER_AUTH_JWT_JWKS_URI",
    "OIDC_ISSUER",
    "FASTMCP_SERVER_AUTH_JWT_AUDIENCE",
    "MCP_ALLOWED_HOSTS",
    "MCP_TLS_SERVER_NAME",
    "GRAPH_SERVICE_ENDPOINTS",
    "ENGINE_TLS_SERVER_NAME",
    "GRAPH_SERVICE_AUTH_SECRET",
    "STATE_DB_URI",
    "GRAPH_DB_CONNECTION_PROFILE_REF",
    "KG_POLICY_VERSION",
    "PERSISTENCE_IDENTITY_HMAC_KEY",
    "OIDC_CLIENT_ID",
    "OIDC_CLIENT_SECRET",
    "OIDC_AUDIENCE",
    "OIDC_TOKEN_URL",
    "ca-bundle.pem",
    "front-tls.crt",
    "front-tls.key",
    "engine-client.crt",
    "engine-client.key",
}
_ENGINE_SECRET_TARGETS = {
    "GRAPH_SERVICE_AUTH_SECRET",
    "ENGINE_TLS_SERVER_NAME",
    "ca-bundle.pem",
    "engine-tls.crt",
    "engine-tls.key",
    "engine-client.crt",
    "engine-client.key",
}
_SECRET_ENV_KEY = re.compile(
    r"(?:SECRET|PASSWORD|TOKEN|PRIVATE_KEY|AUTH_KEY|HMAC_KEY)(?:_REF)?$",
    re.IGNORECASE,
)
_FORBIDDEN_SOURCE = (
    re.compile(r"\b(?:http|tcp)://", re.IGNORECASE),
    re.compile(r"(?:[A-Za-z]:\\|/home/|/Users/|/mnt/[a-z]/|file://)"),
    re.compile(r"\b(?:sk-[A-Za-z0-9]|pk-[A-Za-z0-9]|Bearer\s+)", re.IGNORECASE),
)


def _load(path: Path) -> tuple[str, dict[str, Any]]:
    if not path.is_file() or path.is_symlink():
        raise SwarmAssetError("Swarm stack must be one regular file")
    raw = path.read_text(encoding="utf-8")
    if len(raw.encode("utf-8")) > 512 * 1024:
        raise SwarmAssetError("Swarm stack exceeds the bounded source size")
    value = yaml.safe_load(raw)
    if not isinstance(value, dict):
        raise SwarmAssetError("Swarm stack must be a mapping")
    return raw, value


def _secret_mounts(service: dict[str, Any]) -> dict[str, dict[str, Any]]:
    mounts: dict[str, dict[str, Any]] = {}
    for value in service.get("secrets") or ():
        if not isinstance(value, dict):
            raise SwarmAssetError("all service secrets must use long syntax")
        source = str(value.get("source") or "")
        target = str(value.get("target") or "")
        if not source or not target or target in mounts:
            raise SwarmAssetError("service secret mount is incomplete or duplicated")
        if (
            str(value.get("uid")) != "10001"
            or str(value.get("gid")) != "10001"
            or value.get("mode") not in {0o400, 0o444}
        ):
            raise SwarmAssetError("service secret permissions are not restricted")
        mounts[target] = value
    return mounts


def _validate_service(name: str, service: dict[str, Any]) -> None:
    if service.get("image") != _IMAGE_EXPRESSION:
        raise SwarmAssetError(f"{name} image is not immutable-digest indirection")
    if (
        str(service.get("user")) != "10001:10001"
        or service.get("read_only") is not True
        or service.get("cap_drop") != ["ALL"]
        or "no-new-privileges:true" not in (service.get("security_opt") or ())
    ):
        raise SwarmAssetError(f"{name} container restrictions are incomplete")
    if service.get("networks") != ["graphos-internal"]:
        raise SwarmAssetError(f"{name} is attached to an unexpected network")
    tmpfs = service.get("tmpfs") or ()
    if not tmpfs or any(
        marker not in str(entry)
        for entry in tmpfs
        for marker in ("noexec", "nosuid", "nodev")
    ):
        raise SwarmAssetError(f"{name} temporary storage is not restricted")
    environment = service.get("environment") or {}
    if not isinstance(environment, dict):
        raise SwarmAssetError(f"{name} environment must be a mapping")
    for key, value in environment.items():
        rendered = str(value)
        if "${" in rendered:
            raise SwarmAssetError(f"{name} environment contains runtime interpolation")
        if _SECRET_ENV_KEY.search(str(key)) and not (
            rendered.startswith("/run/secrets/") or rendered.startswith("env://")
        ):
            raise SwarmAssetError(f"{name} places secret material in its environment")
    deploy = service.get("deploy") or {}
    resources = deploy.get("resources") or {}
    if not resources.get("reservations") or not resources.get("limits"):
        raise SwarmAssetError(f"{name} has no resource bounds")
    restart = deploy.get("restart_policy") or {}
    if (
        restart.get("condition") != "on-failure"
        or not isinstance(restart.get("max_attempts"), int)
        or not 1 <= restart["max_attempts"] <= 5
    ):
        raise SwarmAssetError(f"{name} restart policy is not bounded")
    update = deploy.get("update_config") or {}
    if update.get("parallelism") != 1 or update.get("failure_action") != "rollback":
        raise SwarmAssetError(f"{name} update policy does not fail closed")
    if not (deploy.get("placement") or {}).get("constraints"):
        raise SwarmAssetError(f"{name} placement is not constrained")
    health = " ".join(
        str(value) for value in (service.get("healthcheck") or {}).get("test", ())
    )
    if "ssl.create_default_context" not in health or "ca-bundle.pem" not in health:
        raise SwarmAssetError(f"{name} health check does not validate TLS")


def _validate_loaded(stack: dict[str, Any]) -> dict[str, Any]:
    services = stack.get("services")
    if not isinstance(services, dict) or set(services) != {"front", "engine"}:
        raise SwarmAssetError("Swarm stack must contain exactly front and engine")
    front = services["front"]
    engine = services["engine"]
    if not isinstance(front, dict) or not isinstance(engine, dict):
        raise SwarmAssetError("Swarm services must be mappings")
    _validate_service("front", front)
    _validate_service("engine", engine)

    front_environment = front.get("environment") or {}
    for key, expected in _REQUIRED_POLICY.items():
        if str(front_environment.get(key)) != expected:
            raise SwarmAssetError(f"front policy must set {key}={expected}")
    engine_environment = engine.get("environment") or {}
    for key, expected in _REQUIRED_ENGINE_POLICY.items():
        if str(engine_environment.get(key)) != expected:
            raise SwarmAssetError(f"engine policy must set {key}={expected}")

    front_command = " ".join(str(value) for value in front.get("command") or ())
    if (
        "--tls-certfile" not in front_command
        or "--tls-keyfile" not in front_command
        or front_environment.get("AUTH_TYPE") != "jwt"
    ):
        raise SwarmAssetError("front service lacks direct TLS or JWT authentication")
    engine_command = " ".join(str(value) for value in engine.get("command") or ())
    if any(
        flag not in engine_command
        for flag in ("--tcp-tls-cert", "--tcp-tls-key", "--tcp-tls-client-ca")
    ):
        raise SwarmAssetError("native engine mTLS is incomplete")
    if (
        "/run/secrets/GRAPH_SERVICE_AUTH_SECRET" not in engine_command
        or "GRAPH_SERVICE_AUTH_SECRET=" not in engine_command
    ):
        raise SwarmAssetError("native engine session authentication is absent")

    front_mounts = _secret_mounts(front)
    engine_mounts = _secret_mounts(engine)
    if set(front_mounts) != _FRONT_SECRET_TARGETS:
        raise SwarmAssetError("front external-secret contract is incomplete")
    if set(engine_mounts) != _ENGINE_SECRET_TARGETS:
        raise SwarmAssetError("engine external-secret contract is incomplete")
    engine_health = " ".join(
        str(value) for value in (engine.get("healthcheck") or {}).get("test", ())
    )
    if (
        "load_cert_chain" not in engine_health
        or "engine-client.crt" not in engine_health
        or "engine-client.key" not in engine_health
    ):
        raise SwarmAssetError("native engine health check does not prove mTLS")

    declarations = stack.get("secrets")
    if not isinstance(declarations, dict):
        raise SwarmAssetError("external secret declarations are absent")
    sources = {
        str(value["source"])
        for mounts in (front_mounts, engine_mounts)
        for value in mounts.values()
    }
    if set(declarations) != sources or any(
        value != {"external": True} for value in declarations.values()
    ):
        raise SwarmAssetError("all and only mounted secrets must be external")
    network = (stack.get("networks") or {}).get("graphos-internal") or {}
    if (
        network.get("driver") != "overlay"
        or network.get("attachable") is not False
        or "encrypted" not in (network.get("driver_opts") or {})
    ):
        raise SwarmAssetError("Swarm network must be encrypted and non-attachable")
    volume = (stack.get("volumes") or {}).get("engine-data")
    if not isinstance(volume, dict) or volume.get("external") is not True:
        raise SwarmAssetError("engine persistence must use an external volume")
    if engine.get("ports"):
        raise SwarmAssetError("native engine must not publish a host port")
    return {
        "ok": True,
        "services": sorted(services),
        "externalSecrets": len(declarations),
    }


def validate(path: Path) -> dict[str, Any]:
    """Validate a source stack and return a privacy-safe summary."""

    raw, stack = _load(path)
    if any(pattern.search(raw) for pattern in _FORBIDDEN_SOURCE):
        raise SwarmAssetError("Swarm stack contains identifying or plaintext data")
    return _validate_loaded(stack)


def validate_runtime_image(value: str) -> None:
    """Reject tags, mutable names, and malformed digest references."""

    rendered = str(value or "").strip()
    if len(rendered) > 512 or _RUNTIME_IMAGE.fullmatch(rendered) is None:
        raise SwarmAssetError("runtime image must be an immutable sha256 reference")


def self_check(path: Path) -> None:
    """Prove the checker rejects representative security regressions."""

    _, source = _load(path)
    mutations: list[dict[str, Any]] = []
    plaintext = copy.deepcopy(source)
    plaintext["services"]["front"]["environment"]["OIDC_ISSUER"] = (
        "http://identity.invalid"
    )
    mutations.append(plaintext)
    privileged = copy.deepcopy(source)
    privileged["services"]["front"]["cap_drop"] = []
    mutations.append(privileged)
    inline_secret = copy.deepcopy(source)
    inline_secret["services"]["front"]["environment"]["OIDC_CLIENT_SECRET"] = (
        "must-not-appear"
    )
    mutations.append(inline_secret)
    for mutation in mutations:
        try:
            _validate_loaded(mutation)
            values = mutation["services"]["front"].get("environment") or {}
            if any(
                re.search(r"\b(?:http|tcp)://", str(value), re.IGNORECASE)
                for value in values.values()
            ):
                raise SwarmAssetError("plaintext endpoint is forbidden")
        except SwarmAssetError:
            continue
        raise SwarmAssetError("Swarm gate self-check did not reject a mutation")
    try:
        validate_runtime_image("graphos:latest")
    except SwarmAssetError:
        pass
    else:
        raise SwarmAssetError("Swarm gate accepted a mutable runtime image")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="check-graphos-swarm-assets")
    parser.add_argument(
        "--stack",
        type=Path,
        default=Path("deploy/swarm/graphos.stack.yml"),
    )
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument(
        "--runtime-image",
        help="Rendered image reference to require as name@sha256:<64 lowercase hex>",
    )
    args = parser.parse_args(argv)
    try:
        report = validate(args.stack)
        if args.self_check:
            self_check(args.stack)
            report["selfCheck"] = True
        if args.runtime_image is not None:
            validate_runtime_image(args.runtime_image)
            report["runtimeImage"] = True
    except Exception as exc:  # noqa: BLE001 - one privacy-safe CLI boundary
        print(json.dumps({"ok": False, "error": type(exc).__name__}, sort_keys=True))
        return 1
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
