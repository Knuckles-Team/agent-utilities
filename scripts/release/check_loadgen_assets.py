#!/usr/bin/env python3
"""Fail-closed checker for rendered canonical loadgen deployment assets."""

from __future__ import annotations

import argparse
import json
import re
import stat
from pathlib import Path
from typing import Any

import yaml
from jsonschema import Draft202012Validator

from scripts.scale.loadgen_source_authority import source_authority_digest

_DIGEST = re.compile(r"^sha256:(?!0{64}$)[a-f0-9]{64}$")
_IMAGE = re.compile(r"^[a-z0-9][a-z0-9._/:-]*@sha256:[a-f0-9]{64}$")


def _token(name: str) -> str:
    return "$" + "{" + name + ":?required}"


_ALLOWED_COMPOSE_RUNTIME_TOKENS = frozenset(
    {
        _token("GRAPH_SERVICE_ENDPOINTS"),
        _token("LOADGEN_TENANT"),
        _token("LOADGEN_PRINCIPAL"),
        _token("LOADGEN_AUDIENCE"),
    }
)
_SECRET_REFS = {
    "graphos-loadgen-secrets",
    "loadgen-engine-auth",
    "loadgen-engine-tls",
    "loadgen-workload-identity",
}


class LoadgenAssetError(ValueError):
    """A rendered loadgen asset is incomplete or unsafe."""


def _regular(path: Path, *, maximum: int = 4 * 1024 * 1024) -> bytes:
    try:
        metadata = path.lstat()
        if (
            path.is_symlink()
            or not stat.S_ISREG(metadata.st_mode)
            or metadata.st_size > maximum
        ):
            raise LoadgenAssetError("loadgen asset is not a bounded regular file")
        return path.read_bytes()
    except LoadgenAssetError:
        raise
    except OSError as exc:
        raise LoadgenAssetError("loadgen asset is unavailable") from exc


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(_regular(path))
    except (UnicodeError, ValueError) as exc:
        raise LoadgenAssetError("loadgen metadata is invalid") from exc
    if not isinstance(value, dict):
        raise LoadgenAssetError("loadgen metadata must be an object")
    return value


def _yaml_documents(path: Path) -> list[dict[str, Any]]:
    try:
        values = list(yaml.safe_load_all(_regular(path).decode("utf-8")))
    except (UnicodeError, yaml.YAMLError) as exc:
        raise LoadgenAssetError("loadgen YAML is invalid") from exc
    documents = [value for value in values if value is not None]
    if not all(isinstance(value, dict) for value in documents):
        raise LoadgenAssetError("loadgen YAML documents must be objects")
    return documents


def _digest(value: object, field: str) -> str:
    text = str(value or "").strip().casefold()
    if _DIGEST.fullmatch(text) is None:
        raise LoadgenAssetError(f"{field} is not an exact digest")
    return text


def _image(value: object, field: str = "image") -> str:
    text = str(value or "").strip()
    if _IMAGE.fullmatch(text) is None:
        raise LoadgenAssetError(f"{field} is not immutable")
    digest = text.rsplit("@", 1)[1]
    if _DIGEST.fullmatch(digest) is None:
        raise LoadgenAssetError(f"{field} is not a non-zero digest")
    return text


def _reject_unresolved(value: object, *, compose: bool = False) -> None:
    if value is None:
        raise LoadgenAssetError("loadgen output is missing a required value")
    text = str(value)
    if "latest" in text.casefold() or "exact-digest-required" in text:
        raise LoadgenAssetError("loadgen output contains a mutable or sentinel value")
    if compose:
        unresolved = {
            token
            for token in re.findall(r"\$\{[^}]+\}", text)
            if token not in _ALLOWED_COMPOSE_RUNTIME_TOKENS
        }
    else:
        unresolved = set(re.findall(r"\$\{[^}]+\}", text))
    if unresolved:
        raise LoadgenAssetError("loadgen output retains required substitutions")


def _metadata(directory: Path) -> dict[str, Any]:
    metadata = _json(directory / "source-registration.json")
    schema = _json(directory / "source-registration.schema.json")
    try:
        Draft202012Validator(schema).validate(metadata)
    except Exception as exc:  # noqa: BLE001 - one stable gate boundary
        raise LoadgenAssetError(
            "loadgen source-registration metadata is invalid"
        ) from exc
    source = metadata["source"]
    try:
        expected = source_authority_digest(
            source["repository"],
            source["revision"],
            source["manifestDigest"],
        )
    except Exception as exc:  # noqa: BLE001 - path-free gate boundary
        raise LoadgenAssetError(
            "loadgen source-registration authority is invalid"
        ) from exc
    if source["authorityDigest"] != expected:
        raise LoadgenAssetError("loadgen source-registration authority drifted")
    image = _image(metadata["image"], "registered image")
    if metadata["bindings"]["imageDigest"] != image.rsplit("@", 1)[1]:
        raise LoadgenAssetError("loadgen image digest binding drifted")
    if set(metadata["secretRefs"]) != _SECRET_REFS:
        raise LoadgenAssetError("loadgen secret-reference set is not exact")
    return metadata


def _validate_outputs(directory: Path, metadata: dict[str, Any]) -> None:
    """Reject output bundles with omitted, extra, or linked files."""

    actual: list[str] = []
    try:
        for path in directory.rglob("*"):
            if path.is_symlink():
                raise LoadgenAssetError("loadgen output contains a symlink")
            metadata_entry = path.lstat()
            if stat.S_ISREG(metadata_entry.st_mode):
                actual.append(path.relative_to(directory).as_posix())
            elif not stat.S_ISDIR(metadata_entry.st_mode):
                raise LoadgenAssetError("loadgen output contains a non-file")
    except LoadgenAssetError:
        raise
    except (OSError, ValueError) as exc:
        raise LoadgenAssetError("loadgen output inventory is unavailable") from exc
    expected = metadata.get("outputs")
    if not isinstance(expected, list) or sorted(actual) != sorted(expected):
        raise LoadgenAssetError("loadgen output file set is not deterministic")


def _command(document: dict[str, Any]) -> list[str]:
    value = (document.get("spec") or {}).get("template", {})
    pod = value.get("spec") or {}
    container = (pod.get("containers") or [None])[0]
    if not isinstance(container, dict):
        raise LoadgenAssetError("loadgen Job has no container")
    command = container.get("command") or []
    args = container.get("args") or []
    if not isinstance(command, list) or not isinstance(args, list):
        raise LoadgenAssetError("loadgen Job command is invalid")
    return [str(item) for item in [*command, *args]]


def _production_compose(path: Path, metadata: dict[str, Any]) -> None:
    documents = _yaml_documents(path)
    if len(documents) != 1:
        raise LoadgenAssetError("production Compose must have one document")
    compose = documents[0]
    services = compose.get("services")
    if not isinstance(services, dict) or set(services) != {"certification-load"}:
        raise LoadgenAssetError("production Compose service set is not exact")
    service = services["certification-load"]
    if not isinstance(service, dict):
        raise LoadgenAssetError("production Compose service is invalid")
    _reject_unresolved(service.get("image"))
    if service["image"] != metadata["image"]:
        raise LoadgenAssetError("production Compose image drifted")
    if service.get("command") != metadata["commands"]["production"]:
        raise LoadgenAssetError("production Compose command drifted")
    if (service.get("labels") or {}).get(
        "agent-utilities.io/certification"
    ) != "production":
        raise LoadgenAssetError("production Compose certification label is absent")
    environment = service.get("environment") or {}
    if not isinstance(environment, dict):
        raise LoadgenAssetError("production Compose environment is invalid")
    required = {
        "KG_DAEMON_ROLE": "client",
        "KG_AUTH_TOKEN_REF": "secret://loadgen-engine-auth",
        "ENGINE_TLS_PROFILE_REF": "secret://loadgen-engine-tls",
        "LOADGEN_RELEASE_DIGEST": metadata["bindings"]["releaseDigest"],
        "LOADGEN_TOPOLOGY_DIGEST": metadata["bindings"]["topologyDigest"],
        "LOADGEN_IMAGE_DIGEST": metadata["bindings"]["imageDigest"],
        "LOADGEN_WORKLOAD_CONTRACT_DIGEST": metadata["bindings"]["contractDigest"],
        "LOADGEN_SOURCE_REPOSITORY": metadata["source"]["repository"],
        "LOADGEN_SOURCE_REVISION": metadata["source"]["revision"],
        "LOADGEN_SOURCE_MANIFEST_DIGEST": metadata["source"]["manifestDigest"],
        "LOADGEN_SOURCE_AUTHORITY_DIGEST": metadata["source"]["authorityDigest"],
    }
    runtime = {
        "GRAPH_SERVICE_ENDPOINTS",
        "LOADGEN_TENANT",
        "LOADGEN_PRINCIPAL",
        "LOADGEN_AUDIENCE",
    }
    if set(environment) != set(required) | runtime or any(
        environment.get(key) != value for key, value in required.items()
    ):
        raise LoadgenAssetError(
            "production Compose digest or authority binding drifted"
        )
    for key in (
        "GRAPH_SERVICE_ENDPOINTS",
        "LOADGEN_TENANT",
        "LOADGEN_PRINCIPAL",
        "LOADGEN_AUDIENCE",
    ):
        _reject_unresolved(environment.get(key), compose=True)
    secrets = compose.get("secrets") or {}
    if set(secrets) != _SECRET_REFS - {"graphos-loadgen-secrets"} or any(
        value != {"external": True} for value in secrets.values()
    ):
        raise LoadgenAssetError(
            "production Compose Secret declarations are not external"
        )
    mounts = service.get("secrets") or []
    expected_mounts = {
        name: {
            "source": name,
            "target": name,
            "uid": "10001",
            "gid": "10001",
            "mode": 0o400,
        }
        for name in _SECRET_REFS - {"graphos-loadgen-secrets"}
    }
    actual_mounts = {
        str(item.get("source")): item
        for item in mounts
        if isinstance(item, dict) and item.get("source")
    }
    if actual_mounts != expected_mounts:
        raise LoadgenAssetError("production Compose SecretRefs are incomplete")


def _production_kubernetes(path: Path, metadata: dict[str, Any]) -> None:
    documents = _yaml_documents(path)
    by_identity = {
        (
            str(document.get("kind")),
            str((document.get("metadata") or {}).get("name")),
        ): document
        for document in documents
    }
    if len(documents) != 3 or len(by_identity) != len(documents):
        raise LoadgenAssetError("production Kubernetes object set is not exact")
    config = by_identity.get(("ConfigMap", "graphos-loadgen-contract"))
    service_account = by_identity.get(("ServiceAccount", "graphos-certification-load"))
    job = by_identity.get(("Job", "graphos-certification-load"))
    if not config or not service_account or not job:
        raise LoadgenAssetError("production Kubernetes objects are incomplete")
    if config.get("immutable") is not True:
        raise LoadgenAssetError("production loadgen contract must be immutable")
    config_data = config.get("data") or {}
    expected_data = {
        "LOADGEN_RELEASE_DIGEST": metadata["bindings"]["releaseDigest"],
        "LOADGEN_TOPOLOGY_DIGEST": metadata["bindings"]["topologyDigest"],
        "LOADGEN_IMAGE_DIGEST": metadata["bindings"]["imageDigest"],
        "LOADGEN_WORKLOAD_CONTRACT_DIGEST": metadata["bindings"]["contractDigest"],
        "LOADGEN_DURATION_SECONDS": metadata["commands"]["production"][
            metadata["commands"]["production"].index("--duration-s") + 1
        ],
        "LOADGEN_SOURCE_REPOSITORY": metadata["source"]["repository"],
        "LOADGEN_SOURCE_REVISION": metadata["source"]["revision"],
        "LOADGEN_SOURCE_MANIFEST_DIGEST": metadata["source"]["manifestDigest"],
        "LOADGEN_SOURCE_AUTHORITY_DIGEST": metadata["source"]["authorityDigest"],
    }
    if set(config_data) != set(expected_data) or any(
        config_data.get(key) != value for key, value in expected_data.items()
    ):
        raise LoadgenAssetError("production Kubernetes contract binding drifted")
    if service_account.get("automountServiceAccountToken") is not False:
        raise LoadgenAssetError("production loadgen ServiceAccount is not explicit")
    pod = (job.get("spec") or {}).get("template", {}).get("spec") or {}
    containers = pod.get("containers") or []
    if not isinstance(containers, list) or len(containers) != 1:
        raise LoadgenAssetError("production Kubernetes Job container set is not exact")
    container = containers[0]
    if not isinstance(container, dict):
        raise LoadgenAssetError("production Kubernetes Job has no container")
    _reject_unresolved(container.get("image"))
    if container["image"] != metadata["image"]:
        raise LoadgenAssetError("production Kubernetes image drifted")
    if _command(job) != metadata["commands"]["production"]:
        raise LoadgenAssetError("production Kubernetes command drifted")
    labels = (job.get("metadata") or {}).get("labels") or {}
    if labels.get("agent-utilities.io/certification") != "production":
        raise LoadgenAssetError("production Kubernetes certification label is absent")
    if pod.get("serviceAccountName") != "graphos-certification-load":
        raise LoadgenAssetError("production Kubernetes workload identity is absent")
    if pod.get("envFrom") != [
        {
            "configMapRef": {
                "name": "graphos-loadgen-contract",
                "optional": False,
            }
        }
    ]:
        raise LoadgenAssetError("production Kubernetes contract reference is not exact")
    projected = [
        source
        for volume in pod.get("volumes") or []
        if isinstance(volume, dict)
        for source in ((volume.get("projected") or {}).get("sources") or [])
        if isinstance(source, dict) and "serviceAccountToken" in source
    ]
    if (
        not projected
        or projected[0]["serviceAccountToken"].get("audience")
        != metadata["bindings"]["workloadIdentityAudience"]
    ):
        raise LoadgenAssetError("production Kubernetes identity audience is not bound")
    expected_secret_bindings = {
        "GRAPH_SERVICE_ENDPOINTS": "GRAPH_SERVICE_ENDPOINTS",
        "KG_AUTH_TOKEN_REF": "KG_AUTH_TOKEN_REF",
        "ENGINE_TLS_PROFILE_REF": "ENGINE_TLS_PROFILE_REF",
        "LOADGEN_TENANT": "LOADGEN_TENANT",
        "LOADGEN_PRINCIPAL": "LOADGEN_PRINCIPAL",
        "LOADGEN_AUDIENCE": "LOADGEN_AUDIENCE",
    }
    expected_env = {
        "KG_DAEMON_ROLE",
        "WORKLOAD_IDENTITY_TOKEN_FILE",
        *expected_secret_bindings,
    }
    env = container.get("env") or []
    if (
        not isinstance(env, list)
        or len(env) != len(expected_env)
        or {str(entry.get("name")) for entry in env if isinstance(entry, dict)}
        != expected_env
    ):
        raise LoadgenAssetError("production Kubernetes environment is not exact")
    env_by_name = {
        entry["name"]: entry
        for entry in env
        if isinstance(entry, dict) and isinstance(entry.get("name"), str)
    }
    if env_by_name.get("KG_DAEMON_ROLE") != {
        "name": "KG_DAEMON_ROLE",
        "value": "client",
    } or env_by_name.get("WORKLOAD_IDENTITY_TOKEN_FILE") != {
        "name": "WORKLOAD_IDENTITY_TOKEN_FILE",
        "value": "/var/run/secrets/tokens/loadgen-token",
    }:
        raise LoadgenAssetError(
            "production Kubernetes runtime identity binding drifted"
        )
    actual_secret_bindings: dict[str, str] = {}
    for entry in env:
        if (
            not isinstance(entry, dict)
            or entry.get("name") not in expected_secret_bindings
        ):
            continue
        ref = (entry.get("valueFrom") or {}).get("secretKeyRef")
        if not isinstance(ref, dict) or ref.get("optional") is not False:
            raise LoadgenAssetError("production Kubernetes SecretRefs are optional")
        if (
            set(ref) != {"name", "key", "optional"}
            or ref.get("name") != "graphos-loadgen-secrets"
        ):
            raise LoadgenAssetError("production Kubernetes SecretRefs are not exact")
        actual_secret_bindings[entry["name"]] = str(ref.get("key"))
    if actual_secret_bindings != expected_secret_bindings:
        raise LoadgenAssetError("production Kubernetes SecretRefs are not exact")


def _kustomization(path: Path, metadata: dict[str, Any]) -> None:
    documents = _yaml_documents(path)
    if len(documents) != 1:
        raise LoadgenAssetError("loadgen Kustomization must have one document")
    value = documents[0]
    images = value.get("images") or []
    if len(images) != 1 or not isinstance(images[0], dict):
        raise LoadgenAssetError("loadgen Kustomization image binding is incomplete")
    image = images[0]
    _reject_unresolved(image.get("newName"))
    _reject_unresolved(image.get("digest"))
    if (
        image.get("name") != "loadgen-image"
        or image.get("newName") != metadata["image"].rsplit("@", 1)[0]
        or image.get("digest") != metadata["bindings"]["imageDigest"]
    ):
        raise LoadgenAssetError("loadgen Kustomization image drifted")
    if value.get("resources") != ["loadgen.yaml"]:
        raise LoadgenAssetError("loadgen Kustomization resource set is not exact")


def _mock_compose(path: Path, metadata: dict[str, Any]) -> None:
    documents = _yaml_documents(path)
    if len(documents) != 1:
        raise LoadgenAssetError("mock Compose must have one document")
    compose = documents[0]
    services = compose.get("services") or {}
    if not isinstance(services, dict) or set(services) != {"certification-load-mock"}:
        raise LoadgenAssetError("mock Compose service set is not exact")
    service = services.get("certification-load-mock")
    if not isinstance(service, dict):
        raise LoadgenAssetError("mock Compose service is absent")
    _reject_unresolved(service.get("image"))
    if (
        service["image"] != metadata["image"]
        or service.get("command") != metadata["commands"]["mock"]
    ):
        raise LoadgenAssetError("mock Compose command or image drifted")
    labels = service.get("labels") or {}
    if (
        labels.get("agent-utilities.io/certification") != "false"
        or labels.get("agent-utilities.io/non-certifying") != "true"
    ):
        raise LoadgenAssetError("mock Compose is not explicitly non-certifying")
    if service.get("secrets") or compose.get("secrets"):
        raise LoadgenAssetError("mock Compose must not carry live SecretRefs")
    if service.get("environment") != {
        "LOADGEN_MODE": "mock",
        "LOADGEN_WORKLOAD_CONTRACT_DIGEST": metadata["bindings"]["contractDigest"],
    }:
        raise LoadgenAssetError("mock Compose contract binding drifted")


def _mock_kubernetes(path: Path, metadata: dict[str, Any]) -> None:
    documents = _yaml_documents(path)
    identities = {
        (str(document.get("kind")), str((document.get("metadata") or {}).get("name")))
        for document in documents
    }
    if len(documents) != 2 or identities != {
        ("ConfigMap", "graphos-loadgen-mock-contract"),
        ("Job", "graphos-certification-load-mock"),
    }:
        raise LoadgenAssetError("mock Kubernetes object set is not exact")
    job = next(
        (
            document
            for document in documents
            if document.get("kind") == "Job"
            and (document.get("metadata") or {}).get("name")
            == "graphos-certification-load-mock"
        ),
        None,
    )
    if not isinstance(job, dict):
        raise LoadgenAssetError("mock Kubernetes Job is absent")
    pod = (job.get("spec") or {}).get("template", {}).get("spec") or {}
    containers = pod.get("containers") or []
    if not isinstance(containers, list) or len(containers) != 1:
        raise LoadgenAssetError("mock Kubernetes Job container set is not exact")
    container = containers[0]
    if not isinstance(container, dict):
        raise LoadgenAssetError("mock Kubernetes Job has no container")
    _reject_unresolved(container.get("image"))
    if (
        container["image"] != metadata["image"]
        or _command(job) != metadata["commands"]["mock"]
    ):
        raise LoadgenAssetError("mock Kubernetes command or image drifted")
    labels = (job.get("metadata") or {}).get("labels") or {}
    if (
        labels.get("agent-utilities.io/certification") != "false"
        or labels.get("agent-utilities.io/non-certifying") != "true"
    ):
        raise LoadgenAssetError("mock Kubernetes is not explicitly non-certifying")
    if any(
        isinstance(entry, dict) and entry.get("valueFrom", {}).get("secretKeyRef")
        for entry in container.get("env") or []
    ):
        raise LoadgenAssetError("mock Kubernetes must not carry live SecretRefs")
    if container.get("env"):
        raise LoadgenAssetError("mock Kubernetes must not carry runtime environment")
    if pod.get("envFrom") != [
        {
            "configMapRef": {
                "name": "graphos-loadgen-mock-contract",
                "optional": False,
            }
        }
    ]:
        raise LoadgenAssetError("mock Kubernetes contract reference is not exact")


def validate(directory: Path) -> dict[str, Any]:
    """Validate one newly rendered production + mock loadgen bundle."""

    if not directory.is_dir() or directory.is_symlink():
        raise LoadgenAssetError("loadgen output directory is invalid")
    metadata = _metadata(directory)
    _validate_outputs(directory, metadata)
    _production_compose(directory / "production/compose.yml", metadata)
    _kustomization(directory / "production/k8s/kustomization.yaml", metadata)
    _production_kubernetes(directory / "production/k8s/loadgen.yaml", metadata)
    _mock_compose(directory / "mock/compose.yml", metadata)
    _kustomization(directory / "mock/k8s/kustomization.yaml", metadata)
    _mock_kubernetes(directory / "mock/k8s/loadgen.yaml", metadata)
    return {
        "ok": True,
        "image": metadata["bindings"]["imageDigest"],
        "sourceAuthorityDigest": metadata["source"]["authorityDigest"],
        "outputs": metadata["outputs"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="check-graphos-loadgen-assets")
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = validate(args.directory)
    except Exception as exc:  # noqa: BLE001 - privacy-safe CLI boundary
        print(json.dumps({"ok": False, "error": type(exc).__name__}, sort_keys=True))
        return 1
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
