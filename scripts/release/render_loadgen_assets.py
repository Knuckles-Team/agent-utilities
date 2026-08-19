#!/usr/bin/env python3
"""Render canonical loadgen deployment templates from immutable inputs.

This follows the same source-template -> rendered-output boundary as
render_production_cell.py. The committed templates contain no image, secret,
or operator identity values. Rendering is the only path that can produce
deployment-shaped Compose/Kubernetes output, and it refuses mutable images,
missing bindings, source-authority drift, and output reuse.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import stat
from dataclasses import dataclass
from pathlib import Path

from scripts.scale.loadgen_source_authority import (
    LoadgenSourceAuthorityError,
    source_authority_digest,
)

_ROOT = Path(__file__).resolve().parents[2]
_TEMPLATE_ROOT = _ROOT / "deploy" / "loadgen"
_DIGEST = re.compile(r"^sha256:(?!0{64}$)[a-f0-9]{64}$")
_REVISION = re.compile(r"^[a-f0-9]{40}$")
_IMAGE = re.compile(r"^[a-z0-9][a-z0-9._/:-]*@sha256:[a-f0-9]{64}$")
_MAX_TEXT_BYTES = 4096
_MAX_TEMPLATE_BYTES = 4 * 1024 * 1024


def _token(name: str, suffix: str) -> str:
    return "$" + "{" + name + suffix + "}"


_PRODUCTION_ALLOWED_RUNTIME_TOKENS = frozenset(
    {
        _token("GRAPH_SERVICE_ENDPOINTS", ":?required"),
        _token("LOADGEN_TENANT", ":?required"),
        _token("LOADGEN_PRINCIPAL", ":?required"),
        _token("LOADGEN_AUDIENCE", ":?required"),
    }
)
_TEMPLATE_FILES = (
    "compose.yml",
    "mock.compose.yml",
    "k8s/kustomization.yaml",
    "k8s/loadgen.yaml",
    "k8s/mock/kustomization.yaml",
    "k8s/mock/loadgen.yaml",
    "source-registration.schema.json",
)


class LoadgenRenderError(ValueError):
    """The canonical loadgen source cannot be rendered safely."""


def _text(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise LoadgenRenderError(f"{field} is missing or invalid")
    text = value.strip()
    if (
        not text
        or len(text.encode("utf-8")) > _MAX_TEXT_BYTES
        or any(character in text for character in "\x00\r\n")
    ):
        raise LoadgenRenderError(f"{field} is missing or invalid")
    return text


def _digest(value: object, field: str) -> str:
    text = _text(value, field).casefold()
    if _DIGEST.fullmatch(text) is None:
        raise LoadgenRenderError(f"{field} must be a non-zero sha256 digest")
    return text


def _image(value: object) -> tuple[str, str]:
    image = _text(value, "image")
    if _IMAGE.fullmatch(image) is None:
        raise LoadgenRenderError("image must be repository@sha256:<64 lowercase hex>")
    image_digest = "sha256:" + image.rsplit("@sha256:", 1)[1]
    if _DIGEST.fullmatch(image_digest) is None:
        raise LoadgenRenderError("image must carry a non-zero digest")
    return image, image_digest


def _revision(value: object) -> str:
    revision = _text(value, "source revision").casefold()
    if _REVISION.fullmatch(revision) is None:
        raise LoadgenRenderError("source revision must be a full commit")
    return revision


def _directory_is_empty(path: Path) -> None:
    if path.exists() or path.is_symlink():
        try:
            metadata = path.lstat()
        except OSError as exc:
            raise LoadgenRenderError("render output is unavailable") from exc
        if path.is_symlink() or not stat.S_ISDIR(metadata.st_mode):
            raise LoadgenRenderError("render output must be a new directory")
        try:
            if any(path.iterdir()):
                raise LoadgenRenderError("render output must be empty")
        except OSError as exc:
            raise LoadgenRenderError("render output is unavailable") from exc
    else:
        try:
            path.mkdir(parents=True)
        except OSError as exc:
            raise LoadgenRenderError("render output is unavailable") from exc


@dataclass(frozen=True)
class LoadgenRenderInputs:
    image: str
    image_digest: str
    release_digest: str
    topology_digest: str
    contract_digest: str
    source_repository: str
    source_revision: str
    source_manifest_digest: str
    source_authority_digest: str
    workload_identity_audience: str
    duration_seconds: int

    @classmethod
    def from_values(
        cls,
        *,
        image: str,
        release_digest: str,
        topology_digest: str,
        contract_digest: str,
        source_repository: str,
        source_revision: str,
        source_manifest_digest: str,
        source_authority_digest_value: str,
        workload_identity_audience: str,
        duration_seconds: int,
    ) -> "LoadgenRenderInputs":
        image_ref, image_digest = _image(image)
        source_revision_value = _revision(source_revision)
        source_manifest = _digest(source_manifest_digest, "source manifest digest")
        try:
            expected_authority = source_authority_digest(
                source_repository,
                source_revision_value,
                source_manifest,
            )
        except LoadgenSourceAuthorityError as exc:
            raise LoadgenRenderError("source authority inputs are invalid") from exc
        authority = _digest(
            source_authority_digest_value,
            "source authority digest",
        )
        if authority != expected_authority:
            raise LoadgenRenderError("source authority digest does not match pins")
        if type(duration_seconds) is not int or not 86_400 <= duration_seconds <= 259_200:
            raise LoadgenRenderError("duration must be a bounded 24-72 hour campaign")
        return cls(
            image=image_ref,
            image_digest=image_digest,
            release_digest=_digest(release_digest, "release digest"),
            topology_digest=_digest(topology_digest, "topology digest"),
            contract_digest=_digest(contract_digest, "contract digest"),
            source_repository=_text(source_repository, "source repository"),
            source_revision=source_revision_value,
            source_manifest_digest=source_manifest,
            source_authority_digest=authority,
            workload_identity_audience=_text(
                workload_identity_audience,
                "workload identity audience",
            ),
            duration_seconds=duration_seconds,
        )


def _template_digest() -> str:
    digest = hashlib.sha256()
    for relative in _TEMPLATE_FILES:
        path = _TEMPLATE_ROOT / relative
        try:
            metadata = path.lstat()
            if (
                path.is_symlink()
                or not stat.S_ISREG(metadata.st_mode)
                or metadata.st_size > _MAX_TEMPLATE_BYTES
            ):
                raise LoadgenRenderError("loadgen template is not a bounded regular file")
            payload = path.read_bytes()
        except LoadgenRenderError:
            raise
        except OSError as exc:
            raise LoadgenRenderError("loadgen template is unavailable") from exc
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(payload)
        digest.update(b"\0")
    return "sha256:" + digest.hexdigest()


def _replace_tokens(source: str, values: dict[str, str]) -> str:
    rendered = source
    for token, value in values.items():
        rendered = rendered.replace(token, value)
    unresolved = {
        token
        for token in re.findall(r"\$\{[^}]+\}", rendered)
        if token not in _PRODUCTION_ALLOWED_RUNTIME_TOKENS
    }
    if unresolved:
        raise LoadgenRenderError("rendered loadgen output retains required substitutions")
    return rendered


def _write_template(
    relative: str,
    destination: Path,
    values: dict[str, str],
) -> None:
    source = _TEMPLATE_ROOT / relative
    try:
        metadata = source.lstat()
        if (
            source.is_symlink()
            or not stat.S_ISREG(metadata.st_mode)
            or metadata.st_size > _MAX_TEMPLATE_BYTES
        ):
            raise LoadgenRenderError("loadgen template is not a bounded regular file")
        content = source.read_text(encoding="utf-8")
    except LoadgenRenderError:
        raise
    except (OSError, UnicodeError) as exc:
        raise LoadgenRenderError("loadgen template is unavailable") from exc
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        _replace_tokens(content, values),
        encoding="utf-8",
    )


def _values(inputs: LoadgenRenderInputs) -> dict[str, str]:
    return {
        _token("LOADGEN_IMAGE", ":?rendered immutable image required"): inputs.image,
        _token("LOADGEN_IMAGE_REPOSITORY", ":?required"): inputs.image.rsplit("@", 1)[0],
        _token("LOADGEN_IMAGE_DIGEST", ":?rendered image digest required"): inputs.image_digest,
        _token("LOADGEN_RELEASE_DIGEST", ":?rendered release digest required"): inputs.release_digest,
        _token("LOADGEN_TOPOLOGY_DIGEST", ":?rendered topology digest required"): inputs.topology_digest,
        _token("LOADGEN_WORKLOAD_CONTRACT_DIGEST", ":?rendered contract digest required"): inputs.contract_digest,
        _token("LOADGEN_SOURCE_REPOSITORY", ":?rendered source repository required"): inputs.source_repository,
        _token("LOADGEN_SOURCE_REVISION", ":?rendered source revision required"): inputs.source_revision,
        _token("LOADGEN_SOURCE_MANIFEST_DIGEST", ":?rendered source manifest digest required"): inputs.source_manifest_digest,
        _token("LOADGEN_SOURCE_AUTHORITY_DIGEST", ":?rendered source authority digest required"): inputs.source_authority_digest,
        _token("LOADGEN_WORKLOAD_IDENTITY_AUDIENCE", ":?required"): inputs.workload_identity_audience,
        _token("LOADGEN_DURATION_SECONDS", ":?rendered duration required"): str(inputs.duration_seconds),
        "$(LOADGEN_DURATION_SECONDS)": str(inputs.duration_seconds),
        "$(LOADGEN_RELEASE_DIGEST)": inputs.release_digest,
    }


def _commands(inputs: LoadgenRenderInputs) -> dict[str, list[str]]:
    return {
        "production": [
            "graphos-certification-load",
            "--engine",
            "live",
            "--scale",
            "1.0",
            "--duration-s",
            str(inputs.duration_seconds),
            "--report-json",
            "/var/run/loadgen/load-report.json",
            "--release-digest",
            inputs.release_digest,
        ],
        "mock": [
            "graphos-certification-load",
            "--engine",
            "mock",
            "--scale",
            "0.001",
            "--duration-s",
            "5",
            "--report-json",
            "/var/run/loadgen/mock-report.json",
        ],
    }


def render(
    *,
    output_dir: Path,
    inputs: LoadgenRenderInputs,
) -> dict[str, object]:
    if not _TEMPLATE_ROOT.is_dir() or _TEMPLATE_ROOT.is_symlink():
        raise LoadgenRenderError("loadgen template root is unavailable")
    template_digest = _template_digest()
    _directory_is_empty(output_dir)
    values = _values(inputs)
    _write_template("compose.yml", output_dir / "production/compose.yml", values)
    _write_template(
        "k8s/kustomization.yaml",
        output_dir / "production/k8s/kustomization.yaml",
        values,
    )
    _write_template("k8s/loadgen.yaml", output_dir / "production/k8s/loadgen.yaml", values)
    _write_template("mock.compose.yml", output_dir / "mock/compose.yml", values)
    _write_template(
        "k8s/mock/kustomization.yaml",
        output_dir / "mock/k8s/kustomization.yaml",
        values,
    )
    _write_template(
        "k8s/mock/loadgen.yaml",
        output_dir / "mock/k8s/loadgen.yaml",
        values,
    )
    schema_source = _TEMPLATE_ROOT / "source-registration.schema.json"
    (output_dir / "source-registration.schema.json").write_bytes(
        schema_source.read_bytes()
    )
    commands = _commands(inputs)
    metadata = {
        "apiVersion": "agent-utilities.io/v1",
        "kind": "LoadgenDeploymentSource",
        "mode": "production",
        "generator": "scripts/release/render_loadgen_assets.py",
        "templateDigest": template_digest,
        "source": {
            "repository": inputs.source_repository,
            "revision": inputs.source_revision,
            "manifestDigest": inputs.source_manifest_digest,
            "authorityDigest": inputs.source_authority_digest,
        },
        "image": inputs.image,
        "bindings": {
            "releaseDigest": inputs.release_digest,
            "topologyDigest": inputs.topology_digest,
            "contractDigest": inputs.contract_digest,
            "imageDigest": inputs.image_digest,
            "workloadIdentityAudience": inputs.workload_identity_audience,
        },
        "secretRefs": [
            "graphos-loadgen-secrets",
            "loadgen-engine-auth",
            "loadgen-engine-tls",
            "loadgen-workload-identity",
        ],
        "commands": commands,
        "outputs": [
            "production/compose.yml",
            "production/k8s/kustomization.yaml",
            "production/k8s/loadgen.yaml",
            "mock/compose.yml",
            "mock/k8s/kustomization.yaml",
            "mock/k8s/loadgen.yaml",
            "source-registration.json",
            "source-registration.schema.json",
        ],
    }
    (output_dir / "source-registration.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "ok": True,
        "templateDigest": metadata["templateDigest"],
        "sourceAuthorityDigest": inputs.source_authority_digest,
        "outputs": metadata["outputs"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="render-graphos-loadgen-assets")
    parser.add_argument("--image", required=True)
    parser.add_argument("--release-digest", required=True)
    parser.add_argument("--topology-digest", required=True)
    parser.add_argument("--contract-digest", required=True)
    parser.add_argument("--source-repository", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--source-manifest-digest", required=True)
    parser.add_argument("--source-authority-digest", required=True)
    parser.add_argument("--workload-identity-audience", required=True)
    parser.add_argument("--duration-s", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        inputs = LoadgenRenderInputs.from_values(
            image=args.image,
            release_digest=args.release_digest,
            topology_digest=args.topology_digest,
            contract_digest=args.contract_digest,
            source_repository=args.source_repository,
            source_revision=args.source_revision,
            source_manifest_digest=args.source_manifest_digest,
            source_authority_digest_value=args.source_authority_digest,
            workload_identity_audience=args.workload_identity_audience,
            duration_seconds=args.duration_s,
        )
        report = render(output_dir=args.output, inputs=inputs)
    except Exception as exc:  # noqa: BLE001 - privacy-safe CLI boundary
        print(json.dumps({"ok": False, "error": type(exc).__name__}, sort_keys=True))
        return 1
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
